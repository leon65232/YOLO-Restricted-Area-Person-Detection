# save as app.py (or test.py) and run with: streamlit run app.py

import os
import urllib.request
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

# REMOVED the base64 and io imports

# ---------------------------
# Detect device (GPU if available)
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# UI header
st.title("👀 YOLOv11 Person Detection (Streamlit + WebRTC)")
st.write(f"⚡ This is being run on **{device.upper()}**")

# REMOVED the image_to_data_url function

# ---------------------------
# Model download / load
# ---------------------------
MODEL_PATH = "yolov11n.pt"
# Using a more direct URL that doesn't rely on resolves
MODEL_URL = "https://huggingface.co/Ultralytics/YOLOv11/resolve/main/yolov11n.pt?download=true"


if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLOv11 model..."):
        try:
            # Add headers to mimic a browser request
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()


# Load model and move to device
model = YOLO(MODEL_PATH).to(device)

# ---------------------------
# Video transformer (stores last frame)
# ---------------------------
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.last_frame = None  # store last BGR frame

    def transform(self, frame):
        # frame -> numpy BGR
        img = frame.to_ndarray(format="bgr24")
        # store a copy for capture
        self.last_frame = img.copy()

        # Run inference on live frame for overlay (optional)
        try:
            results = model.predict(img, device=device, verbose=False)
            # Draw only "person" boxes for live overlay
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    if name == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{name} {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception:
            pass

        return img

# start webrtc streamer
webrtc_ctx = webrtc_streamer(
    key="person-detection",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

st.write("---")

# ---------------------------
# Capture frame UI
# ---------------------------
col1, col2 = st.columns([1, 1])
with col1:
    captured = st.button("📸 Capture current frame")

with col2:
    st.write("Instructions:")
    st.markdown(
        "- Click **Capture current frame** to grab an image from the stream.\n"
        "- Draw a rectangle on the captured image and resize/move corners as needed.\n"
        "- If any detected person's centroid is inside your rectangle, a red warning will appear."
    )

if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
    
if captured:
    if webrtc_ctx and webrtc_ctx.video_transformer:
        st.session_state.captured_image = webrtc_ctx.video_transformer.last_frame
    else:
        st.warning("Camera not running or transformer not ready.")


if st.session_state.captured_image is not None:
    frame = st.session_state.captured_image
    
    captured_bgr = frame.copy()
    captured_rgb = cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(captured_rgb)
    
    # REMOVED the call to image_to_data_url

    st.subheader("Captured frame — draw a rectangle (restricted area)")
    # --- MODIFIED: Pass the PIL image object directly ---
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=pil_img, # Pass the PIL image object
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="rect",
        key="canvas",
        display_toolbar=True,
    )
    
    restricted_box = None
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        objs = canvas_result.json_data["objects"]
        for o in objs:
            if o.get("type") == "rect":
                left, top, width, height = o.get("left", 0), o.get("top", 0), o.get("width", 0), o.get("height", 0)
                restricted_box = (int(left), int(top), int(left + width), int(top + height))
                break 

    if restricted_box is not None:
        x1_r, y1_r, x2_r, y2_r = restricted_box
        results = model.predict(captured_bgr, device=device, verbose=False)

        disp = captured_bgr.copy()
        person_in_restricted = False
        persons_detected = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                if name == "person":
                    persons_detected += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    if (x1_r <= cx <= x2_r) and (y1_r <= cy <= y2_r):
                        person_in_restricted = True
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(disp, "INSIDE", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.rectangle(disp, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 3)

        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        st.image(disp_rgb, caption=f"Detections (persons: {persons_detected})", use_column_width=True)

        if person_in_restricted:
            st.error("Person detected in restricted area")
        else:
            st.success("No person detected inside the restricted area.")
    else:
        st.info("Draw a rectangle on the image to define the restricted area.")