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
import time

# ---------------------------
# Detect device (GPU if available)
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# UI header
st.title("👀 YOLOv11 Person Detection")
st.write(f"⚡ This is being run on **{device.upper()}**")

# ---------------------------
# Model download / load
# ---------------------------
MODEL_PATH = "yolov11n.pt"
MODEL_URL = "https://huggingface.co/Ultralytics/YOLOv11/resolve/main/yolov11n.pt?download=true"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLOv11 model..."):
        try:
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
# Helper: point-in-polygon (ray-casting)
# ---------------------------
def point_in_polygon(x, y, polygon_pts):
    inside = False
    n = len(polygon_pts)
    if n < 3: return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_pts[i]
        xj, yj = polygon_pts[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

# ---------------------------
# Video transformer
# ---------------------------
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.last_frame = None
        self.restricted_shape = None
        self.person_in_restricted = False
        self.detection_strategy = "Feet (Bottom-Center)"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        person_found = False

        try:
            results = model.predict(img, device=device, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    if name == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        
                        if self.detection_strategy == "Feet (Bottom-Center)":
                            cx, cy = (x1 + x2) / 2.0, y2
                        elif self.detection_strategy == "Head (Top-Center)":
                            cx, cy = (x1 + x2) / 2.0, y1
                        elif self.detection_strategy == "Center (Middle)":
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        elif self.detection_strategy == "Left (Mid-Left)":
                            cx, cy = x1, (y1 + y2) / 2.0
                        elif self.detection_strategy == "Right (Mid-Right)":
                            cx, cy = x2, (y1 + y2) / 2.0
                        else:
                            cx, cy = (x1 + x2) / 2.0, y2
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw the Detection Point (Yellow Dot)
                        cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 255), -1)

                        # Check collision
                        is_inside = False
                        if self.restricted_shape is not None:
                            if self.restricted_shape[0] == "rect":
                                rx1, ry1, rx2, ry2 = self.restricted_shape[1]
                                if (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2):
                                    is_inside = True
                            elif self.restricted_shape[0] == "poly":
                                poly = self.restricted_shape[1]
                                if point_in_polygon(cx, cy, poly):
                                    is_inside = True
                        
                        if is_inside:
                            person_found = True
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                            cv2.putText(img, f"INSIDE {conf:.2f}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw restricted shape
            if self.restricted_shape is not None:
                if self.restricted_shape[0] == "rect":
                    rx1, ry1, rx2, ry2 = map(int, self.restricted_shape[1])
                    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
                elif self.restricted_shape[0] == "poly":
                    poly = [(int(x), int(y)) for (x, y) in self.restricted_shape[1]]
                    cv2.polylines(img, [np.array(poly)], isClosed=True, color=(255, 0, 0), thickness=3)

        except Exception:
            pass

        self.person_in_restricted = bool(person_found)
        return img

# ---------------------------
# Webrtc Streamer
# ---------------------------
webrtc_ctx = webrtc_streamer(
    key="person-detection",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
status_placeholder = st.empty()

st.write("---")

# ---------------------------
# Capture & Draw UI
# ---------------------------
col1, col2 = st.columns([1, 1])
with col1:
    captured = st.button("📸 Capture current frame")

with col2:
    st.write("**Instructions:**")
    st.markdown(
        "- Click **Capture current frame** to grab an image from the stream.  \n"
        "- Draw a **Rectangle** or **Polygon** on the image. \n"
        "- Click **Use ... as restricted area**."
    )

# ---------------------------
# Dropdowns (Drawing Mode & Detection Logic)
# ---------------------------
c_mode1, c_mode2 = st.columns(2)

with c_mode1:
    drawing_mode = st.selectbox("Drawing mode", ["rect", "polygon"], index=1)

with c_mode2:
    detection_mode = st.selectbox(
        "Detection Point Logic",
        ["Feet (Bottom-Center)", "Head (Top-Center)", "Center (Middle)", "Left (Mid-Left)", "Right (Mid-Right)"],
        index=0,
        help="Select which part of the person (yellow dot) triggers the alarm. if its (mid, left), that means that it will be in the middle of axis X and at the left of axis Y."
    )

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "restricted_shape" not in st.session_state:
    st.session_state.restricted_shape = None

# capture last frame when user clicks
if captured:
    if webrtc_ctx and webrtc_ctx.video_transformer:
        frame = webrtc_ctx.video_transformer.last_frame
        if frame is None:
            st.warning("No frame available yet.")
        else:
            st.session_state.captured_image = frame.copy()
            st.success("Frame captured!")
    else:
        st.warning("Camera not running.")

# Extract polygon points from canvas object
def extract_polygon_points(obj):
    pts = obj.get("points") or obj.get("path") or obj.get("polylinePoints") or None
    if pts:
        if isinstance(pts, list) and isinstance(pts[0], dict):
            return [(float(p.get("x", 0)), float(p.get("y", 0))) for p in pts]
    path = obj.get("path")
    if path and isinstance(path, list):
        out = []
        for cmd in path:
            if isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
                if str(cmd[0]).upper() in ("M", "L"):
                    out.append((float(cmd[1]), float(cmd[2])))
        if len(out) > 0: return out
    return None

# show canvas to draw restricted shape
if st.session_state.captured_image is not None:
    frame = st.session_state.captured_image
    captured_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(captured_rgb)

    st.subheader(
        "Define Restricted Area",
        anchor=False,
        help="Left click to add a point, right click to finalize the polygon."
        )
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=pil_img,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode=drawing_mode,
        key="canvas",
        display_toolbar=True,
    )

    restricted_rect = None
    restricted_poly = None

    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        for o in canvas_result.json_data["objects"]:
            t = o.get("type", "").lower()
            if t == "rect" and drawing_mode == "rect":
                left = o.get("left", 0)
                top = o.get("top", 0)
                width = o.get("width", 0)
                height = o.get("height", 0)
                restricted_rect = (int(left), int(top), int(left + width), int(top + height))
            elif drawing_mode == "polygon" and t == "path":
                pts = extract_polygon_points(o)
                if pts and len(pts) >= 3:
                    restricted_poly = pts

    if restricted_poly is not None:
        if st.button("Use current polygon as restricted area"):
            st.session_state.restricted_shape = ("poly", restricted_poly)
            if webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = ("poly", restricted_poly)
            st.success("Polygon applied to live feed.")

    elif restricted_rect is not None:
        if st.button("Use this Rectangle"):
            st.session_state.restricted_shape = ("rect", restricted_rect)
            if webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = ("rect", restricted_rect)
            st.success("Rectangle Applied!")
            
    if st.button("Clear Area"):
        st.session_state.restricted_shape = None
        if webrtc_ctx.video_transformer:
            webrtc_ctx.video_transformer.restricted_shape = None

# ---------------------------
# Status Updater Loop
# ---------------------------
if webrtc_ctx.state.playing:
    try:
        while True:
            if webrtc_ctx.video_transformer:
                # Update Strategy Live
                webrtc_ctx.video_transformer.detection_strategy = detection_mode
                
                flag = webrtc_ctx.video_transformer.person_in_restricted
                if flag:
                    status_placeholder.error("🚨 ALERT: Person INSIDE restricted area!")
                else:
                    status_placeholder.success("✅ Secure: No person detected inside.")
            time.sleep(0.2) 
    except Exception:
        pass