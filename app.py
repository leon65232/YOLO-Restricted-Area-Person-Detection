import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
import time
import tempfile

device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("👀 YOLOv8 Person Detection")
st.write(f"⚡ This is being run on **{device.upper()}**")



def reset_settings():
    st.session_state.captured_image = None
    st.session_state.restricted_shape = None
    st.session_state.canvas_key = f"canvas_{time.time()}"
source_option = st.selectbox(
    "Select Input Source",
    ["Realtime Webcam", "Video Upload"],
    index=0,
    on_change=reset_settings
)

MODEL_PATH = "yolov8n_Trained.pt" 

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH).to(device)

model = load_model()

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

def annotate_frame(img, detection_strategy, restricted_shape):
    person_found_in_zone = False
    try:
        results = model.predict(img, device=device, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                
                if name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    if detection_strategy == "Feet (Bottom-Center)":
                        cx, cy = (x1 + x2) / 2.0, y2
                    elif detection_strategy == "Head (Top-Center)":
                        cx, cy = (x1 + x2) / 2.0, y1
                    elif detection_strategy == "Center (Middle)":
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    elif detection_strategy == "Left (Mid-Left)":
                        cx, cy = x1, (y1 + y2) / 2.0
                    elif detection_strategy == "Right (Mid-Right)":
                        cx, cy = x2, (y1 + y2) / 2.0
                    else:
                        cx, cy = (x1 + x2) / 2.0, y2
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 255), -1)

                    is_inside = False
                    if restricted_shape is not None:
                        if restricted_shape[0] == "rect":
                            rx1, ry1, rx2, ry2 = restricted_shape[1]
                            if (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2):
                                is_inside = True
                        elif restricted_shape[0] == "poly":
                            poly = restricted_shape[1]
                            if point_in_polygon(cx, cy, poly):
                                is_inside = True
                    
                    if is_inside:
                        person_found_in_zone = True
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                        cv2.putText(img, f"INSIDE {conf:.2f}", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if restricted_shape is not None:
            if restricted_shape[0] == "rect":
                rx1, ry1, rx2, ry2 = map(int, restricted_shape[1])
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
            elif restricted_shape[0] == "poly":
                poly = [(int(x), int(y)) for (x, y) in restricted_shape[1]]
                cv2.polylines(img, [np.array(poly)], isClosed=True, color=(255, 0, 0), thickness=3)

    except Exception:
        pass

    return img, person_found_in_zone

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
        annotated_img, is_alert = annotate_frame(img, self.detection_strategy, self.restricted_shape)
        self.person_in_restricted = is_alert
        return annotated_img

webrtc_ctx = None
uploaded_video_path = None

if source_option == "Realtime Webcam":
    webrtc_ctx = webrtc_streamer(
        key="person-detection",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    status_placeholder = st.empty()

elif source_option == "Video Upload":
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        if "last_video_name" not in st.session_state or st.session_state.last_video_name != uploaded_file.name:
            st.session_state.captured_image = None
            st.session_state.restricted_shape = None
            st.session_state.canvas_key = f"canvas_{time.time()}" 
            st.session_state.last_video_name = uploaded_file.name

        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        uploaded_video_path = tfile.name
        
        if "captured_image" not in st.session_state or st.session_state.captured_image is None:
            cap = cv2.VideoCapture(uploaded_video_path)
            ret, frame = cap.read()
            if ret:
                st.session_state.captured_image = frame
            cap.release()

st.write("---")

col1, col2 = st.columns([1, 1])
with col1:
    if source_option == "Realtime Webcam":
        captured = st.button("📸 Capture current frame")
        if captured:
            if webrtc_ctx and webrtc_ctx.video_transformer:
                frame = webrtc_ctx.video_transformer.last_frame
                if frame is not None:
                    st.session_state.captured_image = frame.copy()
                    st.success("Frame captured!")
            else:
                st.warning("Camera not running.")
    else:
        st.info("First frame captured automatically from video.")

with col2:
    st.write("**Instructions:**")
    st.markdown(
        "- Click **Capture current frame** to grab an image from the stream.  \n"
        "- Draw a **Rectangle** or **Polygon** on the image. \n"
        "- Click **Use ... as restricted area**."
    )

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

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"

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

if st.session_state.captured_image is not None:
    original_frame = st.session_state.captured_image
    
    orig_h, orig_w = original_frame.shape[:2]
    display_w = 700
    ratio = orig_w / display_w
    display_h = int(orig_h / ratio)
    
    display_frame = cv2.resize(original_frame, (display_w, display_h))
    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(display_rgb)

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
        height=display_h,
        width=display_w,
        drawing_mode=drawing_mode,
        key=st.session_state.canvas_key,
        display_toolbar=True,
    )

    restricted_rect = None
    restricted_poly = None

    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        for o in canvas_result.json_data["objects"]:
            t = o.get("type", "").lower()
            if t == "rect" and drawing_mode == "rect":
                left = o.get("left", 0) * ratio
                top = o.get("top", 0) * ratio
                width = o.get("width", 0) * ratio
                height = o.get("height", 0) * ratio
                restricted_rect = (int(left), int(top), int(left + width), int(top + height))
            elif drawing_mode == "polygon" and t == "path":
                pts = extract_polygon_points(o)
                if pts and len(pts) >= 3:
                    scaled_pts = [(x * ratio, y * ratio) for x, y in pts]
                    restricted_poly = scaled_pts

    if restricted_poly is not None:
        if st.button("Use current polygon as restricted area"):
            st.session_state.restricted_shape = ("poly", restricted_poly)
            if webrtc_ctx and webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = ("poly", restricted_poly)
            st.success("Polygon applied to live feed.")

    elif restricted_rect is not None:
        if st.button("Use this Rectangle"):
            st.session_state.restricted_shape = ("rect", restricted_rect)
            if webrtc_ctx and webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = ("rect", restricted_rect)
            st.success("Rectangle Applied!")
            
    if st.button("Clear Area"):
        st.session_state.restricted_shape = None
        if webrtc_ctx and webrtc_ctx.video_transformer:
            webrtc_ctx.video_transformer.restricted_shape = None
        st.session_state.canvas_key = f"canvas_{time.time()}"
        st.rerun()

if source_option == "Realtime Webcam" and webrtc_ctx.state.playing:
    try:
        while True:
            if webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.detection_strategy = detection_mode
                flag = webrtc_ctx.video_transformer.person_in_restricted
                if flag:
                    status_placeholder.error("🚨 ALERT: Person INSIDE restricted area!")
                else:
                    status_placeholder.success("✅ Secure: No person detected inside.")
            time.sleep(0.2) 
    except Exception:
        pass

if source_option == "Video Upload" and uploaded_video_path is not None:
    if st.button("▶ Run Detection on Video"):
        st_video_container = st.empty()
        st_alert_container = st.empty()
        
        cap = cv2.VideoCapture(uploaded_video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, is_alert = annotate_frame(frame, detection_mode, st.session_state.restricted_shape)
            
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st_video_container.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if is_alert:
                st_alert_container.error("🚨 ALERT: Person INSIDE restricted area!")
            else:
                st_alert_container.success("✅ Secure: No person detected inside.")
        cap.release()