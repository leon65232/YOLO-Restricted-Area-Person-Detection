import torch
import streamlit as st
from ultralytics import YOLO

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🔧 Running on: {device.upper()}")

model = YOLO("./yolov11n.pt")
model.to(device)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save file temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run detection on the chosen device
    results = model.predict(temp_path, device=device)

    # Show result
    st.image(results[0].plot(), caption="Detection Result")
