import streamlit as st 
from ultralytics import YOLO
from PIL import Image
import numpy as np 
from io import BytesIO
import cv2

# Detect object function 
def detect_image(model, confidence_threshold):
    st.write("Upload an image to detect objects.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image and ensure that the image only has 3 color channels
        image = Image.open(uploaded_file).convert("RGB")

        # Run YOLO inference with confidence threshold
        with st.spinner("Detecting objects..."):
            results = model.predict(np.array(image), conf=confidence_threshold)

        # Get result 
        result = results[0]  
        annotated_image = result.plot()  # Annotated image as numpy array
        
        st.markdown("*You can adjust the **Confidence Threshold** slider on the left sidebar to filter out low-confidence detections!*")
        # 2-column layout
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Uploaded image", use_container_width=True)

        with col2:
            st.image(annotated_image, caption="✅ Detected objects", use_container_width=True)
            # Numpy array to byte
            img_bytes = BytesIO()
            Image.fromarray(annotated_image).save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()  
            
            # Download button
            st.download_button(
                label="💾 Download detected image",
                data=img_bytes,
                file_name="detected_image.jpg",
                mime="image/jpg"
                )
        
def task():
    st.session_state["model"] = YOLO("yolov8m.pt")
    model = st.session_state["model"]

    st.title("🔎 Object Detection with YOLOv8")

    # Choose confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,  # default value 
        step=0.05
    )
    
    # Upload image and detect image
    detect_image(model, confidence_threshold)