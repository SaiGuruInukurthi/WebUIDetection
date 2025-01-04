import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from io import BytesIO

# Load the custom YOLOv5 model
@st.cache_resource
def load_model():
    model_path = "D:\hackathon\best.pt"  # Update with your model's path
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load custom YOLOv5 model
    return model

model = load_model()

# Streamlit app
st.title("WebUI Detection with YOLOv5")
st.write("Upload an image to detect webpage elements.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Perform inference
    st.write("Processing the image...")
    results = model(image_np)

    # Get annotated image
    annotated_image = np.array(results.render()[0])

    # Display the result
    st.image(annotated_image, caption='Processed Image', use_column_width=True)

    # Save the annotated image to BytesIO for download
    buffer = BytesIO()
    result_image = Image.fromarray(annotated_image)
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Download button
    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="processed_image.png",
        mime="image/png"
    )

else:
    st.write("Please upload an image to proceed.")