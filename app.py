import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2

# Set page config
st.set_page_config(page_title="YOLOv8 Web Element Detection", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* General Body Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    /* Title Styles */
    .title-container {
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .title-container h1 {
        color: white;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .title-container p {
        color: #f0f0f0;
        font-size: 1.2rem;
        line-height: 1.6;
        font-weight: 300;
    }

    /* Sidebar Styles */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9ff 100%);
        border-right: 1px solid #e1e5e9;
    }
    
    .css-1d391kg .sidebar-content {
        padding: 1rem;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    h2 {
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* File uploader */
    .stFileUploader > div {
        background-color: #f8f9ff;
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    .stFileUploader > div:hover {
        border-color: #764ba2;
        background-color: #f0f2ff;
    }

    /* Expander/Details Styles */
    .streamlit-expanderHeader {
        background-color: #667eea;
        color: white;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background-color: #f8f9ff;
        border: 1px solid #e1e5e9;
        border-radius: 0 0 10px 10px;
        border-top: none;
    }
    
    /* Warning Box Style */
    div[data-testid="stAlert"] {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 15px;
        color: #856404;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
        font-weight: 500;
    }
    
    div[data-testid="stAlert"] div {
        color: #856404 !important;
        font-weight: 500 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

</style>
""", unsafe_allow_html=True)


# --- UI ELEMENTS ---

# --- Header ---
st.markdown("""
<div class="title-container">
    <h1>🎯 YOLOv8 Web Element Detection</h1>
    <p>Discover the power of AI-driven web element detection! Upload a website screenshot and watch our YOLOv8 model identify various UI components with precision.</p>
</div>
""", unsafe_allow_html=True)


st.sidebar.title("📋 About the Model")
st.sidebar.info(
    "🤖 **YOLOv8 Architecture**: This model has been fine-tuned on a custom dataset to identify various web UI elements.\n\n"
    "🎯 **Detection Capabilities**: Buttons, input fields, labels, navigation elements, and more.\n\n"
    "🎨 **Purpose**: Demonstrating the potential of computer vision for web UI analysis and automation."
)

st.sidebar.title("📊 Training Performance")
st.sidebar.write("📈 Key performance metrics from the training process:")

try:
    st.sidebar.image('results.png', caption='Training Results', use_container_width=True)
    st.sidebar.image('confusion_matrix_normalized.png', caption='Normalized Confusion Matrix', use_container_width=True)
    st.sidebar.image('PR_curve.png', caption='Precision-Recall Curve', use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Could not find performance metric images.")
except Exception as e:
    st.sidebar.error(f"An error occurred while loading performance images: {e}")


# --- MODEL INFERENCE ---
@st.cache_resource
def load_model():
    """
    Loads the YOLOv8 model from the specified path.
    Using st.cache_resource to load the model only once.
    """
    import os
    import requests
    from io import BytesIO
    
    model_path = "weights/best.pt"
    
    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)
    
    # Check if model file exists locally and is valid
    if os.path.exists(model_path):
        try:
            # Try to load the existing model to verify it's valid
            model = YOLO(model_path)
            return model
        except Exception:
            # If loading fails, delete the corrupted file and redownload
            os.remove(model_path)
    
    # Download the model
    try:
        # Google Drive file ID
        file_id = "1q1g2Pd7xff99mrdUYW4aASnxx-Mmkocg"
        
        with st.spinner("Downloading model... This may take a few minutes."):
            # Use the Google Drive download URL that bypasses the virus scan warning
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            
            # Create a session to handle cookies
            session = requests.Session()
            
            # First request to get any cookies/tokens
            response = session.get(download_url, stream=True)
            
            # Check if we got redirected to a confirmation page
            if 'confirm=' in response.url or response.status_code != 200:
                # Try alternative download method
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"
                response = session.get(download_url, stream=True)
            
            # Check if the response is valid
            if response.status_code == 200:
                # Check if we're getting HTML (error page) instead of binary data
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    st.error("❌ Could not download model file. The file might be too large or have download restrictions.")
                    st.info("💡 **Alternative solution**: Please download the model manually from the Google Drive link and upload it to your repository in the 'weights' folder.")
                    return None
                
                # Save the model file
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                st.success("✅ Model downloaded successfully!")
            else:
                raise Exception(f"Failed to download: HTTP {response.status_code}")
                
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")
        st.info("💡 **Alternative solution**: Please download the model manually and add it to your repository.")
        return None
    
    # Load the downloaded model
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        # Clean up corrupted file
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

# Load the model
model = load_model()

if model:
    st.markdown("### 📤 Upload Your Website Screenshot")
      # Image Upload
    uploaded_file = st.file_uploader(
        "Choose a website screenshot to analyze", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear screenshot of a website for best results"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(image, use_container_width=True, caption="Your uploaded screenshot")

        with st.spinner("🔍 Analyzing image and detecting elements..."):
            # Run inference
            results = model(image, conf=0.4) # Can adjust confidence threshold

            # Plot results
            result_image = results[0].plot()  # BGR numpy array
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            with col2:
                st.markdown("#### 🎯 Detected Elements")
                st.image(result_image_rgb, use_container_width=True, caption="AI-detected web elements")

            st.markdown("---")
            
            with st.expander("📊 Detailed Detection Results", expanded=True):
                # Show results as a table
                names = model.names
                boxes = results[0].boxes
                detections_list = []
                for i in range(len(boxes)):
                    conf = boxes.conf[i].item()
                    cls = int(boxes.cls[i].item())
                    label = names[cls]
                    box = boxes.xyxyn[i].cpu().numpy().flatten() # Normalized xyxy
                    detections_list.append([label, conf, *box])

                if detections_list:
                    df = pd.DataFrame(detections_list, columns=["Label", "Confidence", "x1", "y1", "x2", "y2"])
                    st.dataframe(df)
                else:
                    st.write("No elements detected.")
      # --- Disclaimer ---
    st.warning("""
    **Disclaimer:** This model was trained on a limited dataset and is intended for demonstration purposes only.
    It may not be highly accurate for all types of web elements or website layouts.
    """)
else:
    st.warning("Model could not be loaded. Please check the model path and dependencies.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 2rem;">
    <h4 style="color: white; margin-bottom: 1rem;">👨‍💻 Made by</h4>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <a href="https://github.com/igharsha7" target="_blank" style="color: white; text-decoration: none; font-weight: 600; font-size: 1.1rem;">
            🔗 igharsha7
        </a>
        <a href="https://github.com/SaiGuruInukurthi" target="_blank" style="color: white; text-decoration: none; font-weight: 600; font-size: 1.1rem;">
            🔗 SaiGuruInukurthi
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
