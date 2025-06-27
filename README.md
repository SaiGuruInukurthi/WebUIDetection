# 🎯 WebUI Element Detection Model

<div align="center">
  <img src="https://img.shields.io/badge/Hackathon-12%20Hours-ff6b6b?style=for-the-badge&logo=clock" alt="12 Hour Hackathon">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-00d2d3?style=for-the-badge&logo=pytorch" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python" alt="Python">
</div>

## 🚀 Project Overview

This project is a **12-hour hackathon challenge** that demonstrates the power of computer vision for web UI element detection. Using a fine-tuned YOLOv8 model, the system can automatically identify and classify various web elements from website screenshots, including buttons, input fields, labels, navigation elements, and more.

> **⚡ Built in just 12 hours!** This project showcases rapid prototyping and deployment of a machine learning solution for web UI analysis.

## 🎨 Features

- **🤖 AI-Powered Detection**: Uses YOLOv8m architecture for accurate web element identification
- **🌐 Interactive Web Interface**: Clean, modern Streamlit-based UI for easy interaction
- **📊 Real-time Analysis**: Upload screenshots and get instant detection results
- **📈 Performance Metrics**: View training performance and model statistics
- **🎯 Multiple Element Types**: Detects buttons, forms, navigation, text elements, and more

## 🛠️ Technology Stack

### Core Technologies
- **YOLOv8m**: Pre-trained object detection model fine-tuned for web elements
- **Streamlit**: Web application framework for the interactive interface
- **PyTorch**: Deep learning framework with CUDA acceleration
- **OpenCV**: Computer vision library for image processing
- **PIL/Pillow**: Image manipulation and processing

### Libraries and Dependencies
```
torch==2.5.1 (with CUDA 12.4 support)
ultralytics
streamlit
opencv-python-headless
pandas
numpy
pillow
```

## 📊 Model Details

- **Architecture**: YOLOv8m (Medium variant)
- **Training Dataset**: Custom web UI elements dataset from Roboflow
- **Training Duration**: 100 epochs
- **Image Size**: 640x640 pixels
- **Batch Size**: 2 (optimized for available GPU memory)
- **Confidence Threshold**: 0.4 (adjustable)

### Training Configuration
```yaml
Model: yolov8m.pt
Epochs: 100
Batch Size: 2
Image Size: 640
Optimizer: Auto
Learning Rate: 0.01
Device: CUDA (GPU accelerated)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/igharsha7/webui-element-detection.git
cd webui-element-detection
```

2. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics streamlit opencv-python-headless pandas pillow
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📝 Usage

1. **Upload Screenshot**: Use the file uploader to select a website screenshot
2. **View Results**: The model will automatically detect and highlight web elements
3. **Analyze Details**: Expand the detailed results section to see confidence scores and bounding box coordinates
4. **Export Data**: Detection results are displayed in a structured table format

### Supported Image Formats
- PNG
- JPEG/JPG
- WebP

## 📈 Performance Metrics

The model was trained and evaluated with the following key metrics:
- **Training Loss**: Monitored across 100 epochs
- **Validation mAP**: Mean Average Precision for object detection
- **Precision-Recall Curves**: Available in the sidebar of the web interface
- **Confusion Matrix**: Normalized confusion matrix for class predictions

## 🎯 Hackathon Challenge

This project was completed as part of a **12-hour hackathon**, demonstrating:

- ⚡ **Rapid Development**: From concept to deployment in under 12 hours
- 🧠 **Problem-Solving**: Quick adaptation of pre-trained models for specific use cases
- 🎨 **UI/UX Design**: Creating an intuitive interface under time constraints
- 📊 **Model Training**: Efficient fine-tuning with limited time and resources
- 🚀 **Deployment**: Ready-to-use web application with professional UI

## ⚠️ Limitations

- **Dataset Size**: Trained on a limited dataset due to hackathon time constraints
- **Accuracy**: May not be highly accurate for all website layouts and designs
- **Demo Purpose**: Intended primarily for demonstration and proof-of-concept
- **Element Coverage**: Limited to common web UI elements present in training data

## 🔮 Future Improvements

- [ ] Expand training dataset with more diverse web elements
- [ ] Add support for mobile UI detection
- [ ] Implement batch processing for multiple images
- [ ] Add export functionality for detection results
- [ ] Integrate with web scraping tools for automated UI testing

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 👥 Team

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/igharsha7">
          <img src="https://github.com/igharsha7.png" width="100px;" alt="igharsha7"/><br />
          <sub><b>igharsha7</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/SaiGuruInukurthi">
          <img src="https://github.com/SaiGuruInukurthi.png" width="100px;" alt="SaiGuruInukurthi"/><br />
          <sub><b>SaiGuruInukurthi</b></sub>
        </a>
      </td>
    </tr>
  </table>
</div>

---

<div align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
  <br>
  <sub>Built in 12 hours</sub>
</div>
