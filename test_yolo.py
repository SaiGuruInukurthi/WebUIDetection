#!/usr/bin/env python3
"""Test YOLOv8s model availability."""

try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
    
    # This will download the model if not present
    model = YOLO('yolov8s.pt')
    print(f"Model loaded: {model.model}")
    print("YOLOv8s is ready for training!")
except Exception as e:
    print(f"Error: {e}")
    print("\nTo install ultralytics, run: pip install ultralytics")
