# YOLOv8s WebUI Detection - Implementation Plan

## Overview
This document outlines the implementation of a simplified YOLOv8s-based pipeline for WebUI element detection, using accessibility tree roles instead of CSS-based classification.

## Key Changes from Previous Approach

### 1. Label Source: Accessibility Tree Roles
- **Previous**: CSS class pattern matching via `classify_element()` function
- **New**: Direct extraction of semantic roles from `default_1920-1080-axtree.json.gz`
- **Benefit**: Cleaner, browser-native semantic labels without heuristic matching

### 2. Class Selection: Top 10 Semantic Classes
Based on dataset analysis of 500 samples:

| ID | Class | Count | Description |
|----|-------|-------|-------------|
| 0 | link | ~2,000 | Hyperlinks (most common) |
| 1 | button | ~72 | Interactive buttons |
| 2 | img | ~342 | Images |
| 3 | heading | ~693 | H1-H6 headings |
| 4 | listitem | ~933 | List items |
| 5 | textbox | ~16 | Text input fields |
| 6 | navigation | ~89 | Navigation elements |
| 7 | banner | ~123 | Header/banner areas |
| 8 | paragraph | ~1,247 | Text paragraphs |
| 9 | gridcell | ~408 | Grid/table cells |

**Excluded roles**: `none`, `generic`, `StaticText`, `RootWebArea` (non-semantic or root elements)

### 3. Model: YOLOv8s
- **Previous**: YOLOv11n (nano model)
- **New**: YOLOv8s (small model with ~11M parameters)
- **Benefit**: Larger model capacity for better feature extraction

### 4. Data Split: 70/30
- **Previous**: 85% train / 15% validation
- **New**: 70% train / 30% validation
- **Benefit**: More robust validation set for performance evaluation

### 5. Bounding Box Filter: Minimum 10px
- **Previous**: `width > 2` filter only
- **New**: `width >= 10 AND height >= 10`
- **Benefit**: Removes noisy small elements (icons under 10px are rare in UI detection)

## Data Pipeline

### Source Files
For each sample in `webui-7k/train_split_web7k/{timestamp}/`:
- `default_1920-1080-axtree.json.gz`: Accessibility tree with roles and DOM node IDs
- `default_1920-1080-bb.json.gz`: Bounding boxes keyed by element ID
- `default_1920-1080-screenshot.webp`: Screenshot image (1920x1080)

### Data Flow
```
axtree.json.gz → Extract nodes with target roles
       ↓
bb.json.gz → Match backendDOMNodeId to get bounding boxes
       ↓
Filter: w >= 10 AND h >= 10 AND inside viewport
       ↓
Convert to YOLO format (normalized center x/y, width, height)
       ↓
70/30 random split → train/ and val/ directories
```

## Training Configuration

```python
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=4,
    patience=20,
    
    # Disable flips for UI detection (orientation matters)
    fliplr=0.0,
    flipud=0.0,
    
    # Reduced augmentation
    mosaic=0.3,
    mixup=0.0,
    scale=0.3,
)
```

## Expected Improvements

| Metric | Previous | Expected | Reason |
|--------|----------|----------|--------|
| mAP50 | ~16% | 25-40% | Cleaner labels, fewer classes |
| Per-class AP variance | High | Lower | Reduced class imbalance |
| Training stability | Variable | Improved | Semantic labels |

## Files Created

1. `yolov8s_webui_detection.ipynb` - Main training notebook
2. `analyze_dataset.py` - Dataset analysis script
3. `explore_data.py` - Data exploration script
4. `datasets/webui_v2/` - Output dataset directory (created during training)

## Next Steps

1. Run the notebook to process the dataset
2. Train the YOLOv8s model
3. Analyze per-class mAP and confusion matrix
4. If certain classes underperform, consider:
   - Adjusting class weights
   - Data augmentation specific to those classes
   - Investigating label quality for specific roles

## Alternative Approaches to Consider

1. **ARIA roles**: Combine HTML tag roles with ARIA attributes for richer semantics
2. **Hierarchical classification**: Group classes (e.g., interactive elements, text elements)
3. **Multi-scale detection**: Use higher resolution for small elements
