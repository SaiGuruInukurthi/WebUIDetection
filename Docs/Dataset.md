# WebUI Balanced 7K Dataset

## Overview

The **WebUI Balanced 7K** is a comprehensive web user interface detection dataset designed for training and evaluating machine learning models on web page element detection, classification, and accessibility analysis. The dataset contains 7,009 unique web page samples, each captured across 6 different device/resolution configurations, totaling over 42,000 annotated screenshots.

### Dataset Statistics
- **Total Samples**: 7,009 unique web pages
- **Total Screenshots**: 42,054 (7,009 × 6 configurations)
- **Total Files**: 461,993
- **Total Size**: ~7.7 GB (7,009 folders)
- **Device Configurations**: 6 viewports per sample
- **Annotation Types**: Multiple (bounding boxes, accessibility tree, element classes, viewport visibility, CSS styles)

### Dataset Sources
- **Hugging Face**: [biglab/webui-7k](https://huggingface.co/datasets/biglab/webui-7k)
- **Google Drive**: [Download Dataset](https://drive.google.com/drive/folders/1hcO75W2FjsZoibsj2TIbKz67hy9JkOBz)

---

## Directory Structure

```
balanced_7k/
├── balanced_7k.json              # Master index file with all sample IDs
└── [timestamp_directories]/      # 7,009 timestamp-based sample directories
    ├── default_1280-720-*        # Desktop 1280×720 annotations
    ├── default_1366-768-*        # Desktop 1366×768 annotations
    ├── default_1536-864-*        # Desktop 1536×864 annotations
    ├── default_1920-1080-*       # Desktop 1920×1080 annotations
    ├── iPad-Pro-*                # iPad Pro annotations
    └── iPhone-13 Pro-*           # iPhone 13 Pro annotations
```

### Sample Directory Example
Each sample directory (e.g., `1655885112781/`) contains 66 files (11 files × 6 device configurations):

#### Files Per Configuration (11 files each):
1. **axtree.json.gz** - Accessibility tree data
2. **bb.json.gz** - Simple bounding boxes
3. **box.json.gz** - Detailed CSS box model
4. **class.json.gz** - UI element class labels
5. **html.html** - Raw HTML source code
6. **links.json** - Link/hyperlink information
7. **screenshot-full.webp** - Full-page screenshot
8. **screenshot.webp** - Viewport screenshot
9. **style.json.gz** - Computed CSS styles
10. **url.txt** - Source webpage URL
11. **viewport.json.gz** - Element visibility flags

---

## Device/Resolution Configurations

Each web page is captured with 6 different viewport configurations to ensure cross-device compatibility:

| Configuration Name | Resolution/Device | Type |
|-------------------|-------------------|------|
| `default_1280-720` | 1280 × 720 | Desktop |
| `default_1366-768` | 1366 × 768 | Desktop (most common laptop) |
| `default_1536-864` | 1536 × 864 | Desktop |
| `default_1920-1080` | 1920 × 1080 | Desktop (Full HD) |
| `iPad-Pro` | iPad Pro | Tablet |
| `iPhone-13 Pro` | iPhone 13 Pro | Mobile |

---

## File Format Descriptions

### 1. **axtree.json.gz** (Accessibility Tree)
Contains the complete accessibility tree structure captured via Chrome DevTools Protocol.

**Format**: Compressed JSON
```json
{
  "nodes": [
    {
      "nodeId": "1",
      "ignored": false,
      "role": {"type": "internalRole", "value": "RootWebArea"},
      "name": {"type": "computedString", "value": "Page Title"},
      "properties": [...],
      "childIds": ["2"],
      "backendDOMNodeId": 3,
      "frameId": "..."
    },
    ...
  ]
}
```

**Contains**:
- DOM node hierarchy
- ARIA roles and properties
- Accessibility labels and names
- Focus states and ignored elements
- Computed accessibility information

### 2. **bb.json.gz** (Bounding Boxes)
Simple bounding box coordinates for each detected UI element.

**Format**: Compressed JSON (Object with element IDs as keys)
```json
{
  "3": {"x": 0, "y": 0, "width": 1920, "height": 1080},
  "4": {"x": 510, "y": 196.328125, "width": 900, "height": 115.15625},
  ...
}
```

**Properties**:
- `x`: X-coordinate (top-left)
- `y`: Y-coordinate (top-left)
- `width`: Element width in pixels
- `height`: Element height in pixels

### 3. **box.json.gz** (CSS Box Model)
Detailed CSS box model information including content, padding, border, and margin areas.

**Format**: Compressed JSON
```json
{
  "4": {
    "content": [{"x": 510, "y": 196.328125}, ...],
    "padding": [{"x": 510, "y": 196.328125}, ...],
    "border": [{"x": 510, "y": 196.328125}, ...],
    "margin": [{"x": 510, "y": 177.140625}, ...],
    "width": 900,
    "height": 115
  },
  ...
}
```

**Contains**:
- Content box coordinates (4 corner points)
- Padding box coordinates
- Border box coordinates
- Margin box coordinates
- Computed width and height

### 4. **class.json.gz** (UI Element Classes)
Classification labels for detected UI elements.

**Format**: Compressed JSON (Array of numeric class IDs)
```json
[4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 20]
```

**Known Class Labels** (numeric):
- 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 20
- Labels represent different UI element types (buttons, links, inputs, containers, etc.)
- *Note: Full class taxonomy documentation may be available separately*

### 5. **html.html** (Page Source)
Raw HTML source code of the webpage as rendered.

**Format**: Plain HTML file

### 6. **links.json** (Hyperlinks)
Information about hyperlinks and navigational elements on the page.

**Format**: JSON

### 7. **screenshot-full.webp / screenshot.webp** (Screenshots)
Visual captures of the rendered webpage.

**Formats**:
- `screenshot-full.webp`: Full-page screenshot (entire scrollable area)
- `screenshot.webp`: Viewport screenshot (visible area only)

**Image Format**: WebP (compressed, lossless)

### 8. **style.json.gz** (CSS Styles)
Computed CSS styles for elements on the page.

**Format**: Compressed JSON
Contains computed style properties for tracked elements.

### 9. **url.txt** (Source URL)
The original URL of the captured webpage.

**Format**: Plain text file
```
http://semantic-ui.mit-license.org
```

### 10. **viewport.json.gz** (Visibility Flags)
Boolean flags indicating which elements are visible in the current viewport.

**Format**: Compressed JSON
```json
{
  "4": true,
  "5": true,
  "7": true,
  ...
}
```

---

## Master Index File

### balanced_7k.json
Contains an array of all 7,009 sample directory names (timestamps).

**Format**: JSON Array
```json
[
  "1655885112781",
  "1655886032342",
  ...
  "1660285896684"
]
```

---

## Directory Naming Convention

Sample directories are named using Unix timestamps (milliseconds since epoch):
- **Format**: 13-digit timestamp (e.g., `1655885112781`)
- **Range**: `1655885112781` to `1660285896684`
- **Period**: June 22, 2022 to August 12, 2022

---

## Data Usage Guidelines

### Loading a Single Sample

```python
import json
import gzip
from PIL import Image

# Sample directory path
sample_dir = "balanced_7k/1655885112781/"
config = "default_1920-1080"

# Load bounding boxes
with gzip.open(f"{sample_dir}{config}-bb.json.gz", 'rt') as f:
    bounding_boxes = json.load(f)

# Load class labels
with gzip.open(f"{sample_dir}{config}-class.json.gz", 'rt') as f:
    classes = json.load(f)

# Load screenshot
screenshot = Image.open(f"{sample_dir}{config}-screenshot.webp")

# Load URL
with open(f"{sample_dir}{config}-url.txt", 'r') as f:
    url = f.read().strip()
```

### Iterating Through All Samples

```python
import json

# Load master index
with open("balanced_7k/balanced_7k.json", 'r') as f:
    sample_ids = json.load(f)

# Process each sample
for sample_id in sample_ids:
    sample_path = f"balanced_7k/{sample_id}/"
    # Process files...
```

### Multi-Resolution Training

The dataset supports multi-resolution training by providing the same web page across different viewports:

```python
resolutions = [
    "default_1280-720",
    "default_1366-768", 
    "default_1536-864",
    "default_1920-1080",
    "iPad-Pro",
    "iPhone-13 Pro"
]

for resolution in resolutions:
    # Load and process each resolution variant
    pass
```

---

## Potential Applications

1. **Web UI Element Detection**: Train object detection models (YOLO, Faster R-CNN, etc.) to identify buttons, inputs, links, and other UI components

2. **Responsive Design Analysis**: Compare element positions and sizes across different viewport configurations

3. **Accessibility Evaluation**: Use accessibility tree data to train models for WCAG compliance checking

4. **Cross-Device Layout Prediction**: Predict how layouts adapt from desktop to tablet to mobile

5. **Web Scraping Automation**: Train models to identify interactive elements for automated testing

6. **Visual Design Analysis**: Analyze CSS styles and visual properties of modern web interfaces

7. **Semantic Segmentation**: Pixel-level classification of web UI elements using bounding boxes and screenshots

---

## Data Quality Notes

- All annotations are automatically extracted using Chrome DevTools Protocol
- Bounding box coordinates use sub-pixel precision (float values)
- Some elements may have overlapping bounding boxes (nested elements)
- Accessibility tree includes both visible and hidden elements
- Screenshots are captured after full page load (dynamic content may vary)

---

## Citation

If you use this dataset in your research, please cite appropriately:

```
WebUI Balanced 7K Dataset
7,009 web page samples across 6 device configurations
Captured: June-August 2022
```

---

## File System Statistics

- **Total Directories**: 7,010 (7,009 samples + 1 root)
- **Files per Sample**: 66 (11 types × 6 configurations)
- **Total Compressed JSON Files**: ~252,000
- **Total Image Files**: ~84,000
- **Total HTML Files**: ~42,000
- **Compression**: GZIP for JSON files, WebP for images

---

## Technical Details

### Compression Formats
- **JSON Files**: GZIP compressed (`.json.gz`)
- **Images**: WebP format (`.webp`)
- **HTML**: Uncompressed (`.html`)
- **URLs**: Plain text (`.txt`)

### Coordinate System
- Origin: Top-left corner (0, 0)
- X-axis: Left to right
- Y-axis: Top to bottom
- Units: Pixels (CSS pixels, not physical)

### Timestamp Format
```
Format: Unix epoch in milliseconds
Example: 1655885112781
Date: Wed Jun 22 2022 09:31:52 GMT+0000
```

---

## Troubleshooting

### Files Won't Extract
- Ensure 7-Zip or compatible tool is installed
- Python: Use `gzip` module for `.gz` files

### Large Memory Usage
- JSON files are compressed; decompress selectively
- Process samples in batches rather than loading entire dataset

### Missing Files
- Each sample should have exactly 66 files
- Verify extraction completed successfully

---

## Version Information

- **Dataset Version**: 1.0
- **Format Version**: WebUI Balanced 7K
- **Extraction Date**: January 2024
- **Archive Format**: Multi-part ZIP (`.zip-001.001`, `.zip-002.002`)

---

## Contact & Support

For questions, issues, or dataset assistance, please refer to the original data source or research group that provided this dataset.

---

**Last Updated**: January 2024
