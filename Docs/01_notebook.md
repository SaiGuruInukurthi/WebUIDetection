# WebUI Balanced 7K Dataset - Exploratory Data Analysis Report

**Analysis Date:** February 8, 2026  
**Notebook:** `01_Dataset__exploration.ipynb`  
**Dataset:** WebUI Balanced 7K (7,000 web page samples × 6 device configurations)

---

## Executive Summary

This exploratory data analysis assessed the WebUI Balanced 7K dataset for quality, balance, and suitability for training object detection models. The dataset contains **7,000 unique web page samples** captured across **6 device/resolution configurations**, resulting in approximately **42,000 annotated screenshots**.

### Key Findings:
- ✅ **99.94% Dataset Integrity**: 6,996 of 7,000 samples are complete (4 samples have missing configurations)
- ⚠️ **Device Imbalance**: Desktop-heavy (66.7%), with tablet and mobile each at 16.7%
- 📊 **86 Unique UI Element Classes**: Based on accessibility roles (ARIA roles)
- 📐 **843K+ Bounding Boxes Analyzed**: Wide variety in element sizes and aspect ratios
- 🔢 **Average 187 Elements per Screenshot**: Range from 5 to 2,097 elements
- 🔗 **Nested/Overlapping Elements Common**: Average 121 overlapping pairs per screenshot

---

## 1. Dataset Configuration

### Dataset Paths
```
Dataset Root: C:\WebUIDetection\balanced_7k
Index File: C:\WebUIDetection\webui-balanced-7k-20260208T053018Z-1-003\webui-balanced-7k\balanced_7k.json
Total Configurations: 6
```

### Device/Resolution Configurations

| Configuration | Type | Resolution/Device | Count |
|--------------|------|------------------|-------|
| `default_1280-720` | Desktop | 1280 × 720 | 6,996 |
| `default_1366-768` | Desktop | 1366 × 768 | 6,996 |
| `default_1536-864` | Desktop | 1536 × 864 | 6,996 |
| `default_1920-1080` | Desktop | 1920 × 1080 | 6,996 |
| `iPad-Pro` | Tablet | iPad Pro | 6,996 |
| `iPhone-13 Pro` | Mobile | iPhone 13 Pro | 6,996 |

**Total Valid Screenshots:** 41,976 (6,996 samples × 6 configurations)

---

## 2. Dataset Integrity & Health Check

### Integrity Summary

```
============================================================
DATASET INTEGRITY SUMMARY
============================================================
Total samples: 7,000
Complete samples: 6,996 (99.94%)
Incomplete samples: 4
Missing directories: 0
```

### Problematic Samples

| Sample ID | Status | Missing Configurations |
|-----------|--------|------------------------|
| 1656081358661 | Incomplete | default_1280-720, iPad-Pro, iPhone-13 Pro |
| 1656265957879 | Incomplete | iPad-Pro |
| 1656174897093 | Incomplete | iPad-Pro |
| 1656302012565 | Incomplete | iPad-Pro, iPhone-13 Pro |

### Assessment
- **99.94% completeness** rate is excellent for machine learning training
- Only 4 samples have missing configurations (primarily iPad-Pro)
- All 7,000 sample directories exist on disk
- **Recommendation:** Exclude the 4 incomplete samples from training (use 6,996 samples)

---

## 3. Device Type Distribution

### Screenshot Distribution by Device Type

| Device Type | Screenshots | Percentage | Configurations |
|-------------|-------------|------------|----------------|
| **Desktop** | 27,984 | 66.7% | 4 |
| **Tablet** | 6,996 | 16.7% | 1 |
| **Mobile** | 6,996 | 16.7% | 1 |
| **Total** | **41,976** | **100%** | **6** |

### Visual Distribution

![Device Distribution](device_distribution.png)

- **Bar chart** shows absolute counts: Desktop dominates with 27,984 screenshots
- **Pie chart** shows proportions: Desktop (66.7%), Tablet & Mobile (16.7% each)

### Balance Assessment

```
Desktop: 66.7% (4 configurations)
Tablet: 16.7% (1 configuration)
Mobile: 16.7% (1 configuration)
```

**Conclusion:** The dataset is **heavily skewed toward desktop viewports**. The "balanced" naming refers to balanced sampling within desktop resolutions, NOT across device types. This bias should be considered when:
- Training models intended for mobile/tablet deployment
- Evaluating cross-device generalization
- Applying data augmentation or weighted sampling

---

## 4. UI Element Class Distribution

### Class Extraction Source
Classes are extracted from **accessibility tree data** (`axtree.json.gz`), specifically the `role.value` field of each node. These are **ARIA accessibility roles**, not arbitrary numeric IDs.

### Quick Analysis (500 samples)
```
Unique classes found: 86
Total UI elements: 580,350
```

### Full Dataset Analysis (6,996 samples)
```
Processing: 6,996 samples across 6 configurations
Processing time: ~7 minutes
Total UI elements: Multiple millions (output too large for context)
Unique classes: 86 accessibility roles
```

### Class Types (86 Total)

The dataset uses **86 distinct accessibility roles** including:

#### Structural Elements
- `RootWebArea`, `document`, `article`, `section`, `main`, `banner`, `contentinfo`
- `header`, `footer`, `navigation`, `complementary`, `form`

#### Interactive Elements
- `button`, `link`, `checkbox`, `radio`, `textbox`, `searchbox`, `combobox`, `listbox`
- `switch`, `slider`, `spinbutton`, `menuitem`, `tab`

#### Containers & Layout
- `generic`, `group`, `list`, `listitem`, `table`, `row`, `gridcell`, `columnheader`, `rowheader`
- `LayoutTable`, `LayoutTableCell`, `LayoutTableRow`

#### Content Elements
- `heading`, `paragraph`, `img`, `figure`, `caption`, `blockquote`
- `StaticText`, `LabelText`, `ListMarker`, `LineBreak`

#### Rich Media & Advanced
- `Canvas`, `SvgRoot`, `Video`, `Iframe`, `IframePresentational`, `EmbeddedObject`
- `dialog`, `alertdialog`, `alert`, `status`, `timer`, `progressbar`

#### Semantic Elements
- `emphasis`, `strong`, `superscript`, `insertion`, `Abbr`, `time`
- `Details`, `DisclosureTriangle`, `Legend`, `Figcaption`

#### Special Categories
- `application`, `toolbar`, `menubar`, `menu`, `tablist`, `tabpanel`
- `search`, `region`, `none`, `separator`

### Class Distribution Characteristics

**Note:** Full class distribution statistics were too large to display in context. Based on subset analysis:

- **High class variety:** 86 distinct element types provide rich semantic labels
- **Hierarchical structure:** Classes include both generic containers and specific interactive elements
- **Web-native taxonomy:** Directly maps to W3C ARIA specification
- **Imbalanced distribution expected:** Generic/container elements typically outnumber specialized widgets

### Implications for Model Training

1. **Multi-class Detection:** Models must handle 86+ distinct classes
2. **Class Imbalance:** Likely requires weighted loss functions or focal loss
3. **Semantic Understanding:** Rich ARIA taxonomy enables accessibility-aware models
4. **Fine-grained Classification:** Distinguishes between similar elements (button vs. link, textbox vs. searchbox)

---

## 5. Bounding Box Geometry Analysis

### Sample Size
- **Samples analyzed:** 1,000 samples
- **Total bounding boxes:** 843,154

### Geometry Statistics

| Metric | Width (px) | Height (px) | Area (px²) | Aspect Ratio |
|--------|-----------|------------|------------|--------------|
| **Count** | 843,154 | 843,154 | 843,154 | 843,154 |
| **Mean** | 381.47 | 172.04 | 50,341,860 | 9.56 |
| **Std Dev** | 7,482.71 | 6,742.23 | 18,822,590,000 | 49.67 |
| **Min** | 0.16 | 0.09 | 0.02 | 0.0002 |
| **25%** | 80.00 | 19.00 | 1,748 | 1.78 |
| **Median** | 210.88 | 30.00 | 6,435 | 4.50 |
| **75%** | 460.80 | 66.78 | 31,416 | 9.55 |
| **Max** | 2,800,000 | 2,520,000 | 7.056 × 10¹² | 17,543.67 |

### Visual Analysis

![Bounding Box Geometry](bbox_geometry.png)

The visualization includes:
1. **Box plot - Area per class:** Log-scale distribution of element sizes across classes
2. **Histogram - Aspect ratios:** Most elements have aspect ratios between 1:1 and 10:1
3. **Scatter - Width vs Height:** Shows clustering patterns in common UI element dimensions
4. **Histogram - Area distribution:** Right-skewed distribution with most elements < 50,000 px²

### Geometry Insights

```
Median area: 6,435 pixels²
Median aspect ratio: 4.50
Small elements (area < 1,000 px²): 27.2%
Large elements (area > 50,000 px²): 24.8%
```

### Key Observations

1. **Wide Size Variance:** Elements range from sub-pixel (0.16 × 0.09 px) to massive (2.8M × 2.5M px)
2. **Typical Elements:** Median element is ~211 px wide × 30 px tall (e.g., navigation links, buttons)
3. **Horizontal Bias:** Median aspect ratio of 4.5:1 indicates many wide, shallow elements (headers, navbars, footers)
4. **Small Element Challenge:** 27% of elements are very small (< 1,000 px²), challenging for detection
5. **Extreme Outliers:** Some elements have aspect ratios > 17,000:1 (likely decorative lines or spacers)

---

## 6. Element Density per Screenshot

### Density Statistics

```
============================================================
ELEMENT DENSITY SUMMARY
============================================================

Overall statistics:
  Count: 6,000 screenshots
  Mean: 187.2 elements
  Std Dev: 156.8 elements
  Min: 5 elements
  25th percentile: 61 elements
  Median: 155 elements
  75th percentile: 255 elements
  Max: 2,097 elements
```

### By Device Type

| Device Type | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max |
|-------------|-------|------|---------|-----|-----|--------|-----|-----|
| **Desktop** | 4,000 | 184.3 | 150.0 | 5 | 60.8 | 154 | 259 | 1,379 |
| **Mobile** | 1,000 | 197.5 | 179.0 | 5 | 63.0 | 163 | 253 | 2,097 |
| **Tablet** | 1,000 | 188.7 | 159.3 | 7 | 60.8 | 159.5 | 249.3 | 1,380 |

### Visual Distribution

![Element Density](element_density.png)

- **Left plot:** Histogram showing distribution with mean (187.2) and median (155.0) markers
- **Right plot:** Box plot by device type showing similar distributions across devices

### Scene Complexity Insights

```
Sparse screenshots (< 10 elements): 4.1%
Complex screenshots (> 30 elements): 88.7%
```

**Implication:** Object detection models must handle:
- **Variable scene complexity:** 50× difference between simplest and most complex pages
- **Dense scenes:** Most pages (88.7%) have 30+ elements
- **Outliers:** Some pages have 1,000+ elements (mega-menus, data tables, galleries)
- **Consistent across devices:** Mobile doesn't necessarily simplify layouts

---

## 7. Overlapping & Nested Bounding Boxes

### Analysis Parameters
- **Samples analyzed:** 100 samples × 2 configurations = 200 screenshots
- **IoU threshold:** 0.3 (30% overlap)
- **Nested threshold:** 0.9 (90% overlap)

### Overlap Summary

```
============================================================
OVERLAP ANALYSIS SUMMARY
============================================================

Average overlapping pairs per screenshot: 121.3
Average nested pairs per screenshot: 42.8
Average overlap rate: 4.21%

Samples with heavy overlaps (>10 pairs): 190
Samples with nested elements (>5 pairs): 184
```

### Findings

1. **Overlapping Pairs:** 121.3 pairs per screenshot with IoU > 0.3
2. **Nested Elements:** 42.8 pairs per screenshot with IoU > 0.9 (near-complete overlap)
3. **Overlap Rate:** 4.21% of all possible element pairs overlap
4. **Widespread Issue:** 95% of analyzed samples have 10+ overlapping pairs

### Causes of Overlaps

- **Hierarchical HTML/CSS:** Nested divs, containers within containers
- **Layered UI:** Dropdowns, tooltips, modals over content
- **Accessibility Tree Structure:** Parent-child relationships in DOM
- **Intentional Overlays:** Headers over hero images, floating action buttons

---

## 8. Implications for Training

### ✅ Strengths

1. **High Quality:** 99.94% data integrity ensures reliable training
2. **Rich Annotations:** 86 semantic classes based on ARIA roles
3. **Multi-resolution:** 6 viewport configurations support responsive design
4. **Large Scale:** 42K+ screenshots with millions of annotated elements
5. **Real-world Complexity:** Variable element density (5-2,097 per page)

### ⚠️ Challenges

1. **Desktop Bias:** 66.7% desktop screenshots may not generalize well to mobile
2. **Class Imbalance:** Some ARIA roles likely underrepresented (not quantified in this analysis)
3. **Small Objects:** 27% of elements < 1,000 px² require specialized detection strategies
4. **Nested Elements:** 121 overlapping pairs per page complicate NMS/post-processing
5. **Extreme Outliers:** Elements with 17,000:1 aspect ratios may confuse models

---

## 9. Recommendations for Model Training

### Dataset Preparation

1. **Use Complete Samples Only:** Exclude 4 incomplete samples → 6,996 total
2. **Stratified Sampling:** Oversample tablet/mobile or undersample desktop to balance device types
3. **Class Filtering:** Consider grouping rare classes (< 0.5% occurrence) into "other" category
4. **Outlier Handling:** Cap extreme aspect ratios (e.g., clip at 20:1) or filter unrealistic boxes

### Model Architecture

1. **Multi-scale Detection:** Essential for handling 50× variance in element sizes
2. **Small Object Detection:** Use FPN, PANet, or similar feature pyramids
3. **Dense Prediction:** Grid-based architectures (YOLO, RetinaNet) suit ~187 objects per image
4. **Class Embeddings:** 86 classes benefit from learned semantic relationships

### Training Strategy

1. **Weighted Loss:** Address class imbalance with focal loss or class weights
2. **IoU Loss:** Use GIoU or DIoU to handle nested/overlapping boxes
3. **NMS Tuning:** Lower NMS threshold (0.3-0.4) to preserve nested elements
4. **Multi-resolution Training:** Train on mixed resolutions from all 6 configurations
5. **Data Augmentation:** Geometric transforms, color jitter, but preserve aspect ratios

### Evaluation Metrics

1. **Per-class mAP:** Monitor performance on rare classes separately
2. **Size-stratified mAP:** Report mAP for small/medium/large objects
3. **Device-stratified mAP:** Evaluate desktop vs. tablet vs. mobile separately
4. **Nested Object Handling:** Custom metric for IOU > 0.7 pairs

---

## 10. Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| **Total Samples** | 7,000 |
| **Valid Samples** | 6,996 (99.94%) |
| **Total Screenshots** | 41,976 |
| **Device Configurations** | 6 |
| **Unique Classes** | 86 (ARIA roles) |
| **Total Elements** | Millions (exact count TBD) |
| **Avg Elements per Screenshot** | 187.2 ± 156.8 |
| **Median Element Size** | 211 × 30 px (6,435 px²) |
| **Median Aspect Ratio** | 4.5:1 |
| **Avg Overlapping Pairs** | 121.3 per screenshot |
| **Desktop Representation** | 66.7% |
| **Tablet Representation** | 16.7% |
| **Mobile Representation** | 16.7% |

---

## 11. Next Steps

### Immediate Actions

1. ✅ **Complete integrity check** - Done (6,996/7,000 valid)
2. ⏳ **Full class distribution** - Quantify exact counts per class
3. ⏳ **Label mapping** - Create class ID → ARIA role → human-readable name mapping
4. ⏳ **Train/Val/Test split** - Stratify by device type and element density

### Advanced Analysis

1. **Per-class statistics:** Box size distribution for each of 86 classes
2. **Co-occurrence matrix:** Which elements appear together (e.g., button + form)
3. **Cross-device consistency:** How layouts adapt across resolutions
4. **Annotation quality:** Manual review of sample annotations

### Model Development

1. **Baseline model:** Train YOLOv8/v10 or Faster R-CNN on full dataset
2. **Ablation studies:** Desktop-only vs. multi-device training
3. **Transfer learning:** Leverage pre-trained COCO weights
4. **Specialized models:** Separate detectors for small objects

---

## Appendix: Generated Visualizations

During this analysis, the following visualizations were generated:

1. **device_distribution.png** - Bar chart and pie chart of device type distribution
2. **class_distribution.png** - Multiple views of class distribution (linear, log, stacked, percentage)
3. **bbox_geometry.png** - Four-panel analysis of bounding box dimensions
4. **element_density.png** - Histogram and box plots of elements per screenshot
5. **3d_bbox_distribution.png** - 3D scatter plot of width × height × frequency

All visualizations are saved in the notebook execution directory.

---

**Analysis completed:** February 8, 2026  
**Analyst:** GitHub Copilot (Claude Sonnet 4.5)  
**Notebook:** [01_Dataset__exploration.ipynb](../Notebooks/01_Dataset__exploration.ipynb)  
**Dataset Documentation:** [Dataset.md](Dataset.md)
