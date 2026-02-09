# Notebook 03: Model A - Structural Element Dataset

**Execution Date:** February 9, 2026  
**Status:** ✅ All cells executed successfully - Dataset ready for training  
**Purpose:** Convert balanced_7k dataset to YOLO format for structural element detection (Model A)

---

## Overview

This notebook creates a YOLO-format dataset for detecting **9 structural element classes** from web UI screenshots. The dataset is generated from the manifests created in Notebook 02, converting bounding boxes from pixel coordinates to YOLO normalized format while applying viewport clipping and filtering.

### Output Location
- **YOLO Dataset:** `Models and outputs/Outputs/03_notebook/yolo_model_a/`
- **Config File:** `Models and outputs/Models/03_notebook/data_model_a.yaml`
- **Statistics:** `Models and outputs/Outputs/03_notebook/conversion_statistics.json`
- **Visualizations:** `Models and outputs/Outputs/03_notebook/visualizations/`

---

## Environment & Configuration

### Hardware
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)
- **CUDA:** 12.1
- **PyTorch:** 2.5.1+cu121

### Dataset Configuration
- **Source Dataset:** balanced_7k (6,999 unique web pages)
- **Image Resolutions:** 4 desktop configurations
  - 1280×720
  - 1366×768
  - 1536×864
  - 1920×1080
- **Total Screenshots:** 27,996 (6,999 × 4 resolutions)
- **Split Ratio:** 70% train / 15% val / 15% test

### Classes (9 Structural Elements)

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | none | Background/unclassified elements |
| 1 | StaticText | Text nodes without links |
| 2 | link | Hyperlinks (anchor elements) |
| 3 | generic | Generic containers (div, span) |
| 4 | listitem | List item elements |
| 5 | paragraph | Paragraph elements |
| 6 | heading | Heading elements (h1-h6) |
| 7 | img | Image elements |
| 8 | list | List containers (ul, ol) |

**Note:** LineBreak class was removed from the original 10-class configuration as it had zero instances in the dataset.

### Processing Parameters
- **Minimum bbox size:** 2 pixels (width and height after clipping)
- **Viewport clipping:** Enabled (bboxes 95,274 elements filtered out across all splitses)
- **Non-structural elements:** Skipped (127,969 elements filtered out)

---

## Dataset Statistics

### Split Distribution

| Split | Images | Labels | Total Bboxes | Avg per Image | Processing Time |
|-------|--------|--------|--------------|---------------|-----------------|
| Train | 19,596 | 19,596 | 838,854 | 42.8 | 232s (3.9 min) |
| Val | 4,200 | 4,200 | 174,416 | 41.5 | 49s |
| Test | 4,200 | 4,200 | 183,340 | 43.7 | 49s |
| **TOTAL** | **27,996** | **27,996** | **1,196,610** | **42.7** | **330s (5.5 min)** |

### Processing Metrics

**Train Split:**
- Non-structural skipped: 140,289
- Structural (pre-clip): 1,754,640
- Viewport-clipped: 80,079 (4.6%)
- Discarded off-screen: 839,268 (47.8%)
- Discarded too small: 76,518 (4.4%)
- **Final retained:** 838,854 (47.8%)

**Val Split:**
- Non-structural skipped: 27,611
- Structural (pre-clip): 364,406
- Viewport-clipped: 16,622 (4.6%)
- Discarded off-screen: 173,647 (47.6%)
- Discarded too small: 16,343 (4.5%)
- **Final retained:** 174,416 (47.9%)

**Test Split:**
- Non-structural skipped: 27,374
- Structural (pre-clip): 368,698
- Viewport-clipped: 17,629 (4.8%)
- Discarded off-screen: 168,741 (45.8%)
- Discarded too small: 16,617 (4.5%)
- **Final retained:** 183,340 (49.7%)

### Class Distribution

| Class ID | Name | Train | Val | Test | Total | Train % | Val % | Test % |
|----------|------|-------|-----|------|-------|---------|-------|--------|
| 0 | none | 151,183 | 31,650 | 30,019 | 212,852 | 18.0% | 18.1% | 16.4% |
| 1 | StaticText | 282,342 | 57,849 | 64,387 | 404,578 | 33.7% | 33.2% | 35.1% |
| 2 | link | 135,296 | 27,399 | 29,476 | 192,171 | 16.1% | 15.7% | 16.1% |
| 3 | generic | 86,288 | 18,335 | 19,075 | 123,698 | 10.3% | 10.5% | 10.4% |
| 4 | listitem | 80,682 | 17,158 | 17,950 | 115,790 | 9.6% | 9.8% | 9.8% |
| 5 | paragraph | 38,541 | 7,934 | 8,124 | 54,599 | 4.6% | 4.5% | 4.4% |
| 6 | heading | 33,932 | 7,456 | 7,842 | 49,230 | 4.0% | 4.3% | 4.3% |
| 7 | img | 14,792 | 3,259 | 2,887 | 20,938 | 1.8% | 1.9% | 1.6% |
| 8 | list | 15,798 | 3,376 | 3,580 | 22,754 | 1.9% | 1.9% | 2.0% |

**Class Imbalance Ratio:** 19.3× (StaticText: 404,578 vs img: 20,938)

---

## Bounding Box Geometry Analysis

Analysis based on 2,000 randomly sampled label files (89,457 bounding boxes):

### Normalized Dimensions (YOLO format: 0.0-1.0)
- **Width:** Mean=0.275, Median=0.124, Std=0.317
- **Height:** Mean=0.130, Median=0.032, Std=0.251
- **Aspect Ratio:** Mean=5.46, Median=2.55

### Distribution Characteristics
- **Width:** Right-skewed with peak at ~0.10, secondary peak at 1.0 (full-width elements)
- **Height:** Heavily right-skewed with most bboxes <0.1 normalized height
- **Aspect Ratio:** Bimodal distribution
  - Primary peak: ~2.5-3.0 (moderate horizontal elements)
  - Secondary peak: ~20:1 (extreme horizontal elements, likely text snippets)

---

## Data Validation Results

### Label Integrity Check: ✅ **PASS**

| Split | Images | Labels | Matched | Missing Labels | Orphan Labels | Bbox Lines | Empty Files | Errors |
|-------|--------|--------|---------|----------------|---------------|------------|-------------|--------|
| Train | 19,596 | 19,596 | 19,596 | 0 | 0 | 838,854 | 0 | 0 |
| Val | 4,200 | 4,200 | 4,200 | 0 | 0 | 174,416 | 0 | 0 |
| Test | 4,200 | 4,200 | 4,200 | 0 | 0 | 183,340 | 0 | 0 |

**Validation Checks:**
- ✅ All images have corresponding label files
- ✅ No orphaned label files
- ✅ No empty label files
- ✅ All labels have 5 fields (class, x_center, y_center, width, height)
- ✅ All class IDs within valid range [0-9]
- ✅ All coordinates within valid range [8.0-1.0]

---

## Disk Usage

| Split | Images | Labels | Total |
|-------|--------|--------|-------|
| Train | 544.2 MB | 31.2 MB | 575.4 MB |
| Val | 118.7 MB | 6.5 MB | 125.2 MB |
| Test | 116.5 MB | 6.8 MB | 123.3 MB |
| **TOTAL** | **779.4 MB** | **44.5 MB** | **823.9 MB** |

---

## Visualizations Generated

1. **`structural_class_distribution.png`**
   - Dual bar charts showing absolute counts and percentage distribution
   - Clearly shows class imbalance with StaticText dominating
   - LineBreak class completely absent (~34% of all instances)
   - All 9 structural classes represented
2. **`sample_yolo_annotations.png`**
   - 2×3 grid of annotated sample images
   - Color-coded bounding boxes by class
   - Includes diverse page layouts (text-heavy, image-based, structured lists)

3. **`bbox_geometry_structural.png`**
   - 4-panel visualization:
     - Width distribution histogram
     - Height distribution histogram
     - Aspect ratio distribution histogram
     - Width vs Height scatter plot (colored by class)

---

## Quality Assessment

### ✅ Strengths

1. **Perfect Data Integrity**
   - Zero validation errors across all 27,996 images
   - All image-label pairs properly matched
   - All YOLO format constraints satisfied

2. **Good Split Consistency**
   - Class distributions are consistent across train/val/test splits
   - Average bboxes per image similar across splits (41.5-43.7)
   - Processing metrics consistent across splits

3. **Comprehensive Coverage**
   - 1.2M total bounding boxes
   - 28K images at 4 different resolutions
   - Diverse web UI layouts represented

4. **Proper Preprocessing**
   - Viewport clipping applied correctly (~4.5% of boxes clipped)
   - Small bbox filtering effective (removed ~7% of structural elements)
   - Off-screen filtering appropriate (removed ~46% outside viewport)

5. **Excellent Documentation**
   - Comprehensive statistics saved to JSON
   - All conversion parameters recorded
   - Processing time and throughput tracked

### ⚠️ Issues to Address

#### 🟡 **MAJOR: Class Imbalance**

**Issue:** Extreme variation in class frequencies

| Issue | Metric |
|-------|--------|
| Most common | StaticText: 404,578 (33.8%) |
| Least common (excluding LineBreak) | img: 20,938 (1.8%) |
| Imbalance ratio | ~20:1 (StaticText vs img) |
| Underrepresented | img (1.8%), list (1.9%), heading (4.1%) |

**Impact:**
- Model will be biased toward predicting StaticText
- Poor recall for rare classes (img, list, heading)
- May struggle with precision on minority classes

**Mitigation Strategies:**
1. ✅ Use class weights in loss function during training
2. ✅ Consider focal loss to down-weight easy examples
3. ⚠️ Evaluate performance per-class during validation
4. ⚠️ May need targeted data augmentation for rare classes

#### 🟡 **High Discard Rate (46-47%)**

**Issue:** Nearly half of structural elements discarded during conversion

**Breakdown:**
- Off-screen: ~46% (elements outside viewport)
- Too small: ~7% (< 2 pixels after clipping)
- Retained: ~47| img: 20,938 (1.8%) |
| Imbalance ratio | 19.3:1 (StaticText vs img) |
| Underrepresented | img (1.8%), list (1.9%), heading (4.1%), paragraph (4.6
- ✅ **Off-screen filtering is appropriate** - these elements aren't visible
- ✅ **Small bbox filtering is necessary** - prevents training on imperceptible elements
- ℹ️ This is expected behavior for web layouts with scrolling content

**No action needed** - discard rate is within normal range for viewport-based detection

#### 🟢 **MINOR: Extreme Aspect Ratios**
Use class-weighted loss function during training
2. Consider focal loss to down-weight easy examples (dominant classes)
3. Monitor per-class metrics during validation
4. May benefit from targeted augmentation for minority classes (img, list)

#### 🟡 **High Discard Rate (~47-50
- Separator lines
- Footer elements

**Impact:**
- May challenge a-48% (elements outside viewport)
- Too small: ~4.5% (< 2 pixels after clipping)
- Retained: ~48-50s might be harder to regress

**Mitigation:**
- ✅ YOLO architecture should handle this with proper anchor box clustering
- ⚠️ Monitor detection performance on extreme aspect ratio objects
- ℹ️ Consider aspect ratio analysis when selecting/tuning anchor boxes

---

## Files Generated

### Required for Training
1. **`yolo_model_a/data_model_a.yaml`** - YOLO dataset configuration
2. **`yolo_model_a/images/train/*.webp`** - 19,596 training images
3. **`yolo_model_a/images/val/*.webp`** - 4,200 validation images
4. **`yolo_model_a/images/test/*.webp`** - 4,200 test images
5. **`yolo_model_a/labels/train/*.txt`** - 19,596 YOLO label files
6. **`yolo_model_a/labels/val/*.txt`** - 4,200 YOLO label files
7. **`yolo_model_a/labels/test/*.txt`** - 4,200 YOLO label files

### Documentation & Analysis
8. **`conversion_statistics.json`** - Comprehensive conversion metrics
9. **`structural_class_mapping.json`** - Class ID to name mapping
10. **`visualizations/structural_class_distribution.png`** - Class distribution charts
11. **`visualizations/sample_yolo_annotations.png`** - Annotated example images
12. **`visualizations/bbox_geometry_structural.png`** - Bbox geometry analysis

---

## Next Steps & Recommendations

### Immediate Actions Required

1. **🔴 CRITICAL: Address LineBreak class**
   - Investigate why LineBreak has 0 instances
   - Decide on mitigation strategy (remove, augment, or merge)
   - Update class configuration before training

2. **🟡 Plan for class imbalance**
   - Implement class-weighted loss function
   - Consider focal loss (α-balanced focal loss)
   - Prepare per-class evaluation metrics

### Before Training

1. **Configure YOLO anchor boxes**
   - Run k-means clustering on bbox dimensions
   - Ensure anchors cover both typical (2.5:1) and extreme (20:1) aspect ratios
   - Use 3 scales × 3 aspect ratios minimum

2. **Set up training configuration**
   - Batch size appropriate for 4GB VRAM (likely 8-16)
   - Input size: 640×640 or 1280×1280 (depending on memory)
   - Test across all 4 resolutions
   - Analyze failure cases by class
   - Check for bias toward majority classes
3  - Validate on unseen web pages

7. **Error analysis**
   - Identify common misclassifications
   - Analyze performance on extreme aspect ratios
   - Check small object detection capability
   - Evaluate boundary cases (clipped elements)

4--

## Technical Notes

### YOLO Format Conversion
- Bounding boxes converted from pixel coordinates `(x1,y1,x2,y2)` to YOLO normalized format `(x_center, y_center, width, height)`
5 All coordinates normalized to [0.0, 1.0] range
- Viewport clipping applied before normalization
- Minimum size filter applied after clipping

### Processing Performance
- **Throughput:** ~85 images/second
- **Total processing time:** 5.5 minutes for 28K images
- **Efficient I/O:** WebP format for space efficiency

### Reproducibility
- All parameters documented in conversion_statistics.json
- Class mapping saved separately for reference
- Split assignments from Notebook 02 preserved
- Random seed not applicable (deterministic conversion)

---

## Summary

✅ **Dataset successfully prepared for YOLO training**  
⚠️ **Critical issue: LineBreak class must be addressed before training**  
⚠️ **Class imbalance requires mitigation during training**  
✅ **Data integrity validated - zero errors**  
✅ **Comprehensive documentation and visualizations generated**

**Ready for:** Training setup (after addressing LineBreak class issue)  
**Dataset size:** 823.9 MB  
**Total instances:** 1,196,610 bounding boxes across 27,996 images
and validated for YOLO training**  
✅ **9-class configuration finalized (LineBreak class removed)**  
⚠️ **Class imbalance (19.3×) requires weighted loss during training**  
✅ **Data integrity perfect - zero errors across all splits**  
✅ **Comprehensive documentation and visualizations generated**

**Status:** ✅ **Ready for training**  
**Dataset size:** 823.9 MB  
**Total instances:** 1,196,610 bounding boxes across 27,996 images  
**Classes:** 9 structural elements (none, StaticText, link, generic, listitem, paragraph, heading, img, list)