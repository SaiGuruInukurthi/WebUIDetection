# Final Training Method: Two-Stage Detection System

**Project**: WebUI Balanced 7K Desktop Detection  
**Date**: February 9, 2026  
**Approach**: Class-Agnostic Detector + CNN Classifier  
**Branch**: ensemble-model-plan

---

## Executive Summary

We implement a **two-stage detection system** that decomposes the UI element detection problem into:
1. **Stage 1**: Class-agnostic spatial localization (YOLO)
2. **Stage 2**: Multi-class classification (CNN)

This approach addresses our dataset challenges (extreme class imbalance, dense overlap, small elements) while demonstrating proper CNN usage for classification rather than detection.

---

## Approach Overview

### Problem Decomposition

**Traditional Single-Stage Issue**:
- Joint optimization of localization + classification
- Dominant classes (none: 45.97%) overwhelm loss gradients
- Class confusion during bbox regression
- Small elements (27%) get insufficient attention

**Our Two-Stage Solution**:
```
Input Image → [Stage 1: Detector] → Proposals
                                      ↓
                            [Stage 2: Classifier] → Final Detections (class + bbox)
```

**Key Insight**: By separating "where are elements?" from "what are they?", we:
- Simplify learning objectives per stage
- Apply targeted techniques to each subproblem
- Enable independent debugging and optimization

---

## Stage 1: Class-Agnostic Detection

### Objective
Detect **all UI elements** regardless of class, optimizing purely for spatial localization quality.

### Architecture
**Model**: YOLOv8 (single class variant)
- **Input resolution**: 640×640 (scaled from original 1280-1920 width)
- **Backbone**: CSPDarknet53
- **Neck**: PANet (multi-scale feature fusion)
- **Head**: Detection head for single "ui_element" class
- **Output**: [x, y, w, h, confidence] per detection

### Training Configuration

**Dataset Preparation**:
```
All 10 classes → relabeled as class 0 ("ui_element")
Train: 19,596 screenshots, ~1.79M boxes
Val: 4,200 screenshots, ~384K boxes
Test: 4,200 screenshots, ~384K boxes
```

**Hyperparameters**:
```yaml
epochs: 100
batch_size: 16
optimizer: AdamW
learning_rate: 0.001 (with cosine decay)
weight_decay: 0.0005
augmentation:
  - Mosaic: 0.5 probability
  - MixUp: 0.15 probability
  - HSV color jitter
  - Random horizontal flip: 0.5
  - Scale: 0.5-1.5
image_size: 640
confidence_threshold: 0.25 (training)
iou_threshold: 0.7 (NMS)
```

**Loss Function**:
```
L_stage1 = λ₁ * L_box + λ₂ * L_objectness
```
- **L_box**: CIoU loss (handles aspect ratio, distance, overlap)
- **L_objectness**: Binary cross-entropy (element vs background)
- No classification loss (single class)

**Why This Works**:
- ✅ **Cleaner supervision**: No class confusion during bbox regression
- ✅ **Better recall**: Focus on finding all elements, not classifying them
- ✅ **Handles overlap**: Class-agnostic NMS less aggressive
- ✅ **Small objects**: Full loss budget for localization quality

**Expected Stage 1 Performance**:
- **Target Recall@IoU=0.5**: >85% (high recall critical for Stage 2)
- **Precision**: ~70-75% (acceptable, Stage 2 will refine)

---

## Stage 2: CNN Classification

### Objective
Classify each Stage 1 detection into one of 10 ARIA role classes.

### Architecture
**Model**: EfficientNet-B0 (classification backbone)
- **Input**: Cropped bbox regions (224×224 resize)
- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Head**: Global average pooling → FC(1280) → Dropout(0.3) → FC(10)
- **Output**: 10-class probability distribution

**Why EfficientNet**:
- Excellent accuracy/efficiency trade-off
- Good generalization with limited data
- Fast inference per bbox (~5ms on GPU)

### Training Configuration

**Dataset Preparation**:
```python
# From Stage 1 detections on train set
For each image:
    proposals = stage1_detector(image)
    For each proposal:
        if IoU(proposal, any_gt_box) > 0.5:
            crop = extract_bbox(image, proposal)
            label = matched_gt_class
            crops.append((crop, label))

# Result: ~1.5M training crops (from 1.79M gt boxes × ~85% recall)
```

**Class Distribution** (after Stage 1):
```
Before balancing:
  none: ~691K (46%)          list: ~6K (0.4%)
  StaticText: ~287K (19%)    ...

After balanced sampling:
  All classes: ~50K samples each (total 500K)
  Under-sample majority, over-sample minority with augmentation
```

**Hyperparameters**:
```yaml
epochs: 50
batch_size: 128
optimizer: AdamW
learning_rate: 0.001 (ReduceLROnPlateau)
weight_decay: 0.0001
augmentation:
  - Random crop: 0.8-1.0 of bbox
  - Color jitter: brightness, contrast, saturation
  - Random horizontal flip: 0.5
  - Rotation: ±5 degrees
  - Cutout: 0.1 probability
label_smoothing: 0.1
mixup_alpha: 0.2
```

**Loss Function**:
```python
L_stage2 = FocalLoss(α=0.25, γ=2.0) + λ * L_aux

FocalLoss = -α * (1-p)^γ * log(p)  # Down-weight easy examples
L_aux = Optional auxiliary loss (bbox size predictor)
```

**Why Focal Loss**:
- Even after balancing, some classes harder to classify
- Down-weights easy "none" examples
- Focuses on confusing cases (StaticText vs paragraph)

**Expected Stage 2 Performance**:
- **Target Top-1 Accuracy**: >90% on balanced validation set
- **Per-class Precision**: >85% for majority classes, >70% for minority

---

## Joint System Pipeline

### Inference Flow

```python
def detect_ui_elements(image):
    # Stage 1: Detect all elements
    proposals = yolo_detector(image)  # [N, 5]: x,y,w,h,conf
    
    # Filter low-confidence
    proposals = proposals[proposals[:, 4] > 0.3]
    
    # Stage 2: Classify each proposal
    crops = [extract_bbox(image, box) for box in proposals]
    class_probs = efficientnet_classifier(crops)  # [N, 10]
    
    # Combine
    class_ids = class_probs.argmax(dim=1)
    class_confs = class_probs.max(dim=1)
    
    # Final detections
    detections = []
    for i, box in enumerate(proposals):
        detections.append({
            'bbox': box[:4],
            'class': class_ids[i],
            'confidence': box[4] * class_confs[i],  # Combined confidence
            'stage1_conf': box[4],
            'stage2_conf': class_confs[i]
        })
    
    # Multi-class NMS
    final = multi_class_nms(detections, iou_thresh=0.5)
    
    return final
```

### End-to-End Evaluation Metrics

**Primary Metrics**:
- **mAP@IoU=0.5**: Mean average precision across 10 classes
- **mAP@IoU=0.5:0.95**: COCO-style mAP (stricter)
- **Per-class AP**: Individual class performance
- **Recall@IoU=0.5**: Coverage of ground truth elements

**Secondary Metrics**:
- **Inference time**: Total time per image (both stages)
- **Detection breakdown**: Stage 1 failures vs Stage 2 misclassifications
- **Size-stratified AP**: Performance on small/medium/large elements

---

## Implementation Plan

### Notebook Structure

**Notebook 03**: Stage 1 Data Preparation
- Convert 10-class annotations → single-class YOLO format
- Generate train.txt, val.txt, test.txt
- Create data.yaml for YOLO training

**Notebook 04**: Stage 1 Training
- Train YOLOv8 class-agnostic detector
- Evaluate recall and localization quality
- Export best checkpoint

**Notebook 05**: Stage 2 Data Preparation
- Run Stage 1 on train/val sets
- Extract bbox crops matched to ground truth
- Create balanced classification dataset
- Generate PyTorch dataset and dataloaders

**Notebook 06**: Stage 2 Training
- Train EfficientNet-B0 classifier
- Apply focal loss and class balancing
- Validate on balanced and imbalanced splits

**Notebook 07**: Joint Evaluation
- Integrate both stages
- Evaluate on val/test sets
- Generate per-class metrics
- Error analysis (detection vs classification failures)
- Visualization of predictions

**Notebook 08**: Ablation Studies (Optional)
- Baseline single-stage YOLO (10 classes) for comparison
- Stage 2 variants (ResNet-18, MobileNetV3)
- Loss function ablations

---

## Advantages of This Approach

### 1. **Proper CNN Usage**
- ✅ CNNs excel at classification given pre-cropped regions
- ✅ Not forcing CNNs to do detection (wrong application)
- ✅ Academically sound and defensible

### 2. **Class Imbalance Handling**
- ✅ Stage 1: Ignore class, focus on localization
- ✅ Stage 2: Balanced sampling, focal loss, easy to apply
- ✅ Better than joint optimization with class weights

### 3. **Interpretability & Debugging**
- ✅ Can isolate Stage 1 failures (missed elements)
- ✅ Can isolate Stage 2 failures (misclassifications)
- ✅ Clear error attribution for analysis

### 4. **Flexibility**
- ✅ Can swap Stage 1 detector (YOLO → Faster R-CNN)
- ✅ Can swap Stage 2 classifier (EfficientNet → ViT)
- ✅ Can optimize stages independently

### 5. **Ensemble-Like Benefits**
- ✅ Two models with different learning objectives
- ✅ Complementary strengths (localization + classification)
- ✅ Not a naive ensemble (designed decomposition)

---

## Limitations

### 1. **Inference Speed**
- ⚠️ Two forward passes required (detector + N × classifier)
- ⚠️ ~100ms per image (Stage 1: 50ms, Stage 2: 50ms for ~100 boxes)
- **Mitigation**: Batch Stage 2 inference, use faster classifier (MobileNet)

### 2. **Error Propagation**
- ⚠️ Stage 1 missed elements cannot be recovered in Stage 2
- ⚠️ Stage 1 false positives add noise to Stage 2
- **Mitigation**: High recall target (>85%) in Stage 1, confidence filtering

### 3. **Training Complexity**
- ⚠️ Two-stage training (sequential dependency)
- ⚠️ Stage 2 dataset generation requires Stage 1 completion
- **Mitigation**: Clear notebook structure, ~2-3 days total training

### 4. **Memory Overhead**
- ⚠️ Need to store Stage 1 proposals for Stage 2 training
- ⚠️ ~1.5M bbox crops (~100GB if saved as images)
- **Mitigation**: Generate crops on-the-fly during Stage 2 training

### 5. **Hyperparameter Tuning**
- ⚠️ Two sets of hyperparameters to tune
- ⚠️ Stage 1 recall threshold affects Stage 2 dataset quality
- **Mitigation**: Start with standard values, tune Stage 1 first

### 6. **Not End-to-End**
- ⚠️ Cannot jointly optimize both stages with backpropagation
- ⚠️ Potential suboptimality vs. jointly trained system
- **Mitigation**: This is a design choice (decomposition benefits outweigh joint optimization)

---

## Expected Outcomes

### Performance Targets

**Optimistic** (if both stages perform well):
- mAP@50: 78-82%
- Inference: ~100ms per image

**Realistic** (baseline expectations):
- mAP@50: 72-76%
- Inference: ~120ms per image

**Pessimistic** (if Stage 1 struggles):
- mAP@50: 65-70%
- Need to revisit Stage 1 architecture or training

### Comparison to Baseline

**Baseline** (Single YOLOv8, 10 classes, standard training):
- Expected mAP@50: 60-65%

**Our Approach**:
- **Expected gain**: +10-15% mAP
- **Tradeoff**: 2× inference time, more training complexity

### Success Criteria

**Minimum Viable**:
- ✅ Stage 1 recall > 80%
- ✅ Stage 2 accuracy > 85%
- ✅ End-to-end mAP@50 > 70%

**Target**:
- ✅ Stage 1 recall > 85%
- ✅ Stage 2 accuracy > 90%
- ✅ End-to-end mAP@50 > 75%

**Stretch**:
- ✅ Stage 1 recall > 90%
- ✅ Stage 2 accuracy > 92%
- ✅ End-to-end mAP@50 > 80%

---

## Baseline Comparison Strategy

To validate our approach, we will train a **single-stage YOLOv8** baseline with:
- Same 10 classes
- Same train/val/test splits
- Standard hyperparameters
- Focal loss + class weights (fair comparison)

**Comparison Dimensions**:
1. **Overall mAP**: Two-stage vs single-stage
2. **Per-class AP**: Which approach handles minority classes better?
3. **Small object AP**: Which approach detects small elements better?
4. **Inference time**: Speed-accuracy tradeoff
5. **Training time**: Complexity tradeoff

---

## Academic Justification

### Problem Statement
> "Web UI element detection presents dual challenges: (1) dense overlapping layouts with 90+ elements per page, and (2) extreme class imbalance (45.97% background vs 0.42% rare elements). Joint optimization of spatial localization and semantic classification in single-stage detectors leads to conflicting learning objectives, where dominant classes overwhelm gradient signals."

### Proposed Solution
> "We decompose the problem into orthogonal subproblems: **Stage 1** performs class-agnostic detection, optimizing purely for spatial localization quality; **Stage 2** applies a CNN classifier to cropped regions with balanced sampling, focusing on semantic discrimination. This separation enables targeted optimization strategies—CIoU loss for localization, focal loss with balanced sampling for classification."

### Key Contributions
1. **Proper CNN usage**: Classification on pre-cropped regions (not detection)
2. **Class imbalance handling**: Deferred to controlled classification stage
3. **Interpretable errors**: Separate analysis of localization vs classification failures
4. **Empirical validation**: 10-15% mAP improvement over single-stage baseline

### Narrative Arc
1. **Motivation**: Single-stage struggles with our dataset characteristics
2. **Insight**: Decompose into simpler, orthogonal subproblems
3. **Implementation**: Two-stage with targeted techniques per stage
4. **Results**: Quantitative improvement + qualitative interpretability
5. **Analysis**: Error breakdown shows where each stage excels/fails

---

## Risk Mitigation

### Risk 1: Stage 1 Poor Recall
**Impact**: Stage 2 cannot recover missed elements  
**Indicators**: Recall < 80% on validation  
**Mitigation**:
- Increase confidence threshold (trade precision for recall)
- Train longer (more epochs)
- Try Faster R-CNN (higher recall, slower)

### Risk 2: Stage 2 Overfits
**Impact**: Poor generalization to test set  
**Indicators**: Train accuracy 95%, val accuracy 70%  
**Mitigation**:
- More aggressive augmentation
- Higher dropout (0.3 → 0.5)
- Reduce model capacity (EfficientNet-B0 → MobileNetV3)

### Risk 3: Inference Too Slow
**Impact**: Not practical for real applications  
**Indicators**: >200ms per image  
**Mitigation**:
- Batch Stage 2 inference
- Use MobileNetV3 instead of EfficientNet
- Quantize models (FP32 → FP16)

### Risk 4: No Improvement Over Baseline
**Impact**: Approach not justified  
**Indicators**: mAP within 2% of single-stage baseline  
**Mitigation**:
- Analyze error breakdown (is issue in Stage 1 or 2?)
- Try hierarchical approach instead
- Fall back to baseline + better loss

---

## Next Steps

**Immediate** (Week 1):
1. ✅ Document training options and final method (this file)
2. ⏳ Create Notebook 03: Stage 1 YOLO data preparation
3. ⏳ Train Stage 1 detector, validate recall target

**Short-term** (Week 2):
4. ⏳ Create proposals dataset for Stage 2
5. ⏳ Train Stage 2 classifier, validate accuracy target
6. ⏳ Integrate and evaluate joint system

**Medium-term** (Week 3):
7. ⏳ Train baseline single-stage YOLO for comparison
8. ⏳ Error analysis and visualization
9. ⏳ Write final report with results

---

## References

**Architecture Choices**:
- YOLOv8: Ultralytics (2023)
- EfficientNet: Tan & Le, ICML 2019
- Focal Loss: Lin et al., ICCV 2017

**Related Work**:
- Two-stage detection: Girshick et al., Faster R-CNN, NIPS 2015
- Class imbalance: Cui et al., "Class-Balanced Loss", CVPR 2019
- UI element detection: Nguyen et al., ICSE 2023 (WebUI Balanced 7K)

---

**Status**: Ready for implementation (Notebook 03 next)  
**Expected Completion**: 3 weeks (data prep, training, evaluation, analysis)  
**Success Metric**: mAP@50 > 75% on test set, >10% improvement over baseline
