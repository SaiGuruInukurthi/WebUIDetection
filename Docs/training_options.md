# Training Options for WebUI Detection

**Dataset Context**: WebUI Balanced 7K (Desktop), 27,996 screenshots, 10 classes, 2.56M bounding boxes  
**Key Challenges**: Dense overlap, extreme class imbalance (45.97% → 0.42%), 27% small elements  
**Date**: February 9, 2026

---

## Tier 1: Standard, Safe, High-ROI Options

### 1. Single Multi-Class Object Detector (Baseline)

**Description**  
One detector with 10 classes using a shared backbone and head. Standard approach for object detection tasks.

**Examples**
- YOLOv8 / YOLOv10
- Faster R-CNN
- RetinaNet

**Pros**
- ✅ Correct inductive bias for UI element detection
- ✅ Handles overlap via class competition in NMS
- ✅ Simple training pipeline and inference
- ✅ Strong baseline for comparison
- ✅ Well-documented, mature frameworks
- ✅ Fast inference (single forward pass)

**Cons**
- ❌ Class imbalance affects learning (dominant classes dominate loss)
- ❌ Confusion between semantically similar classes (e.g., StaticText vs paragraph)
- ❌ Small element detection may suffer without targeted loss
- ❌ No explicit handling of UI-specific relationships

**When to Use**
- Need a defensible reference baseline
- Limited time or resources
- Require fast training and deployment

**Expected Performance**: mAP@50: 60-70% (dependent on hyperparameters)

---

### 2. Multi-Class Detector + Improved Loss (Baseline+)

**Description**  
Same architecture as #1, but with loss engineering to handle dataset characteristics.

**Enhancements**
- Focal loss (address class imbalance)
- Class weighting (boost minority classes)
- Size-aware loss buckets (small/medium/large elements)
- Optional: Hard negative mining

**Pros**
- ✅ All benefits of baseline detector
- ✅ Directly addresses class imbalance (45.97% none → 0.42% list)
- ✅ Improves small object detection (27% of elements)
- ✅ Minimal architectural complexity
- ✅ More stable training curves
- ✅ Often gives 5-10% mAP improvement over baseline

**Cons**
- ❌ Requires careful hyperparameter tuning (alpha, gamma for focal loss)
- ❌ Class weights need experimentation
- ❌ Still doesn't separate detection from classification conceptually

**When to Use**
- Want better baseline without major architectural changes
- Have time for loss function experimentation
- Dataset shows clear class/size imbalance (✓ your case)

**Expected Performance**: mAP@50: 65-75%

---

## Tier 2: Structured Decomposition

### 3. Semantic Class Grouping (Few Detectors, Not Many)

**Description**  
Group 10 classes into 4-5 semantic categories, train one detector per group.

**Example Grouping**
- **Textual**: StaticText, paragraph, heading (68.48% of elements)
- **Navigation**: link, list, listitem (13.65%)
- **Containers**: generic, none, LineBreak (53.78%)
- **Media**: img (0.74%)

**Pros**
- ✅ Reduces inter-class confusion within logical groups
- ✅ Maintains class competition where it makes sense
- ✅ Lower inference cost than per-class models
- ✅ Fits UI semantic structure naturally
- ✅ Can tune each detector independently
- ✅ Easier to debug category-specific failures

**Cons**
- ❌ Requires domain knowledge for grouping
- ❌ 4-5x training time (parallel training possible)
- ❌ Slightly more complex inference pipeline
- ❌ Overlapping logic between groups needs careful handling
- ❌ Post-processing to merge detections

**When to Use**
- Classes naturally cluster semantically
- Want to leverage UI domain structure
- Can afford multiple training runs

**Expected Performance**: mAP@50: 68-78% (better per-group optimization)

---

### 4. Hierarchical Detection (Coarse → Fine)

**Description**  
Two-stage approach where Stage 1 detects coarse categories, Stage 2 refines within category.

**Pipeline**
- **Stage 1**: Detect 4 coarse categories (text/nav/media/container)
- **Stage 2**: 4 specialist networks refine to specific classes

**Pros**
- ✅ Mirrors DOM hierarchy structure
- ✅ Cleaner learning signal per stage
- ✅ Reduces cross-category misclassification
- ✅ Can use different architectures per stage
- ✅ Academically defensible (hierarchical reasoning)
- ✅ Can incorporate accessibility tree structure

**Cons**
- ❌ Two training stages (sequential dependency)
- ❌ More engineering complexity
- ❌ Error propagation from Stage 1 to Stage 2
- ❌ Slower inference (two passes)
- ❌ Requires careful category boundary definition

**When to Use**
- Dataset has clear hierarchical structure (✓ your case with ARIA roles)
- Research/academic project with novelty requirement
- Time for multi-stage development

**Expected Performance**: mAP@50: 70-80% (if stages are well-optimized)

---

## Tier 3: Two-Stage & Hybrid Models

### 5. Class-Agnostic Detector + CNN Classifier (Recommended)

**Description**  
Stage 1 detects all UI elements (single class), Stage 2 classifies each detection into 10 classes.

**Pipeline**
- **Stage 1**: YOLO/Faster R-CNN (single "ui_element" class)
- **Stage 2**: ResNet/EfficientNet classifier (10 classes)

**Pros**
- ✅ **CNNs used correctly** (classification, not detection)
- ✅ Cleaner detection supervision (no class confusion during localization)
- ✅ Easier class imbalance handling in Stage 2 (balanced sampling)
- ✅ Debuggable (separate detection vs classification errors)
- ✅ Flexible (swap components independently)
- ✅ Can apply different augmentations per stage
- ✅ Good academic narrative (problem decomposition)

**Cons**
- ❌ Two training stages required
- ❌ Slight inference overhead (detector + N × classifier)
- ❌ Needs good proposals from Stage 1 (high recall critical)
- ❌ More code complexity

**When to Use**
- Want to show proper CNN usage (classification )
- Need interpretable error analysis
- Class imbalance is severe (✓ your case: 45.97% vs 0.42%)
- "Ensemble-like" approach without independence issues

**Expected Performance**: mAP@50: 70-82% (best decomposition approach)

**This is our chosen method.**

---

### 6. Shared Backbone, Multi-Head Architecture

**Description**  
One backbone with multiple specialized heads for different tasks or class groups.

**Architecture**
- Shared backbone (ResNet/CSPDarknet)
- Multiple detection heads (per-group or per-task)
- Optional auxiliary heads (density prediction, text-heavy classifier)

**Pros**
- ✅ Efficient inference (single backbone pass)
- ✅ Shared feature learning across tasks
- ✅ Best of specialization + shared context
- ✅ Modern architectural pattern
- ✅ Can add auxiliary supervision (density, visibility)

**Cons**
- ❌ Custom architecture implementation required
- ❌ More complex training (multi-task loss balancing)
- ❌ Difficult to debug (entangled heads)
- ❌ Hyperparameter tuning more involved

**When to Use**
- Building custom architecture from scratch
- Can leverage multi-task learning data
- Need efficient inference above all

**Expected Performance**: mAP@50: 72-85% (if well-tuned)

---

## Tier 4: Advanced / Research-Grade Options

### 7. Transformer-Based Detectors

**Description**  
Use transformer architectures for detection, leveraging self-attention for global context.

**Examples**
- DETR (Detection Transformer)
- Deformable DETR
- Conditional DETR

**Pros**
- ✅ Naturally handle overlap via set prediction
- ✅ Global context reasoning (entire page)
- ✅ No hand-crafted anchors
- ✅ Strong for dense scenes theoretically
- ✅ Research-level novelty

**Cons**
- ❌ Slow convergence (needs 300+ epochs)
- ❌ Requires large datasets (28K may be insufficient)
- ❌ Heavy computational cost (attention is O(n²))
- ❌ Difficult hyperparameter tuning
- ❌ Less mature than CNN detectors

**When to Use**
- Research novelty is primary goal
- Have computational resources for long training
- Dataset size >50K samples

**Expected Performance**: mAP@50: 65-80% (high variance, needs careful tuning)

---

### 8. Anchor-Free Dense Predictors

**Description**  
Direct keypoint-based or center-based detection without anchor heuristics.

**Examples**
- FCOS (Fully Convolutional One-Stage)
- CenterNet
- CornerNet

**Pros**
- ✅ Better for dense layouts (no anchor grid constraints)
- ✅ No anchor hyperparameters
- ✅ Simpler pipeline conceptually
- ✅ Can handle arbitrary aspect ratios

**Cons**
- ❌ Sensitive to small objects (centerness issue)
- ❌ Harder to tune than YOLO
- ❌ Requires careful post-processing
- ❌ Less robust on small elements (27% in your dataset)

**When to Use**
- Anchor tuning is problematic
- Dense packed layouts (✓ your case)
- Research exploration

**Expected Performance**: mAP@50: 60-75%

---

### 9. Multitask Learning with Auxiliary Signals

**Description**  
Train detector with auxiliary tasks using rich dataset annotations.

**Auxiliary Tasks**
- Element density prediction (low/medium/high)
- Visibility flag classification
- Box size category (small/medium/large)
- ARIA role semantic embedding

**Pros**
- ✅ Leverages full dataset richness
- ✅ Better generalization via multi-task learning
- ✅ Can improve primary task performance
- ✅ Rich research narrative

**Cons**
- ❌ Research-level complexity
- ❌ Loss balancing difficult
- ❌ Unclear which auxiliary tasks help
- ❌ More training data preparation

**When to Use**
- Research project with novelty requirement
- Rich auxiliary annotations available (✓ your case)
- Time for experimentation

**Expected Performance**: mAP@50: 68-82% (high variance)

---

## Tier 5: What NOT to Do

### ❌ 10 Independent CNNs (One Per Class)

**Why Not**
- Ignores dataset structure (overlapping elements)
- CNNs for detection is wrong application
- 10× training cost, 10× inference cost
- No shared learning across classes
- Merging predictions is heuristic and fragile

**Never use this approach for object detection.**

---

### ❌ 10 Independent YOLO Models

**Why Not**
- Massive redundancy (shared features wasted)
- 10× resource cost
- Complex ensemble logic needed
- YOLO already handles multi-class natively
- Post-processing NMS across 10 models is messy

---

### ❌ Sliding Window CNN Detection

**Why Not**
- Outdated approach (pre-2015)
- Extremely slow (thousands of windows)
- Poor handling of scale variation
- Modern detectors are strictly better

---

### ❌ Heuristic Ensemble Merging

**Why Not**
- "Average confidences" has no theoretical basis
- Voting schemes fail with overlap
- No guarantee of improvement
- Hard to debug failures
- Modern NMS with multi-class is better

---

## Decision Matrix

| Goal | Best Option | Runner-Up |
|------|-------------|-----------|
| **Strong baseline** | Multi-class detector + improved loss | Single detector baseline |
| **Do something different** | Two-stage (detector + CNN) | Semantic grouping |
| **Use CNNs properly** | Two-stage (detector + CNN) | Shared backbone multi-head |
| **Research novelty** | Hierarchical detection | Multitask learning |
| **Best trade-off** | Two-stage (detector + CNN) | Semantic grouping |
| **Fastest to implement** | Single detector baseline | Baseline + improved loss |
| **Best expected mAP** | Shared backbone multi-head | Two-stage (detector + CNN) |

---

## Performance Expectations Summary

| Approach | Expected mAP@50 | Training Complexity | Inference Speed |
|----------|-----------------|---------------------|-----------------|
| Baseline detector | 60-70% | Low | Fast |
| Baseline + improved loss | 65-75% | Low | Fast |
| Semantic grouping | 68-78% | Medium | Medium |
| Hierarchical | 70-80% | High | Medium-Slow |
| **Two-stage (CNN classifier)** | **70-82%** | **Medium** | **Medium** |
| Multi-head backbone | 72-85% | High | Fast |
| Transformers | 65-80% | Very High | Slow |
| Anchor-free | 60-75% | Medium | Fast |
| Multitask | 68-82% | Very High | Medium |

---

## Recommendation

For the WebUI Balanced 7K Desktop dataset:

**Primary Choice**: **Option 5 - Two-Stage (Class-Agnostic Detector + CNN Classifier)**

**Reasons**:
1. ✅ Proper CNN usage (classification task)
2. ✅ Handles extreme class imbalance cleanly
3. ✅ Debuggable and interpretable
4. ✅ Good trade-off: novelty vs. complexity
5. ✅ Fits "ensemble-model-plan" branch conceptually
6. ✅ Strong academic narrative (problem decomposition)

**Fallback**: Option 2 (Baseline + improved loss) if time-constrained.

---

**Next Steps**: See [final_training_method.md](final_training_method.md) for implementation details.
