# Dataset Quality Monitoring System

## The Problem (Before May 3)

ChatGPT identified three critical weaknesses:

1. **Per-image balancing is incomplete**
   - Class caps prevent domination but don't ensure diversity
   - Pages with 0 dropdowns stay 0 (can't create what's not there)
   - Result: Still have imbalanced images even with caps

2. **No global dataset-level monitoring**
   - No idea if class distribution is balanced until training fails
   - Hidden imbalance (55% links, 45% other) goes undetected
   - No early warning system

3. **No quality rejection or tracking**
   - Every page accepted, including sparse/useless ones (1–2 annotations)
   - Can't distinguish high-quality from low-quality images
   - No data-driven filtering decisions possible

## The Solution (May 3)

Three complementary systems were implemented:

### System 1: Per-Image Quality Assessment ✅

**What it does**: Flags images that fail quality criteria, but still saves them.

**Three checks**:
- Minimum 5 annotations per image
- At least 2 different classes per image
- No single class > 80% of annotations

**Output**: Quality flags in JSON for each image:
```json
{
  "url": "https://example.com",
  "qualityFlags": ["low_annotation_count(3)", "poor_class_diversity(1)"],
  "classDistribution": { "button": 1, "link": 2 },
  "annotations": [...]
}
```

**Why it matters**:
- Know EXACTLY which images are low-quality
- Later filters can exclude flagged images if needed
- ~15–25% expected to have ≥1 flag
- Data-driven decision: "Should I reject these or use weighted loss?"

**Example scenario**:
```
Image 1: 50 annotations, 8 classes, balanced
  → No flags ✓

Image 2: 3 annotations, 1 class (all buttons)
  → FLAGS: low_annotation_count(3), poor_class_diversity(1)
  → Still saved, but marked as low-quality

Image 3: 15 annotations, 1 class (85% links)
  → FLAG: class_imbalance(link:85%)
  → Saved but flagged for imbalance
```

### System 2: Global Dataset Metrics ✅

**What it does**: Tracks class distribution across ALL images and exports statistics.

**How it works**:
- During crawl: aggregate counts from every `classDistribution` field
- Track low-quality image count
- After crawl: write `dataset-metrics.json`

**Output file**: `Dataset/url-sources/dataset-metrics.json`
```json
{
  "crawlSummary": {
    "totalScreenshots": 100000,
    "totalAnnotations": 6500000,
    "lowQualityImages": 16000,
    "lowQualityPercentage": "16.0%"
  },
  "classDistribution": {
    "counts": {
      "link": 1300000,
      "button": 750000,
      "input": 650000,
      "image": 900000,
      "dropdown": 350000,
      "modal": 250000,
      "nav": 450000,
      "form": 400000,
      "header": 200000,
      "footer": 250000
    },
    "percentages": {
      "link": "20.0%",
      "button": "11.5%",
      "input": "10.0%",
      "image": "13.8%",
      "dropdown": "5.4%",
      "modal": "3.8%",
      "nav": "6.9%",
      "form": "6.2%",
      "header": "3.1%",
      "footer": "3.8%"
    },
    "uniqueClasses": 10
  }
}
```

**Why it matters**:
- See global balance BEFORE training starts
- Detect systematic issues: "buttons are only 8%, should be 10%"
- Quantify imbalance: "20% links is worse than the 10% target"
- Make data-driven rebalancing decisions
- Export metrics for reports/analysis

**Example decision**:
```
Current distribution:  button: 11.5%, input: 10.0%, link: 20.0%
Target distribution:   button: 10.0%, input: 10.0%, link: 10.0%

Decision: During training, oversample button/input by 1.15× and downsample link by 2×
```

### System 3: Real-Time Monitoring ✅

**What it does**: Stream quality feedback during crawl.

**Progress output** (every 50 URLs):
```
Processed 100/50000 URLs (200 screenshots, 920 annotations, 15 low-quality)
```

**Final summary** (at completion):
```
Crawl Summary:
Completed 50000/50000 URLs
Saved 100000 screenshots
Extracted 6500000 annotations
Low-quality images: 16000 (16.0%)

Class Distribution (Global):
  link: 1300000 annotations (20.0%)
  button: 750000 annotations (11.5%)
  input: 650000 annotations (10.0%)
  image: 900000 annotations (13.8%)
  nav: 450000 annotations (6.9%)
  form: 400000 annotations (6.2%)
  dropdown: 350000 annotations (5.4%)
  modal: 250000 annotations (3.8%)
  header: 200000 annotations (3.1%)
  footer: 250000 annotations (3.8%)

Dataset metrics saved to: Dataset/url-sources/dataset-metrics.json
```

**Why it matters**:
- Real-time visibility: detect problems while crawl runs
- Early warning: "Imbalance is happening, should I adjust caps?"
- Confidence: "Quality looks good, safe to proceed to training"
- Audit trail: proof of data quality for reproducibility

---

## How They Work Together

```
┌─ CRAWL PHASE ──────────────────────────────────┐
│                                                 │
│  For each URL:                                  │
│    1. Load page, interact, scroll               │
│    2. Extract annotations                       │
│    3. Apply per-image assessment ──┐            │
│       ├─ min annotations?           │           │
│       ├─ class diversity?           ├→ Flags   │
│       └─ class balance?             │  added   │
│    4. Aggregate global counts ──────┤──────┐   │
│    5. Save image + JSON + flags     │      │   │
│    6. Report progress every 50 ────→│      │   │
│                                     │      │   │
└─────────────────────────────────────┼──────┼───┘
                                      │      │
                                      ↓      ↓
                            ┌────────────────────┐
                            │  FINAL REPORT      │
                            ├────────────────────┤
                            │ Per-image flags    │
                            │ (saved in JSON)    │
                            │                    │
                            │ Global metrics     │
                            │ (in .json file)    │
                            │                    │
                            │ Console summary    │
                            │ (printed out)      │
                            └────────────────────┘
                                      │
                                      ↓
                            ┌────────────────────┐
                            │  USER DECISION     │
                            ├────────────────────┤
                            │ Filter flagged?    │
                            │ Adjust thresholds? │
                            │ Oversample rare?   │
                            │ Ready for training?│
                            └────────────────────┘
```

---

## Data Flow Example

**Single Image Processing**:

```typescript
// 1. Capture variant (load page, extract annotations)
const annotations = [
  { class: 'button', x: 10, y: 20, width: 100, height: 40 },
  { class: 'button', x: 120, y: 20, width: 100, height: 40 },
  { class: 'input', x: 10, y: 70, width: 200, height: 30 },
  // Total: 3 annotations, 2 classes
];

// 2. Apply per-image assessment
const { flags, classDistribution } = assessImageQuality(annotations);
// flags = ["low_annotation_count(3)"]  ← has 3, needs 5
// classDistribution = { button: 2, input: 1 }

// 3. Save to JSON
{
  "url": "https://example.com",
  "variant": "light",
  "annotationCount": 3,
  "qualityFlags": ["low_annotation_count(3)"],
  "classDistribution": { "button": 2, "input": 1 },
  "annotations": [...]
}

// 4. Add to manifest
manifestEntries.push({
  fileName: "00001_light_example.com_root_abc123.webp",
  status: "saved",
  qualityFlags: ["low_annotation_count(3)"],
  classDistribution: { "button": 2, "input": 1 }
});

// 5. Aggregate globally
globalClassCounts.button += 2;
globalClassCounts.input += 1;
lowQualityCount += 1;  // has ≥1 flag
```

**After All 100k Images**:

```json
{
  "globalClassCounts": {
    "button": 750000,
    "input": 650000,
    "link": 1300000,
    ...
  },
  "lowQualityCount": 16000,
  "totalAnnotations": 6500000
}

// Export to metrics file
{
  "classDistribution": {
    "percentages": {
      "button": "11.5%",
      "input": "10.0%",
      "link": "20.0%"
    }
  },
  "lowQualityImages": 16000,
  "lowQualityPercentage": "16.0%"
}
```

---

## Quality Thresholds (Tunable)

Current thresholds in [Dataset/src/paths.ts](Dataset/src/paths.ts):

```typescript
export const qualityThresholds = {
  minAnnotationsPerImage: 5,        // Too strict? Lower to 3
  minClassDiversity: 2,             // Too lenient? Raise to 3
  maxSingleClassRatio: 0.8          // Too strict? Raise to 0.9
} as const;
```

**Tuning guide**:
- If > 30% images are flagged: thresholds too strict (lower values)
- If < 5% images are flagged: thresholds too lenient (raise values)
- Target: ~15–20% low-quality images (removes obvious junk, keeps most data)

---

## Usage After Crawl

### 1. Check Metrics
```bash
cat Dataset/url-sources/dataset-metrics.json | jq '.classDistribution'
```

### 2. Analyze Low-Quality Images
```bash
# Count images with quality flags
jq -s '[.[] | select(.qualityFlags)] | length' Dataset/raw/screenshots/manifest.jsonl

# Which flags are most common?
jq -s '[.[].qualityFlags[] | split("(")[0]] | group_by(.) | map({flag: .[0], count: length})' \
  Dataset/raw/screenshots/manifest.jsonl
```

### 3. Filter Dataset (Optional)
```bash
# Create "high-quality only" subset
jq -s 'map(select(.qualityFlags == null or (.qualityFlags | length == 0)))' \
  Dataset/raw/screenshots/manifest.jsonl > Dataset/raw/screenshots/manifest-hq-only.jsonl
```

### 4. Decision: Train or Rebalance?
- If low-quality % is 15–25%: **Good**, proceed with training
- If low-quality % is > 30%: **Adjust thresholds**, re-run quality assessment
- If class imbalance is severe: **Apply weighted loss** or **oversample rare classes**

---

## Expected Results

After full 50k-URL crawl:

| Metric | Value | Status |
|---|---|---|
| Total images | 100,000 | ✅ |
| Total annotations | ~6.5M | ✅ |
| Low-quality images | 15–20% | ✅ Expected |
| Class balance | ~9–20% per class | ✅ Acceptable |
| Most common class | link (20%) | ⚠️ Capped, acceptable |
| Rarest class | header (3%) | ⚠️ Rare but present |

---

## Files Generated

| File | Purpose |
|---|---|
| `Dataset/raw/screenshots/manifest.jsonl` | Per-image metadata + flags |
| `Dataset/url-sources/dataset-metrics.json` | Global distribution + summaries |
| `Dataset/url-sources/crawl-failures.json` | Failed URLs + error messages |
| Console output | Real-time progress + final summary |

---

## Next Steps

1. **Restart crawl** with quality monitoring active
2. **Monitor progress** in console (real-time feedback)
3. **Review metrics** after crawl completes
4. **Make training decisions**:
   - Apply weighted loss for rare classes
   - Oversample rare classes
   - Exclude low-quality images (optional)
   - Adjust thresholds if needed (re-assess only, no re-capture)
5. **Proceed to Phase 3**: Deduplication + format conversion

---

**Status**: ✅ Three quality systems implemented and tested. Ready for full dataset crawl.
