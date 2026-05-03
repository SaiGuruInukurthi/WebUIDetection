# Quick Reference — Dataset Quality Implementation

## TL;DR: What Changed

**3 + 4 improvements for dataset quality**:

### Initial 4 (Balanced Capture)
1. ✅ Class caps (links max 20/image)
2. ✅ Interaction clicks (reveal modals/dropdowns)
3. ✅ Aggressive scrolling (5 steps, footer content)
4. ✅ Higher thresholds (15px min, filter noise)

### New 3 (Quality Monitoring)
5. ✅ Per-image quality flags (sparse/imbalanced images flagged)
6. ✅ Global metrics aggregation (class distribution exported)
7. ✅ Real-time console reports (progress + final summary)

---

## Quality Thresholds

```typescript
// File: Dataset/src/paths.ts

export const qualityThresholds = {
  minAnnotationsPerImage: 5,        // Flag if < 5
  minClassDiversity: 2,             // Flag if only 1 class
  maxSingleClassRatio: 0.8          // Flag if 1 class > 80%
} as const;

export const phase2Limits = {
  minWidth: 15,                     // Increased from 8
  minHeight: 15,                    // Increased from 8
  minArea: 225,                     // Increased from 64 (15×15)
  maxScrollSteps: 5,                // Increased from 3
  // ... other limits
} as const;
```

---

## Expected Results

| Metric | Value |
|---|---|
| Total images | 100,000 |
| Total annotations | 6.5M |
| Low-quality (flagged) | 15–20% |
| High-quality | 80–85% |
| Most balanced class | input, button (~10%) |
| Most capped class | link (20%, capped) |

---

## Quality Flag Examples

### No Flags ✅
```json
{
  "annotationCount": 48,
  "uniqueClasses": 7,
  "maxClassRatio": 0.25,
  "qualityFlags": null
}
```

### With Flags ⚠️
```json
{
  "annotationCount": 3,
  "uniqueClasses": 1,
  "maxClassRatio": 1.0,
  "qualityFlags": [
    "low_annotation_count(3)",
    "poor_class_diversity(1)",
    "class_imbalance(button:100%)"
  ]
}
```

---

## Manifest Entry (per image)

```json
{
  "url": "https://example.com",
  "variant": "light",
  "fileName": "00001_light_example.com_root_abc123.webp",
  "status": "saved",
  "annotationCount": 48,
  "qualityFlags": null,
  "classDistribution": {
    "button": 12,
    "input": 8,
    "link": 15,
    "nav": 5,
    "image": 8
  },
  "timestamp": "2026-05-03T18:15:30.123Z"
}
```

---

## Metrics File (global)

**File**: `Dataset/url-sources/dataset-metrics.json`

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
      "input": 650000
    },
    "percentages": {
      "link": "20.0%",
      "button": "11.5%",
      "input": "10.0%"
    }
  }
}
```

---

## Console Output Pattern

```
Processed 50/50000 URLs (100 screenshots, 480 annotations, 8 low-quality)
Processed 100/50000 URLs (200 screenshots, 920 annotations, 15 low-quality)
...
Processed 50000/50000 URLs (100000 screenshots, 6500000 annotations, 16000 low-quality)

Crawl Summary:
Completed 50000/50000 URLs
Saved 100000 screenshots
Extracted 6500000 annotations
Low-quality images: 16000 (16.0%)

Class Distribution (Global):
  link: 1300000 (20.0%)
  image: 900000 (13.8%)
  button: 750000 (11.5%)
  input: 650000 (10.0%)
  ... (10 classes total)

Dataset metrics saved to: Dataset/url-sources/dataset-metrics.json
```

---

## Run Commands

```bash
cd c:\WebUIDetection\Dataset

# Resume crawl with all improvements
npm run crawl

# Full pipeline (scrape + crawl)
npm run phase2

# Validate code
npm run typecheck

# Run tests
npm test
```

---

## Key Files

| File | Purpose |
|---|---|
| `Dataset/src/paths.ts` | Quality thresholds, metrics path |
| `Dataset/src/crawler.ts` | Assessment logic, aggregation, reporting |
| `Dataset/raw/screenshots/manifest.jsonl` | Per-image metadata + flags |
| `Dataset/url-sources/dataset-metrics.json` | Global metrics (generated after crawl) |
| `QUALITY_IMPROVEMENTS.md` | Full technical documentation |
| `QUALITY_MONITORING_SYSTEM.md` | Architecture & usage guide |
| `IMPLEMENTATION_FINAL_SUMMARY.md` | Comprehensive summary |

---

## Decision Tree: What to Do After Crawl

```
Read dataset-metrics.json
         ↓
    Is low-quality % acceptable (15–25%)?
         ↙          ↘
       YES          NO
       ↓            ↓
   PROCEED    Adjust thresholds
   TO          (in paths.ts)
   TRAINING    Re-run assessment
              (quick, no re-capture)
```

---

## Tuning Quality Thresholds

Current: `minAnnotationsPerImage: 5`
- Too strict? Too many flags? Lower to 3
- Too lenient? Too few flags? Raise to 7

Current: `minClassDiversity: 2`
- Too strict? Lower to 1 (allow single-class images)
- Too lenient? Raise to 3 (require 3+ classes)

Current: `maxSingleClassRatio: 0.8`
- Too strict? Raise to 0.9 (allow up to 90%)
- Too lenient? Lower to 0.7 (require more balance)

**Rule of thumb**: Target ~15–20% flagged images

---

## Quick Diagnostics

```bash
# Count total images
ls -1 Dataset/raw/screenshots/*.webp | wc -l

# Count flagged images
jq -s '[.[] | select(.qualityFlags)] | length' \
  Dataset/raw/screenshots/manifest.jsonl

# Most common flag
jq -s '[.[].qualityFlags[]? | split("(")[0]] | group_by(.) | sort_by(-length)[0]' \
  Dataset/raw/screenshots/manifest.jsonl

# Class distribution from metrics
jq '.classDistribution.percentages' \
  Dataset/url-sources/dataset-metrics.json
```

---

## Checkpoint System

Crawler automatically resumes from last incomplete URL:

```
On startup:
Checkpoint: 80/50000 URLs already completed (160 screenshots).
Resuming from URL 81/50000 (49919 URLs remaining).

On interrupt (Ctrl+C):
Graceful shutdown. Restart same command to continue.
```

---

**Status**: ✅ All systems tested and ready. Crawl can proceed with full quality visibility.
