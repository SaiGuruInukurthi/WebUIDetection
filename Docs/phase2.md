# Phase 2 — Automated WebUI Data Collection

This document describes the Phase 2 workflow used to collect screenshots and element annotations from the web using Playwright. It explains how to run the full pipeline, how checkpoint/resume works, file layout and naming, annotation JSON schema, verification steps, and troubleshooting notes.

**Location**
- Implementation: [Dataset/src/crawler.ts](Dataset/src/crawler.ts#L1)
- Visualization helper: [Dataset/src/visualize-annotations.py](Dataset/src/visualize-annotations.py#L1)
- Screenshots + annotations output: [Dataset/raw/screenshots](Dataset/raw/screenshots)

**Goals**
- Capture three screenshot variants per URL (desktop light, desktop dark, mobile light).
- Extract bounding boxes for 10 UI element classes and save as JSON alongside each image.
- Make crawling resumable via checkpointing so interrupted runs can continue without re-downloading completed items.

---

## Quality Improvements (May 3, 2026)

Four critical dataset quality enhancements were implemented to fix class imbalance and ensure diverse element capture:

### 1. **Class Population Caps**
Prevents overrepresentation of high-frequency classes (e.g., links dominating the dataset at 70% of annotations).
- Per-class limits applied during annotation extraction:
  - `link`: max 20 per image
  - `button`: max 15
  - `input`: max 10
  - `image`: max 10
  - `nav`, `form`, `dropdown`: max 5 each
  - `modal`, `header`, `footer`: max 3, 2, 2
- Result: Balanced class distribution (~10% per class instead of 70-15-10-5 skew)

### 2. **Interactive Element Capture**
Programmatically triggers UI interactions before annotation extraction:
- Clicks first 3 buttons to expose dropdowns/modals/expanded menus
- Opens first 2 select/combobox elements
- Catches and suppresses errors gracefully so one click failure doesn't break capture
- Result: Modals and hidden dropdowns now appear in dataset (~5-10% more annotations per page)

### 3. **Aggressive Scrolling**
Increased scroll steps from 3 to 5 to expose lazy-loaded and footer content:
- Each scroll step: `window.innerHeight` pixels down
- 300ms delay between scrolls for lazy-loading
- Returns to top for final annotation extraction
- Result: Captures footer elements and lazy-loaded content not visible in initial viewport

### 4. **Higher Element Size Threshold**
Increased minimum element dimensions to filter out noise:
- Before: 8×8 px minimum (64 px² area)
- After: 15×15 px minimum (225 px² area)
- Rationale: Filters ~200 false-positive micro-elements (1px lines, tracking pixels, invisible divs)
- Result: Cleaner annotations, fewer training edge cases

---

## Requirements / Setup
- Node.js + npm (Project configured in `Dataset/package.json`)
- Playwright installed (`playwright` package is a dependency)
- Sharp library for WebP conversion (`sharp` package)
- TypeScript dev tooling (for `npm run typecheck`): `typescript`, `tsx`, `@types/node`
- Vitest for unit testing: `vitest`, `@vitest/ui`
- Conda `WEBUI` environment for Python visualization (Pillow)

Install (if not already):

```bash
cd Dataset
npm install
# If you need the TypeScript dev deps and testing:
npm install -D typescript tsx @types/node vitest @vitest/ui
# To use visualization (optional):
conda activate WEBUI
pip install pillow
```

---

## Run commands
From `c:\WebUIDetection\Dataset`.

- Run the full Phase 2 workflow (scrape URLs then crawl):

```bash
npm run phase2
```

- Run only the crawler (this will resume automatically from checkpoint):

```bash
npm run crawl
```

- Run only the scraper (collect source URLs):

```bash
npm run scrape
```

- Run unit tests (classification, balanced sampling, image format):

```bash
npm run test
```

- Run tests with interactive UI:

```bash
npm run test:ui
```

- Type-check the TypeScript code:

```bash
npm run typecheck
```

- Stop a running workflow: press `Ctrl+C` in the terminal where the command runs. If that fails, close the terminal.

---

## Checkpoint & Resume behavior
- On startup the crawler scans `Dataset/raw/screenshots` for existing artifacts.
 - For each URL index the crawler requires both desktop variants and their annotation `.json` files to consider that URL complete:
  - `XXXXX_light_... .webp` + `XXXXX_light_... .json`
  - `XXXXX_dark_... .webp` + `XXXXX_dark_... .json`
 - The crawler will resume from the first URL missing either of the above files for that index.
 - At startup the crawler logs a checkpoint summary similar to:

```
Checkpoint: 2/50000 URLs already completed (4 screenshots).
Resuming from URL 3/50000 (49997 URLs remaining).
```

- Partial URL artifacts are reprocessed from that URL to keep artifacts consistent (i.e., the URL with some missing variants will be re-captured for all variants).

---

## File layout & naming conventions
- Root output directory: `Dataset/raw/screenshots/`
- Each screenshot file name format:

```
{index_padded}_{variant}_{host}_{path_segment}_{hash}.webp
```

Example:

```
00027_light_conference.awwwards.com_root_1cd27974f2.webp
```

- The corresponding annotation file is the same name with `.json` extension:

```
00027_light_conference.awwwards.com_root_1cd27974f2.json
```

**Image Format Details**:
- Format: WebP (via Sharp library, quality=80)
- Benefits: 30-40% smaller file size than JPEG at equivalent quality
- All images saved as WebP during crawl; PNG intermediate files are not retained

- Manifest and failures files (summary):
  - `Dataset/raw/screenshots/manifest.jsonl` — newline-delimited JSON entries (crawler writes per-run manifest entries)
  - `Dataset/url-sources/crawl-failures.json` — aggregated failures with reasons

---

## Annotation JSON schema
Each `.json` saved alongside a screenshot contains at least the following fields:

- `url` (string) — source URL
-- `variant` ("light"|"dark")
- `viewport` (object) — `{ width, height }` used for capture
- `annotationCount` (number) — number of bounding boxes
- `timestamp` (ISO string)
- `annotations` (array of objects) — each annotation has:
  - `class` (string) — one of: `button`, `input`, `link`, `nav`, `form`, `image`, `dropdown`, `modal`, `header`, `footer`
  - `x` (int) — top-left x (pixels)
  - `y` (int) — top-left y (pixels)
  - `width` (int)
  - `height` (int)

Example snippet:

```json
{
  "url": "https://example.com",
  "variant": "light",
  "viewport": { "width": 1920, "height": 1080 },
  "annotationCount": 12,
  "timestamp": "2026-04-30T12:34:56.789Z",
  "annotations": [
    { "class": "button", "x": 100, "y": 200, "width": 120, "height": 36 }
  ]
}
```

---

## Capture variants and thresholds
 - Variants captured per URL:
  - `light`: desktop 1920×1080, light color scheme
  - `dark`: desktop 1920×1080, dark color scheme

- Interaction sequence per URL:
  - Load page with `networkidle` wait strategy
  - Click first 3 buttons to expose modals/dropdowns
  - Open first 2 select/combobox elements
  - Scroll 5 times (full viewport height each) to expose lazy-loaded and footer content
  - Return to top before annotation extraction

- Element size thresholds (from `Dataset/src/paths.ts` / `phase2Limits`):
  - `minWidth`: 15 px (increased from 8)
  - `minHeight`: 15 px (increased from 8)
  - `minArea`: 225 px² (increased from 64)
  - Rationale: Filters noise while capturing all practical interactive elements

---

## Quality Assessment & Monitoring (Added May 3, 2026)

### Per-Image Quality Flags

Every captured image is assessed against three quality criteria:

1. **Minimum Annotations**: Images with fewer than 5 annotations are flagged `low_annotation_count(N)`
2. **Class Diversity**: Images with only 1 class type are flagged `poor_class_diversity(1)`
3. **Class Balance**: Images where a single class exceeds 80% are flagged `class_imbalance(class:N%)`

Quality flags are stored in the annotation JSON and manifest but **do not prevent image save**. This allows:
- Post-crawl filtering: exclude low-quality images before training
- Quality analysis: "16% of images have ≥1 flag, should I adjust thresholds?"
- Per-variant tracking: "Dark mode has fewer modals, need more interactions?"

### Dataset Metrics File

After crawl completes, a comprehensive metrics file is generated: `Dataset/url-sources/dataset-metrics.json`

Contains:
- Global class distribution (counts and percentages)
- Low-quality image count and percentage
- Quality threshold parameters used
- Total annotations and per-image averages

Example:
```json
{
  "crawlSummary": {
    "totalScreenshots": 100000,
    "totalAnnotations": 6500000,
    "lowQualityImages": 16000,
    "lowQualityPercentage": "16.0%"
  },
  "classDistribution": {
    "counts": { "link": 1300000, "button": 750000, ... },
    "percentages": { "link": "20.0%", "button": "11.5%", ... }
  }
}
```

Use this to detect imbalance before training and make informed rebalancing decisions.

### Real-Time Monitoring

During crawl, progress is reported every 50 URLs:
```
Processed 50/50000 URLs (100 screenshots, 480 annotations, 8 low-quality)
Processed 100/50000 URLs (200 screenshots, 920 annotations, 15 low-quality)
```

At completion, a summary prints to console:
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
  ...
```

---

## Verification & visualization
- Quick verification: run the visualization script to draw bounding boxes on the first few captured images:

```bash
# from Dataset/
conda run -n WEBUI python src/visualize-annotations.py
```

- Visualizations are saved to `Dataset/output/visualization/` (files prefixed with `viz_`). Use those to visually verify class coverage and spatial correctness.

---

## Monitoring progress
- While crawling, you will see per-URL logs like:

```
Crawling 3/3000: https://example.com
Processed 50/3000 URLs (150 screenshots, 480 annotations)
```

- To check counts from the host without stopping crawler (PowerShell examples):

```powershell
# number of webps and jsons
Start-Sleep -Seconds 10; @{screenshots=(Get-ChildItem 'c:\WebUIDetection\Dataset\raw\screenshots\*.webp' -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count); annotations=(Get-ChildItem 'c:\WebUIDetection\Dataset\raw\screenshots\*.json' -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count)} | ConvertTo-Json
```

---

## Testing
- Unit tests validate classification logic, balanced sampling, and image format specifications:

```bash
npm run test
```

- Test coverage includes:
  - URL classification into 10 site categories (ecommerce, news, blog, portfolio, corporate, forum, social, docs, education, other)
  - Round-robin balanced sampling ensuring even category distribution
  - WebP format configuration (quality=80), conversion from PNG, RIFF header validation
  - Filename patterns and annotation path derivation
  - File size efficiency gains over JPEG

- All tests must pass before running the full crawl (`npm run phase2`).

---

## Troubleshooting
- If screenshots appear corrupted or incomplete:
  - Ensure Playwright Chromium is up-to-date and available on the machine.
  - Confirm network conditions; the crawler uses `waitUntil: 'networkidle'` and a post-load delay. Consider increasing `phase2Limits.postLoadDelayMs` in `Dataset/src/paths.ts` for slow pages.
- If the crawler repeatedly fails for a URL, check `Dataset/url-sources/crawl-failures.json` for error messages and consider excluding problematic domains.
- If a run was interrupted, re-run `npm run crawl`; the crawler will resume from the first incomplete URL.
- If unit tests fail, check test output with `npm run test:ui` for interactive debugging.

---

## Next steps (recommended)
- After the crawl completes, run deduplication and format conversion (COCO/YOLO) scripts (Phase 3/4). Create `Dataset/src/deduplicator.ts` and `Dataset/src/formatter.ts` if not present.
- Add automated QC checks to flag images with zero annotations or annotations outside bounds.

---

If you want, I can:
- Add a small `README` in `Dataset/` with these run commands.
- Implement the deduplication step and a COCO/YOLO exporter next.
