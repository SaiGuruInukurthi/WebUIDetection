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

## Requirements / Setup
- Node.js + npm (Project configured in `Dataset/package.json`)
- Playwright installed (`playwright` package is a dependency)
- TypeScript dev tooling (for `npm run typecheck`): `typescript`, `tsx`, `@types/node`
- Conda `WEBUI` environment for Python visualization (Pillow)

Install (if not already):

```bash
cd Dataset
npm install
# If you need the TypeScript dev deps:
npm install -D typescript tsx @types/node
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

- Type-check the TypeScript code:

```bash
npm run typecheck
```

- Stop a running workflow: press `Ctrl+C` in the terminal where the command runs. If that fails, close the terminal.

---

## Checkpoint & Resume behavior
- On startup the crawler scans `Dataset/raw/screenshots` for existing artifacts.
- For each URL index the crawler requires all 3 variants and their annotation `.json` files to consider that URL complete:
  - `XXXXX_light_... .jpg` + `XXXXX_light_... .json`
  - `XXXXX_dark_... .jpg` + `XXXXX_dark_... .json`
  - `XXXXX_mobile_... .jpg` + `XXXXX_mobile_... .json`
- The crawler will resume from the first URL missing any of the above 6 files.
- At startup the crawler logs a checkpoint summary similar to:

```
Checkpoint: 2/3000 URLs already completed (6 screenshots).
Resuming from URL 3/3000 (2998 URLs remaining).
```

- Partial URL artifacts are reprocessed from that URL to keep artifacts consistent (i.e., the URL with some missing variants will be re-captured for all variants).

---

## File layout & naming conventions
- Root output directory: `Dataset/raw/screenshots/`
- Each screenshot file name format:

```
{index_padded}_{variant}_{host}_{path_segment}_{hash}.jpg
```

Example:

```
00027_light_conference.awwwards.com_root_1cd27974f2.jpg
```

- The corresponding annotation file is the same name with `.json` extension:

```
00027_light_conference.awwwards.com_root_1cd27974f2.json
```

- Manifest and failures files (summary):
  - `Dataset/raw/screenshots/manifest.jsonl` — newline-delimited JSON entries (crawler writes per-run manifest entries)
  - `Dataset/url-sources/crawl-failures.json` — aggregated failures with reasons

---

## Annotation JSON schema
Each `.json` saved alongside a screenshot contains at least the following fields:

- `url` (string) — source URL
- `variant` ("light"|"dark"|"mobile")
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
  - `mobile`: mobile 390×844, light color scheme

- Element size thresholds (from `Dataset/src/paths.ts` / `phase2Limits`):
  - `minWidth`: 8 px
  - `minHeight`: 8 px
  - `minArea`: 64 px²

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
# number of jpgs and jsons
Start-Sleep -Seconds 10; @{screenshots=(Get-ChildItem 'c:\WebUIDetection\Dataset\raw\screenshots\*.jpg' -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count); annotations=(Get-ChildItem 'c:\WebUIDetection\Dataset\raw\screenshots\*.json' -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count)} | ConvertTo-Json
```

---

## Troubleshooting
- If screenshots appear corrupted or incomplete:
  - Ensure Playwright Chromium is up-to-date and available on the machine.
  - Confirm network conditions; the crawler uses `waitUntil: 'networkidle'` and a post-load delay. Consider increasing `phase2Limits.postLoadDelayMs` in `Dataset/src/paths.ts` for slow pages.
- If the crawler repeatedly fails for a URL, check `Dataset/url-sources/crawl-failures.json` for error messages and consider excluding problematic domains.
- If a run was interrupted, re-run `npm run crawl`; the crawler will resume from the first incomplete URL.

---

## Next steps (recommended)
- After the crawl completes, run deduplication and format conversion (COCO/YOLO) scripts (Phase 3/4). Create `Dataset/src/deduplicator.ts` and `Dataset/src/formatter.ts` if not present.
- Add automated QC checks to flag images with zero annotations or annotations outside bounds.

---

If you want, I can:
- Add a small `README` in `Dataset/` with these run commands.
- Implement the deduplication step and a COCO/YOLO exporter next.
