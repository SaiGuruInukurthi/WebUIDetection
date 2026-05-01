# Dataset Pipeline — WebUI Element Detection

Quick reference for running the data collection pipeline.

## Setup (One-time)

Ensure conda `WEBUI` environment is ready:

```bash
conda activate WEBUI
pip install pycocotools pyyaml pillow
```

From `Dataset/` folder:

```bash
npm install
```

## Running the Pipeline

### 1. Scrape & Collect URLs (50,000 target)

Fetches URLs from multiple sources and produces balanced deduplicated list:

```bash
npm run scrape
```

Output:
- `url-sources/raw-urls.txt` — raw collected URLs
- `url-sources/deduplicated-urls.txt` — final 50,000 balanced URLs
- `url-sources/url-scraper-log.json` — scrape stats

### 2. Crawl & Capture Screenshots (100,000 images)

Captures 2 variants per URL (desktop light + dark) with automatic annotations:

```bash
npm run crawl
```

**Note:** This is long-running (~800–1200 hours for full dataset). The crawler supports resumption—if interrupted, re-run the same command to resume from first incomplete URL.

Output:
- `raw/screenshots/*.webp` — screenshot images
- `raw/screenshots/*.json` — bounding box annotations
- `raw/screenshots/manifest.jsonl` — crawl manifest
- `url-sources/crawl-failures.json` — failed URLs & reasons

**Monitor progress:**
```powershell
# PowerShell: check file counts while crawling
Get-ChildItem 'raw/screenshots/*.webp' | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem 'raw/screenshots/*.json' | Measure-Object | Select-Object -ExpandProperty Count
```

### 3. Verify Annotations (Optional)

Visualize bounding boxes on sample images:

```bash
conda run -n WEBUI python src/visualize-annotations.py
```

Output: `output/visualization/viz_*.png` — annotated screenshots

## Targets & Configuration

| Parameter | Value |
|---|---|
| Unique URLs | 50,000 |
| Screenshot variants | 2 (desktop light + dark) |
| Total images | 100,000 |
| Image format | WebP (quality=80) |
| Element classes | 10 (button, input, link, nav, form, image, dropdown, modal, header, footer) |
| Desktop viewport | 1920×1080 |

## Directory Structure

```
Dataset/
├── src/
│   ├── url-scraper.ts      # URL collection & deduplication
│   ├── crawler.ts          # Screenshot capture & annotation extraction
│   ├── utils.ts            # Helpers (file I/O, URL canonicalization)
│   ├── paths.ts            # Path constants & limits
│   └── regenerate.cjs      # Quick URL regeneration from raw-urls.txt
├── url-sources/
│   ├── raw-urls.txt        # Scraped URLs (before dedup)
│   ├── deduplicated-urls.txt # Final 50k URLs
│   ├── deduplicated-urls-with-category.json
│   ├── url-scraper-log.json
│   └── crawl-failures.json
├── raw/
│   ├── screenshots/        # WebP images + JSON annotations
│   └── manifest.jsonl
├── output/
│   └── visualization/      # Sample annotated images (if visualized)
├── tests/
│   ├── classifier.test.ts
│   ├── crawler.test.ts
│   └── image-format.test.ts
├── config.yaml
├── package.json
├── tsconfig.json
└── README.md (this file)
```

## Testing

Run all tests before starting crawl:

```bash
npm run test              # Run tests once
npm run test:ui          # Run with interactive UI
npm run typecheck        # Type-check TypeScript
```

**Tests validate:**
- URL classification (10 categories)
- Balanced round-robin sampling
- WebP format (quality=80)
- Bounding box size thresholds
- Checkpoint/resume logic

## Notes

- **No mobile captures:** Dataset uses desktop 1920×1080 only (light + dark variants).
- **Resumable crawl:** Checks for existing artifacts; resumes from first incomplete URL.
- **Network errors logged:** Failed URLs and reasons saved to `crawl-failures.json`.
- **Rate limiting:** Some sites may rate-limit; failures are expected and logged.

## Full Pipeline (One Command)

To run scrape + crawl in sequence:

```bash
npm run phase2
```

This will:
1. Scrape URLs and produce deduplicated list
2. Crawl all URLs and capture screenshots + annotations

## Troubleshooting

- **Tests fail:** Check `npm run test:ui` for details.
- **Crawler stuck on single URL:** Check network; may be waiting for slow site. Ctrl+C and resume.
- **Low annotation count on images:** Some sites have minimal interactive elements; this is expected.
- **Missing raw-urls.txt:** Run `npm run scrape` first.
- **Crawl interrupted:** Simply re-run `npm run crawl`; it will resume automatically.

## Next Steps (After Crawl Completes)

- **Deduplication & NMS:** Remove overlapping annotations (Phase 3)
- **Format conversion:** Export to YOLO / COCO format (Phase 4)
- **QC & manual review:** Validate sample annotations (Phase 5)
- **Training:** Use YOLO dataset for model training
