# Dataset Building Plan: WebUI Element Detection with Playwright

## Overview

This plan outlines a structured pipeline to build a high-quality, annotated dataset for training WebUI element detection models (YOLO, Deformable DETR, etc.) using Playwright as the data collection engine.

---

## Phase 1: Environment Setup

### 1.1 Dependencies

```bash
npm install playwright
npm install @playwright/test
npx playwright install chromium
# Run inside the WEBUI conda environment
pip install pycocotools pyyaml pillow  # For annotation conversion
```

All Python steps in this plan should run inside the `WEBUI` conda environment, and training/testing should use Jupyter notebooks rather than standalone `.py` scripts.

### 1.2 Project Structure

```
Dataset/
├── src/
│   ├── url-scraper.ts      # Directory scraper for URL collection
│   ├── crawler.ts          # URL navigation & screenshot logic
│   ├── annotator.ts        # Bounding box extraction
│   ├── deduplicator.ts     # Overlap/nesting resolution
│   └── formatter.ts        # YOLO / COCO conversion
├── url-sources/
│   ├── raw-urls.txt        # Scraped URLs (before deduplication)
│   ├── deduplicated-urls.txt # Final cleaned URL list (3000 URLs)
│   └── url-scraper-log.json # Logs from each scraping run
├── raw/
│   ├── screenshots/        # PNG images (9000 total)
│   └── annotations/        # Raw JSON annotations
├── output/
│   ├── images/             # Final cleaned images
│   ├── labels/             # YOLO .txt label files
│   └── coco.json           # COCO format output
├── config.yaml             # Viewport, selectors, class map
└── README.md               # Dataset pipeline instructions
```

### 1.3 Config File (`config.yaml`)

```yaml
viewport:
  width: 1920
  height: 1080

# Element filtering thresholds
element_filters:
  min_width: 8          # pixels (reject smaller elements)
  min_height: 8         # pixels (reject smaller elements)
  min_area: 64          # pixels^2 (reject ~8x8 and smaller)
  max_area: 2000000     # pixels^2 (2M = 95% of viewport; reject full-screen overlays)

# 10 core interactive element classes
classes:
  - button              # <button>, [role="button"], <a> styled as button
  - input               # <input>, <textarea>, [role="textbox"], [role="combobox"]
  - link                # <a> with href (navigation links)
  - nav                 # <nav>, [role="navigation"], menu containers
  - form                # <form>, [role="form"], form containers
  - image               # <img>, [role="img"], <svg>, icons
  - dropdown            # <select>, [role="listbox"], <ul> dropdowns
  - modal               # [role="dialog"], [role="alertdialog"], modal containers
  - header              # <header>, [role="banner"], page headers
  - footer              # <footer>, [role="contentinfo"], page footers

wait_strategy: networkidle
color_schemes:
  - light
  - dark
mobile_viewport:
  width: 390
  height: 844

max_scroll_steps: 3
navigation_timeout_ms: 30000
post_load_delay_ms: 1000
```

---

## Phase 2: Data Collection

### 2.1 URL Collection Strategy

**Method: Automated directory scraping** (no manual curation due to scale)

Scrape diverse website directories and aggregate deduplicated URLs:

| Source | Category Coverage | Expected Yield |
|---|---|---|
| Crunchbase API | SaaS, startups | 500–800 URLs |
| Product Hunt archive | SaaS, apps, tools | 300–500 URLs |
| Alexa/Similarweb top sites | All categories | 400–600 URLs |
| GitHub Pages hosting | Portfolios, projects | 200–400 URLs |
| Shopify app store links | E-commerce | 300–500 URLs |
| Reddit comments (regex scrape) | All categories | 200–400 URLs |
| Hacker News posts | Tech, SaaS | 100–200 URLs |
| Open directory/DMOZ snapshots | All categories | 500–1000 URLs |

**Target distribution for 3,000 unique URLs:**

| Category | Target Count |
|---|---|
| E-commerce | 600 URLs |
| SaaS dashboards | 450 URLs |
| Blogs / Content | 450 URLs |
| Landing pages | 400 URLs |
| Web apps / Tools | 400 URLs |
| Portfolios / Projects | 350 URLs |
| Social / Communities | 200 URLs |
| Other | 150 URLs |

**Final output: 3,000 unique URLs → 9,000 screenshots (3 variants each)**

### 2.1a URL Scraper Implementation (`url-scraper.ts`)

The scraper aggregates URLs from multiple sources and deduplicates them:

```typescript
import * as fs from 'fs';
import axios from 'axios';
import { JSDOM } from 'jsdom';

const SOURCES = {
  // 1. HackerNews: Scrape top stories & comments
  async hackerNews() {
    const urls = new Set<string>();
    for (let i = 1; i <= 5; i++) {
      const res = await axios.get(`https://news.ycombinator.com/page?p=${i}`);
      const dom = new JSDOM(res.data);
      dom.window.document.querySelectorAll('a.storylink').forEach((el: any) => {
        const href = el.getAttribute('href');
        if (href && href.startsWith('http')) urls.add(href);
      });
    }
    return Array.from(urls);
  },

  // 2. Product Hunt: Scrape featured products
  async productHunt() {
    const urls = new Set<string>();
    for (let i = 0; i < 10; i++) {
      const res = await axios.get(
        `https://api.producthunt.com/v2/posts?order=newest&after=${i * 20}`,
        { headers: { 'Accept': 'application/json' } }
      );
      if (res.data.data) {
        res.data.data.forEach((product: any) => {
          if (product.website) urls.add(product.website);
        });
      }
    }
    return Array.from(urls);
  },

  // 3. Crunchbase: Use free company URLs
  async crunchbase() {
    const urls = new Set<string>();
    const res = await axios.get('https://www.crunchbase.com/search/companies');
    const dom = new JSDOM(res.data);
    dom.window.document.querySelectorAll('a[href*="crunchbase.com/organization"]').forEach((el: any) => {
      const text = el.textContent?.trim();
      if (text) urls.add(`https://${text.replace(/[^a-z0-9.-]/gi, '')}.com`);
    });
    return Array.from(urls);
  },

  // 4. Alexa Top Sites (via cached/mirror sources)
  async alexaTopSites() {
    const urls = [
      // Top e-commerce
      'https://www.amazon.com', 'https://www.ebay.com', 'https://www.etsy.com',
      'https://www.alibaba.com', 'https://www.shopify.com',
      // Top SaaS
      'https://www.notion.so', 'https://www.slack.com', 'https://www.asana.com',
      'https://www.figma.com', 'https://www.github.com', 'https://www.jira.com',
      // Top media/blogs
      'https://www.medium.com', 'https://www.dev.to', 'https://www.wikipedia.org',
      // Add more as needed
    ];
    return urls;
  },

  // 5. GitHub Pages (common portfolio hosting)
  async githubPages() {
    const urls = new Set<string>();
    const domains = ['github.io', 'pages.dev', 'vercel.app', 'netlify.app'];
    // Note: This would require GitHub search API access or cached snapshots
    // For demo, manually add popular GitHub Pages
    return Array.from(urls);
  }
};

async function scrapeAllSources() {
  const allUrls = new Set<string>();
  const sourceLog: Record<string, number> = {};

  for (const [sourceName, sourceFn] of Object.entries(SOURCES)) {
    try {
      const urls = await sourceFn();
      urls.forEach(url => allUrls.add(normalizeUrl(url)));
      sourceLog[sourceName] = urls.length;
      console.log(`✓ ${sourceName}: ${urls.length} URLs`);
    } catch (err) {
      console.warn(`✗ ${sourceName}: ${err.message}`);
      sourceLog[sourceName] = 0;
    }
  }

  return { urls: Array.from(allUrls), sourceLog };
}

function normalizeUrl(url: string): string {
  try {
    const u = new URL(url.startsWith('http') ? url : `https://${url}`);
    // Return domain only (remove path, query, fragment)
    return `${u.protocol}//${u.hostname}`;
  } catch {
    return '';
  }
}

function deduplicateDomains(urls: string[]): string[] {
  const seen = new Set<string>();
  return urls.filter(url => {
    const domain = new URL(url).hostname;
    if (seen.has(domain)) return false;
    seen.add(domain);
    return true;
  });
}

async function main() {
  console.log('Starting URL collection...');
  const { urls, sourceLog } = await scrapeAllSources();
  
  console.log(`\nRaw URLs collected: ${urls.length}`);
  const deduplicated = deduplicateDomains(urls.filter(u => u));
  console.log(`After deduplication: ${deduplicated.length}`);

  // Save raw URLs
  fs.writeFileSync('Dataset/url-sources/raw-urls.txt', urls.join('\n'));
  
  // Save deduplicated (target: 3000)
  const final = deduplicated.slice(0, 3000);
  fs.writeFileSync('Dataset/url-sources/deduplicated-urls.txt', final.join('\n'));

  // Log sources
  fs.writeFileSync('Dataset/url-sources/url-scraper-log.json', JSON.stringify({
    timestamp: new Date().toISOString(),
    sources: sourceLog,
    rawTotal: urls.length,
    dedupTotal: deduplicated.length,
    finalTarget: final.length
  }, null, 2));

  console.log(`\n✓ Saved ${final.length} URLs to deduplicated-urls.txt`);
}

main().catch(console.error);
```

**Run before crawling:**
```bash
npx ts-node src/url-scraper.ts
```

### 2.2 Screenshot Variants per URL

For each URL, capture three variants at zero extra scraping cost:

1. **Desktop light mode** — base viewport (1920×1080)
2. **Desktop dark mode** — `page.emulateMedia({ colorScheme: 'dark' })`
3. **Mobile viewport** — 390×844 (iPhone 14 dimensions)

### 2.3 Crawler Script (`crawler.ts`)

```typescript
import { chromium, Page } from 'playwright';
import * as fs from 'fs';

const VIEWPORT = { width: 1920, height: 1080 };
const MOBILE_VIEWPORT = { width: 390, height: 844 };
const CONFIG_PATH = 'config.yaml'; // Loaded separately for selectors

interface ScreenshotVariant {
  id: string;
  variant: 'light' | 'dark' | 'mobile';
  path: string;
}

async function capturePage(
  page: Page,
  url: string,
  id: string,
  variant: ScreenshotVariant['variant']
): Promise<void> {
  try {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(1000); // Allow animations to settle

    // Scroll to trigger lazy-loaded content
    await page.evaluate(() => window.scrollTo(0, 0));

    const path = variant === 'mobile' 
      ? `Dataset/raw/screenshots/${id}_mobile.png`
      : `Dataset/raw/screenshots/${id}_${variant}.png`;

    await page.screenshot({
      path,
      clip: { x: 0, y: 0, ...(variant === 'mobile' ? MOBILE_VIEWPORT : VIEWPORT) }
    });
  } catch (err) {
    console.warn(`Failed to capture ${variant} for ${id}:`, err.message);
    throw err;
  }
}

(async () => {
  const browser = await chromium.launch();
  const urlFile = 'Dataset/url-sources/deduplicated-urls.txt';
  
  if (!fs.existsSync(urlFile)) {
    console.error(`❌ URL file not found: ${urlFile}`);
    console.error('Run: npx ts-node src/url-scraper.ts first');
    process.exit(1);
  }

  const urls = fs.readFileSync(urlFile, 'utf-8').trim().split('\n').filter(Boolean);
  console.log(`📊 Starting crawler with ${urls.length} URLs`);

  let successful = 0;
  let failed = 0;
  const startTime = Date.now();

  for (const [i, url] of urls.entries()) {
    const id = `img_${String(i).padStart(5, '0')}`;
    const page = await browser.newPage();
    
    try {
      // Variant 1: Desktop Light
      await page.setViewportSize(VIEWPORT);
      await page.emulateMedia({ colorScheme: 'light' });
      await capturePage(page, url, id, 'light');

      // Variant 2: Desktop Dark
      await page.emulateMedia({ colorScheme: 'dark' });
      await capturePage(page, url, id, 'dark');

      // Variant 3: Mobile
      await page.setViewportSize(MOBILE_VIEWPORT);
      await page.emulateMedia({ colorScheme: 'light' });
      await capturePage(page, url, id, 'mobile');

      successful++;
      if ((i + 1) % 100 === 0) {
        console.log(`✓ Processed ${i + 1}/${urls.length} URLs`);
      }
    } catch (err) {
      failed++;
      console.warn(`✗ Skipped ${url}:`, err.message);
    } finally {
      await page.close();
    }
  }

  const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  console.log(`\n✓ Crawler complete in ${elapsed} minutes`);
  console.log(`  Successful: ${successful} URLs (${(successful * 3)} images)`);
  console.log(`  Failed: ${failed} URLs`);

  await browser.close();
})();
```

**Run the crawler:**
```bash
npx ts-node src/crawler.ts
```

This generates 3 screenshots per successful URL, saved to `Dataset/raw/screenshots/`

---

## Phase 3: Annotation Extraction

### 3.1 Element Selection Strategy

Prefer **ARIA roles** over raw HTML tags for semantic consistency. Fall back to HTML tags where ARIA is absent.

```typescript
const SELECTOR_MAP: Record<string, string> = {
  button:   '[role="button"], button',
  input:    'input, textarea, [role="textbox"]',
  nav:      'nav, [role="navigation"]',
  header:   'header, [role="banner"]',
  footer:   'footer, [role="contentinfo"]',
  link:     'a[href]',
  image:    'img, [role="img"], svg',
  form:     'form, [role="form"]',
  dropdown: 'select, [role="listbox"], [role="combobox"]',
  modal:    '[role="dialog"], [role="alertdialog"]',
};
```

### 3.2 Bounding Box Extractor (`annotator.ts`)

```typescript
async function extractAnnotations(page: Page, viewport: { width: number, height: number }) {
  const annotations = [];

  for (const [label, selector] of Object.entries(SELECTOR_MAP)) {
    const locators = page.locator(selector);
    const count = await locators.count();

    for (let i = 0; i < count; i++) {
      const el = locators.nth(i);
      const box = await el.boundingBox();

      if (!box) continue;
      if (!await el.isVisible()) continue;

      // Reject elements outside the viewport bounds
      if (box.x < 0 || box.y < 0) continue;
      if (box.x + box.width > viewport.width) continue;
      if (box.y + box.height > viewport.height) continue;

      // Reject elements that are too small to be meaningful
      if (box.width < 5 || box.height < 5) continue;

      annotations.push({ label, ...box });
    }
  }

  return annotations;
}
```

### 3.3 Deduplication: Resolving Overlapping Boxes

This is the most critical quality step. Apply two strategies in order:

**Strategy A — Parent preference (semantic nesting)**
When a child element is fully contained within a parent of *different* class, keep the parent and discard the child. Example: a `<span>` inside a `<button>` — keep `button`.

**Strategy B — IoU-based NMS (same class)**
When two boxes of the *same* class overlap significantly (IoU > 0.5), keep the larger one.

```typescript
function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - intersection;
  return intersection / union;
}

function deduplicate(annotations: Annotation[]): Annotation[] {
  // Sort by area descending (prefer larger/parent boxes)
  const sorted = [...annotations].sort((a, b) =>
    (b.width * b.height) - (a.width * a.height)
  );

  const kept: Annotation[] = [];
  for (const current of sorted) {
    const dominated = kept.some(existing =>
      existing.label === current.label && iou(existing, current) > 0.5
    );
    if (!dominated) kept.push(current);
  }
  return kept;
}
```

---

## Phase 4: Format Conversion

### 4.1 YOLO Format

Each image gets a `.txt` file with one row per annotation:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

```typescript
function toYOLO(annotations: Annotation[], imgW: number, imgH: number, classMap: string[]): string {
  return annotations.map(ann => {
    const classId = classMap.indexOf(ann.label);
    const xCenter = (ann.x + ann.width / 2) / imgW;
    const yCenter = (ann.y + ann.height / 2) / imgH;
    const w = ann.width / imgW;
    const h = ann.height / imgH;
    return `${classId} ${xCenter.toFixed(6)} ${yCenter.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
  }).join('\n');
}
```

### 4.2 COCO Format

Aggregate all images and annotations into a single `coco.json`:

```json
{
  "info": { "description": "WebUI Element Dataset", "version": "1.0" },
  "categories": [
    { "id": 0, "name": "button" },
    { "id": 1, "name": "input" }
  ],
  "images": [
    { "id": 0, "file_name": "img_00001_light.png", "width": 1280, "height": 720 }
  ],
  "annotations": [
    { "id": 0, "image_id": 0, "category_id": 0, "bbox": [x, y, w, h], "area": "w*h" }
  ]
}
```

---

## Phase 5: Quality Control

### 5.1 Automated Checks

Run these checks before finalizing the dataset:

- [ ] No annotation has coordinates outside image bounds
- [ ] No annotation has zero-area bounding box
- [ ] Every image file has a corresponding label file
- [ ] Class distribution is not severely imbalanced (no class < 2% of total)
- [ ] No duplicate image hashes (perceptual hash check with `imagehash`)

### 5.2 Manual Review Sample

Randomly sample 5% of images and visually inspect annotations using a tool like [LabelStudio](https://labelstud.io/) or [CVAT](https://www.cvat.ai/). Flag images with:

- Boxes that miss the element entirely
- Missing annotations for clearly visible elements
- Incorrect class labels (e.g., a `<div role="button">` mislabeled)

### 5.3 Dataset Split

| Split | Ratio | Notes |
|---|---|---|
| Train | 70% | Main training set |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final holdout — never touch until evaluation |

Split **by URL domain**, not by image, to prevent data leakage (multiple screenshots of the same site should not span train and test).

---

## Phase 6: Augmentation

Beyond dark mode and mobile viewports (already captured in Phase 2), apply these offline augmentations during training:

| Augmentation | Purpose |
|---|---|
| Random crop (keep boxes) | Simulate partial views |
| Horizontal flip | More layout variety |
| Brightness/contrast jitter | Robustness to rendering differences |
| JPEG compression artifacts | Real-world screenshot noise |
| Gaussian blur | Focus/zoom simulation |

> **Do not** apply geometric augmentations that invalidate bounding boxes (e.g., rotation > 15°, extreme perspective warp).

---

## Phase 7: Dataset Statistics & Delivery

Before training, generate and log:

- Total images and total annotations
- Per-class annotation counts and average box sizes
- Images per site category
- Annotation density histogram (annotations per image)

### Target Dataset Scale (3,000 URLs × 3 variants)

| Metric | Target |
|---|---|
| Unique URLs | **3,000** |
| Total images | **9,000** |
| Desktop light mode | 3,000 (1920×1080) |
| Desktop dark mode | 3,000 (1920×1080) |
| Mobile variant | 3,000 (390×844) |
| Expected total annotations | **600,000–800,000** |
| Classes | 10 |
| Formats | YOLO + COCO JSON |
| Storage size | ~40–50 GB (raw + output) |

**Estimated runtime:**
- URL scraping & deduplication: ~30–45 min
- Screenshot crawling (3 variants × 3000 URLs): **80–120 hours** (parallelizable)
- Annotation extraction: ~2–3 hours
- Deduplication & QC: ~4–5 hours
- Format conversion: ~1 hour

---

## Known Limitations & Mitigations

| Limitation | Mitigation |
|---|---|
| iframes not captured | Log iframe-heavy pages; handle separately with `frame.locator()` |
| Shadow DOM elements missed | Use `pierce:` CSS selector prefix where needed |
| JS-rendered content | `waitForLoadState('networkidle')` + 1s buffer |
| Sites blocking headless browsers | Rotate user agents; use `playwright-stealth` |
| Class imbalance (links >> modals) | Oversample rare classes; weighted loss during training |
