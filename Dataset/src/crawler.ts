import { readFile, readdir } from 'node:fs/promises';
import { chromium, type Page } from 'playwright';
import sharp from 'sharp';
import { crawlFailuresPath, crawlManifestPath, deduplicatedUrlsPath, desktopViewport, mobileViewport, phase2Limits, screenshotRoot } from './paths.js';
import { canonicalizeUrl, ensureDir, safeFileSegment, shortHash, splitLines, writeJson, writeText } from './utils.js';

type Variant = 'light' | 'dark' | 'mobile';

type BoundingBox = {
  class: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type CrawlFailure = {
  url: string;
  error: string;
};

type CrawlManifestEntry = {
  url: string;
  variant: Variant;
  fileName: string;
  status: 'saved' | 'skipped' | 'failed';
  attempt: number;
  timestamp: string;
  annotationCount?: number;
};

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
const RETRIES = 2;
const VARIANTS: Variant[] = ['light', 'dark', 'mobile'];
const IMAGE_FORMAT = 'webp';
const IMAGE_QUALITY = 80;

// Element class selectors for annotation extraction
const CLASS_SELECTORS: Record<string, string> = {
  button: 'button, [role="button"], a[role="button"]',
  input: 'input, textarea, [role="textbox"], [role="combobox"]',
  link: 'a[href]',
  nav: 'nav, [role="navigation"]',
  form: 'form, [role="form"]',
  image: 'img, [role="img"], svg',
  dropdown: 'select, [role="listbox"], [role="combobox"]',
  modal: '[role="dialog"], [role="alertdialog"]',
  header: 'header, [role="banner"]',
  footer: 'footer, [role="contentinfo"]'
};

async function extractAnnotations(page: Page, viewport: typeof desktopViewport): Promise<BoundingBox[]> {
  const annotations: BoundingBox[] = [];

  for (const [elementClass, selector] of Object.entries(CLASS_SELECTORS)) {
    try {
      const locators = page.locator(selector);
      const count = await locators.count();

      for (let i = 0; i < count; i++) {
        try {
          const el = locators.nth(i);
          const isVisible = await el.isVisible().catch(() => false);
          if (!isVisible) continue;

          const box = await el.boundingBox().catch(() => null);
          if (!box) continue;

          // Reject elements outside viewport bounds
          if (box.x < 0 || box.y < 0) continue;
          if (box.x + box.width > viewport.width) continue;
          if (box.y + box.height > viewport.height) continue;

          // Reject elements that are too small
          if (box.width < phase2Limits.minWidth || box.height < phase2Limits.minHeight) continue;
          if (box.width * box.height < phase2Limits.minArea) continue;

          annotations.push({
            class: elementClass,
            x: Math.round(box.x),
            y: Math.round(box.y),
            width: Math.round(box.width),
            height: Math.round(box.height)
          });
        } catch {
          // Skip elements that fail during bounding box extraction
          continue;
        }
      }
    } catch {
      // Skip selectors that fail
      continue;
    }
  }

  return annotations;
}

function fileNameFor(url: string, index: number, variant: Variant): string {
  const normalized = canonicalizeUrl(url) ?? url;
  const parsed = new URL(normalized);
  const host = safeFileSegment(parsed.hostname);
  const pathSegment = safeFileSegment(parsed.pathname.replace(/\//g, '-').replace(/^-+|-+$/g, '')) || 'root';
  const hash = shortHash(normalized);
  return `${String(index).padStart(5, '0')}_${variant}_${host}_${pathSegment}_${hash}.${IMAGE_FORMAT}`;
}

function annotationFileNameFor(url: string, index: number, variant: Variant): string {
  return fileNameFor(url, index, variant).replace(/\.\w+$/, '.json');
}

async function detectCheckpoint(urls: string[]): Promise<{ resumeIndex: number; completedUrls: number; completedScreenshots: number }> {
  const files = await readdir(screenshotRoot, { withFileTypes: true });
  const existingFiles = new Set(files.filter(file => file.isFile()).map(file => file.name));

  for (let index = 0; index < urls.length; index += 1) {
    const url = urls[index];
    const allArtifactsPresent = VARIANTS.every(variant => {
      const jpgName = fileNameFor(url, index + 1, variant);
      const jsonName = annotationFileNameFor(url, index + 1, variant);
      return existingFiles.has(jpgName) && existingFiles.has(jsonName);
    });

    if (!allArtifactsPresent) {
      return {
        resumeIndex: index,
        completedUrls: index,
        completedScreenshots: index * VARIANTS.length
      };
    }
  }

  return {
    resumeIndex: urls.length,
    completedUrls: urls.length,
    completedScreenshots: urls.length * VARIANTS.length
  };
}

async function captureVariant(page: Page, url: string, filePath: string, variant: Variant, viewport: typeof desktopViewport): Promise<{ annotationCount: number }> {
  await page.setViewportSize(viewport);
  await page.emulateMedia({ colorScheme: variant === 'dark' ? 'dark' : 'light' });

  await page.goto(url, {
    waitUntil: 'networkidle',
    timeout: phase2Limits.navigationTimeoutMs
  });

  // Wait for critical rendering path and images
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(phase2Limits.postLoadDelayMs);

  // Stabilize rendering: wait for animations and lazy images
  await page.evaluate(() => {
    window.scrollTo(0, 0);
  });
  await page.waitForTimeout(500);

  // Extract annotations in parallel while page is loaded
  const annotations = await extractAnnotations(page, viewport);
  const annotationCount = annotations.length;

  // Save annotation metadata alongside screenshot
  const annotationPath = filePath.replace(/\.\w+$/, '.json');
  await writeJson(annotationPath, {
    url,
    variant,
    viewport,
    annotationCount,
    timestamp: new Date().toISOString(),
    annotations
  });

  // Capture as PNG then convert to WebP via sharp
  const buffer = await page.screenshot({
    fullPage: false,
    type: 'png',
    timeout: 10000
  });

  await sharp(buffer)
    .toFormat(IMAGE_FORMAT, { quality: IMAGE_QUALITY })
    .toFile(filePath);

  return { annotationCount };
}

async function captureWithRetries(page: Page, url: string, filePath: string, variant: Variant, viewport: typeof desktopViewport): Promise<{ status: CrawlManifestEntry['status']; error?: string; annotationCount?: number }> {
  for (let attempt = 1; attempt <= RETRIES + 1; attempt += 1) {
    try {
      // Close and recreate page on retry to clear corrupted state
      if (attempt > 1) {
        await page.close();
        page = await (page.context() as any).newPage({ userAgent: USER_AGENT });
      }
      const result = await captureVariant(page, url, filePath, variant, viewport);
      return { status: 'saved', annotationCount: result.annotationCount };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (attempt > RETRIES) {
        return { status: 'failed', error: errorMsg };
      }
      // Brief delay before retry
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  return { status: 'failed', error: 'unknown capture failure' };
}

async function main(): Promise<void> {
  await ensureDir(crawlManifestPath);
  await ensureDir(crawlFailuresPath);
  await ensureDir(screenshotRoot);

  const urlFile = deduplicatedUrlsPath;
  const urls = splitLines(await readFile(urlFile, 'utf8'));

  if (urls.length === 0) {
    throw new Error(`No URLs found in ${urlFile}`);
  }

  const checkpoint = await detectCheckpoint(urls);
  const pendingUrls = urls.length - checkpoint.resumeIndex;
  console.log(`Checkpoint: ${checkpoint.completedUrls}/${urls.length} URLs already completed (${checkpoint.completedScreenshots} screenshots).`);

  if (pendingUrls === 0) {
    console.log('All URLs are already completed. Nothing to crawl.');
    return;
  }

  console.log(`Resuming from URL ${checkpoint.resumeIndex + 1}/${urls.length} (${pendingUrls} URLs remaining).`);

  const browser = await chromium.launch({ headless: true });
  const manifestEntries: CrawlManifestEntry[] = [];
  const failures: CrawlFailure[] = [];

  let completed = checkpoint.completedUrls;
  let screenshotCount = checkpoint.completedScreenshots;
  let totalAnnotations = 0;

  for (const [index, url] of urls.entries()) {
    if (index < checkpoint.resumeIndex) {
      continue;
    }

    console.log(`Crawling ${index + 1}/${urls.length}: ${url}`);

    const page = await browser.newPage({ userAgent: USER_AGENT });

    try {
      for (const variant of VARIANTS) {
        const viewport = variant === 'mobile' ? mobileViewport : desktopViewport;
        const fileName = fileNameFor(url, index + 1, variant);
        const filePath = `${screenshotRoot}/${fileName}`;

        const result = await captureWithRetries(page, url, filePath, variant, viewport);
        manifestEntries.push({
          url,
          variant,
          fileName,
          status: result.status,
          attempt: result.status === 'saved' ? 1 : RETRIES + 1,
          timestamp: new Date().toISOString(),
          annotationCount: result.annotationCount
        });

        if (result.status === 'saved') {
          screenshotCount += 1;
          if (result.annotationCount) {
            totalAnnotations += result.annotationCount;
          }
        } else if (result.error) {
          failures.push({ url, error: result.error });
        }
      }

      completed += 1;
      if (completed % 50 === 0 || completed === urls.length) {
        console.log(`Processed ${completed}/${urls.length} URLs (${screenshotCount} screenshots, ${totalAnnotations} annotations)`);
      }
    } finally {
      await page.close();
    }
  }

  await browser.close();

  await writeText(crawlManifestPath, `${manifestEntries.map(entry => JSON.stringify(entry)).join('\n')}\n`);
  await writeJson(crawlFailuresPath, {
    generatedAt: new Date().toISOString(),
    totalUrls: urls.length,
    completedUrls: completed,
    totalScreenshots: screenshotCount,
    totalAnnotations,
    failures
  });

  console.log(`Completed ${completed}/${urls.length} URLs`);
  console.log(`Saved ${screenshotCount} screenshots`);
  console.log(`Extracted ${totalAnnotations} annotations`);
  console.log(`Failures: ${failures.length}`);
}

main().catch(error => {
  console.error('Crawler failed:', error);
  process.exitCode = 1;
});
