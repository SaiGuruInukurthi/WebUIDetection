import { readFile, readdir } from 'node:fs/promises';
import { chromium, type Page } from 'playwright';
import sharp from 'sharp';
import { crawlFailuresPath, crawlManifestPath, datasetMetricsPath, deduplicatedUrlsPath, desktopViewport, phase2Limits, qualityThresholds, screenshotRoot } from './paths.js';
import { canonicalizeUrl, ensureDir, safeFileSegment, shortHash, splitLines, writeJson, writeText } from './utils.js';

type Variant = 'light' | 'dark';

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
  qualityFlags?: string[];  // e.g., ["low_annotation_count", "poor_class_diversity", "class_imbalance"]
  classDistribution?: Record<string, number>;  // per-image class counts
};

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
const RETRIES = 2;
const VARIANTS: Variant[] = ['light', 'dark'];
const IMAGE_FORMAT = 'webp';
const IMAGE_QUALITY = 80;

// Class population caps to prevent imbalance (e.g., links dominating dataset)
const CLASS_LIMITS: Record<string, number> = {
  button: 15,
  input: 10,
  link: 20,
  nav: 5,
  form: 5,
  image: 10,
  dropdown: 5,
  modal: 3,
  header: 2,
  footer: 2
};

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

/**
 * Balance annotations by applying per-class population caps.
 * Prevents class imbalance (e.g., links >> modals).
 */
function balanceClasses(annotations: BoundingBox[]): BoundingBox[] {
  const grouped: Record<string, BoundingBox[]> = {};
  
  for (const ann of annotations) {
    if (!grouped[ann.class]) grouped[ann.class] = [];
    grouped[ann.class].push(ann);
  }

  const result: BoundingBox[] = [];
  for (const className in grouped) {
    const limit = CLASS_LIMITS[className] ?? 10;
    result.push(...grouped[className].slice(0, limit));
  }

  return result;
}

/**
 * Check image quality and return any quality flags.
 * Images can still be saved but flagged for optional filtering.
 * Returns:
 *   - Empty array if image passes all checks
 *   - Array of failure reasons (e.g., ["low_annotation_count", "class_imbalance"])
 */
function assessImageQuality(
  annotations: BoundingBox[]
): { flags: string[]; classDistribution: Record<string, number> } {
  const flags: string[] = [];

  // Count annotations per class
  const classCounts: Record<string, number> = {};
  for (const ann of annotations) {
    classCounts[ann.class] = (classCounts[ann.class] ?? 0) + 1;
  }

  const totalAnnotations = annotations.length;
  const uniqueClasses = Object.keys(classCounts).length;

  // Check 1: Minimum annotation count
  if (totalAnnotations < qualityThresholds.minAnnotationsPerImage) {
    flags.push(`low_annotation_count(${totalAnnotations})`);
  }

  // Check 2: Minimum class diversity
  if (uniqueClasses < qualityThresholds.minClassDiversity) {
    flags.push(`poor_class_diversity(${uniqueClasses})`);
  }

  // Check 3: Single class dominance
  const maxClassCount = Math.max(...Object.values(classCounts));
  const maxRatio = maxClassCount / totalAnnotations;
  if (maxRatio > qualityThresholds.maxSingleClassRatio) {
    const dominantClass = Object.entries(classCounts).find(
      ([, count]) => count === maxClassCount
    )?.[0];
    flags.push(`class_imbalance(${dominantClass}:${(maxRatio * 100).toFixed(0)}%)`);
  }

  return { flags, classDistribution: classCounts };
}

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

  // Apply class-level population caps to prevent imbalance
  return balanceClasses(annotations);
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
  // Robust resume using filename indices as the source of truth.
  // For each index, check if both light and dark variants exist.
  // Handles gaps in indices (e.g., 00001, 00003, 00004) when URLs fail.
  
  try {
    const dirents = await readdir(screenshotRoot, { withFileTypes: true });
    const jsonFiles = dirents
      .filter(d => d.isFile() && d.name.endsWith('.json'))
      .map(d => d.name);

    // Build index -> variants map from filenames: XXXXX_variant_...json
    const variantsByIndex = new Map<number, Set<string>>();
    for (const jf of jsonFiles) {
      try {
        // Parse filename format: {5-digit-index}_{variant}_...
        const match = jf.match(/^(\d{5})_(\w+)_/);
        if (!match) continue;

        const index = parseInt(match[1], 10);
        const variant = match[2];

        const set = variantsByIndex.get(index) ?? new Set<string>();
        set.add(variant);
        variantsByIndex.set(index, set);
      } catch {
        continue;
      }
    }

    // Also read URLs from JSON for verification
    const urlsByIndex = new Map<number, string>();
    for (const jf of jsonFiles) {
      try {
        const match = jf.match(/^(\d{5})_/);
        if (!match) continue;
        const index = parseInt(match[1], 10);

        if (urlsByIndex.has(index)) continue; // already have URL for this index

        const content = await readFile(`${screenshotRoot}/${jf}`, 'utf8');
        const parsed = JSON.parse(content);
        if (parsed && typeof parsed.url === 'string') {
          urlsByIndex.set(index, parsed.url);
        }
      } catch {
        continue;
      }
    }

    // Find the first URL index that doesn't have both required variants.
    // Return resume index as 0-based array index (not 1-based file index).
    let completed = 0;
    let screenshotCount = 0;
    let resumeIndex = urls.length; // Default: all done
    let firstIncompleteIndex = -1;

    for (let fileIndex = 1; fileIndex <= urls.length; fileIndex += 1) {
      const variants = variantsByIndex.get(fileIndex);
      const savedUrl = urlsByIndex.get(fileIndex);
      const arrayIndex = fileIndex - 1; // convert to 0-based array index
      const expectedUrl = urls[arrayIndex];

      // Check if this index is complete (has both variants)
      if (variants && VARIANTS.every(v => variants.has(v))) {
        // Verify URL matches (optional, for debugging URL mismatches)
        if (savedUrl && savedUrl !== expectedUrl) {
          console.warn(`⚠ URL mismatch at file index ${fileIndex}:`);
          console.warn(`  Saved: ${savedUrl}`);
          console.warn(`  Expected: ${expectedUrl}`);
        }
        completed += 1;
        screenshotCount += VARIANTS.length;
      } else if (firstIncompleteIndex === -1) {
        // Record the FIRST incomplete index we encounter
        firstIncompleteIndex = arrayIndex;
      }
    }

    // Resume from the first incomplete index (or end of list if all complete)
    if (firstIncompleteIndex !== -1) {
      resumeIndex = firstIncompleteIndex;
    }

    return {
      resumeIndex,
      completedUrls: completed,
      completedScreenshots: screenshotCount
    };
  } catch (error) {
    // If checkpoint detection fails, start from beginning
    console.warn('Failed to detect checkpoint:', error instanceof Error ? error.message : error);
    return {
      resumeIndex: 0,
      completedUrls: 0,
      completedScreenshots: 0
    };
  }
}

async function captureVariant(page: Page, url: string, filePath: string, variant: Variant, viewport: typeof desktopViewport): Promise<{ annotationCount: number; qualityFlags: string[]; classDistribution: Record<string, number> }> {
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

  // Trigger interactions to reveal hidden UI elements (modals, dropdowns, menus)
  // These won't break the page but will expose more interactive elements
  try {
    // Try clicking the first few buttons to trigger dropdowns/modals
    const buttons = page.locator('button');
    const buttonCount = Math.min(3, await buttons.count().catch(() => 0));
    for (let i = 0; i < buttonCount; i += 1) {
      await buttons.nth(i).click({ timeout: 500 }).catch(() => {});
      await page.waitForTimeout(100);
    }
  } catch {}

  try {
    // Try opening any dropdowns or select elements
    const selects = page.locator('select, [role="listbox"], [role="combobox"]');
    const selectCount = Math.min(2, await selects.count().catch(() => 0));
    for (let i = 0; i < selectCount; i += 1) {
      await selects.nth(i).click({ timeout: 500 }).catch(() => {});
      await page.waitForTimeout(100);
    }
  } catch {}

  // Aggressive scrolling to expose lazy-loaded and footer content
  for (let i = 0; i < phase2Limits.maxScrollSteps; i += 1) {
    await page.evaluate(() => {
      window.scrollBy(0, window.innerHeight);
    });
    await page.waitForTimeout(300);
  }

  // Return to top after scrolling
  await page.evaluate(() => {
    window.scrollTo(0, 0);
  });
  await page.waitForTimeout(500);

  // Extract annotations in parallel while page is loaded
  const annotations = await extractAnnotations(page, viewport);
  const annotationCount = annotations.length;

  // Assess image quality
  const { flags: qualityFlags, classDistribution } = assessImageQuality(annotations);

  // Save annotation metadata alongside screenshot
  const annotationPath = filePath.replace(/\.\w+$/, '.json');
  await writeJson(annotationPath, {
    url,
    variant,
    viewport,
    annotationCount,
    qualityFlags: qualityFlags.length > 0 ? qualityFlags : undefined,
    classDistribution,
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

  return { annotationCount, qualityFlags, classDistribution };
}

async function captureWithRetries(page: Page, url: string, filePath: string, variant: Variant, viewport: typeof desktopViewport): Promise<{ status: CrawlManifestEntry['status']; error?: string; annotationCount?: number; qualityFlags?: string[]; classDistribution?: Record<string, number> }> {
  for (let attempt = 1; attempt <= RETRIES + 1; attempt += 1) {
    try {
      // Close and recreate page on retry to clear corrupted state
      if (attempt > 1) {
        await page.close();
        page = await (page.context() as any).newPage({ userAgent: USER_AGENT });
      }
      const result = await captureVariant(page, url, filePath, variant, viewport);
      return { 
        status: 'saved', 
        annotationCount: result.annotationCount,
        qualityFlags: result.qualityFlags,
        classDistribution: result.classDistribution
      };
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
  const globalClassCounts: Record<string, number> = {};
  let lowQualityCount = 0;

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
        const viewport = desktopViewport;
        const fileName = fileNameFor(url, index + 1, variant);
        const filePath = `${screenshotRoot}/${fileName}`;

        const result = await captureWithRetries(page, url, filePath, variant, viewport);
        
        const entry: CrawlManifestEntry = {
          url,
          variant,
          fileName,
          status: result.status,
          attempt: result.status === 'saved' ? 1 : RETRIES + 1,
          timestamp: new Date().toISOString(),
          annotationCount: result.annotationCount
        };

        if (result.qualityFlags && result.qualityFlags.length > 0) {
          entry.qualityFlags = result.qualityFlags;
          lowQualityCount += 1;
        }

        if (result.classDistribution) {
          entry.classDistribution = result.classDistribution;
          // Aggregate global class counts
          for (const [className, count] of Object.entries(result.classDistribution)) {
            globalClassCounts[className] = (globalClassCounts[className] ?? 0) + count;
          }
        }

        manifestEntries.push(entry);

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
        console.log(`Processed ${completed}/${urls.length} URLs (${screenshotCount} screenshots, ${totalAnnotations} annotations, ${lowQualityCount} low-quality)`);
      }
    } finally {
      await page.close();
    }
  }

  await browser.close();

  // Append to manifest (preserve previous entries from resumed runs)
  const existingManifest = (() => {
    try {
      return readFile(crawlManifestPath, 'utf8').then(content => content);
    } catch {
      return Promise.resolve('');
    }
  })();

  const newManifestEntries = manifestEntries.map(entry => JSON.stringify(entry)).join('\n');
  const existingContent = await existingManifest;
  const finalManifest = existingContent ? `${existingContent}${newManifestEntries}\n` : `${newManifestEntries}\n`;
  await writeText(crawlManifestPath, finalManifest);
  await writeJson(crawlFailuresPath, {
    generatedAt: new Date().toISOString(),
    totalUrls: urls.length,
    completedUrls: completed,
    totalScreenshots: screenshotCount,
    totalAnnotations,
    failures
  });

  // Generate dataset metrics for class distribution monitoring
  const sortedClasses = Object.entries(globalClassCounts).sort(([, a], [, b]) => b - a);
  const classPercentages = Object.fromEntries(
    sortedClasses.map(([cls, count]) => [
      cls,
      totalAnnotations > 0 ? ((count / totalAnnotations) * 100).toFixed(1) : '0'
    ])
  );

  await writeJson(datasetMetricsPath, {
    generatedAt: new Date().toISOString(),
    crawlSummary: {
      totalUrls: urls.length,
      completedUrls: completed,
      successfulUrls: completed - (failures.length / 2), // rough estimate (2 variants per URL)
      totalScreenshots: screenshotCount,
      totalAnnotations,
      lowQualityImages: lowQualityCount,
      lowQualityPercentage: ((lowQualityCount / Math.max(1, screenshotCount)) * 100).toFixed(1)
    },
    classDistribution: {
      counts: globalClassCounts,
      percentages: classPercentages,
      uniqueClasses: Object.keys(globalClassCounts).length
    },
    qualityThresholds
  });

  console.log(`\nCrawl Summary:`);
  console.log(`Completed ${completed}/${urls.length} URLs`);
  console.log(`Saved ${screenshotCount} screenshots`);
  console.log(`Extracted ${totalAnnotations} annotations`);
  console.log(`Low-quality images: ${lowQualityCount} (${((lowQualityCount / Math.max(1, screenshotCount)) * 100).toFixed(1)}%)`);
  console.log(`Failures: ${failures.length}`);
  console.log(`\nClass Distribution (Global):`);
  for (const [cls, count] of sortedClasses) {
    const pct = classPercentages[cls];
    console.log(`  ${cls}: ${count} annotations (${pct}%)`);
  }
  console.log(`\nDataset metrics saved to: ${datasetMetricsPath}`);
}

main().catch(error => {
  console.error('Crawler failed:', error);
  process.exitCode = 1;
});
