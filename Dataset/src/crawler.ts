import { readFile } from 'node:fs/promises';
import { chromium, type Page } from 'playwright';
import { crawlFailuresPath, crawlManifestPath, deduplicatedUrlsPath, desktopViewport, mobileViewport, phase2Limits, screenshotRoot } from './paths.js';
import { canonicalizeUrl, ensureDir, safeFileSegment, shortHash, splitLines, writeJson, writeText } from './utils.js';

type Variant = 'light' | 'dark' | 'mobile';

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
};

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
const RETRIES = 2;

function fileNameFor(url: string, index: number, variant: Variant): string {
  const normalized = canonicalizeUrl(url) ?? url;
  const parsed = new URL(normalized);
  const host = safeFileSegment(parsed.hostname);
  const pathSegment = safeFileSegment(parsed.pathname.replace(/\//g, '-').replace(/^-+|-+$/g, '')) || 'root';
  const hash = shortHash(normalized);
  return `${String(index).padStart(5, '0')}_${variant}_${host}_${pathSegment}_${hash}.png`;
}

async function captureVariant(page: Page, url: string, filePath: string, variant: Variant): Promise<void> {
  await page.setViewportSize(variant === 'mobile' ? mobileViewport : desktopViewport);
  await page.emulateMedia({ colorScheme: variant === 'dark' ? 'dark' : 'light' });

  await page.goto(url, {
    waitUntil: 'domcontentloaded',
    timeout: phase2Limits.navigationTimeoutMs
  });

  await page.waitForTimeout(phase2Limits.postLoadDelayMs);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: filePath, fullPage: false });
}

async function captureWithRetries(page: Page, url: string, filePath: string, variant: Variant): Promise<{ status: CrawlManifestEntry['status']; error?: string }> {
  for (let attempt = 1; attempt <= RETRIES + 1; attempt += 1) {
    try {
      await captureVariant(page, url, filePath, variant);
      return { status: 'saved' };
    } catch (error) {
      if (attempt > RETRIES) {
        return { status: 'failed', error: error instanceof Error ? error.message : String(error) };
      }
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

  const browser = await chromium.launch({ headless: true });
  const manifestEntries: CrawlManifestEntry[] = [];
  const failures: CrawlFailure[] = [];

  let completed = 0;
  let screenshotCount = 0;

  for (const [index, url] of urls.entries()) {
    console.log(`Crawling ${index + 1}/${urls.length}: ${url}`);

    const page = await browser.newPage({ userAgent: USER_AGENT });
    const variants: Variant[] = ['light', 'dark', 'mobile'];

    try {
      for (const variant of variants) {
        const fileName = fileNameFor(url, index + 1, variant);
        const filePath = `${screenshotRoot}/${fileName}`;

        const result = await captureWithRetries(page, url, filePath, variant);
        manifestEntries.push({
          url,
          variant,
          fileName,
          status: result.status,
          attempt: result.status === 'saved' ? 1 : RETRIES + 1,
          timestamp: new Date().toISOString()
        });

        if (result.status === 'saved') {
          screenshotCount += 1;
        } else if (result.error) {
          failures.push({ url, error: result.error });
        }
      }

      completed += 1;
      if (completed % 50 === 0) {
        console.log(`Processed ${completed}/${urls.length} URLs (${screenshotCount} screenshots)`);
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
    failures
  });

  console.log(`Completed ${completed}/${urls.length} URLs`);
  console.log(`Saved ${screenshotCount} screenshots`);
  console.log(`Failures: ${failures.length}`);
}

main().catch(error => {
  console.error('Crawler failed:', error);
  process.exitCode = 1;
});