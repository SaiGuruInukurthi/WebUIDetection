import { basename } from 'node:path';
import { canonicalizeUrl, ensureDir, isLikelyAsset, readJson, splitLines, writeJson, writeText } from './utils.js';
import { deduplicatedUrlsPath, deduplicatedUrlsWithCategoryPath, phase2Limits, rawUrlsPath, scraperLogPath, sourcePagesPath } from './paths.js';

type SourcePages = Record<string, string[]>;

type ScrapeLog = {
  generatedAt: string;
  targetUniqueUrls: number;
  rawUrlCount: number;
  uniqueUrlCount: number;
  sourcePageCount: number;
  sourceCounts: Record<string, number>;
  failedPages: Array<{ source: string; page: string; error: string }>;
};

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
const LINK_PATTERN = /href\s*=\s*["']([^"']+)["']/gi;
const URL_PATTERN = /https?:\/\/[^\s"'<>]+/gi;
const MAX_SOURCE_PAGES = 80;
const MAX_SITEMAPS = 120;
const MAX_URLS_PER_PAGE = 400;

const CATEGORIES = ['ecommerce', 'news', 'blog', 'portfolio', 'corporate', 'forum', 'social', 'docs', 'education', 'other'];

function classifyUrl(urlStr: string): string {
  try {
    const u = new URL(urlStr);
    const host = u.hostname.toLowerCase();
    const path = (u.pathname || '').toLowerCase();

    // ecommerce
    if (host.includes('shop') || host.includes('store') || path.includes('/product') || path.includes('/cart') || path.includes('/checkout')) {
      return 'ecommerce';
    }
    // news
    if (host.includes('news') || path.startsWith('/news') || host.includes('nyt') || host.includes('cnn')) {
      return 'news';
    }
    // blog
    if (host.includes('blog') || path.includes('/blog') || host.includes('medium')) {
      return 'blog';
    }
    // portfolio
    if (host.includes('behance') || host.includes('dribbble') || path.includes('/portfolio')) {
      return 'portfolio';
    }
    // forum
    if (path.includes('/forum') || host.includes('reddit') || host.includes('discourse') || path.includes('/thread')) {
      return 'forum';
    }
    // social
    if (host.includes('facebook') || host.includes('twitter') || host.includes('instagram') || host.includes('linkedin')) {
      return 'social';
    }
    // docs
    if (path.startsWith('/docs') || host.includes('readthedocs') || host.includes('doc')) {
      return 'docs';
    }
    // education
    if (host.endsWith('.edu') || path.includes('/course') || path.includes('/university')) {
      return 'education';
    }
    // corporate fallback
    if (path === '/' && host.split('.').length >= 2) {
      return 'corporate';
    }

    return 'other';
  } catch {
    return 'other';
  }
}

function loadDefaultSourcePages(): SourcePages {
  return {
    hackerNews: [
      'https://news.ycombinator.com/',
      'https://news.ycombinator.com/?p=2',
      'https://news.ycombinator.com/?p=3',
      'https://news.ycombinator.com/?p=4',
      'https://news.ycombinator.com/?p=5'
    ],
    productHunt: [
      'https://www.producthunt.com/topics/developer-tools',
      'https://www.producthunt.com/topics/artificial-intelligence',
      'https://www.producthunt.com/topics/productivity'
    ],
    awwwards: [
      'https://www.awwwards.com/websites/',
      'https://www.awwwards.com/websites/page/2/',
      'https://www.awwwards.com/websites/page/3/'
    ],
    siteInspire: [
      'https://www.siteinspire.com/websites',
      'https://www.siteinspire.com/websites?page=2',
      'https://www.siteinspire.com/websites?page=3'
    ],
    onePageLove: [
      'https://onepagelove.com/inspiration',
      'https://onepagelove.com/inspiration/page/2',
      'https://onepagelove.com/inspiration/page/3'
    ]
  };
}

async function loadSourcePages(): Promise<SourcePages> {
  try {
    return await readJson<SourcePages>(sourcePagesPath);
  } catch {
    return loadDefaultSourcePages();
  }
}

async function fetchText(pageUrl: string): Promise<string | null> {
  try {
    const response = await fetch(pageUrl, {
      headers: {
        'user-agent': USER_AGENT,
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
      }
    });

    if (!response.ok) {
      console.warn(`[WARN] fetchText: ${pageUrl} returned ${response.status}`);
      return null;
    }

    const text = await response.text();
    return text;
  } catch (err) {
    console.warn(`[WARN] fetchText: ${pageUrl} error: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  }
}

function extractUrls(html: string, baseUrl: string): string[] {
  const urls = new Set<string>();

  for (const match of html.matchAll(LINK_PATTERN)) {
    const candidate = canonicalizeUrl(match[1], baseUrl);
    if (candidate) {
      urls.add(candidate);
    }
  }

  for (const match of html.matchAll(URL_PATTERN)) {
    const candidate = canonicalizeUrl(match[0]);
    if (candidate) {
      urls.add(candidate);
    }
  }

  return Array.from(urls);
}

function filterPageUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    if (isLikelyAsset(parsed)) {
      return false;
    }

    const lastSegment = basename(parsed.pathname);
    if (!lastSegment && parsed.pathname !== '/') {
      return false;
    }

    return true;
  } catch {
    return false;
  }
}

function normalizeForDeduplication(value: string): string | null {
  const normalized = canonicalizeUrl(value);
  if (!normalized) {
    return null;
  }

  return filterPageUrl(normalized) ? normalized : null;
}

async function extractSitemaps(origin: string): Promise<string[]> {
  const discovered = new Set<string>();
  const robots = await fetchText(`${origin}/robots.txt`);

  if (robots) {
    for (const line of splitLines(robots)) {
      const lower = line.toLowerCase();
      if (lower.startsWith('sitemap:')) {
        const sitemapUrl = canonicalizeUrl(line.slice(8).trim());
        if (sitemapUrl) {
          discovered.add(sitemapUrl);
        }
      }
    }
  }

  discovered.add(`${origin}/sitemap.xml`);
  discovered.add(`${origin}/sitemap_index.xml`);

  return Array.from(discovered);
}

function extractSitemapUrls(xml: string): string[] {
  const urls = new Set<string>();
  const locPattern = /<loc>([^<]+)<\/loc>/gi;

  for (const match of xml.matchAll(locPattern)) {
    const candidate = canonicalizeUrl(match[1]);
    if (candidate) {
      urls.add(candidate);
    }
  }

  return Array.from(urls);
}

async function expandWithSitemaps(candidateUrls: string[]): Promise<string[]> {
  const expanded = new Set<string>(candidateUrls);
  const origins = Array.from(new Set(candidateUrls.map(url => new URL(url).origin))).slice(0, MAX_SITEMAPS);

  for (const origin of origins) {
    const sitemapCandidates = await extractSitemaps(origin);

    for (const sitemapUrl of sitemapCandidates) {
      const sitemapText = await fetchText(sitemapUrl);
      if (!sitemapText) {
        continue;
      }

      const urls = extractSitemapUrls(sitemapText).filter(filterPageUrl);
      for (const url of urls) {
        expanded.add(url);
        if (expanded.size >= phase2Limits.targetUniqueUrls * 3) {
          return Array.from(expanded);
        }
      }
    }
  }

  return Array.from(expanded);
}

async function main(): Promise<void> {
  console.log('[LOG] Scraper starting...');
  
  // If called with --regenerate, use existing raw-urls.txt to build deduplicated list
  if (process.argv.includes('--regenerate')) {
    console.log('[LOG] --regenerate flag detected, using existing raw-urls.txt');
    await ensureDir(rawUrlsPath);
    await ensureDir(deduplicatedUrlsPath);

    let rawText = '';
    try {
      rawText = await (await import('node:fs/promises')).readFile(rawUrlsPath, 'utf8');
    } catch {
      console.error(`[ERROR] Raw URLs file not found: ${rawUrlsPath}`);
      process.exitCode = 1;
      return;
    }

    const rawSet = new Set(splitLines(rawText));
    const deduplicated = Array.from(rawSet).map(normalizeForDeduplication).filter(Boolean) as string[];

    // Buckets by classification
    const buckets: Record<string, string[]> = {};
    for (const url of deduplicated) {
      const category = classifyUrl(url);
      if (!buckets[category]) buckets[category] = [];
      buckets[category].push(url);
    }

    const categories = Object.keys(buckets);
    const target = phase2Limits.targetUniqueUrls;
    const selected: string[] = [];
    let idx = 0;
    while (selected.length < target) {
      let progressed = false;
      for (const cat of categories) {
        const arr = buckets[cat];
        if (arr && arr.length > 0 && selected.length < target) {
          selected.push(arr.shift() as string);
          progressed = true;
        }
      }
      if (!progressed) break;
      idx += 1;
    }

    await writeText(deduplicatedUrlsPath, `${selected.join('\n')}\n`);
    await writeJson(deduplicatedUrlsWithCategoryPath, selected.map(u => ({ url: u, category: classifyUrl(u) })));

    console.log(`[LOG] Regenerated ${selected.length} deduplicated URLs to ${deduplicatedUrlsPath}`);
    return;
  }

  console.log('[LOG] Starting full scrape from source pages...');
  const sourcePages = await loadSourcePages();
  console.log(`[LOG] Loaded source pages: ${Object.keys(sourcePages).length} sources`);
  
  const rawUrls = new Set<string>();
  const sourceCounts: Record<string, number> = {};
  const failedPages: ScrapeLog['failedPages'] = [];

  console.log('[LOG] Ensuring directories...');
  await ensureDir(rawUrlsPath);
  await ensureDir(deduplicatedUrlsPath);
  await ensureDir(scraperLogPath);

  const entries = Object.entries(sourcePages).slice(0, MAX_SOURCE_PAGES);
  console.log(`[LOG] Processing ${entries.length} source entries...`);

  for (const [sourceName, pages] of entries) {
    console.log(`[LOG] Scraping source: ${sourceName} with ${pages.length} pages`);
    let sourceCount = 0;

    for (const pageUrl of pages) {
      console.log(`[LOG]   Fetching: ${pageUrl}`);
      const html = await fetchText(pageUrl);
      if (!html) {
        console.warn(`[WARN] Failed to fetch: ${pageUrl}`);
        failedPages.push({ source: sourceName, page: pageUrl, error: 'fetch failed' });
        continue;
      }

      console.log(`[LOG]   Fetched ${html.length} bytes, extracting URLs...`);
      const extracted = extractUrls(html, pageUrl);
      console.log(`[LOG]   Found ${extracted.length} URLs on page`);
      
      for (const candidate of extracted) {
        const normalized = normalizeForDeduplication(candidate);
        if (normalized) {
          rawUrls.add(normalized);
          sourceCount += 1;
        }
      }
    }

    sourceCounts[sourceName] = sourceCount;
    console.log(`[LOG] Source ${sourceName} complete: ${sourceCount} normalized URLs`);
  }

  const originCandidates = Array.from(rawUrls);
  console.log(`[LOG] Total raw URLs collected: ${originCandidates.length}`);
  
  console.log(`[LOG] Expanding with sitemaps from ${originCandidates.length} origins...`);
  const expanded = await expandWithSitemaps(originCandidates);
  console.log(`[LOG] After sitemap expansion: ${expanded.length} URLs`);

  const deduplicated = Array.from(new Set(expanded))
    .filter(filterPageUrl)
    .sort((left, right) => left.localeCompare(right));
  console.log(`[LOG] After deduplication and filtering: ${deduplicated.length} URLs`);

  // Balanced sampling across categories
  const buckets: Record<string, string[]> = {};
  for (const category of CATEGORIES) {
    buckets[category] = [];
  }

  for (const url of deduplicated) {
    const category = classifyUrl(url);
    buckets[category].push(url);
  }
  console.log(`[LOG] Categorized URLs: ${Object.entries(buckets).map(([cat, urls]) => `${cat}=${urls.length}`).join(', ')}`);

  // Round-robin sampling across categories
  const target = phase2Limits.targetUniqueUrls;
  console.log(`[LOG] Starting round-robin selection targeting ${target} URLs...`);
  const selected: string[] = [];
  let more = true;

  while (selected.length < target && more) {
    more = false;
    for (const category of CATEGORIES) {
      const arr = buckets[category];
      if (arr.length > 0 && selected.length < target) {
        const url = arr.shift();
        if (url) {
          selected.push(url);
          more = true;
        }
      }
    }
  }

  const finalUrls = selected;
  console.log(`[LOG] Selected ${finalUrls.length} URLs via round-robin`);

  // Write deduplicated URLs without category (for crawler)
  console.log(`[LOG] Writing to ${deduplicatedUrlsPath}...`);
  await writeText(deduplicatedUrlsPath, `${finalUrls.join('\n')}\n`);

  // Write deduplicated URLs with category (for analysis)
  console.log(`[LOG] Writing category mappings to ${deduplicatedUrlsWithCategoryPath}...`);
  await writeJson(deduplicatedUrlsWithCategoryPath, finalUrls.map(url => ({ url, category: classifyUrl(url) })));

  const log: ScrapeLog = {
    generatedAt: new Date().toISOString(),
    targetUniqueUrls: phase2Limits.targetUniqueUrls,
    rawUrlCount: originCandidates.length,
    uniqueUrlCount: finalUrls.length,
    sourcePageCount: entries.reduce((total, [, pages]) => total + pages.length, 0),
    sourceCounts,
    failedPages
  };

  console.log(`[LOG] Writing log to ${scraperLogPath}...`);
  await writeJson(scraperLogPath, log);
  console.log(`[LOG] Scraper complete!`);

  console.log(`\n✓ Collected ${originCandidates.length} raw URLs`);
  console.log(`✓ Deduplicated to ${finalUrls.length} URLs`);

  if (finalUrls.length < phase2Limits.targetUniqueUrls) {
    console.warn(`⚠ Target not yet met: ${finalUrls.length}/${phase2Limits.targetUniqueUrls}`);
  }
}

main().catch(error => {
  console.error('[ERROR] URL scraper failed:', error);
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});