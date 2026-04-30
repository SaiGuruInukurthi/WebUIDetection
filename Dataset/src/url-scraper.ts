import { basename } from 'node:path';
import { canonicalizeUrl, ensureDir, isLikelyAsset, readJson, splitLines, writeJson, writeText } from './utils.js';
import { deduplicatedUrlsPath, phase2Limits, rawUrlsPath, scraperLogPath, sourcePagesPath } from './paths.js';

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
      return null;
    }

    return await response.text();
  } catch {
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
  const sourcePages = await loadSourcePages();
  const rawUrls = new Set<string>();
  const sourceCounts: Record<string, number> = {};
  const failedPages: ScrapeLog['failedPages'] = [];

  await ensureDir(rawUrlsPath);
  await ensureDir(deduplicatedUrlsPath);
  await ensureDir(scraperLogPath);

  const entries = Object.entries(sourcePages).slice(0, MAX_SOURCE_PAGES);

  for (const [sourceName, pages] of entries) {
    let sourceCount = 0;

    for (const pageUrl of pages) {
      const html = await fetchText(pageUrl);
      if (!html) {
        failedPages.push({ source: sourceName, page: pageUrl, error: 'fetch failed' });
        continue;
      }

      const extracted = extractUrls(html, pageUrl);
      for (const candidate of extracted) {
        const normalized = normalizeForDeduplication(candidate);
        if (normalized) {
          rawUrls.add(normalized);
          sourceCount += 1;
        }
      }
    }

    sourceCounts[sourceName] = sourceCount;
  }

  const originCandidates = Array.from(rawUrls);
  const expanded = await expandWithSitemaps(originCandidates);

  const deduplicated = Array.from(new Set(expanded))
    .filter(filterPageUrl)
    .sort((left, right) => left.localeCompare(right));

  const finalUrls = deduplicated.slice(0, phase2Limits.targetUniqueUrls);

  await writeText(rawUrlsPath, `${originCandidates.join('\n')}\n`);
  await writeText(deduplicatedUrlsPath, `${finalUrls.join('\n')}\n`);

  const log: ScrapeLog = {
    generatedAt: new Date().toISOString(),
    targetUniqueUrls: phase2Limits.targetUniqueUrls,
    rawUrlCount: originCandidates.length,
    uniqueUrlCount: finalUrls.length,
    sourcePageCount: entries.reduce((total, [, pages]) => total + pages.length, 0),
    sourceCounts,
    failedPages
  };

  await writeJson(scraperLogPath, log);

  console.log(`Collected ${originCandidates.length} raw URLs`);
  console.log(`Deduplicated to ${finalUrls.length} URLs`);

  if (finalUrls.length < phase2Limits.targetUniqueUrls) {
    console.warn(`Target not yet met: ${finalUrls.length}/${phase2Limits.targetUniqueUrls}`);
  }
}

main().catch(error => {
  console.error('URL scraper failed:', error);
  process.exitCode = 1;
});