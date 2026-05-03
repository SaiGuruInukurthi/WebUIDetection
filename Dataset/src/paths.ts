import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const datasetRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const srcRoot = resolve(datasetRoot, 'src');
export const urlSourceRoot = resolve(datasetRoot, 'url-sources');
export const rawRoot = resolve(datasetRoot, 'raw');
export const screenshotRoot = resolve(rawRoot, 'screenshots');
export const annotationRoot = resolve(rawRoot, 'annotations');
export const outputRoot = resolve(datasetRoot, 'output');

export const rawUrlsPath = resolve(urlSourceRoot, 'raw-urls.txt');
export const deduplicatedUrlsPath = resolve(urlSourceRoot, 'deduplicated-urls.txt');
export const deduplicatedUrlsWithCategoryPath = resolve(urlSourceRoot, 'deduplicated-urls-with-category.json');
export const scraperLogPath = resolve(urlSourceRoot, 'url-scraper-log.json');
export const sourcePagesPath = resolve(urlSourceRoot, 'source-pages.json');
export const crawlManifestPath = resolve(screenshotRoot, 'manifest.jsonl');
export const crawlFailuresPath = resolve(urlSourceRoot, 'crawl-failures.json');
export const datasetMetricsPath = resolve(urlSourceRoot, 'dataset-metrics.json');

export const desktopViewport = { width: 1920, height: 1080 };
export const mobileViewport = { width: 390, height: 844 };

export const phase2Limits = {
  targetUniqueUrls: 50000,
  targetImages: 100000,
  navigationTimeoutMs: 30000,
  postLoadDelayMs: 1000,
  maxScrollSteps: 5,
  minWidth: 15,
  minHeight: 15,
  minArea: 225
} as const;

// Quality thresholds: images failing these checks are flagged as "low_quality"
export const qualityThresholds = {
  minAnnotationsPerImage: 5,        // reject images with < 5 annotations (too sparse)
  minClassDiversity: 2,             // reject images with < 2 different classes (no diversity)
  maxSingleClassRatio: 0.8          // reject images where 1 class > 80% of annotations (imbalance)
} as const;