import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export async function ensureDir(filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
}

export async function writeText(filePath: string, content: string): Promise<void> {
  await ensureDir(filePath);
  await writeFile(filePath, content, 'utf8');
}

export async function writeJson(filePath: string, value: unknown): Promise<void> {
  await writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

export async function readJson<T>(filePath: string): Promise<T> {
  const content = await readFile(filePath, 'utf8');
  return JSON.parse(content) as T;
}

export function splitLines(content: string): string[] {
  return content.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
}

export function canonicalizeUrl(input: string, baseUrl?: string): string | null {
  try {
    const resolved = new URL(input, baseUrl);
    if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') {
      return null;
    }

    resolved.hash = '';
    resolved.search = '';
    resolved.hostname = resolved.hostname.toLowerCase();
    resolved.protocol = resolved.protocol.toLowerCase();
    resolved.pathname = resolved.pathname.replace(/\/+/g, '/');

    if (resolved.pathname.length > 1 && resolved.pathname.endsWith('/')) {
      resolved.pathname = resolved.pathname.slice(0, -1);
    }

    return resolved.toString();
  } catch {
    return null;
  }
}

export function isLikelyAsset(url: URL): boolean {
  return /\.(png|jpe?g|gif|webp|svg|ico|css|js|mjs|json|xml|txt|mp4|mp3|pdf|zip|gz|tar)(\?|$)/i.test(url.pathname);
}

export function shortHash(value: string): string {
  return createHash('sha1').update(value).digest('hex').slice(0, 10);
}

export function safeFileSegment(value: string): string {
  return value
    .toLowerCase()
    .replace(/https?:\/\//g, '')
    .replace(/[^a-z0-9.-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
}