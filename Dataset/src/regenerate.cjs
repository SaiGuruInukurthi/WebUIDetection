const fs = require('fs').promises;
const path = require('path');

const datasetRoot = path.resolve(__dirname, '..');
const rawUrlsPath = path.join(datasetRoot, 'url-sources', 'raw-urls.txt');
const dedupPath = path.join(datasetRoot, 'url-sources', 'deduplicated-urls.txt');
const dedupWithCatPath = path.join(datasetRoot, 'url-sources', 'deduplicated-urls-with-category.json');

const TARGET = 50000;

const CATEGORIES = ['ecommerce','news','blog','portfolio','corporate','forum','social','docs','education','other'];

function isLikelyAsset(u) {
  return /\.(png|jpe?g|gif|svg|webp|css|js|ico|woff2?|pdf|zip)(\?|$)/i.test(u);
}

function classifyUrl(urlStr) {
  try {
    const u = new URL(urlStr);
    const host = u.hostname.toLowerCase();
    const pathn = (u.pathname || '').toLowerCase();

    if (host.includes('shop') || host.includes('store') || pathn.includes('/product') || pathn.includes('/cart') || pathn.includes('/checkout')) return 'ecommerce';
    if (host.includes('news') || pathn.startsWith('/news') || host.includes('nyt') || host.includes('cnn')) return 'news';
    if (host.includes('blog') || pathn.includes('/blog') || host.includes('medium')) return 'blog';
    if (host.includes('behance') || host.includes('dribbble') || pathn.includes('/portfolio')) return 'portfolio';
    if (pathn.includes('/forum') || host.includes('reddit') || host.includes('discourse') || pathn.includes('/thread')) return 'forum';
    if (host.includes('facebook') || host.includes('twitter') || host.includes('instagram') || host.includes('linkedin')) return 'social';
    if (pathn.startsWith('/docs') || host.includes('readthedocs') || host.includes('doc')) return 'docs';
    if (host.endsWith('.edu') || pathn.includes('/course') || pathn.includes('/university')) return 'education';
    if (pathn === '/' && host.split('.').length >= 2) return 'corporate';
    return 'other';
  } catch {
    return 'other';
  }
}

async function main() {
  let rawText;
  try {
    rawText = await fs.readFile(rawUrlsPath, 'utf8');
  } catch (err) {
    console.error('raw-urls.txt not found at', rawUrlsPath);
    process.exit(1);
  }

  const lines = rawText.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const seenDomain = new Set();
  const candidates = [];

  for (const l of lines) {
    try {
      const u = new URL(l.startsWith('http') ? l : `https://${l}`);
      if (isLikelyAsset(u.pathname)) continue;
      const domainKey = u.hostname.toLowerCase();
      if (seenDomain.has(domainKey)) continue;
      seenDomain.add(domainKey);
      candidates.push(u.href);
    } catch {
      continue;
    }
  }

  // Bucket by category
  const buckets = {};
  for (const cat of CATEGORIES) buckets[cat] = [];
  for (const url of candidates) {
    const cat = classifyUrl(url) || 'other';
    if (!buckets[cat]) buckets[cat] = [];
    buckets[cat].push(url);
  }

  // Round-robin select
  const selected = [];
  let progress = true;
  while (selected.length < TARGET && progress) {
    progress = false;
    for (const cat of CATEGORIES) {
      if (selected.length >= TARGET) break;
      const arr = buckets[cat];
      if (arr && arr.length > 0) {
        selected.push(arr.shift());
        progress = true;
      }
    }
  }

  await fs.mkdir(path.dirname(dedupPath), { recursive: true });
  await fs.writeFile(dedupPath, selected.join('\n') + '\n', 'utf8');
  const withCat = selected.map(u => ({ url: u, category: classifyUrl(u) }));
  await fs.writeFile(dedupWithCatPath, JSON.stringify(withCat, null, 2), 'utf8');

  console.log(`Wrote ${selected.length} URLs to ${dedupPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
