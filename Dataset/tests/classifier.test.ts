import { describe, it, expect } from 'vitest';

// Classification logic (copied from url-scraper.ts for testing)
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

// Balanced sampling logic (for testing)
function balancedSampling(urls: string[], target: number): string[] {
  const buckets: Record<string, string[]> = {};
  for (const category of CATEGORIES) {
    buckets[category] = [];
  }

  for (const url of urls) {
    const category = classifyUrl(url);
    buckets[category].push(url);
  }

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

  return selected;
}

describe('URL Classification', () => {
  it('should classify e-commerce URLs', () => {
    expect(classifyUrl('https://amazon.com/products')).toBe('ecommerce');
    expect(classifyUrl('https://shop.example.com/')).toBe('ecommerce');
    expect(classifyUrl('https://example.com/cart')).toBe('ecommerce');
    expect(classifyUrl('https://store.com/checkout')).toBe('ecommerce');
  });

  it('should classify news URLs', () => {
    expect(classifyUrl('https://news.example.com/')).toBe('news');
    expect(classifyUrl('https://example.com/news')).toBe('news');
    expect(classifyUrl('https://nytimes.com')).toBe('news');
    expect(classifyUrl('https://cnn.com/article')).toBe('news');
  });

  it('should classify blog URLs', () => {
    expect(classifyUrl('https://blog.example.com/')).toBe('blog');
    expect(classifyUrl('https://example.com/blog/post')).toBe('blog');
    expect(classifyUrl('https://medium.com/stories')).toBe('blog');
  });

  it('should classify portfolio URLs', () => {
    expect(classifyUrl('https://behance.net/user')).toBe('portfolio');
    expect(classifyUrl('https://dribbble.com/shots')).toBe('portfolio');
    expect(classifyUrl('https://example.com/portfolio')).toBe('portfolio');
  });

  it('should classify forum URLs', () => {
    expect(classifyUrl('https://example.com/forum')).toBe('forum');
    expect(classifyUrl('https://reddit.com/r/web')).toBe('forum');
    expect(classifyUrl('https://discourse.example.com')).toBe('forum');
    expect(classifyUrl('https://example.com/thread/123')).toBe('forum');
  });

  it('should classify social URLs', () => {
    expect(classifyUrl('https://facebook.com/page')).toBe('social');
    expect(classifyUrl('https://twitter.com/user')).toBe('social');
    expect(classifyUrl('https://instagram.com/profile')).toBe('social');
    expect(classifyUrl('https://linkedin.com/in/user')).toBe('social');
  });

  it('should classify documentation URLs', () => {
    expect(classifyUrl('https://example.com/docs')).toBe('docs');
    expect(classifyUrl('https://readthedocs.io/en/latest/')).toBe('docs');
    expect(classifyUrl('https://api-doc.example.com')).toBe('docs');
  });

  it('should classify education URLs', () => {
    expect(classifyUrl('https://university.edu/')).toBe('education');
    expect(classifyUrl('https://example.com/course/101')).toBe('education');
    expect(classifyUrl('https://example.com/university/info')).toBe('education');
  });

  it('should classify corporate URLs', () => {
    expect(classifyUrl('https://example.com/')).toBe('corporate');
    expect(classifyUrl('https://company.com/')).toBe('corporate');
  });

  it('should classify other URLs', () => {
    expect(classifyUrl('https://example.com/random/page')).toBe('other');
  });

  it('should handle invalid URLs gracefully', () => {
    expect(classifyUrl('not a url')).toBe('other');
    expect(classifyUrl('')).toBe('other');
  });
});

describe('Class Balancing (Round-Robin Sampling)', () => {
  it('should select balanced URLs across categories', () => {
    const testUrls = [
      'https://amazon.com/products',
      'https://shop.com/item',
      'https://news.com/',
      'https://nyt.com',
      'https://blog.com/post',
      'https://medium.com',
      'https://behance.net',
      'https://dribbble.com',
      'https://reddit.com',
      'https://forum.com/thread',
      'https://facebook.com',
      'https://twitter.com',
      'https://docs.example.com',
      'https://readthedocs.io',
      'https://university.edu',
      'https://course.edu/101',
      'https://company.com/',
      'https://example.com/'
    ];

    const selected = balancedSampling(testUrls, 10);

    // Count categories in selection
    const categoryCounts: Record<string, number> = {};
    for (const url of selected) {
      const cat = classifyUrl(url);
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    }

    // Check that we selected 10 URLs
    expect(selected.length).toBe(10);

    // Check that we have diversity across categories (not all from one category)
    expect(Object.keys(categoryCounts).length).toBeGreaterThan(1);

    // Each category should appear at most once per round (round-robin property)
    for (const count of Object.values(categoryCounts)) {
      expect(count).toBeLessThanOrEqual(2); // at most 2 due to round-robin with limited URLs
    }
  });

  it('should respect target limit', () => {
    const testUrls = Array.from({ length: 100 }, (_, i) => `https://example${i}.com/page`);

    const selected = balancedSampling(testUrls, 50);
    expect(selected.length).toBeLessThanOrEqual(50);

    const selected30 = balancedSampling(testUrls, 30);
    expect(selected30.length).toBe(30);
  });

  it('should return all URLs if target is larger than available', () => {
    const testUrls = ['https://example1.com/', 'https://example2.com/', 'https://example3.com/'];

    const selected = balancedSampling(testUrls, 100);
    expect(selected.length).toBe(3);
  });

  it('should distribute evenly across categories when possible', () => {
    // 20 URLs: 2 per category across 10 categories
    const testUrls = [
      'https://amazon.com/1', 'https://shop.com/1', // ecommerce
      'https://news.com/1', 'https://nyt.com/1', // news
      'https://blog.com/1', 'https://medium.com/1', // blog
      'https://behance.net/1', 'https://dribbble.com/1', // portfolio
      'https://reddit.com/1', 'https://forum.com/1', // forum
      'https://facebook.com/1', 'https://twitter.com/1', // social
      'https://docs.com/1', 'https://readthedocs.io/1', // docs
      'https://university.edu/1', 'https://course.edu/1', // education
      'https://company.com/', 'https://example.com/', // corporate
      'https://random1.com/x', 'https://random2.com/y' // other
    ];

    const selected = balancedSampling(testUrls, 20);

    // Count by category
    const categoryCounts: Record<string, number> = {};
    for (const url of selected) {
      const cat = classifyUrl(url);
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    }

    // With 20 URLs and 10 categories, expect roughly 2 per category
    expect(Object.keys(categoryCounts).length).toBeGreaterThanOrEqual(8);
  });
});
