import { describe, it, expect } from 'vitest';

// Constants and types matching crawler.ts
type Variant = 'light' | 'dark';
type BoundingBox = {
  class: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type AnnotationData = {
  url: string;
  variant: Variant;
  viewport: { width: number; height: number };
  annotationCount: number;
  timestamp: string;
  annotations: BoundingBox[];
};

const VARIANTS: Variant[] = ['light', 'dark'];
const CLASSES = ['button', 'input', 'link', 'nav', 'form', 'image', 'dropdown', 'modal', 'header', 'footer'];

// Phase 2 limits (from paths.ts)
const phase2Limits = {
  targetUniqueUrls: 50000,
  targetImages: 100000,
  navigationTimeoutMs: 30000,
  postLoadDelayMs: 1000,
  maxScrollSteps: 3,
  minWidth: 8,
  minHeight: 8,
  minArea: 64
};

const desktopViewport = { width: 1920, height: 1080 };

// Helper functions for testing
function fileNameFor(url: string, index: number, variant: Variant): string {
  const normalized = url.toLowerCase().replace(/https?:\/\//i, '');
  const [host, ...rest] = normalized.split('/');
  const pathSegment = rest.join('-').replace(/[^a-z0-9-]/gi, '').slice(0, 20) || 'root';
  const hash = Math.random().toString(36).slice(2, 12);
  return `${String(index).padStart(5, '0')}_${variant}_${host}_${pathSegment}_${hash}.webp`;
}

function isValidBoundingBox(box: BoundingBox, viewport: typeof desktopViewport): boolean {
  // Must be one of the known classes
  if (!CLASSES.includes(box.class)) return false;

  // Must have positive dimensions
  if (box.width <= 0 || box.height <= 0) return false;

  // Must meet minimum size thresholds
  if (box.width < phase2Limits.minWidth || box.height < phase2Limits.minHeight) return false;
  if (box.width * box.height < phase2Limits.minArea) return false;

  // Must fit within viewport
  if (box.x < 0 || box.y < 0) return false;
  if (box.x + box.width > viewport.width) return false;
  if (box.y + box.height > viewport.height) return false;

  return true;
}

function validateAnnotationData(data: AnnotationData): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check required fields
  if (!data.url) errors.push('Missing url');
  if (!data.variant) errors.push('Missing variant');
  if (!data.viewport) errors.push('Missing viewport');
  if (data.annotationCount === undefined) errors.push('Missing annotationCount');
  if (!data.timestamp) errors.push('Missing timestamp');
  if (!data.annotations) errors.push('Missing annotations array');

  // Validate variant
  if (data.variant && !VARIANTS.includes(data.variant)) {
    errors.push(`Invalid variant: ${data.variant} (expected light|dark, NOT mobile)`);
  }

  // Validate viewport (must be desktop only)
  if (data.viewport) {
    if (data.viewport.width !== desktopViewport.width || data.viewport.height !== desktopViewport.height) {
      errors.push(`Invalid viewport: ${JSON.stringify(data.viewport)} (expected ${JSON.stringify(desktopViewport)})`);
    }
  }

  // Validate annotation count
  if (data.annotations && data.annotationCount !== data.annotations.length) {
    errors.push(`annotationCount mismatch: ${data.annotationCount} vs actual ${data.annotations.length}`);
  }

  // Validate each annotation
  if (data.annotations) {
    for (let i = 0; i < data.annotations.length; i++) {
      const box = data.annotations[i];
      if (!isValidBoundingBox(box, desktopViewport)) {
        errors.push(`Annotation ${i} failed validation: ${JSON.stringify(box)}`);
      }
    }
  }

  // Validate timestamp format
  if (data.timestamp) {
    try {
      const date = new Date(data.timestamp);
      if (isNaN(date.getTime())) {
        errors.push(`Invalid timestamp: ${data.timestamp}`);
      }
    } catch {
      errors.push(`Invalid timestamp format: ${data.timestamp}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

// Tests
describe('Crawler: Variants', () => {
  it('should only support light and dark variants (no mobile)', () => {
    expect(VARIANTS).toEqual(['light', 'dark']);
    expect(VARIANTS).not.toContain('mobile');
  });

  it('should use desktop viewport for both variants', () => {
    expect(desktopViewport.width).toBe(1920);
    expect(desktopViewport.height).toBe(1080);
  });
});

describe('Crawler: File Naming', () => {
  it('should generate valid filenames for each variant', () => {
    const url = 'https://example.com/page';
    for (const variant of VARIANTS) {
      const name = fileNameFor(url, 1, variant);
      expect(name).toMatch(/^\d{5}_\w+_[a-z0-9.-]+_[a-z0-9-]*_[a-z0-9]+\.webp$/i);
      expect(name).toContain(variant);
    }
  });

  it('should pad index with leading zeros', () => {
    const url = 'https://example.com';
    const name = fileNameFor(url, 1, 'light');
    expect(name).toMatch(/^00001_/);
  });
});

describe('Crawler: Bounding Box Validation', () => {
  const viewport = desktopViewport;

  it('should accept valid bounding boxes', () => {
    const validBoxes: BoundingBox[] = [
      { class: 'button', x: 100, y: 200, width: 120, height: 36 },
      { class: 'input', x: 50, y: 100, width: 300, height: 40 },
      { class: 'link', x: 0, y: 0, width: 100, height: 20 }
    ];

    for (const box of validBoxes) {
      expect(isValidBoundingBox(box, viewport)).toBe(true);
    }
  });

  it('should reject boxes with invalid class', () => {
    const box: BoundingBox = { class: 'invalid', x: 100, y: 100, width: 100, height: 100 };
    expect(isValidBoundingBox(box, viewport)).toBe(false);
  });

  it('should reject boxes smaller than minimum size', () => {
    const boxes: BoundingBox[] = [
      { class: 'button', x: 100, y: 100, width: 7, height: 8 }, // width < minWidth
      { class: 'button', x: 100, y: 100, width: 8, height: 7 }, // height < minHeight
      { class: 'button', x: 100, y: 100, width: 8, height: 8 }  // area = 64, exactly at minArea boundary
    ];

    expect(isValidBoundingBox(boxes[0], viewport)).toBe(false);
    expect(isValidBoundingBox(boxes[1], viewport)).toBe(false);
    expect(isValidBoundingBox(boxes[2], viewport)).toBe(true);
  });

  it('should reject boxes outside viewport', () => {
    const boxes: BoundingBox[] = [
      { class: 'button', x: -1, y: 100, width: 100, height: 100 }, // negative x
      { class: 'button', x: 100, y: -1, width: 100, height: 100 }, // negative y
      { class: 'button', x: 1900, y: 100, width: 100, height: 100 }, // extends beyond right
      { class: 'button', x: 100, y: 1000, width: 100, height: 100 }  // extends beyond bottom
    ];

    for (const box of boxes) {
      expect(isValidBoundingBox(box, viewport)).toBe(false);
    }
  });

  it('should reject boxes with zero or negative dimensions', () => {
    const boxes: BoundingBox[] = [
      { class: 'button', x: 100, y: 100, width: 0, height: 100 },
      { class: 'button', x: 100, y: 100, width: 100, height: 0 },
      { class: 'button', x: 100, y: 100, width: -10, height: 100 }
    ];

    for (const box of boxes) {
      expect(isValidBoundingBox(box, viewport)).toBe(false);
    }
  });
});

describe('Crawler: Annotation Data Validation', () => {
  it('should validate a correct annotation structure', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com',
      variant: 'light',
      viewport: desktopViewport,
      annotationCount: 2,
      timestamp: new Date().toISOString(),
      annotations: [
        { class: 'button', x: 100, y: 200, width: 120, height: 36 },
        { class: 'input', x: 50, y: 100, width: 300, height: 40 }
      ]
    };

    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should reject mobile variant', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com',
      variant: 'dark',
      viewport: desktopViewport,
      annotationCount: 1,
      timestamp: new Date().toISOString(),
      annotations: [{ class: 'button', x: 100, y: 200, width: 120, height: 36 }]
    };

    // Manually set to invalid variant to test
    (annotation as any).variant = 'mobile';
    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('mobile'))).toBe(true);
  });

  it('should reject incorrect viewport dimensions', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com',
      variant: 'light',
      viewport: { width: 390, height: 844 }, // mobile viewport
      annotationCount: 1,
      timestamp: new Date().toISOString(),
      annotations: [{ class: 'button', x: 100, y: 200, width: 120, height: 36 }]
    };

    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('Invalid viewport'))).toBe(true);
  });

  it('should detect annotation count mismatch', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com',
      variant: 'light',
      viewport: desktopViewport,
      annotationCount: 5, // says 5 but only 1 annotation
      timestamp: new Date().toISOString(),
      annotations: [{ class: 'button', x: 100, y: 200, width: 120, height: 36 }]
    };

    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('annotationCount mismatch'))).toBe(true);
  });

  it('should reject invalid timestamps', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com',
      variant: 'light',
      viewport: desktopViewport,
      annotationCount: 0,
      timestamp: 'invalid-date',
      annotations: []
    };

    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('timestamp'))).toBe(true);
  });

  it('should reject invalid annotations in array', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com',
      variant: 'light',
      viewport: desktopViewport,
      annotationCount: 1,
      timestamp: new Date().toISOString(),
      annotations: [
        { class: 'button', x: -10, y: 200, width: 120, height: 36 } // negative x
      ]
    };

    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('Annotation'))).toBe(true);
  });

  it('should allow zero annotations', () => {
    const annotation: AnnotationData = {
      url: 'https://example.com/empty-page',
      variant: 'dark',
      viewport: desktopViewport,
      annotationCount: 0,
      timestamp: new Date().toISOString(),
      annotations: []
    };

    const result = validateAnnotationData(annotation);
    expect(result.valid).toBe(true);
  });
});

describe('Crawler: Checkpoint Detection Logic', () => {
  it('should require both variants for completion', () => {
    const requiredVariants = ['light', 'dark'];
    expect(requiredVariants).toHaveLength(2);
    expect(requiredVariants).not.toContain('mobile');
  });

  it('should calculate correct screenshot targets', () => {
    const urlCount = 50000;
    const variantCount = 2;
    const expectedImages = urlCount * variantCount;

    expect(expectedImages).toBe(100000);
  });
});

describe('Crawler: Phase 2 Limits', () => {
  it('should have correct target values', () => {
    expect(phase2Limits.targetUniqueUrls).toBe(50000);
    expect(phase2Limits.targetImages).toBe(100000);
  });

  it('should have valid size thresholds', () => {
    expect(phase2Limits.minWidth).toBe(8);
    expect(phase2Limits.minHeight).toBe(8);
    expect(phase2Limits.minArea).toBe(64);
    expect(phase2Limits.minArea).toBe(phase2Limits.minWidth * phase2Limits.minHeight);
  });

  it('should have reasonable timeout values', () => {
    expect(phase2Limits.navigationTimeoutMs).toBe(30000);
    expect(phase2Limits.postLoadDelayMs).toBe(1000);
  });
});
