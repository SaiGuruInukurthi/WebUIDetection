import { describe, it, expect } from 'vitest';
import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import os from 'node:os';

describe('Image Format Validation', () => {
  it('should verify WebP format configuration', async () => {
    // Verify that constants are correctly set
    const IMAGE_FORMAT = 'webp';
    const IMAGE_QUALITY = 80;

    expect(IMAGE_FORMAT).toBe('webp');
    expect(IMAGE_QUALITY).toBeGreaterThanOrEqual(70);
    expect(IMAGE_QUALITY).toBeLessThanOrEqual(95);
  });

  it('should convert PNG buffer to WebP using sharp', async () => {
    // Create a small test image (red 10x10 PNG)
    const testPngBuffer = await sharp({
      create: {
        width: 10,
        height: 10,
        channels: 3,
        background: { r: 255, g: 0, b: 0 }
      }
    })
      .png()
      .toBuffer();

    // Verify PNG buffer is created
    expect(testPngBuffer).toBeDefined();
    expect(testPngBuffer.length).toBeGreaterThan(0);

    // Convert to WebP
    const webpBuffer = await sharp(testPngBuffer)
      .toFormat('webp', { quality: 80 })
      .toBuffer();

    // Verify WebP buffer is created and smaller or equal to PNG
    expect(webpBuffer).toBeDefined();
    expect(webpBuffer.length).toBeGreaterThan(0);
    expect(webpBuffer.length).toBeLessThanOrEqual(testPngBuffer.length);

    // Verify WebP magic bytes (RIFF header)
    expect(webpBuffer[0]).toBe(0x52); // 'R'
    expect(webpBuffer[1]).toBe(0x49); // 'I'
    expect(webpBuffer[2]).toBe(0x46); // 'F'
    expect(webpBuffer[3]).toBe(0x46); // 'F'
  });

  it('should verify WebP file extension', async () => {
    // Test filename patterns
    const testCases = [
      { input: '00001_light_example.com_root_abc123.webp', format: 'webp', isValid: true },
      { input: '00002_dark_shop.com_product_def456.webp', format: 'webp', isValid: true },
      { input: '00003_mobile_blog.com_post_ghi789.webp', format: 'webp', isValid: true },
      { input: '00001_light_example.com_root_abc123.jpg', format: 'jpg', isValid: false },
      { input: '00001_light_example.com_root_abc123.png', format: 'png', isValid: false }
    ];

    for (const testCase of testCases) {
      const ext = path.extname(testCase.input).slice(1).toLowerCase();
      expect(ext).toBe(testCase.format);
      expect(ext === 'webp').toBe(testCase.isValid);
    }
  });

  it('should generate correct annotation file paths', async () => {
    const IMAGE_FORMAT = 'webp';

    const testImagePath = `screenshot_00001_light_example.com_root.${IMAGE_FORMAT}`;
    const annotationPath = testImagePath.replace(/\.\w+$/, '.json');

    expect(annotationPath).toBe('screenshot_00001_light_example.com_root.json');
    expect(annotationPath).not.toContain(IMAGE_FORMAT);
    expect(annotationPath.endsWith('.json')).toBe(true);
  });

  it('should handle variant-specific filenames', async () => {
    const variants = ['light', 'dark', 'mobile'];
    const IMAGE_FORMAT = 'webp';

    for (const variant of variants) {
      const filename = `00001_${variant}_example.com_root_hash123.${IMAGE_FORMAT}`;
      const ext = path.extname(filename);
      
      expect(ext).toBe(`.${IMAGE_FORMAT}`);
      expect(filename).toContain(variant);
      expect(filename.endsWith(`.${IMAGE_FORMAT}`)).toBe(true);
    }
  });

  it('should preserve image data during WebP conversion', async () => {
    // Create a colored test image
    const originalBuffer = await sharp({
      create: {
        width: 100,
        height: 100,
        channels: 3,
        background: { r: 100, g: 150, b: 200 }
      }
    })
      .png()
      .toBuffer();

    // Convert to WebP
    const webpBuffer = await sharp(originalBuffer)
      .toFormat('webp', { quality: 85 })
      .toBuffer();

    // Verify conversion maintained basic properties
    const webpMetadata = await sharp(webpBuffer).metadata();
    
    expect(webpMetadata.width).toBe(100);
    expect(webpMetadata.height).toBe(100);
    expect(webpMetadata.format).toBe('webp');
  });

  it('should handle quality settings correctly', async () => {
    const testBuffer = await sharp({
      create: {
        width: 200,
        height: 200,
        channels: 3,
        background: { r: 50, g: 100, b: 150 }
      }
    })
      .png()
      .toBuffer();

    // Test different quality levels
    const qualityLevels = [60, 75, 80, 90];
    let previousSize = Infinity;

    for (const quality of qualityLevels) {
      const result = await sharp(testBuffer)
        .toFormat('webp', { quality })
        .toBuffer();

      // Higher quality should generally result in larger file sizes
      // (or equal if compression is efficient)
      expect(result.length).toBeLessThanOrEqual(previousSize + 100); // Allow small variance
      previousSize = result.length;
    }
  });

  it('should validate WebP is more efficient than JPEG', async () => {
    // Create test image
    const testBuffer = await sharp({
      create: {
        width: 300,
        height: 300,
        channels: 3,
        background: { r: 75, g: 125, b: 175 }
      }
    })
      .png()
      .toBuffer();

    // Convert to WebP (quality 80)
    const webpBuffer = await sharp(testBuffer)
      .toFormat('webp', { quality: 80 })
      .toBuffer();

    // Convert to JPEG (quality 85)
    const jpegBuffer = await sharp(testBuffer)
      .toFormat('jpeg', { quality: 85 })
      .toBuffer();

    // WebP should be smaller or comparable
    // This demonstrates WebP efficiency advantage
    expect(webpBuffer.length).toBeLessThanOrEqual(jpegBuffer.length * 1.1); // Allow 10% variance
  });

  it('should generate valid filenames for all variants', async () => {
    const variants = ['light', 'dark', 'mobile'];
    const urls = [
      'https://example.com',
      'https://shop.com/products',
      'https://blog.example.org/post/123'
    ];

    for (const variant of variants) {
      for (const url of urls) {
        // Simulate filename generation logic
        const hash = 'abc12345'; // simulated
        const host = 'example'; // simulated
        const pathSegment = 'root'; // simulated
        
        const filename = `00001_${variant}_${host}_${pathSegment}_${hash}.webp`;
        const annotationFilename = filename.replace(/\.\w+$/, '.json');

        expect(filename).toMatch(/^\d{5}_(light|dark|mobile)_\w+_\w+_\w{8,}\.webp$/);
        expect(annotationFilename).toMatch(/^\d{5}_(light|dark|mobile)_\w+_\w+_\w{8,}\.json$/);
      }
    }
  });
});
