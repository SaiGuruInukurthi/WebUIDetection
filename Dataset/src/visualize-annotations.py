#!/usr/bin/env python3
"""Visualize bounding box annotations on screenshots."""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

# Color palette for different element classes
CLASS_COLORS = {
    'button': '#FF0000',      # Red
    'input': '#00FF00',       # Green
    'link': '#0000FF',        # Blue
    'nav': '#FFFF00',         # Yellow
    'form': '#FF00FF',        # Magenta
    'image': '#00FFFF',       # Cyan
    'dropdown': '#FF8800',    # Orange
    'modal': '#FF0088',       # Pink
    'header': '#88FF00',      # Lime
    'footer': '#0088FF',      # Sky Blue
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def visualize_screenshot(jpg_path, json_path, output_path):
    """Draw bounding boxes on screenshot and save."""
    if not os.path.exists(jpg_path):
        print(f"  ❌ Screenshot not found: {jpg_path}")
        return False
    
    if not os.path.exists(json_path):
        print(f"  ❌ Annotations not found: {json_path}")
        return False
    
    # Load image
    img = Image.open(jpg_path)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Load annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    annotation_count = data.get('annotationCount', 0)
    
    if not annotations:
        print(f"  ⚠️  No annotations found in {json_path}")
        return False
    
    # Draw boxes
    for ann in annotations:
        elem_class = ann['class']
        x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
        
        # Get color for this class
        color_hex = CLASS_COLORS.get(elem_class, '#FFFFFF')
        color_rgb = hex_to_rgb(color_hex)
        
        # Draw filled rectangle with transparency
        box = [(x, y), (x + w, y + h)]
        draw.rectangle(box, outline=color_rgb + (255,), width=2, fill=color_rgb + (50,))
        
        # Draw label
        label = elem_class[:3].upper()
        try:
            draw.text((x + 2, y + 2), label, fill=color_rgb + (255,))
        except:
            # Fallback if font fails
            pass
    
    # Save visualization
    img.save(output_path)
    print(f"  ✅ Saved: {output_path} ({annotation_count} annotations)")
    return True

def main():
    screenshot_dir = Path('raw/screenshots')
    output_dir = Path('output/visualization')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all screenshot files (WebP, JPG, PNG, AVIF)
    supported_patterns = ['*.webp', '*.jpg', '*.jpeg', '*.png', '*.avif']
    all_files = []
    for pattern in supported_patterns:
        all_files.extend(screenshot_dir.glob(pattern))
    
    # Remove duplicates and sort
    jpg_files = sorted(dict.fromkeys(all_files))
    
    if not jpg_files:
        print("❌ No screenshots found in raw/screenshots/")
        return
    
    print(f"Found {len(jpg_files)} screenshots\n")
    
    # Visualize first 5 screenshots
    visualized = 0
    for jpg_path in jpg_files[:5]:
        json_path = jpg_path.with_suffix('.json')
        output_path = output_dir / f"viz_{jpg_path.name}"
        
        print(f"Visualizing: {jpg_path.name}")
        if visualize_screenshot(str(jpg_path), str(json_path), str(output_path)):
            visualized += 1
    
    print(f"\n✅ Visualization complete! {visualized} images saved to {output_dir}/")
    print("\nClass color legend:")
    for elem_class, color in sorted(CLASS_COLORS.items()):
        print(f"  {elem_class:10} = {color}")

if __name__ == '__main__':
    main()
