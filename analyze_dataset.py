#!/usr/bin/env python3
"""
Analyze the full dataset to find the top 10 most frequent UI element types
from the accessibility tree (axtree) data.
"""

import gzip
import json
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import random

def analyze_dataset():
    data_dir = Path("webui-7k/train_split_web7k")
    
    # Get all sample directories
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Total samples found: {len(sample_dirs)}")
    
    # Use a subset for initial analysis (random sample of 500)
    random.seed(42)
    sample_subset = random.sample(sample_dirs, min(500, len(sample_dirs)))
    
    all_roles = Counter()
    elements_with_bbox = 0
    elements_without_bbox = 0
    
    for sample_dir in tqdm(sample_subset, desc="Analyzing samples"):
        axtree_file = sample_dir / "default_1920-1080-axtree.json.gz"
        bb_file = sample_dir / "default_1920-1080-bb.json.gz"
        
        if not axtree_file.exists() or not bb_file.exists():
            continue
        
        try:
            with gzip.open(axtree_file, 'rt') as f:
                axtree_data = json.load(f)
            
            with gzip.open(bb_file, 'rt') as f:
                bb_data = json.load(f)
            
            # Get set of available bounding box IDs
            bb_ids = set(bb_data.keys())
            
            # Process accessibility tree nodes
            if 'nodes' in axtree_data:
                for node in axtree_data['nodes']:
                    if not isinstance(node, dict):
                        continue
                    
                    # Get role
                    role = node.get('role', {})
                    if isinstance(role, dict):
                        role = role.get('value', 'unknown')
                    else:
                        role = str(role) if role else 'unknown'
                    
                    # Get backend DOM node ID
                    backend_id = str(node.get('backendDOMNodeId', ''))
                    
                    # Check if we have a bounding box for this element
                    if backend_id in bb_ids:
                        bbox = bb_data[backend_id]
                        w = bbox.get('width', 0)
                        h = bbox.get('height', 0)
                        
                        # Filter out very small or zero-size elements
                        if w >= 10 and h >= 10:
                            all_roles[role] += 1
                            elements_with_bbox += 1
                    else:
                        elements_without_bbox += 1
        except Exception as e:
            print(f"Error processing {sample_dir}: {e}")
            continue
    
    print("\n=== DATASET ANALYSIS RESULTS ===")
    print(f"Samples analyzed: {len(sample_subset)}")
    print(f"Elements with bbox (w,h >= 10): {elements_with_bbox}")
    print(f"Elements without matching bbox: {elements_without_bbox}")
    
    print("\n=== TOP 30 ROLES (Potential Classes) ===")
    for role, count in all_roles.most_common(30):
        print(f"  {role}: {count:,}")
    
    print("\n=== RECOMMENDED TOP 10 CLASSES ===")
    # Filter out generic roles like 'none', 'generic', 'StaticText'
    semantic_roles = [
        (r, c) for r, c in all_roles.most_common() 
        if r not in ['none', 'generic', 'StaticText', 'unknown', '']
    ]
    
    for i, (role, count) in enumerate(semantic_roles[:10]):
        print(f"  {i}: {role} ({count:,})")
    
    # Save results
    results = {
        'total_samples': len(sample_subset),
        'all_roles': dict(all_roles),
        'recommended_classes': [r for r, c in semantic_roles[:10]],
        'elements_with_bbox': elements_with_bbox
    }
    
    with open('class_distribution_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to class_distribution_analysis.json")

if __name__ == "__main__":
    analyze_dataset()
