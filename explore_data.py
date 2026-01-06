#!/usr/bin/env python3
"""Explore the raw dataset structure to understand the data format."""

import gzip
import json
import os
from pathlib import Path
from collections import Counter

def explore_sample_data():
    sample_dir = Path("webui-7k/train_split_web7k/1655885421832")
    
    # Load class.json.gz (contains CSS class info)
    class_file = sample_dir / "default_1920-1080-class.json.gz"
    with gzip.open(class_file, 'rt') as f:
        class_data = json.load(f)
    
    # Load bb.json.gz (contains bounding boxes)
    bb_file = sample_dir / "default_1920-1080-bb.json.gz"
    with gzip.open(bb_file, 'rt') as f:
        bb_data = json.load(f)
    
    # Load box.json.gz (may contain additional info)
    box_file = sample_dir / "default_1920-1080-box.json.gz"
    with gzip.open(box_file, 'rt') as f:
        box_data = json.load(f)
    
    # Load axtree.json.gz (accessibility tree with semantic info)
    axtree_file = sample_dir / "default_1920-1080-axtree.json.gz"
    with gzip.open(axtree_file, 'rt') as f:
        axtree_data = json.load(f)
    
    results = []
    
    results.append("=== CLASS DATA STRUCTURE ===")
    results.append(f"Total elements: {len(class_data)}")
    results.append("Sample entries (first 15):")
    for i, (k, v) in enumerate(list(class_data.items())[:15]):
        results.append(f"  ID {k}: {v}")
    
    results.append("\n=== BB (Bounding Box) DATA STRUCTURE ===")
    results.append(f"Total elements: {len(bb_data)}")
    results.append("Sample entries (first 15):")
    for i, (k, v) in enumerate(list(bb_data.items())[:15]):
        results.append(f"  ID {k}: {v}")
    
    results.append("\n=== BOX DATA STRUCTURE ===")
    results.append(f"Total elements: {len(box_data)}")
    if isinstance(box_data, dict):
        results.append(f"Top keys: {list(box_data.keys())[:10]}")
        for i, (k, v) in enumerate(list(box_data.items())[:10]):
            results.append(f"  ID {k}: {str(v)[:200]}")
    
    results.append("\n=== AXTREE (Accessibility Tree) DATA STRUCTURE ===")
    if isinstance(axtree_data, dict):
        results.append(f"Keys: {list(axtree_data.keys())}")
        if 'nodes' in axtree_data:
            nodes = axtree_data['nodes']
            results.append(f"Number of nodes: {len(nodes)}")
            results.append("\nSample nodes (first 20):")
            for i, node in enumerate(nodes[:20]):
                if isinstance(node, dict):
                    role = node.get('role', {}).get('value', 'N/A') if isinstance(node.get('role'), dict) else node.get('role', 'N/A')
                    name = node.get('name', {}).get('value', 'N/A')[:50] if isinstance(node.get('name'), dict) else str(node.get('name', 'N/A'))[:50]
                    backend_id = node.get('backendDOMNodeId', 'N/A')
                    results.append(f"  Node {i}: role={role}, name={name}, backendId={backend_id}")
            
            # Extract all roles
            roles = []
            for node in nodes:
                if isinstance(node, dict):
                    role = node.get('role', {}).get('value', 'unknown') if isinstance(node.get('role'), dict) else node.get('role', 'unknown')
                    roles.append(role)
            
            results.append("\n=== ROLE DISTRIBUTION (axtree) ===")
            role_counts = Counter(roles)
            for role, count in role_counts.most_common(30):
                results.append(f"  {role}: {count}")
    
    # Write results to file
    with open("data_exploration_results.txt", "w") as f:
        f.write("\n".join(results))
    
    print("\n".join(results))

if __name__ == "__main__":
    explore_sample_data()
