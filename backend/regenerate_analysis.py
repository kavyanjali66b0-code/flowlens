#!/usr/bin/env python3
"""Regenerate the response.json analysis file"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer.main import CodebaseAnalyzer

def main():
    print("Starting codebase analysis...")
    
    # Path to analyze
    target_path = r"c:\Users\Vandh\OneDrive\Desktop\Flow lens\flowlens-code-explorer"
    
    # Create analyzer and run
    analyzer = CodebaseAnalyzer()
    result = analyzer.analyze(target_path)
    
    print(f"Analysis complete!")
    print(f"  - Total nodes: {result.get('statistics', {}).get('total_nodes', 0)}")
    print(f"  - Total edges: {result.get('statistics', {}).get('total_edges', 0)}")
    
    # Save to response.json
    output_file = Path(__file__).parent / "response.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    print("Done!")

if __name__ == "__main__":
    main()
