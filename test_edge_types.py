"""Test script to verify edge type preservation."""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from analyzer import CodebaseAnalyzer

def test_edge_types():
    """Analyze a small codebase and check edge type diversity."""
    print("Testing edge type preservation...")
    
    # Analyze the frontend codebase
    target_path = Path(__file__).parent / 'flowlens-code-explorer' / 'src'
    
    if not target_path.exists():
        print(f"Path not found: {target_path}")
        return
    
    print(f"Analyzing: {target_path}")
    
    # Run analysis
    analyzer = CodebaseAnalyzer(str(target_path))
    results = analyzer.analyze()
    
    # Check edge types
    edge_types = {}
    edges_with_metadata = 0
    
    for edge in results['graph']['edges']:
        edge_type = edge['type']
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        if edge.get('metadata') and edge['metadata']:
            edges_with_metadata += 1
    
    # Print results
    print("\nAnalysis Complete!")
    print(f"\nEdge Type Distribution:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge_type}: {count}")
    
    print(f"\nMetadata:")
    print(f"  Edges with metadata: {edges_with_metadata}/{len(results['graph']['edges'])}")
    
    # Check if improvement worked
    unique_types = len(edge_types)
    if unique_types == 1 and 'depends_on' in edge_types:
        print("\nFAILED: All edges are still 'depends_on'")
        print("   The fix may not have worked correctly.")
    elif unique_types > 1:
        print(f"\nSUCCESS: Found {unique_types} different edge types!")
        print("   Edge type preservation is working!")
    
    # Save results
    output_file = Path(__file__).parent / 'backend' / 'response_test.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")

if __name__ == '__main__':
    test_edge_types()
