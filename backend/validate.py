"""
Quick validation script for local testing.

This script helps you validate that the analyzer works correctly
on sample projects during development.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import CodebaseAnalyzer


def validate_project(project_path: str):
    """Run quick validation on a project."""
    print(f"\n{'='*60}")
    print(f"Validating: {project_path}")
    print(f"{'='*60}\n")
    
    if not Path(project_path).exists():
        print(f"✗ FAIL: Project path does not exist: {project_path}")
        return
    
    print("Running analysis...")
    analyzer = CodebaseAnalyzer(enable_enhanced_analysis=False)
    
    try:
        result = analyzer.analyze(project_path)
    except Exception as e:
        print(f"✗ FAIL: Analysis crashed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check results
    nodes = result.get('graph', {}).get('nodes', [])
    edges = result.get('graph', {}).get('edges', [])
    errors = result.get('errors', [])
    warnings = result.get('insights', [])
    analysis_metadata = result.get('analysis_metadata', {})
    
    print(f"\n{'─'*60}")
    print("RESULTS:")
    print(f"{'─'*60}")
    print(f"✓ Nodes found: {len(nodes)}")
    print(f"✓ Edges found: {len(edges)}")
    print(f"  Analysis method: {analysis_metadata.get('analysis_method', 'unknown')}")
    
    if errors:
        print(f"\n⚠ Errors: {len(errors)}")
        for error in errors[:5]:
            print(f"  - {error.get('file')}: {error.get('message')}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    if warnings:
        print(f"\n📋 Insights/Warnings: {len(warnings)}")
        for warning in warnings[:3]:
            print(f"  - {warning.get('title')}")
    
    # Validate no Windows paths in IDs
    print(f"\n{'─'*60}")
    print("VALIDATION CHECKS:")
    print(f"{'─'*60}")
    
    windows_paths = [n['id'] for n in nodes if '\\' in n['id']]
    if windows_paths:
        print(f"✗ FAIL: Found {len(windows_paths)} Windows paths in node IDs:")
        for path in windows_paths[:3]:
            print(f"  - {path}")
        if len(windows_paths) > 3:
            print(f"  ... and {len(windows_paths) - 3} more")
    else:
        print(f"✓ PASS: All paths normalized (no backslashes)")
    
    # Validate no self-referencing edges
    self_refs = [e for e in edges if e['source'] == e['target']]
    if self_refs:
        print(f"✗ FAIL: Found {len(self_refs)} self-referencing edges:")
        for edge in self_refs[:3]:
            print(f"  - {edge['source']} -> {edge['target']} ({edge['type']})")
    else:
        print(f"✓ PASS: No self-referencing edges")
    
    # Check for duplicate edges
    edge_keys = [(e['source'], e['target'], e['type']) for e in edges]
    duplicates = len(edge_keys) - len(set(edge_keys))
    if duplicates > 0:
        print(f"⚠ WARNING: Found {duplicates} duplicate edges")
    else:
        print(f"✓ PASS: No duplicate edges")
    
    # Sample a few nodes
    if nodes:
        print(f"\n{'─'*60}")
        print("SAMPLE NODES:")
        print(f"{'─'*60}")
        for node in nodes[:3]:
            print(f"  {node['type']:15} {node['name']:30} ({node['file']})")
    
    # Sample a few edges
    if edges:
        print(f"\n{'─'*60}")
        print("SAMPLE EDGES:")
        print(f"{'─'*60}")
        for edge in edges[:5]:
            src = next((n for n in nodes if n['id'] == edge['source']), None)
            tgt = next((n for n in nodes if n['id'] == edge['target']), None)
            if src and tgt:
                print(f"  {src['name']:20} --{edge['type']:15}--> {tgt['name']}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate.py <project_path>")
        print("\nExamples:")
        print("  python validate.py ../flowlens-code-explorer")
        print("  python validate.py C:\\Users\\You\\Projects\\my-react-app")
        sys.exit(1)
    
    validate_project(sys.argv[1])
