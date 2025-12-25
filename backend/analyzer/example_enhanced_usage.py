"""
Example: Using Enhanced Semantic Analyzer

This example shows how to integrate the enhanced semantic analyzer
with embeddings into your existing pipeline.
"""

import logging
from pathlib import Path
from typing import List, Dict

from analyzer.parser import LanguageParser
from analyzer.symbol_table import SymbolTable
from analyzer.semantic_analyzer import SemanticAnalyzer
from analyzer.enhanced_semantic_analyzer import EnhancedSemanticAnalyzer
from analyzer.models import Edge, ProjectType

# Configure logging
logging.basicConfig(level=logging.INFO)


def basic_semantic_analysis(project_path: str) -> List[Edge]:
    """
    Example 1: Basic semantic analysis (current approach).
    
    Fast, rule-based, works for most projects.
    """
    print("=== Basic Semantic Analysis ===\n")
    
    # Initialize parser with symbol table
    symbol_table = SymbolTable()
    parser = LanguageParser(
        project_path,
        user_config={},
        symbol_table=symbol_table
    )
    
    # Parse project
    entry_points = [{'value': 'src/main.tsx'}]
    parser.parse_project(entry_points, ProjectType.REACT)
    
    print(f"Parsed {len(parser.nodes)} nodes")
    
    # Run basic semantic analysis
    analyzer = SemanticAnalyzer(
        parser.nodes,
        parser.file_symbols,
        symbol_table
    )
    
    edges = analyzer.analyze()
    
    print(f"Found {len(edges)} relationships")
    print(f"Edge types: {set(e.type.value for e in edges)}")
    
    return edges


def enhanced_semantic_analysis(project_path: str) -> List[Edge]:
    """
    Example 2: Enhanced semantic analysis with embeddings.
    
    Slower but finds more relationships, especially cross-file semantic connections.
    """
    print("\n=== Enhanced Semantic Analysis (with Embeddings) ===\n")
    
    # Initialize parser with symbol table
    symbol_table = SymbolTable()
    parser = LanguageParser(
        project_path,
        user_config={},
        symbol_table=symbol_table
    )
    
    # Parse project
    entry_points = [{'value': 'src/main.tsx'}]
    parser.parse_project(entry_points, ProjectType.REACT)
    
    print(f"Parsed {len(parser.nodes)} nodes")
    
    # Run enhanced semantic analysis
    analyzer = EnhancedSemanticAnalyzer(
        parser.nodes,
        parser.file_symbols,
        symbol_table,
        enable_embeddings=True,
        enable_clustering=True,
        similarity_threshold=0.75
    )
    
    edges = analyzer.analyze()
    
    print(f"Found {len(edges)} relationships")
    print(f"Edge types: {set(e.type.value for e in edges)}")
    
    # Print some example semantic edges
    semantic_edges = [e for e in edges if e.metadata and 'semantic_similarity' in e.metadata]
    print(f"\nFound {len(semantic_edges)} semantic similarity edges")
    
    if semantic_edges:
        print("\nExample semantic connections:")
        for edge in semantic_edges[:5]:
            source_node = next(n for n in parser.nodes if n.id == edge.source)
            target_node = next(n for n in parser.nodes if n.id == edge.target)
            similarity = edge.metadata['semantic_similarity']
            print(f"  {source_node.name} → {target_node.name} (similarity: {similarity:.2f})")
    
    return edges


def hybrid_analysis_with_fallback(project_path: str) -> List[Edge]:
    """
    Example 3: Hybrid approach with fallback (RECOMMENDED).
    
    Try enhanced analysis, fall back to basic if it fails.
    """
    print("\n=== Hybrid Analysis (Recommended) ===\n")
    
    # Initialize parser
    symbol_table = SymbolTable()
    parser = LanguageParser(
        project_path,
        user_config={},
        symbol_table=symbol_table
    )
    
    entry_points = [{'value': 'src/main.tsx'}]
    parser.parse_project(entry_points, ProjectType.REACT)
    
    print(f"Parsed {len(parser.nodes)} nodes")
    
    # Try enhanced analysis first
    try:
        print("Attempting enhanced analysis with embeddings...")
        analyzer = EnhancedSemanticAnalyzer(
            parser.nodes,
            parser.file_symbols,
            symbol_table,
            enable_embeddings=True,
            similarity_threshold=0.75
        )
        edges = analyzer.analyze()
        print("✓ Enhanced analysis successful")
        
    except Exception as e:
        print(f"✗ Enhanced analysis failed: {e}")
        print("Falling back to basic analysis...")
        
        analyzer = SemanticAnalyzer(
            parser.nodes,
            parser.file_symbols,
            symbol_table
        )
        edges = analyzer.analyze()
        print("✓ Basic analysis successful")
    
    print(f"Found {len(edges)} total relationships")
    return edges


def conditional_embeddings_by_project_size(project_path: str) -> List[Edge]:
    """
    Example 4: Conditional embeddings based on project size.
    
    Use embeddings only for projects that benefit from them.
    """
    print("\n=== Conditional Analysis by Project Size ===\n")
    
    # Initialize and parse
    symbol_table = SymbolTable()
    parser = LanguageParser(project_path, user_config={}, symbol_table=symbol_table)
    entry_points = [{'value': 'src/main.tsx'}]
    parser.parse_project(entry_points, ProjectType.REACT)
    
    num_nodes = len(parser.nodes)
    num_files = len(parser.file_symbols)
    
    print(f"Project stats: {num_nodes} nodes, {num_files} files")
    
    # Decision logic
    if num_files < 50:
        print("→ Small project: using basic analysis (fast)")
        use_embeddings = False
    elif num_files < 500:
        print("→ Medium project: using embeddings (balanced)")
        use_embeddings = True
    else:
        print("→ Large project: using embeddings + clustering")
        use_embeddings = True
    
    # Analyze
    if use_embeddings:
        analyzer = EnhancedSemanticAnalyzer(
            parser.nodes,
            parser.file_symbols,
            symbol_table,
            enable_embeddings=True,
            enable_clustering=(num_files > 500)
        )
    else:
        analyzer = SemanticAnalyzer(
            parser.nodes,
            parser.file_symbols,
            symbol_table
        )
    
    edges = analyzer.analyze()
    print(f"Found {len(edges)} relationships")
    
    return edges


def analyze_edge_quality(edges: List[Edge]):
    """
    Example 5: Analyze edge quality and confidence.
    
    Shows how to filter and validate semantic edges.
    """
    print("\n=== Edge Quality Analysis ===\n")
    
    # Group edges by confidence
    high_conf = [e for e in edges if e.metadata and e.metadata.get('confidence', 1.0) >= 0.9]
    med_conf = [e for e in edges if e.metadata and 0.7 <= e.metadata.get('confidence', 1.0) < 0.9]
    low_conf = [e for e in edges if e.metadata and e.metadata.get('confidence', 1.0) < 0.7]
    no_conf = [e for e in edges if not e.metadata or 'confidence' not in e.metadata]
    
    print(f"High confidence (≥0.9): {len(high_conf)} edges")
    print(f"Medium confidence (0.7-0.9): {len(med_conf)} edges")
    print(f"Low confidence (<0.7): {len(low_conf)} edges")
    print(f"No confidence score: {len(no_conf)} edges")
    
    # Group by method
    methods = {}
    for edge in edges:
        if edge.metadata and 'method' in edge.metadata:
            method = edge.metadata['method']
            methods[method] = methods.get(method, 0) + 1
    
    print("\nEdges by detection method:")
    for method, count in methods.items():
        print(f"  {method}: {count} edges")
    
    # Show example high-quality edges
    if high_conf:
        print("\nExample high-quality edges:")
        for edge in high_conf[:3]:
            conf = edge.metadata.get('confidence', 1.0)
            method = edge.metadata.get('method', 'unknown')
            print(f"  {edge.source} → {edge.target}")
            print(f"    Type: {edge.type.value}, Confidence: {conf:.2f}, Method: {method}")


def compare_basic_vs_enhanced(project_path: str):
    """
    Example 6: Compare basic vs enhanced analysis.
    
    Shows the difference in edge detection.
    """
    print("\n=== Comparison: Basic vs Enhanced ===\n")
    
    # Setup
    symbol_table = SymbolTable()
    parser = LanguageParser(project_path, user_config={}, symbol_table=symbol_table)
    entry_points = [{'value': 'src/main.tsx'}]
    parser.parse_project(entry_points, ProjectType.REACT)
    
    # Basic analysis
    print("Running basic analysis...")
    basic_analyzer = SemanticAnalyzer(parser.nodes, parser.file_symbols, symbol_table)
    basic_edges = basic_analyzer.analyze()
    
    # Enhanced analysis
    print("Running enhanced analysis...")
    enhanced_analyzer = EnhancedSemanticAnalyzer(
        parser.nodes,
        parser.file_symbols,
        symbol_table,
        enable_embeddings=True
    )
    enhanced_edges = enhanced_analyzer.analyze()
    
    # Compare
    print(f"\nBasic: {len(basic_edges)} edges")
    print(f"Enhanced: {len(enhanced_edges)} edges")
    print(f"Difference: {len(enhanced_edges) - len(basic_edges)} additional edges")
    
    # Find unique edges in enhanced
    basic_edge_keys = {(e.source, e.target, e.type.value) for e in basic_edges}
    unique_enhanced = [e for e in enhanced_edges 
                      if (e.source, e.target, e.type.value) not in basic_edge_keys]
    
    print(f"\nUnique to enhanced analysis: {len(unique_enhanced)} edges")
    
    if unique_enhanced:
        print("\nExample new relationships found:")
        for edge in unique_enhanced[:5]:
            if edge.metadata:
                method = edge.metadata.get('method', 'unknown')
                conf = edge.metadata.get('confidence', 1.0)
                print(f"  {edge.source} → {edge.target} ({method}, conf={conf:.2f})")


if __name__ == '__main__':
    # Example usage
    project_path = './test_project'  # Replace with actual path
    
    # Uncomment the examples you want to run:
    
    # Example 1: Basic analysis
    # basic_edges = basic_semantic_analysis(project_path)
    
    # Example 2: Enhanced analysis with embeddings
    # enhanced_edges = enhanced_semantic_analysis(project_path)
    
    # Example 3: Hybrid with fallback (RECOMMENDED)
    hybrid_edges = hybrid_analysis_with_fallback(project_path)
    
    # Example 4: Conditional by project size
    # conditional_edges = conditional_embeddings_by_project_size(project_path)
    
    # Example 5: Analyze edge quality
    analyze_edge_quality(hybrid_edges)
    
    # Example 6: Compare basic vs enhanced
    # compare_basic_vs_enhanced(project_path)
    
    print("\n✓ Analysis complete!")
