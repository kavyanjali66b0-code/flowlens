"""
Semantic analyzer module to derive relationships between nodes after parsing.
"""

import logging
from typing import List, Dict

from .models import Node, Edge, EdgeType, NodeType


class SemanticAnalyzer:
    """Derives relationships (edges) from parsed nodes and symbols."""

    def __init__(self, nodes: List[Node], file_symbols: Dict[str, Dict]):
        self.nodes = nodes
        self.file_symbols = file_symbols or {}
        self.edges: List[Edge] = []

    def analyze(self) -> List[Edge]:
        logging.info("Running semantic analysis (relationships)")
        self._analyze_renders()
        self._analyze_calls()
        self._analyze_imports()
        # Placeholders for future analyses
        self._analyze_data_flow()
        self._resolve_dependency_injection()
        return self.edges

    def _analyze_renders(self):
        # RENDERS: use jsx_components captured per file
        for file, meta in list(self.file_symbols.items()):
            jsx_list = (meta or {}).get('jsx_components', [])
            if not jsx_list:
                continue
            source_components = [n for n in self.nodes if n.file == file and n.type == NodeType.COMPONENT]
            if not source_components:
                continue
            for jsx_name in jsx_list:
                targets = [n for n in self.nodes if n.name == jsx_name and n.type == NodeType.COMPONENT]
                for src in source_components:
                    for tgt in targets:
                        self.edges.append(Edge(source=src.id, target=tgt.id, type=EdgeType.RENDERS))

    def _analyze_calls(self):
        # Analyze function calls within the same file
        for file_path, symbols in self.file_symbols.items():
            if not symbols:
                continue
                
            # Get all functions/components in this file
            file_nodes = [n for n in self.nodes if n.file == file_path and n.type in {NodeType.FUNCTION, NodeType.COMPONENT, NodeType.CLASS}]
            
            # Look for calls to other functions in the same file
            for src_node in file_nodes:
                # Find other functions in the same file that could be called
                potential_targets = [n for n in file_nodes if n is not src_node and n.name != src_node.name]
                
                # For now, create edges based on naming patterns
                # In a more sophisticated implementation, we'd parse the actual function calls
                for target in potential_targets:
                    # Only create edge if it makes sense (avoid self-references)
                    if src_node.id != target.id:
                        self.edges.append(Edge(source=src_node.id, target=target.id, type=EdgeType.CALLS))
                        logging.debug(f"Created CALLS edge: {src_node.name} -> {target.name}")

    def _analyze_imports(self):
        """Analyze import relationships between files."""
        for file_path, symbols in self.file_symbols.items():
            if not symbols or 'imports' not in symbols:
                continue
                
            # Get the source module node
            source_module = None
            for node in self.nodes:
                if node.file == file_path and node.type == NodeType.MODULE:
                    source_module = node
                    break
            
            if not source_module:
                continue
                
            # For each import, try to find the target module
            for import_path in symbols['imports']:
                # Skip external dependencies for now
                if not import_path.startswith('.'):
                    continue
                    
                # Find target module by name matching (simplified)
                target_modules = [n for n in self.nodes if n.type == NodeType.MODULE and import_path in n.file]
                for target in target_modules:
                    if target.id != source_module.id:
                        self.edges.append(Edge(source=source_module.id, target=target.id, type=EdgeType.DEPENDS_ON))
                        logging.debug(f"Created DEPENDS_ON edge: {source_module.name} -> {target.name}")

    def _analyze_data_flow(self):
        # Placeholder for future data flow analysis
        pass

    def _resolve_dependency_injection(self):
        # Placeholder for future DI resolution
        pass
