"""
Semantic analyzer module to derive relationships between nodes after parsing.
"""

import logging
from typing import List, Dict, Optional, Tuple

from .models import Node, Edge, EdgeType, NodeType
from .symbol_table import SymbolTable
from .edge_builder import EdgeBuilder


class SemanticAnalyzer:
    """Derives relationships (edges) from parsed nodes and symbols."""

    def __init__(self, nodes: List[Node], file_symbols: Dict[str, Dict], symbol_table: Optional[SymbolTable] = None):
        self.nodes = nodes
        self.file_symbols = file_symbols or {}
        self.symbol_table = symbol_table
        self.edges: List[Edge] = []

    def analyze(self) -> List[Edge]:
        logging.info("Running semantic analysis (relationships)")
        
        # Try EdgeBuilder first if symbol_table with references is available
        if self.symbol_table and self._has_symbol_references():
            logging.info("Using EdgeBuilder for context-aware edge creation")
            self._analyze_with_edge_builder()
        
        # Run traditional analysis methods (these complement EdgeBuilder)
        self._analyze_renders()
        self._analyze_calls()
        self._analyze_imports()
        
        # NEW: Hook dependency tracking
        self._analyze_hook_dependencies()
        
        # Placeholders for future analyses
        self._analyze_data_flow()
        self._resolve_dependency_injection()
        
        # Final cleanup: Remove any self-referencing edges that might have slipped through
        self._remove_self_referencing_edges()
        
        logging.info(f"Semantic analysis complete: {len(self.edges)} edges created")
        return self.edges
    
    def _remove_self_referencing_edges(self):
        """Remove self-referencing edges that shouldn't exist."""
        initial_count = len(self.edges)
        self.edges = [
            e for e in self.edges 
            if e.source != e.target
        ]
        removed = initial_count - len(self.edges)
        if removed > 0:
            logging.warning(f"Removed {removed} self-referencing edges during cleanup")
    
    def _has_symbol_references(self) -> bool:
        """Check if symbol table has any references to process."""
        if not self.symbol_table:
            return False
        return len(self.symbol_table.references) > 0
    
    def _analyze_with_edge_builder(self):
        """
        Use EdgeBuilder to create context-aware edges from symbol references.
        This is the primary edge creation method when symbol tracking is available.
        """
        try:
            builder = EdgeBuilder(self.symbol_table, self.nodes)
            edges = builder.build_edges_from_references()
            
            # Add edges, avoiding duplicates
            existing_edges = {(e.source, e.target, e.type.value) for e in self.edges}
            for edge in edges:
                edge_key = (edge.source, edge.target, edge.type.value)
                if edge_key not in existing_edges:
                    self.edges.append(edge)
                    existing_edges.add(edge_key)
            
            logging.info(f"EdgeBuilder created {len(edges)} context-aware edges")
        except Exception as e:
            logging.warning(f"EdgeBuilder failed, falling back to traditional analysis: {e}")
            # Continue with traditional methods as fallback

    def _analyze_renders(self):
        """
        Analyze React component rendering relationships.
        
        Creates RENDERS edges from files/modules to the components they render.
        """
        renders_created = 0
        
        # DEBUG: Log file_symbols structure
        total_files = len(self.file_symbols)
        files_with_jsx = sum(1 for meta in self.file_symbols.values() if (meta or {}).get('jsx_components'))
        logging.info(f"DEBUG RENDERS: Analyzing {total_files} files, {files_with_jsx} have JSX components")
        
        for file, meta in list(self.file_symbols.items()):
            jsx_list = (meta or {}).get('jsx_components', [])
            if not jsx_list:
                continue
            
            # Get the source node for this file (usually a module)
            source_nodes = [n for n in self.nodes if n.file == file]
            if not source_nodes:
                logging.warning(f"DEBUG RENDERS: No nodes found for file {file}")
                continue
            
            # Use the first node (typically the module node) as the source
            source_node = source_nodes[0]
            
            for jsx_name in jsx_list:
                # Find target nodes that match the JSX component name
                # Look for any node with matching name (could be function, component, or module)
                targets = [n for n in self.nodes if n.name == jsx_name]
                
                if not targets:
                    # Component might not be in the codebase (external library)
                    logging.debug(f"DEBUG RENDERS: Component '{jsx_name}' not found in nodes (likely external)")
                    continue
                
                for tgt in targets:
                    # CRITICAL: Prevent self-referencing edges
                    if source_node.id == tgt.id:
                        logging.debug(f"Skipping self-referencing RENDERS edge: {source_node.name} -> {tgt.name}")
                        continue
                    
                    # Check if edge already exists
                    edge_exists = any(
                        e.source == source_node.id and 
                        e.target == tgt.id and 
                        e.type == EdgeType.RENDERS
                        for e in self.edges
                    )
                    
                    if not edge_exists:
                        self.edges.append(Edge(source=source_node.id, target=tgt.id, type=EdgeType.RENDERS))
                        renders_created += 1
                        logging.info(f"Created RENDERS edge: {source_node.name} ({source_node.file}) -> {tgt.name} ({tgt.file})")
        
        logging.info(f"Created {renders_created} RENDERS edges")


    def _analyze_calls(self):
        """
        Analyze actual function calls from symbol references.
        
        This method now uses symbol_table references (which are populated from AST parsing)
        instead of heuristics. Only creates edges for actual function calls found in the code.
        
        CRITICAL: This replaces the old heuristic that created edges between ALL functions
        in the same file. Now only real function calls from AST parsing create edges.
        """
        if not self.symbol_table:
            logging.debug("No symbol table available for call analysis, skipping")
            return
        
        # DEBUG: Check symbol table population
        total_refs = len(self.symbol_table.references)
        logging.info(f"DEBUG: Symbol table has {total_refs} total references")
        
        # Use symbol references to find actual function calls
        # FIX: javascript_plugin adds references with context 'call' and 'async_call', not 'function_call'
        call_refs = self.symbol_table.get_references_by_type('call')
        async_call_refs = self.symbol_table.get_references_by_type('async_call')
        references = call_refs + async_call_refs
        logging.info(f"DEBUG: Found {len(call_refs)} 'call' + {len(async_call_refs)} 'async_call' = {len(references)} total function references")
        
        call_edges_created = 0
        skipped_self_ref = 0
        skipped_external = 0
        skipped_no_target = 0
        
        for reference in self.symbol_table.references:
            # Only process actual function calls (not imports, extends, etc.)
            if reference.context != 'call' and reference.context != 'async_call':
                continue
            
            # Find source node (where the call occurs) - prefer function/component scope
            source_node = self._find_node_at_location(reference.file_path, reference.line_number)
            if not source_node:
                # If we can't find a specific node, skip this call
                # This prevents creating edges from module-level to everywhere
                continue
            
            # Find target node (the function being called)
            target_node = self._find_function_node_by_name(
                reference.symbol_name, 
                reference.file_path, 
                reference.imported_from
            )
            
            if not target_node:
                # External library call or unresolved reference - skip
                skipped_external += 1
                continue
            
            # CRITICAL FIX: Avoid self-references (function calling itself is valid,
            # but we want to avoid creating edges from component to its own methods incorrectly)
            if source_node.id == target_node.id:
                skipped_self_ref += 1
                logging.debug(f"Skipping self-referencing CALLS edge: {source_node.name} -> {target_node.name}")
                continue
            
            # Check if edge already exists (from EdgeBuilder or previous analysis)
            edge_exists = any(
                e.source == source_node.id and 
                e.target == target_node.id and 
                e.type == EdgeType.CALLS
                for e in self.edges
            )
            
            if not edge_exists:
                # Create edge with high confidence - verified from AST parsing
                confidence = 0.9  # High confidence - from AST parsing
                edge_metadata = {
                    'confidence_score': confidence,
                    'context': 'calls relationship',
                    'flow_type': 'function_call',
                    'verified_from_ast': True,
                    'is_async': reference.context == 'async_call',
                    # Add location and context for visualization
                    'line_number': reference.line_number if hasattr(reference, 'line_number') else None,
                    'source_file': source_node.file,
                    'target_file': target_node.file,
                    'call_site': getattr(reference, 'context_snippet', f"{source_node.name}() -> {target_node.name}()")
                }
                
                edge = Edge(
                    source=source_node.id,
                    target=target_node.id,
                    type=EdgeType.ASYNC_CALLS if reference.context == 'async_call' else EdgeType.CALLS,
                    metadata=edge_metadata
                )
                self.edges.append(edge)
                call_edges_created += 1
                logging.debug(f"Created CALLS edge from AST: {source_node.name} -> {target_node.name}")
        
        logging.info(f"CALLS analysis: Created {call_edges_created} edges, "
                     f"skipped {skipped_self_ref} self-refs, {skipped_external} external calls")
    
    def _find_node_at_location(self, file_path: str, line_number: int) -> Optional[Node]:
        """
        Find the most specific node containing a given line number.
        
        Prefers functions/components over modules to ensure we're creating
        edges at the correct scope level.
        """
        # Get all nodes in the file
        file_nodes = [n for n in self.nodes if n.file == file_path]
        if not file_nodes:
            return None
        
        # Try to find exact match first using line_range metadata
        candidates_with_range = []
        for node in file_nodes:
            # Check if metadata has line_range (NodeMetadata dataclass)
            if hasattr(node.metadata, 'line_range') and node.metadata.line_range:
                start_line, end_line = node.metadata.line_range
                if start_line <= line_number <= end_line:
                    candidates_with_range.append((node, end_line - start_line))
            # Check if metadata is a dict with line_range
            elif isinstance(node.metadata, dict) and node.metadata.get('line_range'):
                start_line, end_line = node.metadata['line_range']
                if start_line <= line_number <= end_line:
                    candidates_with_range.append((node, end_line - start_line))
        
        # If we found candidates with line ranges, pick the most specific (smallest range)
        if candidates_with_range:
            candidates_with_range.sort(key=lambda x: x[1])  # Sort by range size
            return candidates_with_range[0][0]
        
        # Fallback: prefer functions/components over modules
        # This ensures function calls are attributed to functions, not modules
        function_nodes = [n for n in file_nodes if n.type in {NodeType.FUNCTION, NodeType.COMPONENT}]
        if function_nodes:
            # If multiple functions in file, return the first one
            # (In a more sophisticated implementation, we'd use scope analysis)
            return function_nodes[0]
        
        # Last resort: return module node (but this is less ideal)
        module_nodes = [n for n in file_nodes if n.type == NodeType.MODULE]
        return module_nodes[0] if module_nodes else None
    
    def _find_function_node_by_name(self, function_name: str, source_file: str, imported_from: Optional[str] = None) -> Optional[Node]:
        """Find a function/component node by name, considering imports."""
        # If imported_from is specified, look in that file first
        if imported_from:
            for node in self.nodes:
                if node.file == imported_from and node.name == function_name:
                    if node.type in {NodeType.FUNCTION, NodeType.COMPONENT}:
                        return node
        
        # Otherwise, look in the same file
        for node in self.nodes:
            if node.file == source_file and node.name == function_name:
                if node.type in {NodeType.FUNCTION, NodeType.COMPONENT}:
                    return node
        
        # Last resort: search all files (for global functions)
        for node in self.nodes:
            if node.name == function_name and node.type in {NodeType.FUNCTION, NodeType.COMPONENT}:
                return node
        
        return None

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
            for imp in symbols['imports']:
                # Handle both old string format and new dict format
                if isinstance(imp, dict):
                    import_path = imp['path']
                else:
                    import_path = imp
                
                # Skip external dependencies for now
                if not import_path.startswith('.'):
                    continue
                    
                # Find target module by name matching (simplified)
                target_modules = [n for n in self.nodes if n.type == NodeType.MODULE and import_path in n.file]
                for target in target_modules:
                    if target.id != source_module.id:
                        self.edges.append(Edge(source=source_module.id, target=target.id, type=EdgeType.IMPORTS))
                        logging.debug(f"Created IMPORTS edge: {source_module.name} -> {target.name}")

    def _analyze_data_flow(self):
        # Placeholder for future data flow analysis
        pass

    def _analyze_hook_dependencies(self):
        """
        Create edges from components to hooks they use.
        
        Analyzes hook_info metadata in component nodes to create USES edges
        for React hooks (useState, useEffect, custom hooks, etc.).
        """
        hook_edges_created = 0
        
        for node in self.nodes:
            # Only process components (they use hooks)
            if node.type != NodeType.COMPONENT:
                continue
            
            # Get hook info from metadata
            metadata = node.metadata
            if not metadata:
                continue
            
            # Handle both dict and NodeMetadata dataclass
            if hasattr(metadata, 'hook_info'):
                hook_info = metadata.hook_info
            elif isinstance(metadata, dict):
                hook_info = metadata.get('hook_info')
            else:
                continue
            
            if not hook_info:
                continue
            
            # Handle both dict and HookInfo dataclass
            if hasattr(hook_info, 'hook_name'):
                hook_name = hook_info.hook_name
            elif isinstance(hook_info, dict):
                hook_name = hook_info.get('hook_name')
            else:
                continue
            
            if not hook_name:
                continue
            
            # Find the hook node (could be a custom hook or built-in)
            hook_node = self._find_hook_node(hook_name, node.file)
            
            if hook_node:
                # Create USES edge from component to hook
                edge_exists = any(
                    e.source == node.id and 
                    e.target == hook_node.id and 
                    e.type == EdgeType.USES
                    for e in self.edges
                )
                
                if not edge_exists:
                    edge_metadata = {
                        'confidence_score': 1.0,  # High confidence - from hook detection
                        'context': 'hook_usage',
                        'hook_name': hook_name
                    }
                    
                    edge = Edge(
                        source=node.id,
                        target=hook_node.id,
                        type=EdgeType.USES,
                        metadata=edge_metadata
                    )
                    self.edges.append(edge)
                    hook_edges_created += 1
                    logging.debug(f"Created hook dependency edge: {node.name} -> {hook_name}")
        
        logging.info(f"Created {hook_edges_created} hook dependency edges")
    
    def _find_hook_node(self, hook_name: str, source_file: str) -> Optional[Node]:
        """Find a hook node by name (custom hooks start with 'use')."""
        # First, try to find in the same file
        for node in self.nodes:
            if node.file == source_file and node.name == hook_name:
                if node.type == NodeType.FUNCTION:
                    # Check if it's actually a hook (name starts with 'use' or has hook_info)
                    if hook_name.startswith('use') or (hasattr(node.metadata, 'hook_info') or 
                                                       (isinstance(node.metadata, dict) and node.metadata.get('hook_info'))):
                        return node
        
        # Search all files for the hook
        for node in self.nodes:
            if node.name == hook_name and node.type == NodeType.FUNCTION:
                # Verify it's a hook
                if hook_name.startswith('use') or (hasattr(node.metadata, 'hook_info') or 
                                                   (isinstance(node.metadata, dict) and node.metadata.get('hook_info'))):
                    return node
        
        return None
    
    def _resolve_dependency_injection(self):
        # Placeholder for future DI resolution
        pass
