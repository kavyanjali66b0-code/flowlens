"""
EdgeBuilder: Converts symbol references into typed edges with rich context.

This module provides the EdgeBuilder class that:
- Processes SymbolReference objects from the symbol table
- Determines appropriate EdgeType based on reference context
- Creates Edge objects with full EdgeContext metadata
- Resolves symbol definitions to enrich edge information
"""

from typing import List, Dict, Optional, Set
from pathlib import Path
import re

from .models import Edge, EdgeType, Node
from .symbol_table import SymbolTable, Symbol, SymbolReference, SymbolType
from .edge_context import EdgeContext

# Regex to validate valid JavaScript/TypeScript identifier names
# Valid identifiers: start with letter, underscore, or $; followed by alphanumeric, _, or $
VALID_IDENTIFIER = re.compile(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$')

# Common JavaScript built-in globals that should be tagged for optional filtering
JS_BUILTINS = {
    # Primitives & constructors
    'Number', 'String', 'Boolean', 'Symbol', 'BigInt', 'Object', 'Array', 'Function',
    'Date', 'RegExp', 'Error', 'Map', 'Set', 'WeakMap', 'WeakSet', 'Promise',
    # Global functions
    'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'decodeURI', 'encodeURI',
    'decodeURIComponent', 'encodeURIComponent', 'eval', 'setTimeout', 'setInterval',
    'clearTimeout', 'clearInterval', 'fetch', 'alert', 'confirm', 'prompt',
    # Array methods (common)
    'map', 'filter', 'reduce', 'forEach', 'find', 'findIndex', 'some', 'every',
    'includes', 'indexOf', 'slice', 'splice', 'concat', 'join', 'push', 'pop',
    'shift', 'unshift', 'sort', 'reverse', 'flat', 'flatMap',
    # String methods (common)
    'toString', 'valueOf', 'charAt', 'charCodeAt', 'substring', 'substr', 'slice',
    'replace', 'replaceAll', 'match', 'search', 'split', 'trim', 'toLowerCase', 'toUpperCase',
    'startsWith', 'endsWith', 'includes', 'repeat', 'padStart', 'padEnd',
    # Object methods
    'keys', 'values', 'entries', 'assign', 'freeze', 'seal', 'hasOwnProperty',
    # Math
    'abs', 'ceil', 'floor', 'round', 'max', 'min', 'pow', 'sqrt', 'random',
    # JSON
    'parse', 'stringify',
    # Console
    'log', 'warn', 'error', 'info', 'debug', 'trace', 'table', 'dir',
    # Events
    'addEventListener', 'removeEventListener',
    # DOM
    'getElementById', 'querySelector', 'querySelectorAll', 'createElement', 'appendChild',
    # Date methods
    'getTime', 'getFullYear', 'getMonth', 'getDate', 'getDay', 'getHours', 'getMinutes',
    'getSeconds', 'toISOString', 'toLocaleDateString', 'toLocaleTimeString',
    # Number methods
    'toFixed', 'toPrecision', 'toExponential'
}


class EdgeBuilder:
    """
    Builds typed edges with context from symbol references.
    
    The EdgeBuilder converts raw symbol references (function calls, class instantiations,
    inheritance) into strongly-typed Edge objects with rich EdgeContext metadata.
    
    Usage:
        builder = EdgeBuilder(symbol_table, nodes)
        edges = builder.build_edges_from_references()
    """
    
    def __init__(self, symbol_table: SymbolTable, nodes: List[Node]):
        """
        Initialize EdgeBuilder.
        
        Args:
            symbol_table: SymbolTable containing symbols and references
            nodes: List of Node objects (for node lookup by file/line)
        """
        self.symbol_table = symbol_table
        self.nodes = nodes
        
        # Build lookup indexes for fast access
        self._node_by_file: Dict[str, List[Node]] = {}
        self._node_by_id: Dict[str, Node] = {}
        self._build_node_indexes()
    
    def _build_node_indexes(self):
        """Build indexes for fast node lookup."""
        for node in self.nodes:
            # Index by file path
            if node.file not in self._node_by_file:
                self._node_by_file[node.file] = []
            self._node_by_file[node.file].append(node)
            
            # Index by ID
            self._node_by_id[node.id] = node
    
    def build_edges_from_references(self) -> List[Edge]:
        """
        Build edges from all symbol references in the symbol table.
        
        Returns:
            List of Edge objects with EdgeContext metadata
        """
        edges = []
        
        # Process each reference
        for reference in self.symbol_table.references:
            edge = self._build_edge_from_reference(reference)
            if edge:
                edges.append(edge)
        
        return edges
    
    def _build_edge_from_reference(self, reference: SymbolReference) -> Optional[Edge]:
        """
        Build an edge from a single symbol reference.
        
        Args:
            reference: SymbolReference object
            
        Returns:
            Edge object with EdgeContext, or None if edge cannot be built
        """
        # Determine edge type from reference context
        edge_type = self._determine_edge_type(reference)
        if not edge_type:
            return None
        
        # Find source node (where the reference occurs)
        source_node = self._find_node_at_location(
            reference.file_path,
            reference.line_number
        )
        if not source_node:
            return None
        
        # Find target node (the symbol being referenced)
        target_node = self._find_target_node(reference)
        if not target_node:
            # Create a placeholder target for external references
            target_node = self._create_placeholder_node(reference)
            # Skip if placeholder creation failed (malformed symbol)
            if not target_node:
                return None
        
        # Build EdgeContext with rich metadata
        edge_context = self._create_edge_context(
            edge_type=edge_type,
            reference=reference,
            source_node=source_node,
            target_node=target_node
        )
        
        # Create Edge with merged metadata
        # Start with edge context, then merge in reference metadata
        edge_metadata = edge_context.to_dict()
        if reference.metadata:
            edge_metadata.update(reference.metadata)
        
        edge = Edge(
            source=source_node.id,
            target=target_node.id,
            type=edge_type,
            metadata=edge_metadata
        )
        
        return edge
    
    def _determine_edge_type(self, reference: SymbolReference) -> Optional[EdgeType]:
        """
        Determine EdgeType from reference context.
        
        Args:
            reference: SymbolReference object
            
        Returns:
            EdgeType enum value, or None if type cannot be determined
        """
        context = reference.context
        metadata = reference.metadata
        
        # Map reference context to EdgeType
        if context == 'call':
            # Check if async call
            if metadata.get('is_async'):
                return EdgeType.ASYNC_CALLS
            return EdgeType.CALLS
        
        elif context == 'async_call':
            return EdgeType.ASYNC_CALLS
        
        elif context == 'instantiate':
            return EdgeType.INSTANTIATES
        
        elif context == 'extends':
            return EdgeType.EXTENDS
        
        elif context == 'implements':
            return EdgeType.IMPLEMENTS
        
        elif context == 'import':
            return EdgeType.IMPORTS
        
        elif context == 'export':
            return EdgeType.EXPORTS
        
        elif context == 'access':
            return EdgeType.USES
        
        else:
            # Unknown context, default to CALLS
            return EdgeType.CALLS
    
    def _find_node_at_location(self, file_path: str, line_number: int) -> Optional[Node]:
        """
        Find the node at a specific file location.
        
        Args:
            file_path: Relative file path
            line_number: Line number (1-indexed)
            
        Returns:
            Node object at that location, or None
        """
        # Get all nodes in the file
        nodes_in_file = self._node_by_file.get(file_path, [])
        
        # Find the most specific node containing this line
        # Prefer smaller nodes (more specific scope)
        best_match = None
        best_match_size = float('inf')
        
        for node in nodes_in_file:
            # Check if node contains this line
            # Note: Some nodes might not have line info
            node_line = (node.metadata or {}).get('line')
            if node_line and node_line == line_number:
                return node
            
            # If no exact match, find containing node
            if node.file == file_path:
                # Calculate "size" based on metadata
                size = 1  # Default size
                if best_match is None or size < best_match_size:
                    best_match = node
                    best_match_size = size
        
        return best_match
    
    def _find_target_node(self, reference: SymbolReference) -> Optional[Node]:
        """
        Find the target node (symbol definition) for a reference.
        
        Args:
            reference: SymbolReference object
            
        Returns:
            Node representing the symbol definition, or None
        """
        # Try to find the symbol definition in the symbol table
        symbols = self.symbol_table.get_symbols_by_name(reference.symbol_name)
        
        if not symbols:
            return None
        
        # If imported_from is specified, prefer that file
        if reference.imported_from:
            for symbol in symbols:
                if symbol.file == reference.imported_from:
                    return self._find_node_for_symbol(symbol)
        
        # Otherwise, take the first exported symbol
        for symbol in symbols:
            if symbol.is_exported:
                return self._find_node_for_symbol(symbol)
        
        # Fall back to any symbol with this name
        if symbols:
            return self._find_node_for_symbol(symbols[0])
        
        return None
    
    def _find_node_for_symbol(self, symbol: Symbol) -> Optional[Node]:
        """
        Find the Node object corresponding to a Symbol.
        
        Args:
            symbol: Symbol object
            
        Returns:
            Node object, or None
        """
        # Find node at the symbol's location
        return self._find_node_at_location(symbol.file_path, symbol.line_number)
    
    def _create_placeholder_node(self, reference: SymbolReference) -> Optional[Node]:
        """
        Create a placeholder node for external/unresolved references.
        
        Args:
            reference: SymbolReference object
            
        Returns:
            Placeholder Node object, or None if symbol is malformed
        """
        from .models import NodeType
        
        symbol_name = reference.symbol_name
        
        # Validate symbol name - skip malformed identifiers
        if not symbol_name or not VALID_IDENTIFIER.match(symbol_name):
            # Skip malformed symbols like 'etFullYear(' or 'Fixed(2'
            return None
        
        # Check if it's a JS built-in
        is_builtin = symbol_name in JS_BUILTINS
        
        # Determine external type for better filtering
        external_type = 'builtin' if is_builtin else 'library'
        
        # Create a virtual node for the external reference
        node = Node(
            id=f"external:{symbol_name}",
            name=symbol_name,
            type=NodeType.MODULE,  # Default to MODULE for external refs
            file=reference.imported_from or "external",
            metadata={
                'external': True,
                'is_builtin': is_builtin,
                'external_type': external_type,
                'hide_by_default': is_builtin,  # UI hint: hide built-ins by default
                'referenced_as': symbol_name
            }
        )
        
        return node
    
    def _create_edge_context(
        self,
        edge_type: EdgeType,
        reference: SymbolReference,
        source_node: Node,
        target_node: Node
    ) -> EdgeContext:
        """
        Create EdgeContext with rich metadata.
        
        Args:
            edge_type: EdgeType enum value
            reference: SymbolReference object
            source_node: Source Node (where reference occurs)
            target_node: Target Node (symbol being referenced)
            
        Returns:
            EdgeContext object with full metadata
        """
        # Extract metadata from reference
        metadata = reference.metadata or {}
        
        # Determine symbol types
        source_symbol_type = self._infer_symbol_type(source_node)
        target_symbol_type = self._infer_symbol_type(target_node)
        
        # Build EdgeContext
        context = EdgeContext(
            edge_type=edge_type,
            source_file=source_node.file,
            target_file=target_node.file,
            source_symbol=source_node.name,
            target_symbol=target_node.name,
            source_symbol_type=source_symbol_type,
            target_symbol_type=target_symbol_type,
            source_line=reference.line_number,
            target_line=(target_node.metadata or {}).get('line'),
            context=reference.context,
            is_async=metadata.get('is_async', False),
            is_default=metadata.get('is_default', False),
            is_dynamic=metadata.get('is_dynamic', False),
            imported_names=metadata.get('imported_names', []),
            exported_names=metadata.get('exported_names', []),
            import_path=reference.imported_from,
            metadata=metadata
        )
        
        return context
    
    def _infer_symbol_type(self, node: Node) -> Optional[SymbolType]:
        """
        Infer SymbolType from Node type.
        
        Args:
            node: Node object
            
        Returns:
            SymbolType enum value, or None
        """
        from .models import NodeType
        
        # Map NodeType to SymbolType
        type_mapping = {
            NodeType.FUNCTION: SymbolType.FUNCTION,
            NodeType.CLASS: SymbolType.CLASS,
            NodeType.COMPONENT: SymbolType.CLASS,
            NodeType.MODULE: SymbolType.MODULE,
            NodeType.MODEL: SymbolType.CLASS,
            NodeType.SERVICE: SymbolType.CLASS,
            NodeType.CONTROLLER: SymbolType.CLASS,
        }
        
        return type_mapping.get(node.type)
