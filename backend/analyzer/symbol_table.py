"""
Symbol Table for tracking symbols and references across files.

This module provides a comprehensive symbol tracking system for:
- Functions, classes, variables, constants
- Cross-file symbol resolution
- Import-aware reference resolution
- Scope-aware symbol lookup
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path


class SymbolType(Enum):
    """Type of symbol being tracked."""
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    MODULE = "module"


@dataclass
class Symbol:
    """
    Represents a symbol definition (function, class, variable, etc.).
    
    Attributes:
        name: Symbol name (e.g., 'MyClass', 'myFunction')
        symbol_type: Type of symbol (FUNCTION, CLASS, etc.)
        file_path: Absolute path to file containing the symbol
        line_number: Line number where symbol is defined
        scope: Scope of the symbol (e.g., 'global', 'class:MyClass', 'function:myFunc')
        is_exported: Whether the symbol is exported (ES6 export, module.exports)
        is_default_export: Whether it's a default export
        metadata: Additional metadata (params, return type, parent class, etc.)
    """
    name: str
    symbol_type: SymbolType
    file_path: str
    line_number: int
    scope: str = "global"
    is_exported: bool = False
    is_default_export: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make Symbol hashable for use in sets."""
        return hash((self.name, self.file_path, self.line_number, self.scope))
    
    def __eq__(self, other):
        """Symbols are equal if they have same name, file, line, and scope."""
        if not isinstance(other, Symbol):
            return False
        return (
            self.name == other.name and
            self.file_path == other.file_path and
            self.line_number == other.line_number and
            self.scope == other.scope
        )


@dataclass
class SymbolReference:
    """
    Represents a reference to a symbol (function call, class instantiation, etc.).
    
    Attributes:
        symbol_name: Name of the referenced symbol
        file_path: File containing the reference
        line_number: Line number of the reference
        context: How the symbol is used ('call', 'import', 'extends', 'instantiate', etc.)
        imported_from: Absolute file path if symbol was imported
        metadata: Additional context (arguments, type annotations, etc.)
    """
    symbol_name: str
    file_path: str
    line_number: int
    context: str  # "call", "import", "extends", "instantiate", "assignment", "access"
    imported_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make SymbolReference hashable."""
        return hash((self.symbol_name, self.file_path, self.line_number, self.context))


class SymbolTable:
    """
    Comprehensive symbol tracking system.
    
    Tracks:
    - Symbol definitions (functions, classes, variables)
    - Symbol references (calls, imports, extends)
    - Import mappings (local name -> absolute file path)
    - Cross-file symbol resolution
    """
    
    def __init__(self):
        """Initialize empty symbol table."""
        # Symbol storage
        self.symbols: Dict[str, List[Symbol]] = {}  # name -> [Symbol, ...]
        self.symbols_by_file: Dict[str, List[Symbol]] = {}  # file -> [Symbol, ...]
        
        # Reference tracking
        self.references: List[SymbolReference] = []
        
        # Import mapping: file -> {local_name: absolute_file_path}
        # e.g., {'/app.js': {'Button': '/components/Button.jsx'}}
        self.import_map: Dict[str, Dict[str, str]] = {}
        
        # Export tracking: file -> {symbol_name: is_default}
        self.exports_by_file: Dict[str, Dict[str, bool]] = {}
    
    def add_symbol(self, symbol: Symbol) -> None:
        """
        Add a symbol to the table.
        
        Args:
            symbol: Symbol to add
        """
        # Add to name-based index
        if symbol.name not in self.symbols:
            self.symbols[symbol.name] = []
        
        # Avoid duplicates
        if symbol not in self.symbols[symbol.name]:
            self.symbols[symbol.name].append(symbol)
        
        # Add to file-based index
        if symbol.file_path not in self.symbols_by_file:
            self.symbols_by_file[symbol.file_path] = []
        
        if symbol not in self.symbols_by_file[symbol.file_path]:
            self.symbols_by_file[symbol.file_path].append(symbol)
        
        # Track exports
        if symbol.is_exported:
            if symbol.file_path not in self.exports_by_file:
                self.exports_by_file[symbol.file_path] = {}
            self.exports_by_file[symbol.file_path][symbol.name] = symbol.is_default_export
    
    def add_reference(self, reference: SymbolReference) -> None:
        """
        Add a symbol reference to the table.
        
        Args:
            reference: Symbol reference to add
        """
        self.references.append(reference)
    
    def get_symbols_by_name(self, name: str) -> List[Symbol]:
        """
        Get all symbols with a given name.
        
        Args:
            name: Symbol name to search for
            
        Returns:
            List of symbols with that name (may be from different files)
        """
        return self.symbols.get(name, [])
    
    def get_symbols_by_file(self, file_path: str) -> List[Symbol]:
        """
        Get all symbols defined in a specific file.
        
        Args:
            file_path: Absolute path to file
            
        Returns:
            List of symbols defined in that file
        """
        return self.symbols_by_file.get(file_path, [])
    
    def get_exported_symbols(self, file_path: str) -> List[Symbol]:
        """
        Get only exported symbols from a file.
        
        Args:
            file_path: Absolute path to file
            
        Returns:
            List of exported symbols from that file
        """
        all_symbols = self.get_symbols_by_file(file_path)
        return [s for s in all_symbols if s.is_exported]
    
    def build_import_map(
        self, 
        file_path: str, 
        imports: List[Any],
        resolver: Optional[Any] = None
    ) -> None:
        """
        Build import mapping for a file.
        
        Maps local symbol names to the absolute file paths they're imported from.
        
        Args:
            file_path: File containing the imports
            imports: List of ImportStatement or RequireStatement objects
            resolver: ES6ImportResolver or CommonJSResolver instance
        """
        if file_path not in self.import_map:
            self.import_map[file_path] = {}
        
        for imp in imports:
            # Resolve import to absolute file path
            if resolver and hasattr(imp, 'module_path'):
                if hasattr(resolver, 'resolve_import'):
                    # ES6ImportResolver
                    resolved_path = resolver.resolve_import(imp.module_path, file_path)
                elif hasattr(resolver, 'resolve_require'):
                    # CommonJSResolver
                    resolved_path = resolver.resolve_require(imp.module_path, file_path)
                else:
                    resolved_path = None
                
                if resolved_path:
                    # Handle different import types
                    if hasattr(imp, 'imported_names') and imp.imported_names:
                        # Named imports: import { A, B } from 'mod'
                        for name in imp.imported_names:
                            self.import_map[file_path][name] = resolved_path
                    
                    if hasattr(imp, 'default_import') and imp.default_import:
                        # Default import: import X from 'mod'
                        self.import_map[file_path][imp.default_import] = resolved_path
                    
                    if hasattr(imp, 'namespace_import') and imp.namespace_import:
                        # Namespace import: import * as X from 'mod'
                        self.import_map[file_path][imp.namespace_import] = resolved_path
                    
                    if hasattr(imp, 'assigned_to') and imp.assigned_to:
                        # CommonJS: const X = require('mod')
                        self.import_map[file_path][imp.assigned_to] = resolved_path
                    
                    if hasattr(imp, 'destructured_names') and imp.destructured_names:
                        # CommonJS: const { A, B } = require('mod')
                        for name in imp.destructured_names:
                            self.import_map[file_path][name] = resolved_path
    
    def resolve_reference(self, reference: SymbolReference) -> Optional[Symbol]:
        """
        Resolve a symbol reference to its definition.
        
        Resolution strategy:
        1. If reference has imported_from, look in that file
        2. Check import map to see if symbol was imported
        3. Look in same file as reference
        4. Return first match or None
        
        Args:
            reference: Symbol reference to resolve
            
        Returns:
            Resolved Symbol or None if not found
        """
        # Strategy 1: Use explicit imported_from if available
        if reference.imported_from:
            symbols_in_file = self.get_symbols_by_file(reference.imported_from)
            for symbol in symbols_in_file:
                if symbol.name == reference.symbol_name and symbol.is_exported:
                    return symbol
        
        # Strategy 2: Check import map
        if reference.file_path in self.import_map:
            import_mapping = self.import_map[reference.file_path]
            if reference.symbol_name in import_mapping:
                imported_file = import_mapping[reference.symbol_name]
                symbols_in_file = self.get_symbols_by_file(imported_file)
                for symbol in symbols_in_file:
                    if symbol.name == reference.symbol_name and symbol.is_exported:
                        return symbol
        
        # Strategy 3: Look in same file
        symbols_in_file = self.get_symbols_by_file(reference.file_path)
        for symbol in symbols_in_file:
            if symbol.name == reference.symbol_name:
                return symbol
        
        # Strategy 4: Search all symbols (as fallback)
        matching_symbols = self.get_symbols_by_name(reference.symbol_name)
        if matching_symbols:
            # Prefer exported symbols from other files
            exported = [s for s in matching_symbols if s.is_exported]
            if exported:
                return exported[0]
            # Otherwise return first match
            return matching_symbols[0]
        
        return None
    
    def get_references_to_symbol(self, symbol: Symbol) -> List[SymbolReference]:
        """
        Get all references to a specific symbol.
        
        Args:
            symbol: Symbol to find references for
            
        Returns:
            List of references to that symbol
        """
        matching_refs = []
        for ref in self.references:
            resolved = self.resolve_reference(ref)
            if resolved and resolved == symbol:
                matching_refs.append(ref)
        return matching_refs
    
    def clear(self) -> None:
        """Clear all data from the symbol table."""
        self.symbols.clear()
        self.symbols_by_file.clear()
        self.references.clear()
        self.import_map.clear()
        self.exports_by_file.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the symbol table.
        
        Returns:
            Dict with counts of symbols, references, files, etc.
        """
        total_symbols = sum(len(syms) for syms in self.symbols.values())
        exported_count = sum(
            len([s for s in syms if s.is_exported]) 
            for syms in self.symbols_by_file.values()
        )
        
        return {
            "total_symbols": total_symbols,
            "unique_names": len(self.symbols),
            "files_with_symbols": len(self.symbols_by_file),
            "total_references": len(self.references),
            "exported_symbols": exported_count,
            "files_with_imports": len(self.import_map),
        }
