"""
Edge context dataclass for rich edge metadata.

Provides structured metadata about relationships between code entities,
including symbol information, location data, and context-specific details.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .models import EdgeType
from .symbol_table import SymbolType


@dataclass
class EdgeContext:
    """
    Rich metadata about an edge relationship between code entities.
    
    This class captures detailed information about relationships discovered
    during code analysis, including symbol types, locations, and context.
    """
    
    # Core information
    edge_type: EdgeType
    source_file: str
    target_file: str
    
    # Symbol information
    source_symbol: Optional[str] = None
    target_symbol: Optional[str] = None
    source_symbol_type: Optional[SymbolType] = None
    target_symbol_type: Optional[SymbolType] = None
    
    # Location information
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    
    # Context information
    context: Optional[str] = None  # "call", "import", "extends", etc.
    is_async: bool = False
    is_default: bool = False
    is_dynamic: bool = False  # Dynamic import/require
    
    # Import/Export specific
    imported_names: List[str] = field(default_factory=list)
    exported_names: List[str] = field(default_factory=list)
    import_path: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict with all edge context data, suitable for JSON serialization.
        """
        return {
            "edge_type": self.edge_type.value,
            "source_file": self.source_file,
            "target_file": self.target_file,
            "source_symbol": self.source_symbol,
            "target_symbol": self.target_symbol,
            "source_symbol_type": self.source_symbol_type.value if self.source_symbol_type else None,
            "target_symbol_type": self.target_symbol_type.value if self.target_symbol_type else None,
            "source_line": self.source_line,
            "target_line": self.target_line,
            "context": self.context,
            "is_async": self.is_async,
            "is_default": self.is_default,
            "is_dynamic": self.is_dynamic,
            "imported_names": self.imported_names,
            "exported_names": self.exported_names,
            "import_path": self.import_path,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeContext':
        """
        Create EdgeContext from dictionary.
        
        Args:
            data: Dictionary with edge context data
            
        Returns:
            EdgeContext instance
        """
        # Convert string enum values back to enums
        data = data.copy()  # Don't mutate input
        data['edge_type'] = EdgeType(data['edge_type'])
        if data.get('source_symbol_type'):
            data['source_symbol_type'] = SymbolType(data['source_symbol_type'])
        if data.get('target_symbol_type'):
            data['target_symbol_type'] = SymbolType(data['target_symbol_type'])
        
        return cls(**data)
    
    def is_module_relationship(self) -> bool:
        """Check if this is a module-level relationship (imports/exports)."""
        return self.edge_type in [EdgeType.IMPORTS, EdgeType.EXPORTS, EdgeType.RE_EXPORTS]
    
    def is_function_relationship(self) -> bool:
        """Check if this is a function-related relationship."""
        return self.edge_type in [
            EdgeType.CALLS, EdgeType.ASYNC_CALLS, EdgeType.INVOKES, EdgeType.RETURNS
        ]
    
    def is_class_relationship(self) -> bool:
        """Check if this is a class-related relationship."""
        return self.edge_type in [
            EdgeType.EXTENDS, EdgeType.IMPLEMENTS, EdgeType.INSTANTIATES, 
            EdgeType.USES_METHOD, EdgeType.INHERITS
        ]
    
    def is_type_relationship(self) -> bool:
        """Check if this is a type-related relationship."""
        return self.edge_type in [
            EdgeType.HAS_TYPE, EdgeType.TYPE_ALIAS, EdgeType.GENERIC_PARAM
        ]
    
    def __str__(self) -> str:
        """String representation for debugging."""
        symbol_info = ""
        if self.source_symbol or self.target_symbol:
            symbol_info = f" ({self.source_symbol or '?'} → {self.target_symbol or '?'})"
        
        return (
            f"EdgeContext({self.edge_type.value}: "
            f"{self.source_file} → {self.target_file}{symbol_info})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return self.__str__()
