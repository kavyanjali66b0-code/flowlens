"""
Structured metadata classes for nodes and edges.

This module provides type-safe metadata structures to replace Dict[str, Any],
making the codebase more maintainable and enabling IDE autocomplete.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class C4Level(Enum):
    """C4 model architecture levels."""
    SYSTEM = "system"
    CONTAINER = "container"
    COMPONENT = "component"
    CODE = "code"


@dataclass
class TypeInfo:
    """Type information for a node."""
    name: str  # e.g., "string", "React.Component", "Promise<User>"
    category: str  # e.g., "primitive", "function", "class", "generic"
    confidence: float  # 0.0-1.0
    source: str  # e.g., "annotation", "inference", "literal"
    is_array: bool = False
    is_optional: bool = False
    generic_params: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HookInfo:
    """React hook information."""
    hook_name: str  # e.g., "useState", "useEffect"
    hook_type: str  # e.g., "state", "effect", "context", "custom"
    hook_category: str  # e.g., "core", "custom"
    dependencies: List[str] = field(default_factory=list)
    is_stable: bool = True
    effect_cleanup: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConfidenceScore:
    """Confidence score for analysis results."""
    overall_score: float  # 0.0-1.0
    confidence_level: str  # "very_high", "high", "medium", "low", "very_low"
    factors: Dict[str, float] = field(default_factory=dict)  # factor -> score
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LocationInfo:
    """Source code location information."""
    file: str
    line_start: int
    line_end: int
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NodeMetadata:
    """
    Structured metadata for nodes.
    
    This replaces the previous Dict[str, Any] with typed fields,
    providing better IDE support and validation.
    """
    # Core fields
    c4_level: C4Level
    is_entry: bool = False
    
    # Location information
    location: Optional[LocationInfo] = None
    
    # Type information (for functions, variables)
    type_info: Optional[TypeInfo] = None
    
    # React-specific information
    hook_info: Optional[HookInfo] = None
    is_react_component: bool = False
    jsx_components_used: List[str] = field(default_factory=list)
    
    # Analysis metadata
    confidence_score: Optional[ConfidenceScore] = None
    complexity_score: Optional[int] = None
    
    # Dependency information
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # Plugin-specific data (extensible)
    plugin_data: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Handles nested dataclasses and enums properly.
        """
        result = {}
        
        # C4 level
        result['c4_level'] = self.c4_level.value if isinstance(self.c4_level, C4Level) else self.c4_level
        result['is_entry'] = self.is_entry
        
        # Location
        if self.location:
            result['location'] = self.location.to_dict()
        
        # Type info
        if self.type_info:
            result['type_info'] = self.type_info.to_dict()
        
        # Hook info
        if self.hook_info:
            result['hook_info'] = self.hook_info.to_dict()
        
        # React info
        if self.is_react_component:
            result['is_react_component'] = True
        if self.jsx_components_used:
            result['jsx_components_used'] = self.jsx_components_used
        
        # Analysis metadata
        if self.confidence_score:
            result['confidence_score'] = self.confidence_score.to_dict()
        if self.complexity_score is not None:
            result['complexity_score'] = self.complexity_score
        
        # Dependencies
        if self.dependencies:
            result['dependencies'] = self.dependencies
        if self.dependents:
            result['dependents'] = self.dependents
        
        # Plugin data
        if self.plugin_data:
            result['plugin_data'] = self.plugin_data
        
        # Tags
        if self.tags:
            result['tags'] = self.tags
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeMetadata':
        """
        Create NodeMetadata from dictionary.
        
        Handles backward compatibility with old Dict[str, Any] format.
        """
        # Handle C4 level
        c4_level_value = data.get('c4_level', 'code')
        if isinstance(c4_level_value, str):
            try:
                c4_level = C4Level(c4_level_value)
            except ValueError:
                c4_level = C4Level.CODE
        else:
            c4_level = c4_level_value
        
        # Handle location
        location = None
        if 'location' in data:
            loc_data = data['location']
            location = LocationInfo(**loc_data) if isinstance(loc_data, dict) else None
        
        # Handle type info
        type_info = None
        if 'type_info' in data:
            ti_data = data['type_info']
            type_info = TypeInfo(**ti_data) if isinstance(ti_data, dict) else None
        
        # Handle hook info
        hook_info = None
        if 'hook_info' in data:
            hi_data = data['hook_info']
            hook_info = HookInfo(**hi_data) if isinstance(hi_data, dict) else None
        
        # Handle confidence score
        confidence_score = None
        if 'confidence_score' in data:
            cs_data = data['confidence_score']
            confidence_score = ConfidenceScore(**cs_data) if isinstance(cs_data, dict) else None
        
        return cls(
            c4_level=c4_level,
            is_entry=data.get('is_entry', False),
            location=location,
            type_info=type_info,
            hook_info=hook_info,
            is_react_component=data.get('is_react_component', False),
            jsx_components_used=data.get('jsx_components_used', []),
            confidence_score=confidence_score,
            complexity_score=data.get('complexity_score'),
            dependencies=data.get('dependencies', []),
            dependents=data.get('dependents', []),
            plugin_data=data.get('plugin_data', {}),
            tags=data.get('tags', [])
        )


@dataclass
class EdgeMetadata:
    """
    Structured metadata for edges.
    
    Provides context about the relationship between two nodes.
    """
    # Confidence
    confidence: float = 1.0  # 0.0-1.0
    
    # Context information
    context: Optional[str] = None  # "call", "import", "extends", etc.
    flow_type: Optional[str] = None  # "data", "control", "dependency"
    is_async: bool = False
    
    # Call-specific metadata
    call_args: List[str] = field(default_factory=list)
    call_result: Optional[str] = None
    
    # Import-specific metadata
    import_source: Optional[str] = None  # Module path
    is_default_import: bool = False
    imported_names: List[str] = field(default_factory=list)
    
    # Type-specific metadata
    type_relationship: Optional[str] = None  # "extends", "implements", "instance_of"
    
    # Analysis metadata
    detected_by: str = "parser"  # "parser", "semantic_analyzer", "ml_model"
    validation_status: str = "unvalidated"  # "validated", "suspicious", "unvalidated"
    
    # Plugin-specific data
    plugin_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {'confidence': self.confidence}
        
        if self.context:
            result['context'] = self.context
        if self.flow_type:
            result['flow_type'] = self.flow_type
        if self.is_async:
            result['is_async'] = True
        
        if self.call_args:
            result['call_args'] = self.call_args
        if self.call_result:
            result['call_result'] = self.call_result
        
        if self.import_source:
            result['import_source'] = self.import_source
        if self.is_default_import:
            result['is_default_import'] = True
        if self.imported_names:
            result['imported_names'] = self.imported_names
        
        if self.type_relationship:
            result['type_relationship'] = self.type_relationship
        
        result['detected_by'] = self.detected_by
        result['validation_status'] = self.validation_status
        
        if self.plugin_data:
            result['plugin_data'] = self.plugin_data
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeMetadata':
        """Create EdgeMetadata from dictionary."""
        return cls(
            confidence=data.get('confidence', 1.0),
            context=data.get('context'),
            flow_type=data.get('flow_type'),
            is_async=data.get('is_async', False),
            call_args=data.get('call_args', []),
            call_result=data.get('call_result'),
            import_source=data.get('import_source'),
            is_default_import=data.get('is_default_import', False),
            imported_names=data.get('imported_names', []),
            type_relationship=data.get('type_relationship'),
            detected_by=data.get('detected_by', 'parser'),
            validation_status=data.get('validation_status', 'unvalidated'),
            plugin_data=data.get('plugin_data', {})
        )

