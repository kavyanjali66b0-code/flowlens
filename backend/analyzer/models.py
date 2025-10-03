"""
Data models for the codebase analyzer.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


class ProjectType(Enum):
    """Enumeration of supported project types."""
    MAVEN_JAVA = "maven_java"
    GRADLE_JAVA = "gradle_java"
    REACT_VITE = "react_vite"
    ANGULAR = "angular"
    DJANGO = "django"
    PYTHON_APP = "python_app"
    ANDROID = "android"
    SPRING_BOOT = "spring_boot"
    EXPRESS_NODE = "express_node"
    UNKNOWN = "unknown"


class NodeType(Enum):
    """Enumeration of node types in the workflow graph."""
    COMPONENT = "component"
    CLASS = "class"
    FUNCTION = "function"
    API_ENDPOINT = "api_endpoint"
    ROUTE = "route"
    MODULE = "module"
    SERVICE = "service"
    CONTROLLER = "controller"
    MODEL = "model"
    VIEW = "view"
    TEMPLATE = "template"


class EdgeType(Enum):
    """Enumeration of edge types in the workflow graph."""
    # Legacy types (keep for backward compatibility)
    IMPORTS = "imports"
    RENDERS = "renders"
    CALLS = "calls"
    INHERITS = "inherits"
    ROUTES_TO = "routes_to"
    CALLS_API = "calls_api"
    DEPENDS_ON = "depends_on"
    USES = "uses"
    
    # Module relationships
    EXPORTS = "exports"              # A exports symbol
    RE_EXPORTS = "re_exports"        # A re-exports from B
    
    # Function relationships
    ASYNC_CALLS = "async_calls"      # A awaits async function B
    INVOKES = "invokes"              # A invokes callback B
    RETURNS = "returns"              # A returns type/value B
    
    # Class relationships
    EXTENDS = "extends"              # Class A extends Class B
    IMPLEMENTS = "implements"        # Class A implements Interface B
    INSTANTIATES = "instantiates"    # A creates instance of B
    USES_METHOD = "uses_method"      # A calls method of B
    
    # Type relationships (TypeScript/Flow)
    HAS_TYPE = "has_type"            # Variable A has type B
    TYPE_ALIAS = "type_alias"        # Type A is alias of B
    GENERIC_PARAM = "generic_param"  # Type A has generic param B
    
    # Composition relationships
    HAS_PROPERTY = "has_property"    # Class A has property B
    HAS_METHOD = "has_method"        # Class A has method B
    CONTAINS = "contains"            # Module A contains B
    
    # Reference relationships
    REFERENCES = "references"        # A references B (generic)
    DECORATES = "decorates"          # Decorator A decorates B
    ANNOTATES = "annotates"          # Annotation A annotates B


@dataclass
class ConfigFile:
    """Represents a configuration file found in the project."""
    path: str
    type: str
    content: Optional[Dict] = None


@dataclass
class Node:
    """Represents a node in the workflow graph."""
    id: str
    type: NodeType
    file: str
    name: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "file": self.file,
            "name": self.name,
            "metadata": self.metadata or {}
        }


@dataclass
class Edge:
    """Represents an edge in the workflow graph."""
    source: str
    target: str
    type: EdgeType
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert edge to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "metadata": self.metadata or {}
        }
