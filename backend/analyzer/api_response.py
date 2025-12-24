"""
API Response structures for the analyzer.

This module defines the structured JSON format returned by the analyzer API.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum


class InsightType(Enum):
    """Types of insights generated from analysis."""
    CIRCULAR_DEPENDENCY = "circular_dependency"
    UNUSED_CODE = "unused_code"
    LARGE_FILE = "large_file"
    HIGH_COMPLEXITY = "high_complexity"
    MISSING_TESTS = "missing_tests"
    SECURITY_CONCERN = "security_concern"
    PERFORMANCE_ISSUE = "performance_issue"
    ARCHITECTURE_SMELL = "architecture_smell"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


@dataclass
class ParseError:
    """Represents an error that occurred during parsing."""
    file: str
    error_type: str  # "encoding", "syntax", "unsupported", "timeout"
    message: str
    severity: str  # "error", "warning"
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'file': self.file,
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity
        }
        if self.line_number is not None:
            result['line_number'] = self.line_number
        return result


@dataclass
class Insight:
    """Represents an analysis insight or suggestion."""
    type: InsightType
    severity: InsightSeverity
    title: str
    description: str
    affected_nodes: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'affected_nodes': self.affected_nodes,
            'affected_files': self.affected_files,
            'metadata': self.metadata
        }


@dataclass
class ProjectInfo:
    """Project-level metadata."""
    name: str
    path: str
    type: str  # "react_vite", "django", "spring_boot", etc.
    language: str
    framework: str
    config_files: List[Dict[str, str]] = field(default_factory=list)
    entry_points: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'path': self.path,
            'type': self.type,
            'language': self.language,
            'framework': self.framework,
            'config_files': self.config_files,
            'entry_points': self.entry_points
        }


@dataclass
class GraphLevel:
    """Graph data for a specific C4 level."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'metadata': self.metadata
        }


@dataclass
class Graph:
    """Complete graph structure with optional C4 levels."""
    # All nodes and edges (flat structure for backward compatibility)
    all: GraphLevel = field(default_factory=GraphLevel)
    
    # Hierarchical C4 levels (optional, for advanced visualization)
    levels: Optional[Dict[str, GraphLevel]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'nodes': self.all.nodes,
            'edges': self.all.edges,
            'metadata': self.all.metadata
        }
        
        # Add hierarchical levels if present
        if self.levels:
            result['levels'] = {
                level: graph_level.to_dict()
                for level, graph_level in self.levels.items()
            }
        
        return result


@dataclass
class Statistics:
    """Analysis statistics."""
    total_files: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_components: int = 0
    
    # Breakdown by type
    node_types: Dict[str, int] = field(default_factory=dict)
    edge_types: Dict[str, int] = field(default_factory=dict)
    
    # Complexity metrics (optional)
    avg_complexity: Optional[float] = None
    max_complexity: Optional[int] = None
    
    # Coverage metrics (optional)
    test_coverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'total_files': self.total_files,
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'total_functions': self.total_functions,
            'total_classes': self.total_classes,
            'total_components': self.total_components,
            'node_types': self.node_types,
            'edge_types': self.edge_types
        }
        
        if self.avg_complexity is not None:
            result['avg_complexity'] = self.avg_complexity
        if self.max_complexity is not None:
            result['max_complexity'] = self.max_complexity
        if self.test_coverage is not None:
            result['test_coverage'] = self.test_coverage
        
        return result


@dataclass
class AnalysisResponse:
    """
    Complete analysis response structure.
    
    This is the top-level structure returned by the analyzer API.
    """
    # Project metadata
    project_info: ProjectInfo
    
    # Graph structure
    graph: Graph
    
    # Analysis insights (warnings, suggestions, issues)
    insights: List[Insight] = field(default_factory=list)
    
    # Parse errors encountered during analysis
    errors: List[ParseError] = field(default_factory=list)
    
    # Statistics
    statistics: Statistics = field(default_factory=Statistics)
    
    # Analysis metadata
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This is the final format sent to the frontend.
        """
        return {
            'project_info': self.project_info.to_dict(),
            'graph': self.graph.to_dict(),
            'insights': [insight.to_dict() for insight in self.insights],
            'errors': [error.to_dict() for error in self.errors],
            'statistics': self.statistics.to_dict(),
            'analysis_metadata': self.analysis_metadata
        }
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Alias for to_dict() for clarity."""
        return self.to_dict()


class ResponseBuilder:
    """
    Helper class to build AnalysisResponse from analyzer results.
    
    Usage:
        builder = ResponseBuilder()
        builder.set_project_info(name, path, type, ...)
        builder.add_nodes(nodes)
        builder.add_edges(edges)
        builder.add_insight(insight)
        response = builder.build()
    """
    
    def __init__(self):
        """Initialize empty response builder."""
        self.project_info: Optional[ProjectInfo] = None
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.insights: List[Insight] = []
        self.errors: List[ParseError] = []
        self.analysis_metadata: Dict[str, Any] = {}
    
    def set_project_info(
        self,
        name: str,
        path: str,
        project_type: str,
        language: str,
        framework: str,
        config_files: List[Dict[str, str]] = None,
        entry_points: List[Dict[str, str]] = None
    ) -> 'ResponseBuilder':
        """Set project information."""
        self.project_info = ProjectInfo(
            name=name,
            path=path,
            type=project_type,
            language=language,
            framework=framework,
            config_files=config_files or [],
            entry_points=entry_points or []
        )
        return self
    
    def add_nodes(self, nodes: List[Any]) -> 'ResponseBuilder':
        """Add nodes (Node objects or dicts)."""
        for node in nodes:
            if hasattr(node, 'to_dict'):
                self.nodes.append(node.to_dict())
            else:
                self.nodes.append(node)
        return self
    
    def add_edges(self, edges: List[Any]) -> 'ResponseBuilder':
        """Add edges (Edge objects or dicts)."""
        for edge in edges:
            if hasattr(edge, 'to_dict'):
                self.edges.append(edge.to_dict())
            else:
                self.edges.append(edge)
        return self
    
    def add_insight(self, insight: Insight) -> 'ResponseBuilder':
        """Add an analysis insight."""
        self.insights.append(insight)
        return self
    
    def add_error(self, error: ParseError) -> 'ResponseBuilder':
        """Add a parse error."""
        self.errors.append(error)
        return self
    
    def set_metadata(self, key: str, value: Any) -> 'ResponseBuilder':
        """Set analysis metadata."""
        self.analysis_metadata[key] = value
        return self
    
    def build(self) -> AnalysisResponse:
        """Build the final AnalysisResponse."""
        if not self.project_info:
            raise ValueError("Project info must be set before building response")
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Build graph structure
        graph = Graph(
            all=GraphLevel(
                nodes=self.nodes,
                edges=self.edges,
                metadata={
                    'total_nodes': len(self.nodes),
                    'total_edges': len(self.edges)
                }
            )
        )
        
        # Optionally organize by C4 levels
        levels = self._organize_by_c4_levels()
        if levels:
            graph.levels = levels
        
        return AnalysisResponse(
            project_info=self.project_info,
            graph=graph,
            insights=self.insights,
            errors=self.errors,
            statistics=stats,
            analysis_metadata=self.analysis_metadata
        )
    
    def _calculate_statistics(self) -> Statistics:
        """Calculate statistics from nodes and edges."""
        stats = Statistics(
            total_files=len(set(node.get('file', '') for node in self.nodes)),
            total_nodes=len(self.nodes),
            total_edges=len(self.edges)
        )
        
        # Count by node type
        for node in self.nodes:
            node_type = node.get('type', 'unknown')
            stats.node_types[node_type] = stats.node_types.get(node_type, 0) + 1
            
            # Count specific types
            if node_type == 'function':
                stats.total_functions += 1
            elif node_type == 'class':
                stats.total_classes += 1
            elif node_type == 'component':
                stats.total_components += 1
        
        # Count by edge type
        for edge in self.edges:
            edge_type = edge.get('type', 'unknown')
            stats.edge_types[edge_type] = stats.edge_types.get(edge_type, 0) + 1
        
        return stats
    
    def _organize_by_c4_levels(self) -> Optional[Dict[str, GraphLevel]]:
        """Organize nodes/edges into C4 hierarchy."""
        # Group nodes by C4 level
        levels_data: Dict[str, Dict[str, List]] = {
            'system': {'nodes': [], 'edges': []},
            'container': {'nodes': [], 'edges': []},
            'component': {'nodes': [], 'edges': []},
            'code': {'nodes': [], 'edges': []}
        }
        
        # Organize nodes
        node_ids_by_level: Dict[str, set] = {
            'system': set(),
            'container': set(),
            'component': set(),
            'code': set()
        }
        
        for node in self.nodes:
            c4_level = node.get('metadata', {}).get('c4_level', 'code')
            if c4_level in levels_data:
                levels_data[c4_level]['nodes'].append(node)
                node_ids_by_level[c4_level].add(node['id'])
        
        # Organize edges (edge belongs to a level if both source and target are in that level)
        for edge in self.edges:
            source_id = edge['source']
            target_id = edge['target']
            
            for level in ['system', 'container', 'component', 'code']:
                if source_id in node_ids_by_level[level] and target_id in node_ids_by_level[level]:
                    levels_data[level]['edges'].append(edge)
                    break
        
        # Convert to GraphLevel objects
        levels = {}
        for level, data in levels_data.items():
            if data['nodes']:  # Only include levels with nodes
                levels[level] = GraphLevel(
                    nodes=data['nodes'],
                    edges=data['edges'],
                    metadata={
                        'level': level,
                        'total_nodes': len(data['nodes']),
                        'total_edges': len(data['edges'])
                    }
                )
        
        return levels if levels else None

