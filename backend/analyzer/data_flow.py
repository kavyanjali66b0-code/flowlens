"""
Data Flow Analysis Models

This module provides the core data structures for tracking how data flows through
a codebase, including variable assignments, function calls, returns, and transformations.

Classes:
    DataFlowNodeType: Enum for types of data flow nodes
    DataFlowType: Enum for types of data flow edges
    DataFlowNode: Represents data at a specific program point
    DataFlowEdge: Represents data transformation or movement
    DataFlowGraph: Tracks complete data flow through codebase
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Set


class DataFlowNodeType(Enum):
    """Types of data flow nodes."""
    PARAMETER = auto()      # Function parameter
    RETURN = auto()         # Return value
    ASSIGNMENT = auto()     # Variable assignment
    DECLARATION = auto()    # Variable declaration
    PROPERTY = auto()       # Object property
    ARGUMENT = auto()       # Function call argument
    IMPORT = auto()         # Imported value
    EXPORT = auto()         # Exported value
    LITERAL = auto()        # Literal value
    OPERATION = auto()      # Result of operation
    DESTRUCTURE = auto()    # Destructured value
    SPREAD = auto()         # Spread operation
    CONDITIONAL = auto()    # Conditional expression result


class DataFlowType(Enum):
    """Types of data flow edges."""
    DIRECT = auto()         # Direct assignment (a = b)
    TRANSFORM = auto()      # Transformation (a = f(b))
    CONDITIONAL = auto()    # Conditional flow (if/else)
    PARAMETER = auto()      # Function parameter passing
    RETURN = auto()         # Function return
    PROPERTY_ACCESS = auto() # Object property access
    PROPERTY_SET = auto()   # Object property mutation
    ARRAY_ACCESS = auto()   # Array element access
    DESTRUCTURE = auto()    # Destructuring assignment
    SPREAD = auto()         # Spread operation
    AWAIT = auto()          # Async await


@dataclass
class DataFlowNode:
    """
    Represents data at a specific program point.
    
    Attributes:
        id: Unique identifier for this node
        variable_name: Name of the variable/property
        source_location: (file_path, line_number) where node appears
        node_type: Type of data flow node
        inferred_type: Inferred type of the data (optional)
        scope: Scope identifier (function/class name, optional)
        metadata: Additional metadata
    """
    id: str
    variable_name: str
    source_location: Tuple[str, int]
    node_type: DataFlowNodeType
    inferred_type: Optional[str] = None
    scope: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Make node hashable for use in sets/dicts."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Compare nodes by ID."""
        if not isinstance(other, DataFlowNode):
            return False
        return self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'id': self.id,
            'variable_name': self.variable_name,
            'source_location': {
                'file': self.source_location[0],
                'line': self.source_location[1]
            },
            'node_type': self.node_type.name,
            'inferred_type': self.inferred_type,
            'scope': self.scope,
            'metadata': self.metadata
        }


@dataclass
class DataFlowEdge:
    """
    Represents data transformation or movement.
    
    Attributes:
        source: Source node ID
        target: Target node ID
        flow_type: Type of data flow
        operation: Name of operation/function (optional)
        metadata: Additional metadata
    """
    source: str
    target: str
    flow_type: DataFlowType
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Make edge hashable."""
        return hash((self.source, self.target, self.flow_type.name))
    
    def __eq__(self, other: object) -> bool:
        """Compare edges."""
        if not isinstance(other, DataFlowEdge):
            return False
        return (self.source == other.source and 
                self.target == other.target and
                self.flow_type == other.flow_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            'source': self.source,
            'target': self.target,
            'flow_type': self.flow_type.name,
            'operation': self.operation,
            'metadata': self.metadata
        }


class DataFlowGraph:
    """
    Tracks how data flows through the codebase.
    
    This graph maintains nodes representing data at various program points
    and edges representing how data moves and transforms between those points.
    """
    
    def __init__(self):
        """Initialize empty data flow graph."""
        self.nodes: Dict[str, DataFlowNode] = {}
        self.edges: List[DataFlowEdge] = []
        # Index for fast lookups
        self._outgoing_edges: Dict[str, List[DataFlowEdge]] = {}
        self._incoming_edges: Dict[str, List[DataFlowEdge]] = {}
    
    def add_node(self, node: DataFlowNode) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: DataFlowNode to add
        """
        self.nodes[node.id] = node
        if node.id not in self._outgoing_edges:
            self._outgoing_edges[node.id] = []
        if node.id not in self._incoming_edges:
            self._incoming_edges[node.id] = []
    
    def add_flow(self, source: DataFlowNode, target: DataFlowNode, 
                 flow_type: DataFlowType, operation: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> DataFlowEdge:
        """
        Add a data flow edge between two nodes.
        
        Args:
            source: Source node
            target: Target node
            flow_type: Type of data flow
            operation: Optional operation name
            metadata: Optional edge metadata
            
        Returns:
            The created DataFlowEdge
        """
        # Ensure nodes exist in graph
        if source.id not in self.nodes:
            self.add_node(source)
        if target.id not in self.nodes:
            self.add_node(target)
        
        # Create edge
        edge = DataFlowEdge(
            source=source.id,
            target=target.id,
            flow_type=flow_type,
            operation=operation,
            metadata=metadata or {}
        )
        
        # Add to edge list and indexes
        self.edges.append(edge)
        self._outgoing_edges[source.id].append(edge)
        self._incoming_edges[target.id].append(edge)
        
        return edge
    
    def get_node(self, node_id: str) -> Optional[DataFlowNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            DataFlowNode if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def trace_data_origin(self, node_id: str, 
                         visited: Optional[Set[str]] = None) -> List[DataFlowNode]:
        """
        Trace data back to its origins (sources with no incoming edges).
        
        Args:
            node_id: Starting node ID
            visited: Set of visited node IDs (for cycle detection)
            
        Returns:
            List of origin nodes
        """
        if visited is None:
            visited = set()
        
        # Prevent cycles
        if node_id in visited:
            return []
        visited.add(node_id)
        
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        # If no incoming edges, this is an origin
        incoming = self._incoming_edges.get(node_id, [])
        if not incoming:
            return [node]
        
        # Recursively trace origins
        origins = []
        for edge in incoming:
            origins.extend(self.trace_data_origin(edge.source, visited))
        
        return origins
    
    def find_data_consumers(self, node_id: str,
                           visited: Optional[Set[str]] = None) -> List[DataFlowNode]:
        """
        Find all nodes that consume data from this node.
        
        Args:
            node_id: Starting node ID
            visited: Set of visited node IDs (for cycle detection)
            
        Returns:
            List of consumer nodes
        """
        if visited is None:
            visited = set()
        
        # Prevent cycles
        if node_id in visited:
            return []
        visited.add(node_id)
        
        consumers = []
        
        # Get all outgoing edges
        outgoing = self._outgoing_edges.get(node_id, [])
        for edge in outgoing:
            target_node = self.nodes.get(edge.target)
            if target_node:
                consumers.append(target_node)
                # Recursively find consumers of consumers
                consumers.extend(self.find_data_consumers(edge.target, visited))
        
        return consumers
    
    def get_flow_chain(self, start_id: str, end_id: str,
                      max_depth: int = 100) -> List[List[DataFlowNode]]:
        """
        Find all flow paths from start node to end node.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_depth: Maximum path depth to prevent infinite recursion
            
        Returns:
            List of paths, where each path is a list of nodes
        """
        def _find_paths(current_id: str, target_id: str, 
                       current_path: List[str], depth: int) -> List[List[str]]:
            """Recursive helper to find all paths."""
            if depth > max_depth:
                return []
            
            if current_id == target_id:
                return [current_path + [current_id]]
            
            if current_id in current_path:  # Cycle detection
                return []
            
            paths = []
            outgoing = self._outgoing_edges.get(current_id, [])
            for edge in outgoing:
                sub_paths = _find_paths(
                    edge.target, 
                    target_id,
                    current_path + [current_id],
                    depth + 1
                )
                paths.extend(sub_paths)
            
            return paths
        
        # Find all paths (as node IDs)
        id_paths = _find_paths(start_id, end_id, [], 0)
        
        # Convert to node objects
        node_paths = []
        for id_path in id_paths:
            node_path = [self.nodes[node_id] for node_id in id_path 
                        if node_id in self.nodes]
            if node_path:
                node_paths.append(node_path)
        
        return node_paths
    
    def get_outgoing_edges(self, node_id: str) -> List[DataFlowEdge]:
        """Get all edges flowing out from a node."""
        return self._outgoing_edges.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List[DataFlowEdge]:
        """Get all edges flowing into a node."""
        return self._incoming_edges.get(node_id, [])
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the data flow graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {
                node_type.name: sum(1 for n in self.nodes.values() 
                                   if n.node_type == node_type)
                for node_type in DataFlowNodeType
            },
            'edge_types': {
                flow_type.name: sum(1 for e in self.edges 
                                   if e.flow_type == flow_type)
                for flow_type in DataFlowType
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire graph to dictionary representation.
        
        Returns:
            Dictionary with nodes and edges
        """
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges],
            'stats': self.get_stats()
        }
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self._outgoing_edges.clear()
        self._incoming_edges.clear()
