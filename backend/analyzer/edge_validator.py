"""
Edge validation module for detecting incorrect or suspicious relationships.

This module helps catch semantic analysis bugs during development by validating
that edges make logical sense based on the codebase structure.
"""

import logging
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .models import Node, Edge, EdgeType, NodeType


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"  # Definitely wrong
    WARNING = "warning"  # Probably wrong
    INFO = "info"  # Might be worth checking


@dataclass
class ValidationIssue:
    """Represents a validation issue found in an edge."""
    edge_id: str
    severity: ValidationSeverity
    issue_type: str
    message: str
    edge: Edge
    source_node: Optional[Node] = None
    target_node: Optional[Node] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'edge_id': self.edge_id,
            'severity': self.severity.value,
            'issue_type': self.issue_type,
            'message': self.message,
            'edge': {
                'source': self.edge.source,
                'target': self.edge.target,
                'type': self.edge.type.value
            },
            'metadata': self.metadata
        }


class EdgeValidator:
    """
    Validates edges to catch incorrect relationships.
    
    This is primarily a development/debugging tool to ensure the semantic
    analysis layer is working correctly.
    """
    
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        """
        Initialize edge validator.
        
        Args:
            nodes: List of nodes in the graph
            edges: List of edges to validate
        """
        self.nodes = nodes
        self.edges = edges
        
        # Build lookup indexes
        self.node_by_id: Dict[str, Node] = {node.id: node for node in nodes}
        self.nodes_by_file: Dict[str, List[Node]] = {}
        self._build_file_index()
        
        # Validation results
        self.issues: List[ValidationIssue] = []
    
    def _build_file_index(self):
        """Build index of nodes by file path."""
        for node in self.nodes:
            if node.file not in self.nodes_by_file:
                self.nodes_by_file[node.file] = []
            self.nodes_by_file[node.file].append(node)
    
    def validate_all(self) -> List[ValidationIssue]:
        """
        Run all validation checks.
        
        Returns:
            List of validation issues found
        """
        logging.info(f"Starting edge validation for {len(self.edges)} edges...")
        
        self.issues = []
        
        # Run validation checks
        self._validate_node_existence()
        self._validate_cross_file_dependencies()
        self._validate_renders_relationships()
        self._validate_calls_relationships()
        self._validate_circular_dependencies()
        self._validate_orphaned_nodes()
        
        # Log summary
        errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
        info = [i for i in self.issues if i.severity == ValidationSeverity.INFO]
        
        logging.info(f"Edge validation complete: {len(errors)} errors, "
                    f"{len(warnings)} warnings, {len(info)} info")
        
        return self.issues
    
    def _validate_node_existence(self):
        """Validate that all referenced nodes exist."""
        for edge in self.edges:
            if edge.source not in self.node_by_id:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.ERROR,
                    issue_type="missing_source_node",
                    message=f"Edge references non-existent source node: {edge.source}",
                    edge=edge
                ))
            
            if edge.target not in self.node_by_id:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.ERROR,
                    issue_type="missing_target_node",
                    message=f"Edge references non-existent target node: {edge.target}",
                    edge=edge
                ))
    
    def _validate_cross_file_dependencies(self):
        """
        Validate that cross-file edges have corresponding imports.
        
        If file A imports from file B, there should be an import edge or
        the import should be recorded in file metadata.
        """
        for edge in self.edges:
            # Skip if nodes don't exist
            source_node = self.node_by_id.get(edge.source)
            target_node = self.node_by_id.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # Check if it's a cross-file edge
            if source_node.file != target_node.file:
                # For dependency-type edges, check if there's an import
                if edge.type in [EdgeType.CALLS, EdgeType.USES, EdgeType.RENDERS]:
                    # Check if there's an import edge
                    has_import = self._has_import_edge(source_node.file, target_node.file)
                    
                    if not has_import:
                        self.issues.append(ValidationIssue(
                            edge_id=f"{edge.source}->{edge.target}",
                            severity=ValidationSeverity.WARNING,
                            issue_type="missing_import",
                            message=f"Cross-file {edge.type.value} edge without import: "
                                   f"{source_node.file} -> {target_node.file}",
                            edge=edge,
                            source_node=source_node,
                            target_node=target_node,
                            metadata={
                                'source_file': source_node.file,
                                'target_file': target_node.file
                            }
                        ))
    
    def _has_import_edge(self, source_file: str, target_file: str) -> bool:
        """Check if there's an import edge from source_file to target_file."""
        source_nodes = self.nodes_by_file.get(source_file, [])
        target_nodes = self.nodes_by_file.get(target_file, [])
        
        # Check if any node in source_file imports any node in target_file
        for edge in self.edges:
            if edge.type in [EdgeType.IMPORTS, EdgeType.DEPENDS_ON]:
                source_node = self.node_by_id.get(edge.source)
                target_node = self.node_by_id.get(edge.target)
                
                if (source_node and target_node and
                    source_node.file == source_file and
                    target_node.file == target_file):
                    return True
        
        return False
    
    def _validate_renders_relationships(self):
        """
        Validate RENDERS edges.
        
        RENDERS should only be between components.
        Also checks for self-referencing RENDERS edges.
        """
        for edge in self.edges:
            if edge.type != EdgeType.RENDERS:
                continue
            
            source_node = self.node_by_id.get(edge.source)
            target_node = self.node_by_id.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # CRITICAL: Check for self-referencing RENDERS edges
            if edge.source == edge.target:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.ERROR,
                    issue_type="self_referencing_render",
                    message=f"Self-referencing RENDERS edge: {source_node.name} renders itself",
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node
                ))
                continue
            
            # Source should be a component
            if source_node.type != NodeType.COMPONENT:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.WARNING,
                    issue_type="invalid_renders_source",
                    message=f"RENDERS edge from non-component: {source_node.type.value}",
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node
                ))
            
            # Target should be a component
            if target_node.type != NodeType.COMPONENT:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.INFO,
                    issue_type="renders_non_component",
                    message=f"RENDERS edge to non-component: {target_node.type.value} "
                           f"(might be HTML tag or library component)",
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node
                ))
    
    def _validate_calls_relationships(self):
        """
        Validate CALLS edges.
        
        CALLS should be to functions, methods, or components.
        Also checks for self-referencing CALLS edges.
        """
        for edge in self.edges:
            if edge.type not in [EdgeType.CALLS, EdgeType.ASYNC_CALLS]:
                continue
            
            source_node = self.node_by_id.get(edge.source)
            target_node = self.node_by_id.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # CRITICAL: Check for self-referencing CALLS edges
            if edge.source == edge.target:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.ERROR,
                    issue_type="self_referencing_call",
                    message=f"Self-referencing CALLS edge: {source_node.name} calls itself",
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node
                ))
                continue
            
            # Target should be callable
            valid_target_types = {
                NodeType.FUNCTION,
                NodeType.COMPONENT,
                NodeType.CLASS,
                NodeType.API_ENDPOINT
            }
            
            if target_node.type not in valid_target_types:
                self.issues.append(ValidationIssue(
                    edge_id=f"{edge.source}->{edge.target}",
                    severity=ValidationSeverity.WARNING,
                    issue_type="calls_non_callable",
                    message=f"CALLS edge to non-callable: {target_node.type.value}",
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node
                ))
    
    def _validate_circular_dependencies(self):
        """
        Detect circular dependencies in import/dependency edges.
        """
        # Build dependency graph
        dep_graph: Dict[str, Set[str]] = {}
        
        for edge in self.edges:
            if edge.type in [EdgeType.IMPORTS, EdgeType.DEPENDS_ON]:
                if edge.source not in dep_graph:
                    dep_graph[edge.source] = set()
                dep_graph[edge.source].add(edge.target)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str, path: List[str]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for neighbor in dep_graph.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    self._report_circular_dependency(cycle)
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in dep_graph.keys():
            if node_id not in visited:
                has_cycle(node_id, [])
    
    def _report_circular_dependency(self, cycle: List[str]):
        """Report a circular dependency."""
        # Get node names for readability
        node_names = []
        for node_id in cycle:
            node = self.node_by_id.get(node_id)
            if node:
                node_names.append(f"{node.name} ({node.file})")
            else:
                node_names.append(node_id)
        
        self.issues.append(ValidationIssue(
            edge_id=f"cycle:{cycle[0]}",
            severity=ValidationSeverity.INFO,
            issue_type="circular_dependency",
            message=f"Circular dependency detected: {' -> '.join(node_names[:4])}...",
            edge=Edge(source=cycle[0], target=cycle[1], type=EdgeType.DEPENDS_ON),
            metadata={'cycle': cycle, 'cycle_length': len(cycle)}
        ))
    
    def _validate_orphaned_nodes(self):
        """
        Find nodes with no incoming or outgoing edges.
        
        These might be dead code or indicate missing edges.
        """
        # Build sets of nodes with edges
        nodes_with_outgoing = set()
        nodes_with_incoming = set()
        
        for edge in self.edges:
            nodes_with_outgoing.add(edge.source)
            nodes_with_incoming.add(edge.target)
        
        # Find orphaned nodes (no edges at all)
        for node in self.nodes:
            # Skip entry points and modules (they're supposed to have no incoming edges)
            if node.type in [NodeType.MODULE]:
                continue
            
            is_entry = False
            if isinstance(node.metadata, dict):
                is_entry = node.metadata.get('is_entry', False)
            
            if is_entry:
                continue
            
            has_outgoing = node.id in nodes_with_outgoing
            has_incoming = node.id in nodes_with_incoming
            
            if not has_outgoing and not has_incoming:
                self.issues.append(ValidationIssue(
                    edge_id=f"orphan:{node.id}",
                    severity=ValidationSeverity.INFO,
                    issue_type="orphaned_node",
                    message=f"Node has no edges: {node.name} ({node.type.value})",
                    edge=Edge(source=node.id, target=node.id, type=EdgeType.DEPENDS_ON),
                    source_node=node,
                    metadata={'node_id': node.id, 'node_name': node.name}
                ))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'total_edges': len(self.edges),
            'total_nodes': len(self.nodes),
            'total_issues': len(self.issues),
            'errors': len([i for i in self.issues if i.severity == ValidationSeverity.ERROR]),
            'warnings': len([i for i in self.issues if i.severity == ValidationSeverity.WARNING]),
            'info': len([i for i in self.issues if i.severity == ValidationSeverity.INFO]),
            'issues_by_type': self._group_issues_by_type()
        }
    
    def _group_issues_by_type(self) -> Dict[str, int]:
        """Group issues by type."""
        by_type = {}
        for issue in self.issues:
            by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1
        return by_type

