"""
Dependency Graph

Builds and analyzes dependency graphs using NetworkX.
Provides operations like circular dependency detection, depth calculation,
and dependency analysis.

Classes:
    DependencyGraph: Manages service/component dependencies
    CircularDependency: Represents a circular dependency chain
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, Tuple
import networkx as nx
from analyzer.di_resolver import Dependency, DIType


@dataclass
class CircularDependency:
    """Represents a circular dependency chain."""
    cycle: List[str]  # List of components in the cycle
    severity: str  # 'critical', 'warning', 'info'
    
    def __str__(self) -> str:
        return " → ".join(self.cycle + [self.cycle[0]])


class DependencyGraph:
    """
    Manages service/component dependencies using NetworkX.
    
    Provides analysis operations including circular dependency detection,
    dependency depth calculation, and unused dependency identification.
    """
    
    def __init__(self, dependencies: Dict[str, List[Dependency]]):
        """
        Initialize dependency graph.
        
        Args:
            dependencies: Dictionary mapping components to their dependencies
        """
        self.dependencies = dependencies
        self.graph = nx.DiGraph()
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build NetworkX graph from dependencies."""
        # Add all components as nodes
        all_components = set(self.dependencies.keys())
        
        # Add providers as nodes too
        for deps in self.dependencies.values():
            for dep in deps:
                all_components.add(dep.provider)
        
        for component in all_components:
            self.graph.add_node(component)
        
        # Add edges for each dependency
        for consumer, deps in self.dependencies.items():
            for dep in deps:
                # Add edge from consumer to provider
                # Store dependency info as edge attribute
                self.graph.add_edge(
                    consumer,
                    dep.provider,
                    injection_type=dep.injection_type.name,
                    location=dep.location,
                    metadata=dep.metadata
                )
    
    def find_circular_dependencies(self) -> List[CircularDependency]:
        """
        Detect circular dependency chains.
        
        Returns:
            List of CircularDependency objects representing cycles
        """
        try:
            cycles = list(nx.simple_cycles(self.graph))
        except:
            cycles = []
        
        circular_deps = []
        for cycle in cycles:
            # Determine severity based on cycle length
            if len(cycle) == 2:
                severity = 'critical'  # Direct circular dependency
            elif len(cycle) <= 4:
                severity = 'warning'  # Short cycle
            else:
                severity = 'info'  # Longer cycle
            
            circular_deps.append(CircularDependency(
                cycle=cycle,
                severity=severity
            ))
        
        return circular_deps
    
    def get_dependency_depth(self, component: str) -> int:
        """
        Calculate maximum depth of dependency tree for a component.
        
        The depth is the longest path from the component to any leaf node
        (component with no dependencies).
        
        Args:
            component: Component name
            
        Returns:
            Maximum depth, or 0 if component not found
        """
        if component not in self.graph:
            return 0
        
        # Find all paths from component to nodes with no outgoing edges
        max_depth = 0
        
        try:
            # Get all descendants (reachable nodes)
            descendants = nx.descendants(self.graph, component)
            
            if not descendants:
                return 0
            
            # Find leaf nodes (no outgoing edges)
            leaf_nodes = [n for n in descendants 
                         if self.graph.out_degree(n) == 0]
            
            if not leaf_nodes:
                # If no leaf nodes in descendants, use arbitrary depth
                return len(descendants)
            
            # Calculate depth to each leaf
            for leaf in leaf_nodes:
                try:
                    paths = list(nx.all_simple_paths(self.graph, component, leaf))
                    for path in paths:
                        depth = len(path) - 1  # Path length is nodes, depth is edges
                        max_depth = max(max_depth, depth)
                except nx.NetworkXNoPath:
                    continue
        except:
            return 0
        
        return max_depth
    
    def find_unused_dependencies(self) -> List[str]:
        """
        Find components that are declared but never used.
        
        Returns:
            List of unused component names
        """
        unused = []
        
        for node in self.graph.nodes():
            # A component is unused if:
            # 1. It has no incoming edges (nothing depends on it)
            # 2. It's not in the root dependencies (not a top-level component)
            if self.graph.in_degree(node) == 0 and node not in self.dependencies:
                unused.append(node)
        
        return unused
    
    def get_dependents(self, component: str) -> List[str]:
        """
        Get all components that depend on the given component.
        
        Args:
            component: Component name
            
        Returns:
            List of dependent component names
        """
        if component not in self.graph:
            return []
        
        # Get all predecessors (nodes that point to this component)
        return list(self.graph.predecessors(component))
    
    def get_dependencies_of(self, component: str) -> List[str]:
        """
        Get all direct dependencies of a component.
        
        Args:
            component: Component name
            
        Returns:
            List of dependency names
        """
        if component not in self.graph:
            return []
        
        # Get all successors (nodes this component points to)
        return list(self.graph.successors(component))
    
    def get_all_transitive_dependencies(self, component: str) -> Set[str]:
        """
        Get all transitive dependencies (dependencies of dependencies).
        
        Args:
            component: Component name
            
        Returns:
            Set of all transitive dependency names
        """
        if component not in self.graph:
            return set()
        
        try:
            # Get all descendants (reachable nodes)
            return nx.descendants(self.graph, component)
        except:
            return set()
    
    def find_missing_dependencies(self, available_components: Set[str]) -> List[Tuple[str, str]]:
        """
        Find dependencies that are referenced but not available.
        
        Args:
            available_components: Set of components that exist/are available
            
        Returns:
            List of (consumer, missing_provider) tuples
        """
        missing = []
        
        for consumer, deps in self.dependencies.items():
            for dep in deps:
                if dep.provider not in available_components:
                    missing.append((consumer, dep.provider))
        
        return missing
    
    def suggest_injection_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements to dependency injection patterns.
        
        Returns:
            List of suggestions with details
        """
        suggestions = []
        
        # Suggest breaking circular dependencies
        circular = self.find_circular_dependencies()
        for circ in circular:
            suggestions.append({
                'type': 'circular_dependency',
                'severity': circ.severity,
                'message': f"Circular dependency detected: {circ}",
                'components': circ.cycle,
                'suggestion': 'Consider using dependency inversion or breaking the cycle'
            })
        
        # Suggest removing unused dependencies
        unused = self.find_unused_dependencies()
        for component in unused:
            suggestions.append({
                'type': 'unused_dependency',
                'severity': 'info',
                'message': f"Component '{component}' is never used",
                'component': component,
                'suggestion': 'Remove if not needed, or ensure it\'s properly imported'
            })
        
        # Suggest reducing high dependency counts
        for component in self.dependencies.keys():
            deps = self.get_dependencies_of(component)
            if len(deps) > 10:  # Arbitrary threshold
                suggestions.append({
                    'type': 'high_dependency_count',
                    'severity': 'warning',
                    'message': f"Component '{component}' has {len(deps)} dependencies",
                    'component': component,
                    'count': len(deps),
                    'suggestion': 'Consider splitting into smaller components or using facades'
                })
        
        # Suggest reducing deep dependency chains
        for component in self.dependencies.keys():
            depth = self.get_dependency_depth(component)
            if depth > 5:  # Arbitrary threshold
                suggestions.append({
                    'type': 'deep_dependency_chain',
                    'severity': 'warning',
                    'message': f"Component '{component}' has dependency depth of {depth}",
                    'component': component,
                    'depth': depth,
                    'suggestion': 'Deep chains make testing difficult. Consider flattening the hierarchy'
                })
        
        return suggestions
    
    def get_dependency_path(self, source: str, target: str) -> List[List[str]]:
        """
        Find all paths from source component to target component.
        
        Args:
            source: Starting component
            target: Target component
            
        Returns:
            List of paths, where each path is a list of component names
        """
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []
        except:
            return []
    
    def get_dependency_tree(self, component: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get dependency tree starting from a component.
        
        Args:
            component: Root component
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary representing the tree structure
        """
        if component not in self.graph:
            return {}
        
        def build_tree(node: str, depth: int, visited: Set[str]) -> Dict[str, Any]:
            """Recursively build tree."""
            if depth >= max_depth or node in visited:
                return {'name': node, 'children': [], 'truncated': depth >= max_depth}
            
            visited.add(node)
            
            children = []
            for child in self.graph.successors(node):
                children.append(build_tree(child, depth + 1, visited.copy()))
            
            return {
                'name': node,
                'children': children,
                'depth': depth,
                'truncated': False
            }
        
        return build_tree(component, 0, set())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dependency graph.
        
        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'total_components': len(self.graph.nodes()),
            'total_dependencies': len(self.graph.edges()),
            'circular_dependencies': len(self.find_circular_dependencies()),
            'unused_components': len(self.find_unused_dependencies()),
            'average_dependencies': 0.0,
            'max_dependencies': 0,
            'max_dependency_depth': 0
        }
        
        # Calculate average dependencies per component
        if self.dependencies:
            dep_counts = [len(deps) for deps in self.dependencies.values()]
            stats['average_dependencies'] = sum(dep_counts) / len(dep_counts)
            stats['max_dependencies'] = max(dep_counts) if dep_counts else 0
        
        # Find maximum dependency depth
        for component in self.dependencies.keys():
            depth = self.get_dependency_depth(component)
            stats['max_dependency_depth'] = max(stats['max_dependency_depth'], depth)
        
        return stats
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export graph to dictionary format.
        
        Returns:
            Dictionary representation of the graph
        """
        nodes = []
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            nodes.append({
                'id': node,
                'dependents': in_degree,
                'dependencies': out_degree
            })
        
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'type': data.get('injection_type', 'unknown')
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': self.get_stats(),
            'suggestions': self.suggest_injection_improvements()
        }
    
    def visualize_ascii(self, component: str, max_depth: int = 3) -> str:
        """
        Create ASCII visualization of dependency tree.
        
        Args:
            component: Root component to visualize
            max_depth: Maximum depth to show
            
        Returns:
            ASCII art representation
        """
        if component not in self.graph:
            return f"Component '{component}' not found"
        
        lines = []
        
        def add_node(node: str, depth: int, prefix: str, visited: Set[str]):
            """Recursively add nodes to visualization."""
            if depth >= max_depth or node in visited:
                if depth >= max_depth:
                    lines.append(f"{prefix}└── {node} [...]")
                return
            
            visited.add(node)
            lines.append(f"{prefix}└── {node}")
            
            children = list(self.graph.successors(node))
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                add_node(child, depth + 1, child_prefix, visited.copy())
        
        lines.append(component)
        children = list(self.graph.successors(component))
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            child_prefix = "    " if is_last else "│   "
            add_node(child, 1, child_prefix, {component})
        
        return "\n".join(lines)
