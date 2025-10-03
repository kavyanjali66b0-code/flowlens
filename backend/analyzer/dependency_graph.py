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
import logging
import json
import hashlib
from pathlib import Path
import networkx as nx
from analyzer.di_resolver import Dependency, DIType

logger = logging.getLogger(__name__)


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
        """
        Build optimized NetworkX graph from dependencies.
        
        Performance optimizations:
        - Uses batch node addition
        - Deduplicates edges using sets
        - Pre-computes all components for O(1) lookup
        """
        logger.debug(f"Building dependency graph from {len(self.dependencies)} consumers")
        
        # Collect all unique components (consumers + providers)
        all_components = set(self.dependencies.keys())
        for deps in self.dependencies.values():
            all_components.update(dep.provider for dep in deps)
        
        logger.debug(f"Found {len(all_components)} unique components")
        
        # Batch add all nodes at once (faster than individual adds)
        self.graph.add_nodes_from(all_components)
        
        # Build edges with deduplication
        # Use set to track unique (source, target) pairs
        seen_edges = set()
        edges_to_add = []
        duplicate_count = 0
        
        for consumer, deps in self.dependencies.items():
            for dep in deps:
                edge_key = (consumer, dep.provider)
                
                # Skip duplicate edges (keep first occurrence)
                if edge_key in seen_edges:
                    duplicate_count += 1
                    continue
                
                seen_edges.add(edge_key)
                edges_to_add.append((
                    consumer,
                    dep.provider,
                    {
                        'injection_type': dep.injection_type.name,
                        'location': dep.location,
                        'metadata': dep.metadata
                    }
                ))
        
        # Batch add edges
        self.graph.add_edges_from(edges_to_add)
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate edges during graph building")
        
        logger.info(f"Built dependency graph: {len(all_components)} nodes, {len(edges_to_add)} edges")
    
    @staticmethod
    def _compute_cache_key(dependencies: Dict[str, List[Dependency]]) -> str:
        """
        Compute a deterministic hash key for the dependency data.
        
        Args:
            dependencies: Dependency dictionary to hash
            
        Returns:
            SHA256 hash string
        """
        # Create deterministic representation
        components = []
        for consumer in sorted(dependencies.keys()):
            deps = dependencies[consumer]
            for dep in sorted(deps, key=lambda d: (d.provider, d.injection_type.name)):
                components.append(f"{consumer}|{dep.provider}|{dep.injection_type.name}")
        
        data_str = "\n".join(components)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def save_to_cache(self, cache_dir: Path) -> bool:
        """
        Save the built graph to cache.
        
        Args:
            cache_dir: Directory to store cache files
            
        Returns:
            True if saved successfully
        """
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute cache key
            cache_key = self._compute_cache_key(self.dependencies)
            cache_file = cache_dir / f"graph-{cache_key}.json"
            
            # Serialize graph using NetworkX's node_link_data
            graph_data = nx.node_link_data(self.graph)
            
            # Save to file
            with open(cache_file, 'w') as f:
                json.dump({
                    'graph': graph_data,
                    'cache_key': cache_key,
                    'version': '1.0'
                }, f)
            
            logger.info(f"Saved dependency graph to cache: {cache_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save graph cache: {e}")
            return False
    
    @classmethod
    def load_from_cache(
        cls,
        dependencies: Dict[str, List[Dependency]],
        cache_dir: Path
    ) -> Optional['DependencyGraph']:
        """
        Load graph from cache if available and valid.
        
        Args:
            dependencies: Current dependency data (for validation)
            cache_dir: Directory containing cache files
            
        Returns:
            DependencyGraph instance if loaded successfully, None otherwise
        """
        try:
            # Compute expected cache key
            cache_key = cls._compute_cache_key(dependencies)
            cache_file = cache_dir / f"graph-{cache_key}.json"
            
            if not cache_file.exists():
                logger.debug(f"Graph cache miss: {cache_key}")
                return None
            
            # Load cache file
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache key
            if cache_data.get('cache_key') != cache_key:
                logger.warning("Graph cache key mismatch")
                return None
            
            # Deserialize graph
            graph = nx.node_link_graph(cache_data['graph'])
            
            # Create instance without rebuilding
            instance = cls.__new__(cls)
            instance.dependencies = dependencies
            instance.graph = graph
            
            logger.info(f"Loaded dependency graph from cache: {cache_key}")
            return instance
            
        except Exception as e:
            logger.warning(f"Failed to load graph cache: {e}")
            return None
    
    def merge_incremental_changes(
        self,
        changed_dependencies: Dict[str, List[Dependency]],
        removed_components: Set[str]
    ) -> None:
        """
        Incrementally update graph with changes instead of full rebuild.
        
        Args:
            changed_dependencies: Components with changed dependencies
            removed_components: Components that were removed
        """
        logger.debug(f"Merging incremental changes: {len(changed_dependencies)} changed, {len(removed_components)} removed")
        
        # Remove deleted components
        for component in removed_components:
            if component in self.graph:
                self.graph.remove_node(component)
        
        # Update changed components
        for consumer, deps in changed_dependencies.items():
            # Remove old edges from this consumer
            if consumer in self.graph:
                old_edges = list(self.graph.out_edges(consumer))
                self.graph.remove_edges_from(old_edges)
            else:
                # Add new node
                self.graph.add_node(consumer)
            
            # Add new edges with deduplication
            seen_providers = set()
            for dep in deps:
                if dep.provider in seen_providers:
                    continue
                
                seen_providers.add(dep.provider)
                
                # Add provider node if doesn't exist
                if dep.provider not in self.graph:
                    self.graph.add_node(dep.provider)
                
                # Add edge
                self.graph.add_edge(
                    consumer,
                    dep.provider,
                    injection_type=dep.injection_type.name,
                    location=dep.location,
                    metadata=dep.metadata
                )
        
        # Update dependencies dictionary
        self.dependencies.update(changed_dependencies)
        for component in removed_components:
            self.dependencies.pop(component, None)
        
        logger.info(f"Merged incremental changes: graph now has {self.graph.number_of_nodes()} nodes")
    
    def get_topological_order(self) -> List[str]:
        """
        Get components in topological order (dependencies before dependents).
        
        This is useful for:
        - Visualizing dependency flow (bottom-up)
        - Determining build/initialization order
        - Organizing code structure
        
        Returns:
            List of component names in topological order.
            Returns empty list if graph has cycles.
        """
        try:
            # NetworkX's topological_sort returns iterator
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Graph has cycles - cannot produce topological order
            return []
    
    def get_layers(self) -> List[List[str]]:
        """
        Group components into dependency layers.
        
        Layer 0: Components with no dependencies (leaf nodes)
        Layer 1: Components that only depend on Layer 0
        Layer N: Components that depend on any layer < N
        
        Returns:
            List of layers, where each layer is a list of component names.
            Returns empty list if graph has cycles.
        """
        try:
            # Get topological generations (layers)
            layers = list(nx.topological_generations(self.graph))
            return layers
        except nx.NetworkXError:
            # Graph has cycles
            return []
    
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
        # Calculate basic stats from graph structure
        total_components = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        
        stats = {
            'total_components': total_components,
            'total_dependencies': total_edges,
            'circular_dependencies': len(self.find_circular_dependencies()),
            'unused_components': len(self.find_unused_dependencies()),
            'average_dependencies': 0.0,
            'max_dependencies': 0,
            'max_dependency_depth': 0,
            'has_cycles': not nx.is_directed_acyclic_graph(self.graph)
        }
        
        # Calculate dependency statistics efficiently
        if total_components > 0:
            # Use out_degree for dependency counts (more efficient than iterating self.dependencies)
            out_degrees = [d for n, d in self.graph.out_degree()]
            if out_degrees:
                stats['average_dependencies'] = sum(out_degrees) / len(out_degrees)
                stats['max_dependencies'] = max(out_degrees)
        
        # Find maximum dependency depth (only for DAGs)
        if not stats['has_cycles']:
            # For DAGs, use longest path from source nodes
            try:
                stats['max_dependency_depth'] = nx.dag_longest_path_length(self.graph)
            except:
                stats['max_dependency_depth'] = 0
        else:
            # For cyclic graphs, sample some nodes
            for component in list(self.dependencies.keys())[:10]:  # Sample first 10
                depth = self.get_dependency_depth(component)
                stats['max_dependency_depth'] = max(stats['max_dependency_depth'], depth)
        
        return stats
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export graph to dictionary format with topological ordering.
        
        Returns:
            Dictionary representation of the graph
        """
        # Get topological order for better visualization
        topo_order = self.get_topological_order()
        layers = self.get_layers()
        
        nodes = []
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            # Find layer for this node
            layer = -1
            for idx, layer_nodes in enumerate(layers):
                if node in layer_nodes:
                    layer = idx
                    break
            
            nodes.append({
                'id': node,
                'dependents': in_degree,
                'dependencies': out_degree,
                'layer': layer  # Add layer information for visualization
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
            'topological_order': topo_order,  # For build/init order
            'layers': layers,  # For hierarchical visualization
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
