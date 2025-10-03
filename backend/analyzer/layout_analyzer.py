"""
Layout Hints Analyzer for M4.0

Provides intelligent layout hints for visualization by detecting:
1. Directory-based clustering (nodes in same folder)
2. Dependency islands (tightly coupled components)
3. Architectural layers (presentation, business logic, data)
4. Entry point hierarchy (app entries positioned at top)
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from .models import Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class LayoutHintsAnalyzer:
    """
    Analyzes codebase structure to provide layout hints for visualization.
    
    Features:
    - Directory-based clustering
    - Dependency island detection
    - Architectural layer inference
    - Entry point hierarchy
    """
    
    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        entry_points: List[str],
        project_root: str
    ):
        """
        Initialize the layout analyzer.
        
        Args:
            nodes: List of all nodes in the graph
            edges: List of all edges in the graph
            entry_points: List of entry point node IDs
            project_root: Root path of the project
        """
        self.nodes = nodes
        self.edges = edges
        self.entry_points = set(entry_points)
        self.project_root = Path(project_root)
        
        # Build lookup structures
        self._node_by_id = {node.id: node for node in nodes}
        self._edges_by_source = defaultdict(list)
        self._edges_by_target = defaultdict(list)
        
        for edge in edges:
            self._edges_by_source[edge.source].append(edge)
            self._edges_by_target[edge.target].append(edge)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform full layout analysis.
        
        Returns:
            Dictionary with layout hints
        """
        logger.info("Starting layout hints analysis")
        
        # Analyze all layout aspects
        directory_clusters = self._analyze_directory_clusters()
        dependency_islands = self._detect_dependency_islands()
        architectural_layers = self._infer_architectural_layers()
        entry_hierarchy = self._build_entry_hierarchy()
        
        # Calculate statistics
        stats = {
            'total_clusters': len(directory_clusters),
            'total_islands': len(dependency_islands),
            'total_layers': len(architectural_layers),
            'entry_point_levels': len(entry_hierarchy),
            'largest_cluster_size': max([c['size'] for c in directory_clusters], default=0),
            'largest_island_size': max([len(i['nodes']) for i in dependency_islands], default=0)
        }
        
        return {
            'directory_clusters': directory_clusters,
            'dependency_islands': dependency_islands,
            'architectural_layers': architectural_layers,
            'entry_hierarchy': entry_hierarchy,
            'statistics': stats
        }
    
    def _analyze_directory_clusters(self) -> List[Dict[str, Any]]:
        """
        Group nodes by directory for spatial clustering.
        
        Returns:
            List of directory clusters with node IDs
        """
        clusters_by_dir = defaultdict(list)
        
        for node in self.nodes:
            # Get directory path relative to project root
            try:
                file_path = Path(node.file)
                if file_path.is_absolute():
                    try:
                        rel_path = file_path.relative_to(self.project_root)
                    except ValueError:
                        rel_path = file_path
                else:
                    rel_path = file_path
                
                # Get parent directory
                directory = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
                clusters_by_dir[directory].append(node.id)
            except Exception as e:
                logger.warning(f"Error processing file path {node.file}: {e}")
                clusters_by_dir['unknown'].append(node.id)
        
        # Convert to list format with metadata
        clusters = []
        for directory, node_ids in clusters_by_dir.items():
            # Calculate cluster depth (nesting level)
            depth = directory.count('/') + directory.count('\\') if directory != 'root' else 0
            
            # Determine cluster type based on directory name
            cluster_type = self._infer_cluster_type(directory)
            
            clusters.append({
                'directory': directory,
                'nodes': node_ids,
                'size': len(node_ids),
                'depth': depth,
                'type': cluster_type
            })
        
        # Sort by directory path for consistent ordering
        clusters.sort(key=lambda x: x['directory'])
        
        logger.info(f"Identified {len(clusters)} directory clusters")
        return clusters
    
    def _infer_cluster_type(self, directory: str) -> str:
        """
        Infer the purpose of a directory cluster.
        
        Args:
            directory: Directory path
            
        Returns:
            Cluster type (components, utils, services, etc.)
        """
        dir_lower = directory.lower()
        
        if 'component' in dir_lower:
            return 'components'
        elif 'service' in dir_lower or 'api' in dir_lower:
            return 'services'
        elif 'util' in dir_lower or 'helper' in dir_lower:
            return 'utilities'
        elif 'model' in dir_lower or 'type' in dir_lower:
            return 'models'
        elif 'hook' in dir_lower:
            return 'hooks'
        elif 'page' in dir_lower or 'view' in dir_lower or 'screen' in dir_lower:
            return 'views'
        elif 'store' in dir_lower or 'state' in dir_lower:
            return 'state_management'
        elif 'test' in dir_lower or 'spec' in dir_lower:
            return 'tests'
        elif 'config' in dir_lower:
            return 'configuration'
        else:
            return 'general'
    
    def _detect_dependency_islands(self) -> List[Dict[str, Any]]:
        """
        Detect groups of tightly coupled nodes (dependency islands).
        
        Uses connected components algorithm to find clusters of nodes
        that heavily depend on each other but are loosely coupled to
        the rest of the codebase.
        
        Returns:
            List of dependency islands with metrics
        """
        # Build undirected graph for connectivity analysis
        adjacency = defaultdict(set)
        for edge in self.edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        
        # Find connected components
        visited = set()
        islands = []
        
        for node in self.nodes:
            if node.id not in visited:
                island_nodes = self._dfs_component(node.id, adjacency, visited)
                
                # Only consider as "island" if it has multiple nodes
                if len(island_nodes) > 1:
                    island_info = self._analyze_island(island_nodes)
                    islands.append(island_info)
        
        # Sort by size (largest first)
        islands.sort(key=lambda x: x['size'], reverse=True)
        
        logger.info(f"Detected {len(islands)} dependency islands")
        return islands
    
    def _dfs_component(
        self,
        start_node: str,
        adjacency: Dict[str, Set[str]],
        visited: Set[str]
    ) -> List[str]:
        """
        DFS to find connected component.
        
        Args:
            start_node: Starting node ID
            adjacency: Adjacency list
            visited: Set of visited nodes
            
        Returns:
            List of node IDs in the component
        """
        component = []
        stack = [start_node]
        
        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                visited.add(node_id)
                component.append(node_id)
                
                # Add unvisited neighbors
                for neighbor in adjacency.get(node_id, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return component
    
    def _analyze_island(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze a dependency island to extract metrics.
        
        Args:
            node_ids: List of node IDs in the island
            
        Returns:
            Dictionary with island metrics
        """
        # Count internal vs external edges
        internal_edges = 0
        external_edges = 0
        node_set = set(node_ids)
        
        for node_id in node_ids:
            for edge in self._edges_by_source.get(node_id, []):
                if edge.target in node_set:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Calculate cohesion (internal edges / total edges)
        total_edges = internal_edges + external_edges
        cohesion = internal_edges / total_edges if total_edges > 0 else 0.0
        
        # Get node types in island
        node_types = [self._node_by_id[nid].type.value for nid in node_ids if nid in self._node_by_id]
        type_distribution = {}
        for nt in set(node_types):
            type_distribution[nt] = node_types.count(nt)
        
        # Determine island purpose
        purpose = self._infer_island_purpose(node_ids, type_distribution)
        
        return {
            'nodes': node_ids,
            'size': len(node_ids),
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'cohesion': round(cohesion, 3),
            'type_distribution': type_distribution,
            'purpose': purpose
        }
    
    def _infer_island_purpose(
        self,
        node_ids: List[str],
        type_distribution: Dict[str, int]
    ) -> str:
        """
        Infer the purpose of a dependency island.
        
        Args:
            node_ids: Node IDs in the island
            type_distribution: Distribution of node types
            
        Returns:
            Purpose description
        """
        # Check if mostly components
        if type_distribution.get('component', 0) > len(node_ids) * 0.6:
            return 'ui_feature'
        
        # Check if mostly services/API
        if (type_distribution.get('service', 0) + type_distribution.get('api_endpoint', 0)) > len(node_ids) * 0.5:
            return 'backend_service'
        
        # Check if mostly models/classes
        if (type_distribution.get('class', 0) + type_distribution.get('model', 0)) > len(node_ids) * 0.5:
            return 'data_layer'
        
        # Check if mostly utilities
        if type_distribution.get('function', 0) > len(node_ids) * 0.7:
            return 'utilities'
        
        return 'mixed_module'
    
    def _infer_architectural_layers(self) -> List[Dict[str, Any]]:
        """
        Infer architectural layers (presentation, business, data).
        
        Uses heuristics based on:
        - Node types (components are presentation)
        - Dependency direction (data layer has no dependencies)
        - Directory structure
        - Naming patterns
        
        Returns:
            List of architectural layers with nodes
        """
        layers = {
            'presentation': [],
            'business_logic': [],
            'data_access': [],
            'infrastructure': []
        }
        
        for node in self.nodes:
            layer = self._classify_node_layer(node)
            layers[layer].append(node.id)
        
        # Convert to list format with metadata
        layer_list = []
        for layer_name, node_ids in layers.items():
            if node_ids:  # Only include non-empty layers
                layer_list.append({
                    'layer': layer_name,
                    'nodes': node_ids,
                    'size': len(node_ids),
                    'level': self._get_layer_level(layer_name)
                })
        
        # Sort by level (top to bottom)
        layer_list.sort(key=lambda x: x['level'])
        
        logger.info(f"Identified {len(layer_list)} architectural layers")
        return layer_list
    
    def _classify_node_layer(self, node: Node) -> str:
        """
        Classify a node into an architectural layer.
        
        Args:
            node: Node to classify
            
        Returns:
            Layer name
        """
        node_type = node.type.value.lower()
        file_path = node.file.lower()
        
        # Presentation layer: UI components, views, templates
        if node_type in ['component', 'view', 'template', 'route']:
            return 'presentation'
        
        # Data access layer: models, repositories, API endpoints
        if node_type in ['model', 'api_endpoint'] or 'model' in file_path or 'repository' in file_path:
            return 'data_access'
        
        # Infrastructure: config, utilities, services
        if 'config' in file_path or 'util' in file_path or 'service' in file_path:
            return 'infrastructure'
        
        # Business logic: controllers, services, functions
        return 'business_logic'
    
    def _get_layer_level(self, layer_name: str) -> int:
        """
        Get the vertical level of a layer for layout.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Level number (0 = top)
        """
        levels = {
            'presentation': 0,
            'business_logic': 1,
            'data_access': 2,
            'infrastructure': 3
        }
        return levels.get(layer_name, 99)
    
    def _build_entry_hierarchy(self) -> List[Dict[str, Any]]:
        """
        Build hierarchy from entry points using BFS.
        
        Entry points are at level 0, direct dependencies at level 1, etc.
        This creates a natural top-down layout for visualization.
        
        Returns:
            List of hierarchy levels with nodes
        """
        if not self.entry_points:
            logger.warning("No entry points provided for hierarchy")
            return []
        
        # BFS from entry points
        levels = defaultdict(list)
        visited = set()
        queue = [(ep, 0) for ep in self.entry_points if ep in self._node_by_id]
        
        while queue:
            node_id, level = queue.pop(0)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            levels[level].append(node_id)
            
            # Add dependencies (outgoing edges)
            for edge in self._edges_by_source.get(node_id, []):
                if edge.target not in visited:
                    queue.append((edge.target, level + 1))
        
        # Convert to list format
        hierarchy = []
        for level, node_ids in sorted(levels.items()):
            hierarchy.append({
                'level': level,
                'nodes': node_ids,
                'size': len(node_ids),
                'is_entry': level == 0
            })
        
        logger.info(f"Built entry hierarchy with {len(hierarchy)} levels")
        return hierarchy
    
    def get_layout_recommendations(self) -> Dict[str, str]:
        """
        Get high-level layout recommendations for the frontend.
        
        Returns:
            Dictionary with layout suggestions
        """
        num_nodes = len(self.nodes)
        num_clusters = len(self._analyze_directory_clusters())
        
        # Recommend layout algorithm based on structure
        if num_nodes < 20:
            layout_type = 'force-directed'
            reasoning = 'Small graph works well with force-directed layout'
        elif num_clusters > 5:
            layout_type = 'hierarchical-clusters'
            reasoning = 'Many clusters suggest grouped hierarchical layout'
        elif len(self.entry_points) > 0:
            layout_type = 'top-down-hierarchy'
            reasoning = 'Entry points present, use top-down flow'
        else:
            layout_type = 'force-directed'
            reasoning = 'Default to force-directed for general graphs'
        
        return {
            'recommended_layout': layout_type,
            'reasoning': reasoning,
            'node_count': num_nodes,
            'cluster_count': num_clusters
        }
