"""
Module Boundaries Analyzer for M4.0

Enhances module boundary analysis with metrics:
1. Cohesion score - how well module components work together
2. Coupling score - how dependent module is on others
3. Public exports - what module exposes
4. Internal-only components - private implementation details
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from .models import Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class ModuleBoundariesAnalyzer:
    """
    Analyzes module boundaries and calculates cohesion/coupling metrics.
    
    A module is typically a directory containing related components.
    """
    
    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        project_root: str
    ):
        """
        Initialize the module boundaries analyzer.
        
        Args:
            nodes: List of all nodes in the graph
            edges: List of all edges in the graph
            project_root: Root path of the project
        """
        self.nodes = nodes
        self.edges = edges
        self.project_root = Path(project_root)
        
        # Build lookup structures
        self._node_by_id = {node.id: node for node in nodes}
        self._edges_by_source = defaultdict(list)
        self._edges_by_target = defaultdict(list)
        
        for edge in edges:
            self._edges_by_source[edge.source].append(edge)
            self._edges_by_target[edge.target].append(edge)
        
        # Group nodes by module (directory)
        self._modules = self._identify_modules()
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform module boundary analysis.
        
        Returns:
            Dictionary with module boundaries and metrics
        """
        logger.info("Starting module boundaries analysis")
        
        module_metrics = []
        coupling_matrix = {}
        hotspots = []
        
        for module_path, module_nodes in self._modules.items():
            metrics = self._analyze_module(module_path, module_nodes)
            module_metrics.append(metrics)
            
            # Build coupling matrix
            for dep_module, dep_count in metrics['dependencies'].items():
                key = f"{module_path} -> {dep_module}"
                coupling_matrix[key] = dep_count
        
        # Identify coupling hotspots (highly coupled modules)
        hotspots = self._identify_coupling_hotspots(module_metrics)
        
        # Calculate overall statistics
        stats = self._calculate_statistics(module_metrics)
        
        return {
            'modules': module_metrics,
            'coupling_matrix': coupling_matrix,
            'hotspots': hotspots,
            'statistics': stats
        }
    
    def _identify_modules(self) -> Dict[str, List[str]]:
        """
        Group nodes into modules based on directory structure.
        
        Returns:
            Dictionary mapping module path to node IDs
        """
        modules = defaultdict(list)
        
        for node in self.nodes:
            try:
                file_path = Path(node.file)
                if file_path.is_absolute():
                    try:
                        rel_path = file_path.relative_to(self.project_root)
                    except ValueError:
                        rel_path = file_path
                else:
                    rel_path = file_path
                
                # Module is the parent directory
                module_path = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
                modules[module_path].append(node.id)
            except Exception as e:
                logger.warning(f"Error processing file path {node.file}: {e}")
                modules['unknown'].append(node.id)
        
        return dict(modules)
    
    def _analyze_module(self, module_path: str, node_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze a single module and calculate metrics.
        
        Args:
            module_path: Path to the module
            node_ids: List of node IDs in the module
            
        Returns:
            Dictionary with module metrics
        """
        # Get nodes in this module
        module_node_set = set(node_ids)
        
        # Count edges
        internal_edges = 0  # Edges within module
        outgoing_edges = 0  # Edges to other modules
        incoming_edges = 0  # Edges from other modules
        
        # Track dependencies (which modules this depends on)
        dependencies = defaultdict(int)
        dependents = defaultdict(int)  # Which modules depend on this
        
        # Track exports (nodes used by other modules)
        exported_nodes = set()
        internal_only_nodes = set(node_ids)
        
        for node_id in node_ids:
            # Outgoing edges from this node
            for edge in self._edges_by_source.get(node_id, []):
                if edge.target in module_node_set:
                    internal_edges += 1
                else:
                    outgoing_edges += 1
                    # Track which module we depend on
                    target_node = self._node_by_id.get(edge.target)
                    if target_node:
                        target_module = self._get_module_for_node(target_node)
                        if target_module and target_module != module_path:
                            dependencies[target_module] += 1
            
            # Incoming edges to this node
            for edge in self._edges_by_target.get(node_id, []):
                if edge.source not in module_node_set:
                    incoming_edges += 1
                    exported_nodes.add(node_id)
                    internal_only_nodes.discard(node_id)
                    
                    # Track which module depends on us
                    source_node = self._node_by_id.get(edge.source)
                    if source_node:
                        source_module = self._get_module_for_node(source_node)
                        if source_module and source_module != module_path:
                            dependents[source_module] += 1
        
        # Calculate cohesion (0.0 to 1.0)
        # Cohesion = internal edges / (internal + outgoing edges)
        total_outbound = internal_edges + outgoing_edges
        cohesion = internal_edges / total_outbound if total_outbound > 0 else 0.0
        
        # Calculate coupling (0.0 to 1.0)
        # Coupling = outgoing edges / total edges
        total_edges = internal_edges + outgoing_edges + incoming_edges
        coupling = (outgoing_edges + incoming_edges) / total_edges if total_edges > 0 else 0.0
        
        # Get node types in module
        node_types = []
        for nid in node_ids:
            node = self._node_by_id.get(nid)
            if node:
                node_types.append(node.type.value)
        
        type_distribution = {}
        for nt in set(node_types):
            type_distribution[nt] = node_types.count(nt)
        
        # Determine module type based on content
        module_type = self._infer_module_type(module_path, type_distribution)
        
        return {
            'path': module_path,
            'size': len(node_ids),
            'nodes': node_ids,
            'cohesion_score': round(cohesion, 3),
            'coupling_score': round(coupling, 3),
            'internal_edges': internal_edges,
            'outgoing_edges': outgoing_edges,
            'incoming_edges': incoming_edges,
            'dependencies': dict(dependencies),
            'dependents': dict(dependents),
            'dependency_count': len(dependencies),
            'dependent_count': len(dependents),
            'public_exports': list(exported_nodes),
            'export_count': len(exported_nodes),
            'internal_only': list(internal_only_nodes),
            'internal_count': len(internal_only_nodes),
            'type_distribution': type_distribution,
            'module_type': module_type
        }
    
    def _get_module_for_node(self, node: Node) -> Optional[str]:
        """
        Get the module path for a node.
        
        Args:
            node: Node to find module for
            
        Returns:
            Module path or None
        """
        try:
            file_path = Path(node.file)
            if file_path.is_absolute():
                try:
                    rel_path = file_path.relative_to(self.project_root)
                except ValueError:
                    rel_path = file_path
            else:
                rel_path = file_path
            
            module_path = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
            return module_path
        except Exception:
            return None
    
    def _infer_module_type(
        self,
        module_path: str,
        type_distribution: Dict[str, int]
    ) -> str:
        """
        Infer the type/purpose of a module.
        
        Args:
            module_path: Path to the module
            type_distribution: Distribution of node types
            
        Returns:
            Module type
        """
        path_lower = module_path.lower()
        
        # Check path-based hints first
        if 'component' in path_lower:
            return 'ui_components'
        elif 'service' in path_lower or 'api' in path_lower:
            return 'services'
        elif 'util' in path_lower or 'helper' in path_lower:
            return 'utilities'
        elif 'model' in path_lower or 'type' in path_lower:
            return 'models'
        elif 'hook' in path_lower:
            return 'hooks'
        elif 'page' in path_lower or 'view' in path_lower:
            return 'views'
        elif 'store' in path_lower or 'state' in path_lower:
            return 'state_management'
        elif 'test' in path_lower:
            return 'tests'
        elif 'config' in path_lower:
            return 'configuration'
        
        # Infer from content
        total = sum(type_distribution.values())
        if total == 0:
            return 'empty'
        
        if type_distribution.get('component', 0) / total > 0.6:
            return 'ui_components'
        elif type_distribution.get('function', 0) / total > 0.7:
            return 'utilities'
        elif type_distribution.get('class', 0) / total > 0.5:
            return 'classes'
        
        return 'mixed'
    
    def _identify_coupling_hotspots(
        self,
        module_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify modules with high coupling (potential refactoring targets).
        
        Args:
            module_metrics: List of module metrics
            
        Returns:
            List of coupling hotspots
        """
        hotspots = []
        
        for module in module_metrics:
            coupling = module['coupling_score']
            cohesion = module['cohesion_score']
            
            # High coupling OR low cohesion indicates a problem
            is_hotspot = False
            issues = []
            
            if coupling > 0.7:
                is_hotspot = True
                issues.append('high_coupling')
            
            if cohesion < 0.3 and module['internal_edges'] > 0:
                is_hotspot = True
                issues.append('low_cohesion')
            
            # Too many dependencies
            if module['dependency_count'] > 5:
                is_hotspot = True
                issues.append('many_dependencies')
            
            # God module (too many nodes)
            if module['size'] > 20:
                is_hotspot = True
                issues.append('large_module')
            
            if is_hotspot:
                hotspots.append({
                    'module': module['path'],
                    'issues': issues,
                    'coupling_score': module['coupling_score'],
                    'cohesion_score': module['cohesion_score'],
                    'size': module['size'],
                    'dependency_count': module['dependency_count'],
                    'severity': self._calculate_severity(issues, coupling, cohesion)
                })
        
        # Sort by severity
        hotspots.sort(key=lambda x: x['severity'], reverse=True)
        
        return hotspots
    
    def _calculate_severity(
        self,
        issues: List[str],
        coupling: float,
        cohesion: float
    ) -> float:
        """
        Calculate severity score for a hotspot.
        
        Args:
            issues: List of issue types
            coupling: Coupling score
            cohesion: Cohesion score
            
        Returns:
            Severity score (0.0 to 1.0)
        """
        severity = 0.0
        
        # Base severity from issue count
        severity += len(issues) * 0.15
        
        # Coupling contribution
        if coupling > 0.7:
            severity += (coupling - 0.7) * 0.5
        
        # Cohesion contribution (inverse)
        if cohesion < 0.3:
            severity += (0.3 - cohesion) * 0.5
        
        return min(1.0, severity)
    
    def _calculate_statistics(
        self,
        module_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall statistics across all modules.
        
        Args:
            module_metrics: List of module metrics
            
        Returns:
            Statistics dictionary
        """
        if not module_metrics:
            return {}
        
        total_modules = len(module_metrics)
        avg_cohesion = sum(m['cohesion_score'] for m in module_metrics) / total_modules
        avg_coupling = sum(m['coupling_score'] for m in module_metrics) / total_modules
        avg_size = sum(m['size'] for m in module_metrics) / total_modules
        
        # Find extremes
        most_cohesive = max(module_metrics, key=lambda m: m['cohesion_score'])
        least_cohesive = min(module_metrics, key=lambda m: m['cohesion_score'])
        most_coupled = max(module_metrics, key=lambda m: m['coupling_score'])
        least_coupled = min(module_metrics, key=lambda m: m['coupling_score'])
        largest = max(module_metrics, key=lambda m: m['size'])
        
        # Count module types
        type_counts = defaultdict(int)
        for m in module_metrics:
            type_counts[m['module_type']] += 1
        
        return {
            'total_modules': total_modules,
            'avg_cohesion': round(avg_cohesion, 3),
            'avg_coupling': round(avg_coupling, 3),
            'avg_module_size': round(avg_size, 1),
            'most_cohesive_module': {
                'path': most_cohesive['path'],
                'score': most_cohesive['cohesion_score']
            },
            'least_cohesive_module': {
                'path': least_cohesive['path'],
                'score': least_cohesive['cohesion_score']
            },
            'most_coupled_module': {
                'path': most_coupled['path'],
                'score': most_coupled['coupling_score']
            },
            'least_coupled_module': {
                'path': least_coupled['path'],
                'score': least_coupled['coupling_score']
            },
            'largest_module': {
                'path': largest['path'],
                'size': largest['size']
            },
            'module_types': dict(type_counts)
        }
