"""
Runtime Instrumentation Metadata for M4.0

Provides metadata for future runtime instrumentation and n8n-like workflow features:
1. Hookable points - functions/methods that can be instrumented
2. Observable state - state variables to track at runtime
3. Entry-to-exit paths - execution flows from entry points to exits
"""

import logging
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from .models import Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class RuntimeInstrumentationAnalyzer:
    """
    Analyzes code structure to identify instrumentation points for runtime monitoring.
    
    This enables future features like:
    - n8n-style workflow execution tracking
    - Runtime state observation
    - Execution path monitoring
    - Performance profiling hooks
    """
    
    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        entry_points: List[str],
        data_flows: Dict[str, Any]
    ):
        """
        Initialize the runtime instrumentation analyzer.
        
        Args:
            nodes: List of all nodes in the graph
            edges: List of all edges in the graph
            entry_points: List of entry point node IDs
            data_flows: Data flow analysis results
        """
        self.nodes = nodes
        self.edges = edges
        self.entry_points = set(entry_points)
        self.data_flows = data_flows
        
        # Build lookup structures
        self._node_by_id = {node.id: node for node in nodes}
        self._edges_by_source = defaultdict(list)
        self._edges_by_target = defaultdict(list)
        
        for edge in edges:
            self._edges_by_source[edge.source].append(edge)
            self._edges_by_target[edge.target].append(edge)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform runtime instrumentation analysis.
        
        Returns:
            Dictionary with instrumentation metadata
        """
        logger.info("Starting runtime instrumentation analysis")
        
        hookable_points = self._identify_hookable_points()
        observable_state = self._identify_observable_state()
        execution_paths = self._trace_execution_paths()
        
        # Calculate statistics
        stats = {
            'total_hookable_points': len(hookable_points),
            'total_state_variables': len(observable_state),
            'total_execution_paths': len(execution_paths),
            'hookable_by_type': self._count_by_type(hookable_points, 'hook_type'),
            'state_by_type': self._count_by_type(observable_state, 'state_type')
        }
        
        return {
            'hookable_points': hookable_points,
            'observable_state': observable_state,
            'execution_paths': execution_paths,
            'statistics': stats
        }
    
    def _identify_hookable_points(self) -> List[Dict[str, Any]]:
        """
        Identify functions and methods that can be instrumented.
        
        Hookable points are ideal for:
        - Before/after execution hooks
        - Performance monitoring
        - Logging/debugging
        - Workflow step tracking
        
        Returns:
            List of hookable points with metadata
        """
        hookable = []
        
        for node in self.nodes:
            hook_info = self._analyze_hookability(node)
            if hook_info:
                hookable.append(hook_info)
        
        # Sort by priority (higher priority first)
        hookable.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Identified {len(hookable)} hookable points")
        return hookable
    
    def _analyze_hookability(self, node: Node) -> Optional[Dict[str, Any]]:
        """
        Analyze if a node is hookable and calculate its priority.
        
        Args:
            node: Node to analyze
            
        Returns:
            Hookable point info or None
        """
        node_type = node.type.value.lower()
        
        # Skip non-executable nodes
        if node_type in ['module', 'template', 'model']:
            return None
        
        # Determine hook type
        hook_type = None
        base_priority = 0.0
        
        if node_type == 'function':
            hook_type = 'function_call'
            base_priority = 0.5
        elif node_type == 'component':
            hook_type = 'component_render'
            base_priority = 0.7
        elif node_type == 'api_endpoint':
            hook_type = 'api_request'
            base_priority = 0.9
        elif node_type == 'route':
            hook_type = 'route_handler'
            base_priority = 0.8
        elif node_type in ['service', 'controller']:
            hook_type = 'service_method'
            base_priority = 0.6
        elif node_type == 'class':
            hook_type = 'method_call'
            base_priority = 0.4
        else:
            hook_type = 'generic'
            base_priority = 0.3
        
        # Calculate priority based on various factors
        priority = base_priority
        
        # Entry points are high priority
        if node.id in self.entry_points:
            priority += 0.2
        
        # API calls are high priority
        metadata = getattr(node, 'metadata', {}) or {}
        if metadata.get('has_api_call'):
            priority += 0.15
        
        # State involvement increases priority
        if metadata.get('hooks') or metadata.get('has_state'):
            priority += 0.1
        
        # High fan-out (many dependencies) increases priority
        outgoing_edges = len(self._edges_by_source.get(node.id, []))
        if outgoing_edges > 5:
            priority += 0.1
        
        # Cap at 1.0
        priority = min(1.0, priority)
        
        # Determine instrumentation suggestions
        suggestions = []
        if hook_type == 'function_call':
            suggestions = ['measure_execution_time', 'log_arguments', 'log_return_value']
        elif hook_type == 'component_render':
            suggestions = ['track_render_count', 'measure_render_time', 'capture_props']
        elif hook_type == 'api_request':
            suggestions = ['log_request', 'log_response', 'measure_latency', 'track_errors']
        elif hook_type == 'route_handler':
            suggestions = ['log_navigation', 'track_params', 'measure_load_time']
        else:
            suggestions = ['log_execution', 'measure_time']
        
        return {
            'node_id': node.id,
            'name': node.name,
            'file': node.file,
            'hook_type': hook_type,
            'priority': round(priority, 2),
            'is_entry_point': node.id in self.entry_points,
            'has_api_call': metadata.get('has_api_call', False),
            'has_state': bool(metadata.get('hooks')),
            'outgoing_edges': outgoing_edges,
            'suggested_instrumentation': suggestions
        }
    
    def _identify_observable_state(self) -> List[Dict[str, Any]]:
        """
        Identify state variables that can be observed at runtime.
        
        Observable state includes:
        - React hooks (useState, useReducer, etc.)
        - Class properties
        - Module-level variables
        - Store/context state
        
        Returns:
            List of observable state with metadata
        """
        observable = []
        
        for node in self.nodes:
            state_vars = self._extract_state_variables(node)
            observable.extend(state_vars)
        
        logger.info(f"Identified {len(observable)} observable state variables")
        return observable
    
    def _extract_state_variables(self, node: Node) -> List[Dict[str, Any]]:
        """
        Extract state variables from a node.
        
        Args:
            node: Node to extract state from
            
        Returns:
            List of state variable metadata
        """
        state_vars = []
        metadata = getattr(node, 'metadata', {}) or {}
        
        # React hooks
        hooks = metadata.get('hooks', [])
        for hook in hooks:
            hook_str = str(hook)
            
            state_type = None
            if 'useState' in hook_str:
                state_type = 'react_state'
            elif 'useReducer' in hook_str:
                state_type = 'react_reducer'
            elif 'useContext' in hook_str:
                state_type = 'react_context'
            elif 'useRef' in hook_str:
                state_type = 'react_ref'
            else:
                state_type = 'react_hook'
            
            state_vars.append({
                'node_id': node.id,
                'component': node.name,
                'file': node.file,
                'state_type': state_type,
                'hook_name': hook_str,
                'observable': True,
                'mutable': state_type in ['react_state', 'react_reducer', 'react_ref'],
                'suggested_tracking': ['track_value_changes', 'log_updates', 'performance_impact']
            })
        
        # Class properties (if metadata contains them)
        properties = metadata.get('properties', [])
        for prop in properties:
            state_vars.append({
                'node_id': node.id,
                'class': node.name,
                'file': node.file,
                'state_type': 'class_property',
                'property_name': str(prop),
                'observable': True,
                'mutable': True,
                'suggested_tracking': ['track_mutations', 'log_access']
            })
        
        return state_vars
    
    def _trace_execution_paths(self) -> List[Dict[str, Any]]:
        """
        Trace execution paths from entry points to exit points.
        
        Execution paths show:
        - Flow from entry point to various endpoints
        - Critical paths (through API calls, state changes)
        - Branching points
        
        Returns:
            List of execution paths
        """
        paths = []
        
        for entry_id in self.entry_points:
            if entry_id not in self._node_by_id:
                continue
            
            # BFS to find all paths from this entry point
            entry_paths = self._find_paths_from_entry(entry_id, max_depth=10)
            paths.extend(entry_paths)
        
        logger.info(f"Traced {len(paths)} execution paths")
        return paths
    
    def _find_paths_from_entry(
        self,
        entry_id: str,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find execution paths from an entry point.
        
        Args:
            entry_id: Entry point node ID
            max_depth: Maximum path depth to trace
            
        Returns:
            List of path metadata
        """
        paths = []
        
        # BFS with path tracking
        queue = [(entry_id, [entry_id], 0)]
        visited_paths = set()
        
        while queue:
            node_id, path, depth = queue.pop(0)
            
            if depth >= max_depth:
                # Reached max depth, save this path
                path_key = '->'.join(path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append(self._create_path_metadata(path, 'max_depth_reached'))
                continue
            
            # Get outgoing edges
            outgoing = self._edges_by_source.get(node_id, [])
            
            if not outgoing:
                # Exit point reached, save this path
                path_key = '->'.join(path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append(self._create_path_metadata(path, 'exit_point'))
                continue
            
            # Continue traversal
            for edge in outgoing:
                target = edge.target
                if target not in path:  # Avoid cycles
                    new_path = path + [target]
                    queue.append((target, new_path, depth + 1))
        
        return paths
    
    def _create_path_metadata(
        self,
        path: List[str],
        termination_reason: str
    ) -> Dict[str, Any]:
        """
        Create metadata for an execution path.
        
        Args:
            path: List of node IDs in the path
            termination_reason: Why the path ended
            
        Returns:
            Path metadata
        """
        # Analyze path characteristics
        has_api_call = False
        has_state_change = False
        node_types = []
        
        for node_id in path:
            node = self._node_by_id.get(node_id)
            if node:
                node_types.append(node.type.value)
                metadata = getattr(node, 'metadata', {}) or {}
                if metadata.get('has_api_call'):
                    has_api_call = True
                if metadata.get('hooks'):
                    has_state_change = True
        
        # Determine path criticality
        criticality = 0.5
        if has_api_call:
            criticality += 0.3
        if has_state_change:
            criticality += 0.2
        if path[0] in self.entry_points:
            criticality += 0.1
        criticality = min(1.0, criticality)
        
        return {
            'path': path,
            'length': len(path),
            'entry_point': path[0],
            'exit_point': path[-1],
            'termination_reason': termination_reason,
            'has_api_call': has_api_call,
            'has_state_change': has_state_change,
            'criticality': round(criticality, 2),
            'node_types': node_types,
            'suggested_monitoring': self._suggest_path_monitoring(has_api_call, has_state_change)
        }
    
    def _suggest_path_monitoring(
        self,
        has_api_call: bool,
        has_state_change: bool
    ) -> List[str]:
        """
        Suggest monitoring strategies for an execution path.
        
        Args:
            has_api_call: Whether path includes API calls
            has_state_change: Whether path includes state changes
            
        Returns:
            List of monitoring suggestions
        """
        suggestions = ['measure_total_execution_time', 'count_executions']
        
        if has_api_call:
            suggestions.extend(['track_api_latency', 'log_api_errors'])
        
        if has_state_change:
            suggestions.extend(['track_state_mutations', 'log_state_snapshots'])
        
        return suggestions
    
    def _count_by_type(
        self,
        items: List[Dict[str, Any]],
        type_key: str
    ) -> Dict[str, int]:
        """
        Count items by a type field.
        
        Args:
            items: List of items with type field
            type_key: Key to group by
            
        Returns:
            Count by type
        """
        counts = defaultdict(int)
        for item in items:
            item_type = item.get(type_key, 'unknown')
            counts[item_type] += 1
        return dict(counts)
