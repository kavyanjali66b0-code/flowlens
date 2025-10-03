"""
Edge Intelligence: Enriches edges with metadata for better visualization.

This module analyzes edges and adds intelligence metadata including:
- Weight: Importance/strength of the relationship
- Frequency: How often the edge is traversed (call count)
- Criticality: How critical the edge is to application flow
- Data dependencies: What data flows through this edge
"""

import logging
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict

from .models import Edge, EdgeType, Node


class EdgeIntelligenceEnricher:
    """
    Enriches edges with intelligence metadata for visualization.
    
    The enricher analyzes:
    - Call patterns and frequency
    - Data flow relationships
    - API call dependencies
    - State propagation
    - Critical paths
    """
    
    def __init__(self, nodes: List[Node], edges: List[Edge], 
                 data_flows: Optional[Dict] = None,
                 api_calls: Optional[Dict] = None):
        """
        Initialize edge intelligence enricher.
        
        Args:
            nodes: List of Node objects
            edges: List of Edge objects to enrich
            data_flows: Optional data flow analysis results
            api_calls: Optional API call tracking results
        """
        self.nodes = nodes
        self.edges = edges
        self.data_flows = data_flows or {}
        self.api_calls = api_calls or {}
        
        # Build lookup indexes
        self._node_by_id = {node.id: node for node in nodes}
        self._edges_by_source = defaultdict(list)
        self._edges_by_target = defaultdict(list)
        
        for edge in edges:
            self._edges_by_source[edge.source].append(edge)
            self._edges_by_target[edge.target].append(edge)
        
        logging.info(f"EdgeIntelligenceEnricher initialized with {len(nodes)} nodes, "
                    f"{len(edges)} edges")
    
    def enrich_all_edges(self) -> List[Edge]:
        """
        Enrich all edges with intelligence metadata.
        
        Returns:
            List of enriched Edge objects
        """
        logging.info("Enriching edges with intelligence metadata...")
        
        enriched_count = 0
        for edge in self.edges:
            if self._enrich_single_edge(edge):
                enriched_count += 1
        
        logging.info(f"Enriched {enriched_count}/{len(self.edges)} edges")
        return self.edges
    
    def _enrich_single_edge(self, edge: Edge) -> bool:
        """
        Enrich a single edge with intelligence metadata.
        
        Args:
            edge: Edge to enrich
            
        Returns:
            True if enrichment was successful
        """
        try:
            # Initialize metadata if not present
            if edge.metadata is None:
                edge.metadata = {}
            
            # Calculate weight (0.0 - 1.0)
            weight = self._calculate_edge_weight(edge)
            edge.metadata['weight'] = weight
            
            # Calculate frequency (call count)
            frequency = self._calculate_edge_frequency(edge)
            edge.metadata['frequency'] = frequency
            
            # Calculate criticality (0.0 - 1.0)
            criticality = self._calculate_edge_criticality(edge)
            edge.metadata['criticality'] = criticality
            
            # Identify data dependencies
            data_deps = self._identify_data_dependencies(edge)
            if data_deps:
                edge.metadata['data_dependencies'] = data_deps
            
            # Check if edge involves API calls
            has_api = self._check_api_involvement(edge)
            edge.metadata['has_api_call'] = has_api
            
            # Check if edge involves state
            has_state = self._check_state_involvement(edge)
            edge.metadata['has_state'] = has_state
            
            logging.debug(f"Enriched edge {edge.source} -> {edge.target}: "
                         f"weight={weight:.2f}, criticality={criticality:.2f}")
            
            return True
            
        except Exception as e:
            logging.warning(f"Failed to enrich edge {edge.source} -> {edge.target}: {e}", exc_info=True)
            return False
    
    def _calculate_edge_weight(self, edge: Edge) -> float:
        """
        Calculate edge weight based on relationship type and patterns.
        
        Higher weight = more important relationship
        
        Returns:
            Weight value between 0.0 and 1.0
        """
        # Base weight by edge type
        type_weights = {
            EdgeType.CALLS: 0.6,
            EdgeType.ASYNC_CALLS: 0.7,  # Async calls are important
            EdgeType.IMPORTS: 0.3,
            EdgeType.EXPORTS: 0.4,
            EdgeType.RE_EXPORTS: 0.35,
            EdgeType.EXTENDS: 0.8,  # Inheritance is strong
            EdgeType.IMPLEMENTS: 0.7,
            EdgeType.INSTANTIATES: 0.6,
            EdgeType.USES: 0.4,
            EdgeType.USES_METHOD: 0.5,
            EdgeType.CALLS_API: 0.85,  # API calls are important
            EdgeType.RENDERS: 0.75,
            EdgeType.DEPENDS_ON: 0.65,
            EdgeType.INVOKES: 0.6,
            EdgeType.ROUTES_TO: 0.7,
            EdgeType.HAS_TYPE: 0.4,
            EdgeType.TYPE_ALIAS: 0.3,
            EdgeType.GENERIC_PARAM: 0.35
        }
        
        # Handle both EdgeType enum and string type
        edge_type = edge.type
        if not isinstance(edge_type, EdgeType):
            # Try to convert string to EdgeType
            try:
                edge_type = EdgeType(edge_type) if isinstance(edge_type, str) else EdgeType.CALLS
            except (ValueError, TypeError):
                logging.debug(f"Unknown edge type: {edge.type}, using CALLS")
                edge_type = EdgeType.CALLS
        
        base_weight = type_weights.get(edge_type, 0.5)
        
        # Boost weight if source or target is an entry point
        source_node = self._node_by_id.get(edge.source)
        target_node = self._node_by_id.get(edge.target)
        
        if source_node and source_node.metadata.get('is_entry'):
            base_weight += 0.1
        if target_node and target_node.metadata.get('is_entry'):
            base_weight += 0.1
        
        # Boost weight if it's a component relationship
        if source_node and source_node.type.value == 'component':
            base_weight += 0.05
        if target_node and target_node.type.value == 'component':
            base_weight += 0.05
        
        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, base_weight))
    
    def _calculate_edge_frequency(self, edge: Edge) -> int:
        """
        Calculate how often an edge is traversed.
        
        This is an estimate based on:
        - Call graph data
        - Data flow counts
        - Multiple references
        
        Returns:
            Estimated frequency count
        """
        frequency = 1  # Base frequency
        
        # Check if we have call graph data
        metadata = edge.metadata or {}
        
        # If edge has explicit call count
        if 'call_count' in metadata:
            frequency = metadata['call_count']
        
        # Count how many times this edge appears in data flows
        if self.data_flows:
            flow_stats = self.data_flows.get('statistics', {})
            # Rough estimate: edges involved in more flows are used more
            if edge.type in [EdgeType.CALLS_API, EdgeType.DEPENDS_ON, EdgeType.ASYNC_CALLS]:
                frequency += 5  # Data flow edges are traversed often
        
        # Check for API calls (they're usually called multiple times)
        if self._check_api_involvement(edge):
            frequency += 3
        
        return frequency
    
    def _calculate_edge_criticality(self, edge: Edge) -> float:
        """
        Calculate edge criticality (how important it is to app functionality).
        
        Critical edges:
        - Connect to entry points
        - Involve API calls
        - Involve state changes
        - Are on critical paths
        
        Returns:
            Criticality score between 0.0 and 1.0
        """
        criticality = 0.3  # Base criticality
        
        source_node = self._node_by_id.get(edge.source)
        target_node = self._node_by_id.get(edge.target)
        
        # Entry points are critical
        if source_node and source_node.metadata.get('is_entry'):
            criticality += 0.2
        if target_node and target_node.metadata.get('is_entry'):
            criticality += 0.1
        
        # API calls are critical
        if self._check_api_involvement(edge):
            criticality += 0.3
        
        # Handle both EdgeType enum and string type
        edge_type = edge.type
        if not isinstance(edge_type, EdgeType):
            try:
                edge_type = EdgeType(edge_type) if isinstance(edge_type, str) else EdgeType.CALLS
            except (ValueError, TypeError):
                edge_type = EdgeType.CALLS
        
        # State changes and data flows are critical (check via metadata or type)
        if edge_type in [EdgeType.CALLS_API, EdgeType.DEPENDS_ON]:
            criticality += 0.2
        
        # Async operations are often critical
        if edge_type == EdgeType.ASYNC_CALLS:
            criticality += 0.1
        
        # Component renders are critical in UI
        if edge_type == EdgeType.RENDERS:
            criticality += 0.15
        
        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, criticality))
    
    def _identify_data_dependencies(self, edge: Edge) -> List[str]:
        """
        Identify what data flows through this edge.
        
        Returns:
            List of data dependency names (variable names, props, etc.)
        """
        dependencies = []
        
        # Check edge metadata for explicit data
        metadata = edge.metadata or {}
        
        # Props data
        if 'props' in metadata:
            props = metadata['props']
            if isinstance(props, list):
                dependencies.extend(props)
            elif isinstance(props, str):
                dependencies.append(props)
        
        # State data
        if 'state_var' in metadata:
            dependencies.append(metadata['state_var'])
        
        # Parameter data
        if 'parameters' in metadata:
            params = metadata['parameters']
            if isinstance(params, list):
                dependencies.extend(params)
        
        # Return data
        if 'return_type' in metadata:
            dependencies.append(f"returns:{metadata['return_type']}")
        
        return dependencies
    
    def _check_api_involvement(self, edge: Edge) -> bool:
        """
        Check if this edge involves an API call.
        
        Returns:
            True if edge is related to API calls
        """
        if not self.api_calls:
            return False
        
        # Get source and target nodes
        source_node = self._node_by_id.get(edge.source)
        target_node = self._node_by_id.get(edge.target)
        
        if not source_node or not target_node:
            return False
        
        # Check if either node is mentioned in API calls
        endpoints = self.api_calls.get('endpoints', [])
        
        for endpoint in endpoints:
            source_file = endpoint.get('source_file', '')
            
            # Check if edge nodes are in the same file as API call
            if source_node.file == source_file or target_node.file == source_file:
                # Check if function names match (rough heuristic)
                if (source_node.name in source_file or 
                    target_node.name in source_file):
                    return True
        
        return False
    
    def _check_state_involvement(self, edge: Edge) -> bool:
        """
        Check if this edge involves state management.
        
        Returns:
            True if edge is related to state
        """
        # Check edge type for API/dependency edges (often involve state)
        edge_type = edge.type
        if not isinstance(edge_type, EdgeType):
            try:
                edge_type = EdgeType(edge_type) if isinstance(edge_type, str) else EdgeType.CALLS
            except (ValueError, TypeError):
                edge_type = EdgeType.CALLS
        
        if edge_type in [EdgeType.CALLS_API, EdgeType.DEPENDS_ON]:
            return True
        
        # Check if source or target has React hooks
        source_node = self._node_by_id.get(edge.source)
        target_node = self._node_by_id.get(edge.target)
        
        if source_node:
            hooks = source_node.metadata.get('hooks', [])
            if any('useState' in str(h) or 'useReducer' in str(h) for h in hooks):
                return True
        
        if target_node:
            hooks = target_node.metadata.get('hooks', [])
            if any('useState' in str(h) or 'useReducer' in str(h) for h in hooks):
                return True
        
        return False
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about edge enrichment.
        
        Returns:
            Dict with enrichment statistics
        """
        stats = {
            'total_edges': len(self.edges),
            'enriched_edges': 0,
            'avg_weight': 0.0,
            'avg_criticality': 0.0,
            'edges_with_data_deps': 0,
            'edges_with_api': 0,
            'edges_with_state': 0,
            'frequency_distribution': defaultdict(int)
        }
        
        total_weight = 0.0
        total_criticality = 0.0
        
        for edge in self.edges:
            if edge.metadata:
                stats['enriched_edges'] += 1
                
                if 'weight' in edge.metadata:
                    total_weight += edge.metadata['weight']
                
                if 'criticality' in edge.metadata:
                    total_criticality += edge.metadata['criticality']
                
                if edge.metadata.get('data_dependencies'):
                    stats['edges_with_data_deps'] += 1
                
                if edge.metadata.get('has_api_call'):
                    stats['edges_with_api'] += 1
                
                if edge.metadata.get('has_state'):
                    stats['edges_with_state'] += 1
                
                freq = edge.metadata.get('frequency', 0)
                stats['frequency_distribution'][freq] += 1
        
        if stats['enriched_edges'] > 0:
            stats['avg_weight'] = total_weight / stats['enriched_edges']
            stats['avg_criticality'] = total_criticality / stats['enriched_edges']
        
        return dict(stats)
