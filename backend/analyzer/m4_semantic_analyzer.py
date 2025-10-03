"""
M4.0 Semantic Analyzer - Production API Adapter

This adapter integrates M4.0 components into the production API by:
1. Maintaining API compatibility with the existing SemanticAnalyzer interface
2. Enriching nodes and edges with M4.0 rich metadata
3. Using TypeInferenceEngine, HookDetector, and ConfidenceScorer

The adapter takes parsed nodes/edges and enhances them with M4.0 features:
- Rich type information in node metadata
- React hook detection in node metadata
- Confidence scores for all entities
- Complete edge metadata with flow types and context
"""

import logging
from typing import List, Dict, Optional, Any

from .models import Node, Edge, EdgeType, NodeType
from .symbol_table import SymbolTable
from .semantic_analyzer import SemanticAnalyzer

# Import M4.0 components
from .type_inference_engine import TypeInferenceEngine, TypeCategory
from .hook_detector import HookDetector
from .confidence_scorer import ConfidenceScorer, EntityCategory, ConfidenceFactor


class M4SemanticAnalyzer:
    """
    M4.0-enhanced semantic analyzer that enriches nodes and edges with rich metadata.
    
    This adapter:
    1. Uses the existing SemanticAnalyzer to create base edges
    2. Enriches nodes with M4.0 type inference and hook detection
    3. Enriches edges with M4.0 confidence and flow information
    4. Maintains full API compatibility
    """
    
    def __init__(
        self,
        nodes: List[Node],
        file_symbols: Dict[str, Dict],
        symbol_table: Optional[SymbolTable] = None
    ):
        """
        Initialize M4SemanticAnalyzer.
        
        Args:
            nodes: Parsed nodes from LanguageParser
            file_symbols: Symbol information per file
            symbol_table: Optional symbol table for reference tracking
        """
        self.nodes = nodes
        self.file_symbols = file_symbols
        self.symbol_table = symbol_table
        
        # Initialize M4.0 components
        self.type_engine = TypeInferenceEngine()
        self.hook_detector = HookDetector()
        self.confidence_scorer = ConfidenceScorer()
        
        # Use base semantic analyzer for edge creation
        self.base_analyzer = SemanticAnalyzer(nodes, file_symbols, symbol_table)
        
        logging.info("M4SemanticAnalyzer initialized with M4.0 components")
    
    def analyze(self) -> List[Edge]:
        """
        Perform semantic analysis with M4.0 enhancements.
        
        This method:
        1. Runs base semantic analysis to create edges
        2. Enriches nodes with M4.0 metadata
        3. Enriches edges with M4.0 metadata
        4. Returns enhanced edges
        
        Returns:
            List of edges with rich M4.0 metadata
        """
        logging.info("Starting M4.0-enhanced semantic analysis")
        
        # Step 1: Run base semantic analysis to create edges
        edges = self.base_analyzer.analyze()
        logging.info(f"Base analysis created {len(edges)} edges")
        
        # Step 2: Enrich nodes with M4.0 metadata
        self._enrich_nodes_with_m4_metadata()
        logging.info(f"Enriched {len(self.nodes)} nodes with M4.0 metadata")
        
        # Step 3: Enrich edges with M4.0 metadata
        enriched_edges = self._enrich_edges_with_m4_metadata(edges)
        logging.info(f"Enriched {len(enriched_edges)} edges with M4.0 metadata")
        
        logging.info("M4.0-enhanced semantic analysis complete")
        return enriched_edges
    
    def _enrich_nodes_with_m4_metadata(self):
        """
        Enrich all nodes with M4.0 metadata.
        
        Adds to node.metadata:
        - type_info: Rich type information (name, category, confidence, source)
        - hook_info: React hook detection (if applicable)
        - confidence_score: Entity confidence score
        """
        for node in self.nodes:
            if node.metadata is None:
                node.metadata = {}
            
            # Enrich based on node type
            if node.type == NodeType.FUNCTION and not node.name.startswith('use'):
                # Treat non-hook functions as potential variables
                self._enrich_variable_node(node)
            elif node.type == NodeType.FUNCTION:
                self._enrich_function_node(node)
            elif node.type == NodeType.COMPONENT:
                self._enrich_component_node(node)
            elif node.type == NodeType.CLASS:
                self._enrich_class_node(node)
            
            # Add confidence score for all nodes
            self._add_confidence_score(node)
    
    def _enrich_variable_node(self, node: Node):
        """Enrich variable node with type information."""
        metadata = node.metadata or {}
        
        # Try to infer type from name/context
        type_info = None
        
        # Check if it's a React state variable (contains 'state', 'count', 'data', etc.)
        if any(keyword in node.name.lower() for keyword in ['state', 'count', 'data', 'value', 'text']):
            # Likely a state variable - infer as unknown initially
            type_info = self.type_engine.infer_from_literal(None)
            metadata['react_info'] = {
                'likely_state': True,
                'pattern_matched': True
            }
        
        # Add type info if inferred
        if type_info:
            metadata['type_info'] = {
                'name': type_info.name,
                'category': type_info.category.value,
                'confidence': type_info.confidence,
                'source': type_info.source.value
            }
        
        node.metadata = metadata
    
    def _enrich_function_node(self, node: Node):
        """Enrich function node with type and hook information."""
        metadata = node.metadata or {}
        
        # Check if it's a React hook
        if node.name.startswith('use') and len(node.name) > 3 and node.name[3].isupper():
            # Detect hook
            hook_call = self.hook_detector.detect_hook_call(
                hook_id=node.id,
                hook_name=node.name,
                arguments=[],
                line_number=0
            )
            
            if hook_call:
                metadata['hook_info'] = {
                    'is_hook': True,
                    'hook_name': hook_call.hook_name,
                    'hook_type': hook_call.hook_type.value,
                    'hook_category': hook_call.hook_category.value,
                    'confidence': hook_call.confidence
                }
        
        # Infer function type
        type_info = self.type_engine.infer_function_type(
            params=[],
            return_expr=None,
            is_arrow=False
        )
        
        metadata['type_info'] = {
            'name': type_info.name,
            'category': type_info.category.value,
            'confidence': type_info.confidence,
            'source': type_info.source.value
        }
        
        node.metadata = metadata
    
    def _enrich_component_node(self, node: Node):
        """Enrich React component node with rich metadata."""
        metadata = node.metadata or {}
        
        # Infer component type
        type_info = self.type_engine.infer_jsx_element_type(node.name)
        
        metadata['type_info'] = {
            'name': type_info.name,
            'category': type_info.category.value,
            'confidence': type_info.confidence,
            'source': type_info.source.value,
            'is_component': True
        }
        
        metadata['react_info'] = {
            'component_type': 'functional',  # Default assumption
            'is_react_component': True
        }
        
        node.metadata = metadata
    
    def _enrich_class_node(self, node: Node):
        """Enrich class node with type information."""
        metadata = node.metadata or {}
        
        # Infer class type
        type_info = self.type_engine.infer_function_type(
            params=[],
            return_expr=None,
            is_arrow=False
        )
        
        metadata['type_info'] = {
            'name': node.name,
            'category': 'class',
            'confidence': 0.9,
            'source': 'definition'
        }
        
        node.metadata = metadata
    
    def _add_confidence_score(self, node: Node):
        """Add confidence score to node metadata."""
        metadata = node.metadata or {}
        
        # Determine entity category
        category_map = {
            NodeType.FUNCTION: EntityCategory.FUNCTION,
            NodeType.COMPONENT: EntityCategory.COMPONENT,
            NodeType.CLASS: EntityCategory.CLASS,
        }
        
        entity_category = category_map.get(node.type, EntityCategory.FUNCTION)
        
        # Collect confidence factors
        factors = {}
        
        # Type inference factor
        if 'type_info' in metadata:
            factors[ConfidenceFactor.TYPE_INFERENCE] = metadata['type_info'].get('confidence', 0.5)
        
        # Hook pattern factor
        if 'hook_info' in metadata:
            factors[ConfidenceFactor.HOOK_PATTERN] = metadata['hook_info'].get('confidence', 0.8)
        
        # Evidence factor (has metadata)
        if metadata:
            factors[ConfidenceFactor.EVIDENCE] = min(1.0, len(metadata) * 0.2)
        
        # Default factor if no specific factors
        if not factors:
            factors[ConfidenceFactor.EVIDENCE] = 0.5
        
        # Score entity
        score = self.confidence_scorer.score_entity(
            entity_id=node.id,
            entity_name=node.name,
            entity_category=entity_category,
            factors=factors,
            metadata=metadata
        )
        
        # Add confidence score to metadata
        metadata['confidence_score'] = {
            'score': score.overall_score,
            'level': score.confidence_level.value,
            'factors': [fs.factor.value for fs in score.factor_scores],
            'explanation': '; '.join(score.recommendations)
        }
        
        node.metadata = metadata
    
    def _enrich_edges_with_m4_metadata(self, edges: List[Edge]) -> List[Edge]:
        """
        Enrich edges with M4.0 metadata.
        
        Adds to edge.metadata:
        - flow_type: Type of data flow
        - confidence_score: Edge confidence
        - type_propagation: Type information flow
        - context: Additional context
        """
        enriched_edges = []
        
        for edge in edges:
            if edge.metadata is None:
                edge.metadata = {}
            
            # Add flow type information
            edge.metadata['flow_type'] = self._infer_flow_type(edge)
            
            # Add confidence score
            edge.metadata['confidence_score'] = self._calculate_edge_confidence(edge)
            
            # Add type propagation if both nodes have type info
            source_node = next((n for n in self.nodes if n.id == edge.source), None)
            target_node = next((n for n in self.nodes if n.id == edge.target), None)
            
            if source_node and target_node:
                source_type = (source_node.metadata or {}).get('type_info')
                target_type = (target_node.metadata or {}).get('type_info')
                
                if source_type and target_type:
                    edge.metadata['type_propagation'] = {
                        'from': source_type.get('name', 'unknown'),
                        'to': target_type.get('name', 'unknown'),
                        'source_confidence': source_type.get('confidence', 0.5),
                        'target_confidence': target_type.get('confidence', 0.5)
                    }
                
                # Add hook info if relevant
                source_hook = (source_node.metadata or {}).get('hook_info')
                if source_hook:
                    edge.metadata['hook_info'] = {
                        'hook_name': source_hook.get('hook_name'),
                        'hook_type': source_hook.get('hook_type'),
                        'confidence': source_hook.get('confidence', 0.8)
                    }
            
            # Add context
            edge.metadata['context'] = f"{edge.type.value} relationship"
            
            enriched_edges.append(edge)
        
        return enriched_edges
    
    def _infer_flow_type(self, edge: Edge) -> str:
        """Infer the flow type for an edge."""
        # Map edge types to flow types
        flow_type_map = {
            EdgeType.RENDERS: 'component_render',
            EdgeType.CALLS: 'function_call',
            EdgeType.IMPORTS: 'module_import',
            EdgeType.USES: 'dependency_use',
            EdgeType.DEPENDS_ON: 'dependency',
            EdgeType.EXTENDS: 'class_extension',
            EdgeType.IMPLEMENTS: 'interface_implementation'
        }
        
        return flow_type_map.get(edge.type, 'generic_relationship')
    
    def _calculate_edge_confidence(self, edge: Edge) -> float:
        """Calculate confidence score for an edge."""
        # Get source and target nodes
        source_node = next((n for n in self.nodes if n.id == edge.source), None)
        target_node = next((n for n in self.nodes if n.id == edge.target), None)
        
        if not source_node or not target_node:
            return 0.5
        
        # Average confidence of source and target
        source_conf = (source_node.metadata or {}).get('confidence_score', {}).get('score', 0.5)
        target_conf = (target_node.metadata or {}).get('confidence_score', {}).get('score', 0.5)
        
        # Edge type confidence
        type_confidence = {
            EdgeType.RENDERS: 0.9,
            EdgeType.CALLS: 0.85,
            EdgeType.IMPORTS: 0.95,
            EdgeType.DEPENDS_ON: 0.8,
            EdgeType.USES: 0.75
        }.get(edge.type, 0.7)
        
        # Calculate weighted average
        confidence = (source_conf * 0.3 + target_conf * 0.3 + type_confidence * 0.4)
        
        return round(confidence, 2)
