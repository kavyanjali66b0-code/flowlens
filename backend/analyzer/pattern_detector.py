"""
Architectural Pattern Detector for Semantic Analysis

This module detects common architectural patterns in codebases, enabling
understanding of system design and architectural relationships between
components.

Classes:
    ArchitecturalPatternDetector: Main detector for architectural patterns
    PatternResult: Result of pattern detection
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

from .models import Node, Edge, NodeType, EdgeType


class PatternType(Enum):
    """Types of architectural patterns."""
    MVC = "mvc"
    REPOSITORY = "repository"
    FACTORY = "factory"
    SINGLETON = "singleton"
    OBSERVER = "observer"
    DEPENDENCY_INJECTION = "dependency_injection"
    COMMAND = "command"
    STRATEGY = "strategy"
    ADAPTER = "adapter"
    DECORATOR = "decorator"
    FACADE = "facade"
    PROXY = "proxy"
    BUILDER = "builder"
    CHAIN_OF_RESPONSIBILITY = "chain_of_responsibility"
    MEDIATOR = "mediator"
    STATE = "state"
    TEMPLATE_METHOD = "template_method"
    VISITOR = "visitor"
    MICROSERVICE = "microservice"
    LAYERED = "layered"
    PIPE_AND_FILTER = "pipe_and_filter"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    UNKNOWN = "unknown"


@dataclass
class PatternResult:
    """Result of pattern detection."""
    pattern_type: PatternType
    confidence: float
    components: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_type': self.pattern_type.value,
            'confidence': round(self.confidence, 3),
            'components': self.components,
            'relationships': self.relationships,
            'evidence': self.evidence,
            'violations': self.violations,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class ArchitecturalPatternDetector:
    """
    Detector for architectural patterns in codebases.
    
    Analyzes nodes and edges to identify common design patterns and
    architectural structures, providing insights into system design.
    """
    
    def __init__(self):
        """Initialize the pattern detector."""
        self.logger = logging.getLogger(__name__)
        self._initialize_pattern_rules()
    
    def _initialize_pattern_rules(self):
        """Initialize pattern detection rules."""
        self.pattern_rules = {
            PatternType.MVC: {
                'components': {
                    'controllers': ['controller', 'ctrl', 'handler', 'action'],
                    'models': ['model', 'entity', 'domain', 'data'],
                    'views': ['view', 'component', 'template', 'page', 'ui']
                },
                'relationships': [
                    ('controller', 'model', EdgeType.USES),
                    ('controller', 'view', EdgeType.RENDERS),
                    ('view', 'model', EdgeType.USES)
                ],
                'naming_patterns': [
                    r'.*Controller$', r'.*Model$', r'.*View$', r'.*Component$'
                ]
            },
            PatternType.REPOSITORY: {
                'components': {
                    'repositories': ['repository', 'repo', 'dao', 'dataaccess'],
                    'entities': ['entity', 'model', 'domain'],
                    'services': ['service', 'business', 'logic']
                },
                'relationships': [
                    ('service', 'repository', EdgeType.USES),
                    ('repository', 'entity', EdgeType.USES)
                ],
                'naming_patterns': [
                    r'.*Repository$', r'.*DAO$', r'.*DataAccess$'
                ]
            },
            PatternType.FACTORY: {
                'components': {
                    'factories': ['factory', 'builder', 'creator', 'maker'],
                    'products': ['product', 'object', 'instance', 'item']
                },
                'relationships': [
                    ('factory', 'product', EdgeType.CREATES)
                ],
                'naming_patterns': [
                    r'.*Factory$', r'.*Builder$', r'.*Creator$'
                ]
            },
            PatternType.SINGLETON: {
                'components': {
                    'singletons': ['singleton', 'instance', 'manager', 'registry']
                },
                'relationships': [],
                'naming_patterns': [
                    r'.*Singleton$', r'.*Manager$', r'.*Registry$'
                ],
                'code_patterns': [
                    r'getInstance\s*\(', r'instance\s*=\s*null', r'private.*constructor'
                ]
            },
            PatternType.OBSERVER: {
                'components': {
                    'subjects': ['subject', 'observable', 'publisher', 'event'],
                    'observers': ['observer', 'listener', 'subscriber', 'handler']
                },
                'relationships': [
                    ('subject', 'observer', EdgeType.NOTIFIES)
                ],
                'naming_patterns': [
                    r'.*Observer$', r'.*Listener$', r'.*Subscriber$'
                ]
            },
            PatternType.DEPENDENCY_INJECTION: {
                'components': {
                    'services': ['service', 'component', 'bean'],
                    'injectors': ['injector', 'container', 'registry', 'module']
                },
                'relationships': [
                    ('injector', 'service', EdgeType.INJECTS)
                ],
                'naming_patterns': [
                    r'.*Service$', r'.*Component$', r'.*Bean$'
                ],
                'code_patterns': [
                    r'@Inject', r'@Autowired', r'@Component', r'@Service'
                ]
            },
            PatternType.COMMAND: {
                'components': {
                    'commands': ['command', 'action', 'operation', 'task'],
                    'invokers': ['invoker', 'executor', 'dispatcher'],
                    'receivers': ['receiver', 'handler', 'processor']
                },
                'relationships': [
                    ('invoker', 'command', EdgeType.EXECUTES),
                    ('command', 'receiver', EdgeType.ACTUATES)
                ],
                'naming_patterns': [
                    r'.*Command$', r'.*Action$', r'.*Operation$'
                ]
            },
            PatternType.STRATEGY: {
                'components': {
                    'strategies': ['strategy', 'algorithm', 'policy', 'behavior'],
                    'contexts': ['context', 'client', 'executor']
                },
                'relationships': [
                    ('context', 'strategy', EdgeType.USES)
                ],
                'naming_patterns': [
                    r'.*Strategy$', r'.*Algorithm$', r'.*Policy$'
                ]
            },
            PatternType.ADAPTER: {
                'components': {
                    'adapters': ['adapter', 'wrapper', 'converter'],
                    'targets': ['target', 'interface', 'contract'],
                    'adaptees': ['adaptee', 'legacy', 'external']
                },
                'relationships': [
                    ('adapter', 'target', EdgeType.IMPLEMENTS),
                    ('adapter', 'adaptee', EdgeType.WRAPS)
                ],
                'naming_patterns': [
                    r'.*Adapter$', r'.*Wrapper$', r'.*Converter$'
                ]
            },
            PatternType.DECORATOR: {
                'components': {
                    'decorators': ['decorator', 'wrapper', 'enhancer'],
                    'components': ['component', 'base', 'core']
                },
                'relationships': [
                    ('decorator', 'component', EdgeType.DECORATES)
                ],
                'naming_patterns': [
                    r'.*Decorator$', r'.*Wrapper$', r'.*Enhancer$'
                ]
            },
            PatternType.FACADE: {
                'components': {
                    'facades': ['facade', 'interface', 'api', 'gateway'],
                    'subsystems': ['subsystem', 'module', 'service']
                },
                'relationships': [
                    ('facade', 'subsystem', EdgeType.ORCHESTRATES)
                ],
                'naming_patterns': [
                    r'.*Facade$', r'.*Gateway$', r'.*Interface$'
                ]
            },
            PatternType.PROXY: {
                'components': {
                    'proxies': ['proxy', 'stub', 'surrogate'],
                    'subjects': ['subject', 'real', 'target']
                },
                'relationships': [
                    ('proxy', 'subject', EdgeType.PROXIES)
                ],
                'naming_patterns': [
                    r'.*Proxy$', r'.*Stub$', r'.*Surrogate$'
                ]
            },
            PatternType.BUILDER: {
                'components': {
                    'builders': ['builder', 'constructor', 'assembler'],
                    'products': ['product', 'object', 'result']
                },
                'relationships': [
                    ('builder', 'product', EdgeType.BUILDS)
                ],
                'naming_patterns': [
                    r'.*Builder$', r'.*Constructor$', r'.*Assembler$'
                ]
            },
            PatternType.MICROSERVICE: {
                'components': {
                    'services': ['service', 'microservice', 'api'],
                    'gateways': ['gateway', 'router', 'proxy']
                },
                'relationships': [
                    ('gateway', 'service', EdgeType.ROUTES_TO)
                ],
                'naming_patterns': [
                    r'.*Service$', r'.*API$', r'.*Gateway$'
                ]
            },
            PatternType.LAYERED: {
                'components': {
                    'presentation': ['presentation', 'ui', 'view', 'controller'],
                    'business': ['business', 'service', 'logic', 'domain'],
                    'data': ['data', 'repository', 'dao', 'persistence']
                },
                'relationships': [
                    ('presentation', 'business', EdgeType.USES),
                    ('business', 'data', EdgeType.USES)
                ],
                'naming_patterns': [
                    r'.*Controller$', r'.*Service$', r'.*Repository$'
                ]
            }
        }
    
    def detect_patterns(self, nodes: List[Node], edges: List[Edge]) -> List[PatternResult]:
        """
        Detect architectural patterns in the codebase.
        
        Args:
            nodes: List of nodes in the codebase
            edges: List of edges between nodes
            
        Returns:
            List of detected patterns with confidence scores
        """
        detected_patterns = []
        
        for pattern_type, rules in self.pattern_rules.items():
            result = self._detect_single_pattern(pattern_type, rules, nodes, edges)
            if result and result.confidence > 0.3:  # Only include patterns with reasonable confidence
                detected_patterns.append(result)
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return detected_patterns
    
    def _detect_single_pattern(self, pattern_type: PatternType, rules: Dict, 
                             nodes: List[Node], edges: List[Edge]) -> Optional[PatternResult]:
        """Detect a single architectural pattern."""
        try:
            # Find components for this pattern
            components = self._find_pattern_components(pattern_type, rules, nodes)
            
            if not any(components.values()):
                return None
            
            # Check relationships
            relationships = self._check_pattern_relationships(rules, components, edges)
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                pattern_type, rules, components, relationships, nodes
            )
            
            # Generate evidence
            evidence = self._generate_pattern_evidence(pattern_type, rules, components, relationships)
            
            # Check for violations
            violations = self._check_pattern_violations(pattern_type, rules, components, relationships)
            
            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(
                pattern_type, components, violations
            )
            
            return PatternResult(
                pattern_type=pattern_type,
                confidence=confidence,
                components=components,
                relationships=relationships,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations,
                metadata={
                    'component_count': sum(len(comps) for comps in components.values()),
                    'relationship_count': len(relationships)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern {pattern_type}: {e}")
            return None
    
    def _find_pattern_components(self, pattern_type: PatternType, rules: Dict, 
                               nodes: List[Node]) -> Dict[str, List[str]]:
        """Find components that match pattern rules."""
        components = defaultdict(list)
        
        for component_type, patterns in rules.get('components', {}).items():
            for node in nodes:
                if self._node_matches_pattern(node, patterns, rules.get('naming_patterns', [])):
                    components[component_type].append(node.name)
        
        return dict(components)
    
    def _node_matches_pattern(self, node: Node, patterns: List[str], 
                            naming_patterns: List[str]) -> bool:
        """Check if a node matches pattern criteria."""
        node_name_lower = node.name.lower()
        
        # Check keyword patterns
        for pattern in patterns:
            if pattern in node_name_lower:
                return True
        
        # Check regex naming patterns
        for naming_pattern in naming_patterns:
            if re.match(naming_pattern, node.name, re.IGNORECASE):
                return True
        
        # Check node type
        if node.type in [NodeType.CLASS, NodeType.COMPONENT, NodeType.SERVICE]:
            return True
        
        return False
    
    def _check_pattern_relationships(self, rules: Dict, components: Dict[str, List[str]], 
                                   edges: List[Edge]) -> List[Dict[str, Any]]:
        """Check if relationships match pattern expectations."""
        relationships = []
        expected_relationships = rules.get('relationships', [])
        
        for source_type, target_type, edge_type in expected_relationships:
            source_components = components.get(source_type, [])
            target_components = components.get(target_type, [])
            
            if not source_components or not target_components:
                continue
            
            # Find edges between these component types
            for edge in edges:
                source_name = self._get_node_name_by_id(edge.source)
                target_name = self._get_node_name_by_id(edge.target)
                
                if (source_name in source_components and 
                    target_name in target_components and 
                    edge.type == edge_type):
                    relationships.append({
                        'source': source_name,
                        'target': target_name,
                        'type': edge_type.value,
                        'source_type': source_type,
                        'target_type': target_type
                    })
        
        return relationships
    
    def _get_node_name_by_id(self, node_id: str) -> str:
        """Get node name by ID (placeholder - would need node lookup)."""
        # This is a simplified version - in practice, you'd maintain a node lookup
        return node_id
    
    def _calculate_pattern_confidence(self, pattern_type: PatternType, rules: Dict,
                                    components: Dict[str, List[str]], 
                                    relationships: List[Dict[str, Any]], 
                                    nodes: List[Node]) -> float:
        """Calculate confidence score for pattern detection."""
        confidence = 0.0
        
        # Component presence (40% weight)
        component_score = 0.0
        total_expected_components = len(rules.get('components', {}))
        present_components = len([comp for comp in components.values() if comp])
        
        if total_expected_components > 0:
            component_score = present_components / total_expected_components
        confidence += component_score * 0.4
        
        # Relationship presence (30% weight)
        relationship_score = 0.0
        expected_relationships = len(rules.get('relationships', []))
        if expected_relationships > 0:
            relationship_score = len(relationships) / expected_relationships
        confidence += relationship_score * 0.3
        
        # Component balance (20% weight)
        balance_score = 0.0
        if components:
            component_counts = [len(comps) for comps in components.values()]
            if component_counts:
                # Prefer balanced component distribution
                max_count = max(component_counts)
                min_count = min(component_counts)
                if max_count > 0:
                    balance_score = min_count / max_count
        confidence += balance_score * 0.2
        
        # Naming convention adherence (10% weight)
        naming_score = 0.0
        naming_patterns = rules.get('naming_patterns', [])
        if naming_patterns:
            total_nodes = sum(len(comps) for comps in components.values())
            matching_nodes = 0
            
            for comp_list in components.values():
                for comp_name in comp_list:
                    for pattern in naming_patterns:
                        if re.match(pattern, comp_name, re.IGNORECASE):
                            matching_nodes += 1
                            break
            
            if total_nodes > 0:
                naming_score = matching_nodes / total_nodes
        confidence += naming_score * 0.1
        
        return min(confidence, 1.0)
    
    def _generate_pattern_evidence(self, pattern_type: PatternType, rules: Dict,
                                 components: Dict[str, List[str]], 
                                 relationships: List[Dict[str, Any]]) -> List[str]:
        """Generate evidence for pattern detection."""
        evidence = []
        
        # Component evidence
        for comp_type, comp_list in components.items():
            if comp_list:
                evidence.append(f"Found {len(comp_list)} {comp_type}: {', '.join(comp_list[:3])}")
        
        # Relationship evidence
        if relationships:
            evidence.append(f"Found {len(relationships)} expected relationships")
            for rel in relationships[:2]:  # Show first 2 relationships
                evidence.append(f"  {rel['source']} -> {rel['target']} ({rel['type']})")
        
        # Naming evidence
        naming_patterns = rules.get('naming_patterns', [])
        if naming_patterns:
            matching_names = []
            for comp_list in components.values():
                for comp_name in comp_list:
                    for pattern in naming_patterns:
                        if re.match(pattern, comp_name, re.IGNORECASE):
                            matching_names.append(comp_name)
                            break
            
            if matching_names:
                evidence.append(f"Naming convention matches: {', '.join(matching_names[:3])}")
        
        return evidence
    
    def _check_pattern_violations(self, pattern_type: PatternType, rules: Dict,
                                components: Dict[str, List[str]], 
                                relationships: List[Dict[str, Any]]) -> List[str]:
        """Check for pattern violations."""
        violations = []
        
        # Check for missing components
        expected_components = rules.get('components', {})
        for comp_type, comp_list in expected_components.items():
            if not components.get(comp_type):
                violations.append(f"Missing {comp_type} components")
        
        # Check for missing relationships
        expected_relationships = rules.get('relationships', [])
        for source_type, target_type, edge_type in expected_relationships:
            source_components = components.get(source_type, [])
            target_components = components.get(target_type, [])
            
            if source_components and target_components:
                # Check if any relationship exists
                has_relationship = any(
                    rel['source_type'] == source_type and rel['target_type'] == target_type
                    for rel in relationships
                )
                if not has_relationship:
                    violations.append(f"Missing {source_type} -> {target_type} relationships")
        
        return violations
    
    def _generate_pattern_recommendations(self, pattern_type: PatternType,
                                        components: Dict[str, List[str]], 
                                        violations: List[str]) -> List[str]:
        """Generate recommendations for improving pattern implementation."""
        recommendations = []
        
        # Address violations
        for violation in violations:
            if "Missing" in violation and "components" in violation:
                recommendations.append(f"Add missing components to complete {pattern_type.value} pattern")
            elif "Missing" in violation and "relationships" in violation:
                recommendations.append(f"Establish missing relationships for {pattern_type.value} pattern")
        
        # Pattern-specific recommendations
        if pattern_type == PatternType.MVC:
            if not components.get('controllers'):
                recommendations.append("Add controller classes to handle user interactions")
            if not components.get('models'):
                recommendations.append("Add model classes to represent data")
            if not components.get('views'):
                recommendations.append("Add view components for user interface")
        
        elif pattern_type == PatternType.REPOSITORY:
            if not components.get('repositories'):
                recommendations.append("Add repository classes for data access")
            if not components.get('services'):
                recommendations.append("Add service classes for business logic")
        
        elif pattern_type == PatternType.SINGLETON:
            recommendations.append("Ensure singleton classes have private constructors")
            recommendations.append("Implement getInstance() method for singleton access")
        
        return recommendations
    
    def get_pattern_statistics(self, results: List[PatternResult]) -> Dict[str, Any]:
        """Get statistics about detected patterns."""
        if not results:
            return {}
        
        # Count patterns
        pattern_counts = Counter(result.pattern_type.value for result in results)
        
        # Confidence statistics
        confidences = [result.confidence for result in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Component statistics
        total_components = sum(
            sum(len(comps) for comps in result.components.values()) 
            for result in results
        )
        
        return {
            'total_patterns': len(results),
            'pattern_distribution': dict(pattern_counts),
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_patterns': len([r for r in results if r.confidence > 0.7]),
            'total_components': total_components,
            'patterns_with_violations': len([r for r in results if r.violations])
        }

