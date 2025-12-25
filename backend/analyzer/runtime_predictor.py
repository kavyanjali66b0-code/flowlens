"""
Runtime Behavior Predictor for Semantic Analysis

This module predicts runtime behavior of code entities, enabling proactive
identification of potential issues, performance bottlenecks, and behavioral
patterns without actually executing the code.

Classes:
    RuntimeBehaviorPredictor: Main predictor for runtime behavior
    BehaviorPrediction: Result of behavior prediction
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

from .models import Node, Edge, NodeType, EdgeType


class BehaviorType(Enum):
    """Types of runtime behaviors."""
    ASYNC_OPERATION = "async_operation"
    ERROR_PRONE = "error_prone"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    SIDE_EFFECT = "side_effect"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_OPERATION = "network_operation"
    FILE_OPERATION = "file_operation"
    DATABASE_OPERATION = "database_operation"
    CACHING_OPERATION = "caching_operation"
    LOGGING_OPERATION = "logging_operation"
    SECURITY_SENSITIVE = "security_sensitive"
    THREAD_SAFE = "thread_safe"
    BLOCKING_OPERATION = "blocking_operation"
    RECURSIVE = "recursive"
    INFINITE_LOOP_RISK = "infinite_loop_risk"
    RESOURCE_LEAK = "resource_leak"
    UNKNOWN = "unknown"


@dataclass
class BehaviorPrediction:
    """Result of behavior prediction."""
    entity_name: str
    entity_type: str
    predicted_behaviors: Dict[BehaviorType, float] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'predicted_behaviors': {behavior.value: score for behavior, score in self.predicted_behaviors.items()},
            'confidence': round(self.confidence, 3),
            'evidence': self.evidence,
            'risks': self.risks,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class RuntimeBehaviorPredictor:
    """
    Predictor for runtime behavior of code entities.
    
    Analyzes code patterns to predict potential runtime behaviors,
    performance issues, and security concerns without execution.
    """
    
    def __init__(self):
        """Initialize the behavior predictor."""
        self.logger = logging.getLogger(__name__)
        self._initialize_behavior_patterns()
    
    def _initialize_behavior_patterns(self):
        """Initialize behavior detection patterns."""
        self.behavior_patterns = {
            BehaviorType.ASYNC_OPERATION: {
                'keywords': ['async', 'await', 'promise', 'then', 'catch', 'finally'],
                'patterns': [
                    r'async\s+function', r'await\s+\w+', r'Promise\.', r'\.then\s*\(',
                    r'setTimeout\s*\(', r'setInterval\s*\(', r'fetch\s*\('
                ],
                'context': ['asynchronous', 'non-blocking', 'concurrent']
            },
            BehaviorType.ERROR_PRONE: {
                'keywords': ['throw', 'catch', 'error', 'exception', 'try', 'finally'],
                'patterns': [
                    r'throw\s+new\s+\w*Error', r'try\s*\{', r'catch\s*\(', r'\.catch\s*\(',
                    r'if\s*\(\s*!\w+\)\s*throw', r'assert\s*\('
                ],
                'context': ['error', 'exception', 'failure', 'invalid']
            },
            BehaviorType.PERFORMANCE_BOTTLENECK: {
                'keywords': ['loop', 'for', 'while', 'foreach', 'map', 'filter', 'reduce'],
                'patterns': [
                    r'for\s*\([^)]*\)\s*\{', r'while\s*\([^)]*\)\s*\{', r'\.forEach\s*\(',
                    r'\.map\s*\(', r'\.filter\s*\(', r'\.reduce\s*\('
                ],
                'context': ['iteration', 'processing', 'calculation']
            },
            BehaviorType.SIDE_EFFECT: {
                'keywords': ['console', 'log', 'print', 'alert', 'confirm', 'prompt'],
                'patterns': [
                    r'console\.\w+', r'print\s*\(', r'alert\s*\(', r'confirm\s*\(',
                    r'window\.', r'document\.', r'global\.'
                ],
                'context': ['output', 'interaction', 'global']
            },
            BehaviorType.MEMORY_INTENSIVE: {
                'keywords': ['new', 'Array', 'Object', 'Buffer', 'allocate', 'malloc'],
                'patterns': [
                    r'new\s+Array\s*\(', r'new\s+Object\s*\(', r'new\s+Buffer\s*\(',
                    r'Array\s*\(\s*\d+', r'\.push\s*\(', r'\.concat\s*\('
                ],
                'context': ['allocation', 'memory', 'buffer']
            },
            BehaviorType.NETWORK_OPERATION: {
                'keywords': ['http', 'fetch', 'request', 'response', 'ajax', 'xhr'],
                'patterns': [
                    r'fetch\s*\(', r'XMLHttpRequest', r'\.get\s*\(', r'\.post\s*\(',
                    r'http\.', r'https\.', r'ajax\s*\('
                ],
                'context': ['network', 'api', 'http', 'request']
            },
            BehaviorType.FILE_OPERATION: {
                'keywords': ['read', 'write', 'file', 'fs', 'path', 'stream'],
                'patterns': [
                    r'fs\.\w+', r'readFile\s*\(', r'writeFile\s*\(', r'createReadStream',
                    r'createWriteStream', r'path\.'
                ],
                'context': ['file', 'filesystem', 'path', 'stream']
            },
            BehaviorType.DATABASE_OPERATION: {
                'keywords': ['sql', 'query', 'select', 'insert', 'update', 'delete'],
                'patterns': [
                    r'SELECT\s+', r'INSERT\s+INTO', r'UPDATE\s+', r'DELETE\s+FROM',
                    r'\.query\s*\(', r'\.execute\s*\(', r'\.find\s*\('
                ],
                'context': ['database', 'sql', 'query', 'table']
            },
            BehaviorType.CACHING_OPERATION: {
                'keywords': ['cache', 'memoize', 'store', 'get', 'set', 'redis'],
                'patterns': [
                    r'cache\.\w+', r'memoize\s*\(', r'\.get\s*\(', r'\.set\s*\(',
                    r'redis\.', r'localStorage\.', r'sessionStorage\.'
                ],
                'context': ['cache', 'storage', 'memoization']
            },
            BehaviorType.LOGGING_OPERATION: {
                'keywords': ['log', 'debug', 'info', 'warn', 'error', 'trace'],
                'patterns': [
                    r'console\.log', r'console\.debug', r'console\.info', r'console\.warn',
                    r'console\.error', r'logger\.\w+', r'log\.\w+'
                ],
                'context': ['logging', 'debug', 'trace']
            },
            BehaviorType.SECURITY_SENSITIVE: {
                'keywords': ['password', 'token', 'auth', 'encrypt', 'decrypt', 'hash'],
                'patterns': [
                    r'password\s*=', r'token\s*=', r'encrypt\s*\(', r'decrypt\s*\(',
                    r'hash\s*\(', r'bcrypt\.', r'jwt\.', r'oauth\.'
                ],
                'context': ['security', 'authentication', 'authorization']
            },
            BehaviorType.THREAD_SAFE: {
                'keywords': ['lock', 'mutex', 'semaphore', 'synchronized', 'atomic'],
                'patterns': [
                    r'lock\s*\(', r'mutex\s*\(', r'synchronized\s+', r'atomic\s+',
                    r'\.lock\s*\(', r'\.unlock\s*\('
                ],
                'context': ['concurrency', 'threading', 'synchronization']
            },
            BehaviorType.BLOCKING_OPERATION: {
                'keywords': ['sleep', 'wait', 'block', 'sync', 'synchronous'],
                'patterns': [
                    r'sleep\s*\(', r'wait\s*\(', r'block\s*\(', r'sync\s+',
                    r'Thread\.sleep', r'\.join\s*\('
                ],
                'context': ['blocking', 'waiting', 'synchronous']
            },
            BehaviorType.RECURSIVE: {
                'keywords': ['recursive', 'recursion', 'self', 'this'],
                'patterns': [
                    r'function\s+\w+.*\w+\s*\(', r'return\s+\w+\s*\(', r'this\.\w+\s*\('
                ],
                'context': ['recursive', 'self-calling']
            },
            BehaviorType.INFINITE_LOOP_RISK: {
                'keywords': ['while', 'for', 'loop', 'infinite'],
                'patterns': [
                    r'while\s*\(\s*true\s*\)', r'for\s*\(\s*;\s*;\s*\)', r'while\s*\(\s*1\s*\)',
                    r'for\s*\(\s*;\s*true\s*;\s*\)'
                ],
                'context': ['infinite', 'endless', 'loop']
            },
            BehaviorType.RESOURCE_LEAK: {
                'keywords': ['open', 'close', 'connect', 'disconnect', 'alloc', 'free'],
                'patterns': [
                    r'open\s*\(', r'close\s*\(', r'connect\s*\(', r'disconnect\s*\(',
                    r'alloc\s*\(', r'free\s*\(', r'\.open\s*\(', r'\.close\s*\('
                ],
                'context': ['resource', 'connection', 'allocation']
            }
        }
    
    def predict_behavior(self, node: Node, code: str = "", 
                        call_graph: Dict[str, List[str]] = None) -> BehaviorPrediction:
        """
        Predict runtime behavior of a code entity.
        
        Args:
            node: Node representing the entity
            code: Code content of the entity
            call_graph: Call graph for context
            
        Returns:
            BehaviorPrediction with predicted behaviors
        """
        if not code:
            code = self._extract_code_from_node(node)
        
        # Analyze code for behavior patterns
        behavior_scores = self._analyze_behavior_patterns(code, node.name)
        
        # Analyze call graph for additional context
        if call_graph:
            behavior_scores = self._enhance_with_call_graph(
                behavior_scores, node.name, call_graph
            )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(behavior_scores)
        
        # Generate evidence
        evidence = self._generate_evidence(code, behavior_scores)
        
        # Identify risks
        risks = self._identify_risks(behavior_scores, code)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(behavior_scores, risks)
        
        return BehaviorPrediction(
            entity_name=node.name,
            entity_type=node.type.value if hasattr(node.type, 'value') else str(node.type),
            predicted_behaviors=behavior_scores,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
            recommendations=recommendations,
            metadata={
                'code_length': len(code),
                'pattern_matches': len([score for score in behavior_scores.values() if score > 0])
            }
        )
    
    def _extract_code_from_node(self, node: Node) -> str:
        """Extract code content from node (placeholder implementation)."""
        # In practice, this would extract the actual code content
        # For now, return a placeholder
        return f"// Code for {node.name}"
    
    def _analyze_behavior_patterns(self, code: str, entity_name: str) -> Dict[BehaviorType, float]:
        """Analyze code for behavior patterns."""
        behavior_scores = {}
        code_lower = code.lower()
        
        for behavior_type, patterns in self.behavior_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in code_lower)
            score += min(keyword_matches * 0.2, 1.0)
            
            # Check regex patterns
            pattern_matches = sum(1 for pattern in patterns['patterns'] 
                                if re.search(pattern, code, re.IGNORECASE))
            score += min(pattern_matches * 0.3, 1.0)
            
            # Check context
            context_matches = sum(1 for ctx in patterns['context'] 
                                if ctx in code_lower)
            score += min(context_matches * 0.1, 1.0)
            
            # Entity name bonus
            if any(keyword in entity_name.lower() for keyword in patterns['keywords']):
                score += 0.2
            
            behavior_scores[behavior_type] = min(score, 1.0)
        
        return behavior_scores
    
    def _enhance_with_call_graph(self, behavior_scores: Dict[BehaviorType, float],
                               entity_name: str, call_graph: Dict[str, List[str]]) -> Dict[BehaviorType, float]:
        """Enhance behavior scores using call graph context."""
        enhanced_scores = behavior_scores.copy()
        
        # Get functions called by this entity
        called_functions = call_graph.get(entity_name, [])
        
        # Analyze called functions for additional behaviors
        for called_func in called_functions:
            # This is a simplified analysis - in practice, you'd analyze the called function
            if 'async' in called_func.lower():
                enhanced_scores[BehaviorType.ASYNC_OPERATION] = min(
                    enhanced_scores.get(BehaviorType.ASYNC_OPERATION, 0) + 0.1, 1.0
                )
            if 'error' in called_func.lower() or 'exception' in called_func.lower():
                enhanced_scores[BehaviorType.ERROR_PRONE] = min(
                    enhanced_scores.get(BehaviorType.ERROR_PRONE, 0) + 0.1, 1.0
                )
        
        return enhanced_scores
    
    def _calculate_confidence(self, behavior_scores: Dict[BehaviorType, float]) -> float:
        """Calculate overall confidence in predictions."""
        if not behavior_scores:
            return 0.0
        
        # Confidence based on number of high-scoring behaviors
        high_confidence_behaviors = len([score for score in behavior_scores.values() if score > 0.5])
        total_behaviors = len(behavior_scores)
        
        if total_behaviors == 0:
            return 0.0
        
        # Base confidence on proportion of high-confidence behaviors
        base_confidence = high_confidence_behaviors / total_behaviors
        
        # Boost confidence if we have very high scores
        max_score = max(behavior_scores.values()) if behavior_scores else 0
        score_boost = max_score * 0.3
        
        return min(base_confidence + score_boost, 1.0)
    
    def _generate_evidence(self, code: str, behavior_scores: Dict[BehaviorType, float]) -> List[str]:
        """Generate evidence for behavior predictions."""
        evidence = []
        
        for behavior_type, score in behavior_scores.items():
            if score > 0.3:  # Only include behaviors with reasonable confidence
                patterns = self.behavior_patterns.get(behavior_type, {})
                
                # Find specific evidence in code
                found_keywords = []
                for keyword in patterns.get('keywords', []):
                    if keyword in code.lower():
                        found_keywords.append(keyword)
                
                if found_keywords:
                    evidence.append(f"{behavior_type.value}: Found keywords {', '.join(found_keywords[:3])}")
                
                # Find pattern matches
                found_patterns = []
                for pattern in patterns.get('patterns', []):
                    matches = re.findall(pattern, code, re.IGNORECASE)
                    if matches:
                        found_patterns.extend(matches[:2])
                
                if found_patterns:
                    evidence.append(f"{behavior_type.value}: Found patterns {', '.join(found_patterns[:2])}")
        
        return evidence
    
    def _identify_risks(self, behavior_scores: Dict[BehaviorType, float], code: str) -> List[str]:
        """Identify potential risks based on predicted behaviors."""
        risks = []
        
        # High-risk behaviors
        if behavior_scores.get(BehaviorType.INFINITE_LOOP_RISK, 0) > 0.5:
            risks.append("Potential infinite loop detected")
        
        if behavior_scores.get(BehaviorType.RESOURCE_LEAK, 0) > 0.5:
            risks.append("Potential resource leak - ensure proper cleanup")
        
        if behavior_scores.get(BehaviorType.MEMORY_INTENSIVE, 0) > 0.7:
            risks.append("Memory-intensive operation - monitor memory usage")
        
        if behavior_scores.get(BehaviorType.SECURITY_SENSITIVE, 0) > 0.6:
            risks.append("Security-sensitive operation - ensure proper validation")
        
        if behavior_scores.get(BehaviorType.ERROR_PRONE, 0) > 0.6:
            risks.append("Error-prone operation - ensure proper error handling")
        
        # Performance risks
        if behavior_scores.get(BehaviorType.PERFORMANCE_BOTTLENECK, 0) > 0.6:
            risks.append("Potential performance bottleneck - consider optimization")
        
        if behavior_scores.get(BehaviorType.BLOCKING_OPERATION, 0) > 0.5:
            risks.append("Blocking operation - consider async alternatives")
        
        return risks
    
    def _generate_recommendations(self, behavior_scores: Dict[BehaviorType, float], 
                                risks: List[str]) -> List[str]:
        """Generate recommendations for improving code."""
        recommendations = []
        
        # Address specific behaviors
        if behavior_scores.get(BehaviorType.ASYNC_OPERATION, 0) > 0.5:
            recommendations.append("Consider using async/await for better error handling")
        
        if behavior_scores.get(BehaviorType.ERROR_PRONE, 0) > 0.5:
            recommendations.append("Add comprehensive error handling and validation")
        
        if behavior_scores.get(BehaviorType.PERFORMANCE_BOTTLENECK, 0) > 0.5:
            recommendations.append("Consider optimizing loops and data processing")
        
        if behavior_scores.get(BehaviorType.MEMORY_INTENSIVE, 0) > 0.5:
            recommendations.append("Monitor memory usage and consider streaming for large data")
        
        if behavior_scores.get(BehaviorType.SECURITY_SENSITIVE, 0) > 0.5:
            recommendations.append("Implement proper security measures and input validation")
        
        if behavior_scores.get(BehaviorType.RESOURCE_LEAK, 0) > 0.5:
            recommendations.append("Ensure proper resource cleanup in finally blocks")
        
        # General recommendations
        if len(risks) > 3:
            recommendations.append("Consider refactoring to reduce complexity and risks")
        
        return recommendations
    
    def predict_multiple_entities(self, entities: List[Tuple[Node, str]], 
                                call_graph: Dict[str, List[str]] = None) -> List[BehaviorPrediction]:
        """
        Predict behavior for multiple entities.
        
        Args:
            entities: List of (node, code) tuples
            call_graph: Call graph for context
            
        Returns:
            List of BehaviorPrediction objects
        """
        predictions = []
        
        for node, code in entities:
            prediction = self.predict_behavior(node, code, call_graph)
            predictions.append(prediction)
        
        return predictions
    
    def get_behavior_statistics(self, predictions: List[BehaviorPrediction]) -> Dict[str, Any]:
        """Get statistics about behavior predictions."""
        if not predictions:
            return {}
        
        # Count behaviors
        behavior_counts = defaultdict(int)
        confidence_scores = []
        risk_counts = []
        
        for prediction in predictions:
            confidence_scores.append(prediction.confidence)
            risk_counts.append(len(prediction.risks))
            
            for behavior_type, score in prediction.predicted_behaviors.items():
                if score > 0.5:  # Only count high-confidence behaviors
                    behavior_counts[behavior_type.value] += 1
        
        return {
            'total_entities': len(predictions),
            'average_confidence': round(sum(confidence_scores) / len(confidence_scores), 3),
            'behavior_distribution': dict(behavior_counts),
            'average_risks_per_entity': round(sum(risk_counts) / len(risk_counts), 2),
            'high_risk_entities': len([p for p in predictions if len(p.risks) > 2]),
            'low_confidence_entities': len([p for p in predictions if p.confidence < 0.3])
        }

