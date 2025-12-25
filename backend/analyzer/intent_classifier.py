"""
Function Intent Classifier for Semantic Analysis

This module provides intent classification for code functions, enabling
understanding of what functions do beyond their syntax. It identifies
patterns like data fetching, validation, authentication, etc.

Classes:
    FunctionIntentClassifier: Main classifier for function intents
    IntentResult: Result of intent classification
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# ML imports with fallback handling
try:
    from transformers import pipeline
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class IntentType(Enum):
    """Types of function intents."""
    DATA_FETCHING = "data_fetching"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ERROR_HANDLING = "error_handling"
    UI_RENDERING = "ui_rendering"
    BUSINESS_LOGIC = "business_logic"
    UTILITY = "utility"
    CONFIGURATION = "configuration"
    LOGGING = "logging"
    CACHING = "caching"
    NETWORK = "network"
    FILE_OPERATIONS = "file_operations"
    DATABASE = "database"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    function_name: str
    primary_intent: IntentType
    confidence: float
    all_intents: Dict[IntentType, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    patterns_found: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'primary_intent': self.primary_intent.value,
            'confidence': round(self.confidence, 3),
            'all_intents': {intent.value: score for intent, score in self.all_intents.items()},
            'evidence': self.evidence,
            'patterns_found': self.patterns_found,
            'metadata': self.metadata
        }


class FunctionIntentClassifier:
    """
    Classifier for determining function intents and purposes.
    
    Uses pattern matching and ML models to understand what functions do,
    enabling better semantic analysis and relationship detection.
    """
    
    def __init__(self, use_ml: bool = True):
        """
        Initialize the intent classifier.
        
        Args:
            use_ml: Whether to use ML models (requires transformers)
        """
        self.logger = logging.getLogger(__name__)
        self.use_ml = use_ml and ML_AVAILABLE
        
        # Initialize pattern-based classifier
        self._initialize_patterns()
        
        # Initialize ML model if available
        if self.use_ml:
            self._initialize_ml_model()
        else:
            self.logger.info("Running in pattern-only mode")
    
    def _initialize_patterns(self):
        """Initialize pattern-based classification rules."""
        self.intent_patterns = {
            IntentType.DATA_FETCHING: {
                'keywords': ['fetch', 'get', 'load', 'retrieve', 'query', 'find', 'search', 'fetchdata', 'getdata'],
                'patterns': [r'fetch.*data', r'get.*from', r'load.*from', r'query.*database', r'api.*get'],
                'context': ['http', 'api', 'database', 'repository', 'service']
            },
            IntentType.DATA_PROCESSING: {
                'keywords': ['process', 'transform', 'convert', 'parse', 'format', 'map', 'filter', 'reduce', 'calculate'],
                'patterns': [r'process.*data', r'transform.*to', r'convert.*from', r'parse.*json', r'format.*string'],
                'context': ['data', 'array', 'object', 'json', 'xml']
            },
            IntentType.VALIDATION: {
                'keywords': ['validate', 'check', 'verify', 'ensure', 'assert', 'isvalid', 'validateinput'],
                'patterns': [r'validate.*input', r'check.*valid', r'verify.*data', r'ensure.*correct'],
                'context': ['input', 'data', 'form', 'parameter', 'argument']
            },
            IntentType.AUTHENTICATION: {
                'keywords': ['auth', 'login', 'authenticate', 'signin', 'signin', 'loginuser', 'checkauth'],
                'patterns': [r'login.*user', r'auth.*token', r'signin.*with', r'authenticate.*user'],
                'context': ['user', 'password', 'token', 'session', 'credential']
            },
            IntentType.AUTHORIZATION: {
                'keywords': ['authorize', 'permission', 'access', 'role', 'privilege', 'checkpermission'],
                'patterns': [r'check.*permission', r'authorize.*access', r'has.*role', r'can.*access'],
                'context': ['permission', 'role', 'access', 'privilege', 'authorization']
            },
            IntentType.ERROR_HANDLING: {
                'keywords': ['handle', 'catch', 'error', 'exception', 'try', 'catch', 'throw', 'errorhandler'],
                'patterns': [r'handle.*error', r'catch.*exception', r'throw.*error', r'try.*catch'],
                'context': ['error', 'exception', 'failure', 'invalid', 'error']
            },
            IntentType.UI_RENDERING: {
                'keywords': ['render', 'display', 'show', 'draw', 'paint', 'rendercomponent', 'displayui'],
                'patterns': [r'render.*component', r'display.*ui', r'show.*page', r'draw.*element'],
                'context': ['component', 'ui', 'view', 'page', 'element', 'dom']
            },
            IntentType.BUSINESS_LOGIC: {
                'keywords': ['calculate', 'compute', 'execute', 'run', 'process', 'business', 'logic', 'algorithm'],
                'patterns': [r'calculate.*total', r'compute.*result', r'execute.*business', r'run.*algorithm'],
                'context': ['business', 'logic', 'rule', 'algorithm', 'calculation']
            },
            IntentType.UTILITY: {
                'keywords': ['util', 'helper', 'common', 'shared', 'tool', 'utility', 'helperfunction'],
                'patterns': [r'util.*function', r'helper.*method', r'common.*tool', r'shared.*utility'],
                'context': ['util', 'helper', 'common', 'shared', 'tool']
            },
            IntentType.CONFIGURATION: {
                'keywords': ['config', 'setup', 'init', 'initialize', 'configure', 'settings', 'setupconfig'],
                'patterns': [r'config.*setup', r'init.*config', r'setup.*environment', r'configure.*app'],
                'context': ['config', 'setting', 'environment', 'initialization']
            },
            IntentType.LOGGING: {
                'keywords': ['log', 'logger', 'debug', 'info', 'warn', 'error', 'trace', 'logmessage'],
                'patterns': [r'log.*message', r'debug.*info', r'logger.*debug', r'console.*log'],
                'context': ['log', 'debug', 'info', 'warning', 'error', 'trace']
            },
            IntentType.CACHING: {
                'keywords': ['cache', 'memoize', 'store', 'buffer', 'cache', 'memoization'],
                'patterns': [r'cache.*data', r'memoize.*function', r'store.*cache', r'buffer.*data'],
                'context': ['cache', 'memoization', 'buffer', 'storage']
            },
            IntentType.NETWORK: {
                'keywords': ['http', 'request', 'response', 'api', 'endpoint', 'network', 'socket'],
                'patterns': [r'http.*request', r'api.*call', r'network.*request', r'socket.*connection'],
                'context': ['http', 'api', 'network', 'request', 'response', 'endpoint']
            },
            IntentType.FILE_OPERATIONS: {
                'keywords': ['file', 'read', 'write', 'save', 'load', 'upload', 'download', 'fileoperation'],
                'patterns': [r'read.*file', r'write.*file', r'save.*data', r'upload.*file'],
                'context': ['file', 'path', 'directory', 'upload', 'download']
            },
            IntentType.DATABASE: {
                'keywords': ['database', 'db', 'sql', 'query', 'insert', 'update', 'delete', 'select'],
                'patterns': [r'database.*query', r'sql.*statement', r'insert.*into', r'update.*set'],
                'context': ['database', 'sql', 'table', 'record', 'query']
            }
        }
    
    def _initialize_ml_model(self):
        """Initialize ML model for intent classification."""
        try:
            # Use a lightweight model for text classification
            self.ml_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                return_all_scores=True
            )
            self.logger.info("ML model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ML model: {e}")
            self.use_ml = False
    
    def classify_function_intent(self, function_code: str, function_name: str = "", 
                               context: str = "") -> IntentResult:
        """
        Classify the intent of a function.
        
        Args:
            function_code: The function code
            function_name: Name of the function
            context: Surrounding context (class, module, etc.)
            
        Returns:
            IntentResult with classification results
        """
        # Combine all text for analysis
        full_text = f"{function_name} {context} {function_code}".lower()
        
        # Pattern-based classification
        pattern_scores = self._pattern_based_classification(full_text, function_name)
        
        # ML-based classification (if available)
        ml_scores = {}
        if self.use_ml:
            ml_scores = self._ml_based_classification(full_text)
        
        # Combine scores
        combined_scores = self._combine_scores(pattern_scores, ml_scores)
        
        # Find primary intent
        primary_intent = max(combined_scores.items(), key=lambda x: x[1])
        
        # Generate evidence
        evidence = self._generate_evidence(full_text, primary_intent[0])
        
        # Find patterns
        patterns_found = self._find_patterns(full_text, primary_intent[0])
        
        return IntentResult(
            function_name=function_name,
            primary_intent=primary_intent[0],
            confidence=primary_intent[1],
            all_intents=combined_scores,
            evidence=evidence,
            patterns_found=patterns_found,
            metadata={
                'pattern_scores': {intent.value: score for intent, score in pattern_scores.items()},
                'ml_scores': ml_scores,
                'text_length': len(full_text)
            }
        )
    
    def _pattern_based_classification(self, text: str, function_name: str) -> Dict[IntentType, float]:
        """Classify based on naming patterns and keywords."""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in text)
            score += min(keyword_matches * 0.3, 1.0)
            
            # Check regex patterns
            pattern_matches = sum(1 for pattern in patterns['patterns'] 
                                if re.search(pattern, text, re.IGNORECASE))
            score += min(pattern_matches * 0.4, 1.0)
            
            # Check context
            context_matches = sum(1 for ctx in patterns['context'] if ctx in text)
            score += min(context_matches * 0.2, 1.0)
            
            # Function name bonus
            if any(keyword in function_name.lower() for keyword in patterns['keywords']):
                score += 0.3
            
            scores[intent] = min(score, 1.0)
        
        return scores
    
    def _ml_based_classification(self, text: str) -> Dict[str, float]:
        """Classify using ML model."""
        if not self.use_ml or not self.ml_classifier:
            return {}
        
        try:
            # Truncate text for model input
            truncated_text = text[:512]
            
            # Get predictions
            predictions = self.ml_classifier(truncated_text)
            
            # Convert to intent scores
            ml_scores = {}
            for prediction in predictions:
                label = prediction['label'].lower()
                score = prediction['score']
                
                # Map model labels to our intent types
                intent_mapping = {
                    'positive': IntentType.BUSINESS_LOGIC,
                    'negative': IntentType.ERROR_HANDLING,
                    'neutral': IntentType.UTILITY
                }
                
                for model_label, intent_type in intent_mapping.items():
                    if model_label in label:
                        ml_scores[intent_type.value] = score
                        break
            
            return ml_scores
            
        except Exception as e:
            self.logger.error(f"ML classification failed: {e}")
            return {}
    
    def _combine_scores(self, pattern_scores: Dict[IntentType, float], 
                       ml_scores: Dict[str, float]) -> Dict[IntentType, float]:
        """Combine pattern and ML scores."""
        combined = {}
        
        # Start with pattern scores
        for intent, score in pattern_scores.items():
            combined[intent] = score
        
        # Add ML scores with weight
        ml_weight = 0.3  # ML contributes 30% to final score
        for intent_str, score in ml_scores.items():
            try:
                intent = IntentType(intent_str)
                if intent in combined:
                    combined[intent] = combined[intent] * (1 - ml_weight) + score * ml_weight
                else:
                    combined[intent] = score * ml_weight
            except ValueError:
                # Unknown intent type from ML
                continue
        
        return combined
    
    def _generate_evidence(self, text: str, primary_intent: IntentType) -> List[str]:
        """Generate evidence for the classification."""
        evidence = []
        
        if primary_intent in self.intent_patterns:
            patterns = self.intent_patterns[primary_intent]
            
            # Find matching keywords
            found_keywords = [kw for kw in patterns['keywords'] if kw in text]
            if found_keywords:
                evidence.append(f"Keywords found: {', '.join(found_keywords[:3])}")
            
            # Find matching patterns
            found_patterns = []
            for pattern in patterns['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    found_patterns.append(pattern)
            if found_patterns:
                evidence.append(f"Patterns matched: {', '.join(found_patterns[:2])}")
            
            # Find context
            found_context = [ctx for ctx in patterns['context'] if ctx in text]
            if found_context:
                evidence.append(f"Context found: {', '.join(found_context[:2])}")
        
        return evidence
    
    def _find_patterns(self, text: str, primary_intent: IntentType) -> List[str]:
        """Find specific patterns in the text."""
        patterns_found = []
        
        if primary_intent in self.intent_patterns:
            for pattern in self.intent_patterns[primary_intent]['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    patterns_found.extend(matches[:2])  # Limit to 2 matches
        
        return patterns_found
    
    def classify_multiple_functions(self, functions: List[Dict[str, str]]) -> List[IntentResult]:
        """
        Classify multiple functions at once.
        
        Args:
            functions: List of dicts with 'name', 'code', 'context' keys
            
        Returns:
            List of IntentResult objects
        """
        results = []
        for func in functions:
            result = self.classify_function_intent(
                func.get('code', ''),
                func.get('name', ''),
                func.get('context', '')
            )
            results.append(result)
        
        return results
    
    def get_intent_statistics(self, results: List[IntentResult]) -> Dict[str, Any]:
        """
        Get statistics about intent classifications.
        
        Args:
            results: List of IntentResult objects
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        # Count intents
        intent_counts = {}
        confidence_scores = []
        
        for result in results:
            intent = result.primary_intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            confidence_scores.append(result.confidence)
        
        # Calculate averages
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'total_functions': len(results),
            'intent_distribution': intent_counts,
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_count': len([r for r in results if r.confidence > 0.7]),
            'low_confidence_count': len([r for r in results if r.confidence < 0.3])
        }

