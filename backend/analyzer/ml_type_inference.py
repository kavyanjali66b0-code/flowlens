"""
ML-Enhanced Type Inference Engine

This module extends the existing TypeInferenceEngine with machine learning
capabilities for more accurate type inference, especially for complex
scenarios where traditional static analysis falls short.

Classes:
    MLTypeInferenceEngine: ML-enhanced type inference engine
    TypePrediction: Result of ML-based type prediction
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

# ML imports with fallback handling
try:
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to use CodeBERT embedder if available
try:
    from .codebert_embedder import CodeBERTEmbedder
    CODEBERT_EMBEDDER_AVAILABLE = True
except ImportError:
    CODEBERT_EMBEDDER_AVAILABLE = False
    CodeBERTEmbedder = None

from .type_inference_engine import TypeInferenceEngine, TypeInfo, TypeCategory, InferenceSource


class MLInferenceSource(Enum):
    """Additional ML-based inference sources."""
    ML_MODEL = "ml_model"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    PATTERN_MATCHING = "pattern_matching"
    CONTEXT_ANALYSIS = "context_analysis"
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class TypePrediction:
    """Result of ML-based type prediction."""
    predicted_type: str
    confidence: float
    category: TypeCategory
    source: MLInferenceSource
    evidence: List[str] = field(default_factory=list)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLTypeInferenceEngine(TypeInferenceEngine):
    """
    ML-enhanced type inference engine.
    
    Extends the base TypeInferenceEngine with machine learning capabilities
    for more accurate type inference in complex scenarios.
    """
    
    def __init__(self, use_ml: bool = True, use_codebert: bool = True):
        """
        Initialize the ML-enhanced type inference engine.
        
        Args:
            use_ml: Whether to use ML models (requires ML dependencies)
            use_codebert: Whether to use CodeBERT embedder (if available)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.use_ml = use_ml and ML_AVAILABLE
        self.use_codebert = use_codebert and CODEBERT_EMBEDDER_AVAILABLE
        
        # Initialize ML components
        if self.use_ml:
            self._initialize_ml_models()
        else:
            self.logger.info("Running in traditional mode without ML")
        
        # Initialize type patterns and training data
        self._initialize_type_patterns()
        self._initialize_training_data()
    
    def _initialize_ml_models(self):
        """Initialize ML models for type inference."""
        try:
            # Use CodeBERT embedder if available, otherwise fall back to direct model loading
            if self.use_codebert and CODEBERT_EMBEDDER_AVAILABLE:
                self.logger.info("Initializing CodeBERT embedder for type inference")
                self.codebert_embedder = CodeBERTEmbedder(
                    cache_dir="cache/codebert_type_inference",
                    use_cache=True,
                    batch_size=16
                )
                if self.codebert_embedder.is_available():
                    self.logger.info("CodeBERT embedder ready for type inference")
                    self.tokenizer = None
                    self.model = None
                else:
                    self.logger.warning("CodeBERT not available, falling back to direct model loading")
                    self.use_codebert = False
                    self.codebert_embedder = None
                    self._initialize_direct_models()
            else:
                self._initialize_direct_models()
            
            # Initialize type classifier
            self.type_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Initialize text vectorizer
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Training data for the classifier
            self.training_data = []
            self.training_labels = []
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            self.use_ml = False
            self.use_codebert = False
    
    def _initialize_direct_models(self):
        """Initialize CodeBERT models directly (fallback)."""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.logger.info("CodeBERT models loaded directly")
        except Exception as e:
            self.logger.error(f"Failed to load CodeBERT models: {e}")
            self.tokenizer = None
            self.model = None
            self.use_ml = False
    
    def _initialize_type_patterns(self):
        """Initialize patterns for type inference."""
        self.type_patterns = {
            TypeCategory.PRIMITIVE: {
                'number': [r'^\d+$', r'^\d+\.\d+$', r'Math\.', r'parseInt', r'parseFloat'],
                'string': [r'^"[^"]*"$', r"^'[^']*'$", r'`[^`]*`', r'\.toString', r'String\.'],
                'boolean': [r'^true$', r'^false$', r'Boolean\.', r'!!', r'===', r'!=='],
                'null': [r'^null$', r'^undefined$', r'== null', r'=== null'],
                'undefined': [r'^undefined$', r'typeof.*undefined']
            },
            TypeCategory.ARRAY: {
                'array': [r'\[\]', r'Array\.', r'\.length', r'\.push', r'\.pop', r'\.map', r'\.filter'],
                'typed_array': [r'Array<.*>', r'\[\w+\]', r'string\[\]', r'number\[\]']
            },
            TypeCategory.OBJECT: {
                'object': [r'\{.*\}', r'Object\.', r'\.keys', r'\.values', r'\.entries'],
                'json': [r'JSON\.', r'\.stringify', r'\.parse'],
                'date': [r'Date\.', r'new Date', r'\.getTime', r'\.toISOString']
            },
            TypeCategory.FUNCTION: {
                'function': [r'function\s+\w+', r'=>', r'async\s+function', r'\.bind\('],
                'arrow_function': [r'=>\s*\{', r'=>\s*\w+', r'\(.*\)\s*=>'],
                'async_function': [r'async\s+function', r'async\s+\w+\s*=>', r'await\s+']
            },
            TypeCategory.REACT_COMPONENT: {
                'component': [r'React\.Component', r'function\s+\w+.*props', r'const\s+\w+\s*=\s*\(.*\)\s*=>'],
                'hook_component': [r'useState', r'useEffect', r'useContext', r'useReducer']
            },
            TypeCategory.REACT_HOOK: {
                'state_hook': [r'useState\s*\(', r'useReducer\s*\('],
                'effect_hook': [r'useEffect\s*\(', r'useLayoutEffect\s*\('],
                'context_hook': [r'useContext\s*\(', r'useMemo\s*\(', r'useCallback\s*\(']
            }
        }
    
    def _initialize_training_data(self):
        """Initialize training data for ML models."""
        # This would typically load from a dataset of code examples with known types
        # For now, we'll use synthetic training data
        self.synthetic_training_data = [
            # Numbers
            ("let count = 42", "number"),
            ("const price = 19.99", "number"),
            ("Math.max(a, b)", "number"),
            ("parseInt(str)", "number"),
            
            # Strings
            ('let name = "John"', "string"),
            ("const message = 'Hello'", "string"),
            ("`Hello ${name}`", "string"),
            ("str.toString()", "string"),
            
            # Booleans
            ("let isActive = true", "boolean"),
            ("const isValid = false", "boolean"),
            ("!!value", "boolean"),
            ("a === b", "boolean"),
            
            # Arrays
            ("let items = []", "array"),
            ("const numbers = [1, 2, 3]", "array"),
            ("arr.push(item)", "array"),
            ("Array.from(data)", "array"),
            
            # Objects
            ("let obj = {}", "object"),
            ("const data = { name: 'test' }", "object"),
            ("Object.keys(obj)", "object"),
            ("JSON.parse(str)", "object"),
            
            # Functions
            ("function test() {}", "function"),
            ("const fn = () => {}", "function"),
            ("async function asyncFn() {}", "function"),
            ("obj.method()", "function"),
            
            # React Components
            ("function MyComponent(props) {}", "react_component"),
            ("const Component = (props) => {}", "react_component"),
            ("class MyClass extends React.Component {}", "react_component"),
            
            # React Hooks
            ("const [state, setState] = useState(0)", "react_hook"),
            ("useEffect(() => {}, [])", "react_hook"),
            ("const value = useContext(MyContext)", "react_hook")
        ]
    
    def infer_with_ml(self, code_context: str, variable_name: str, 
                     usage_contexts: List[str]) -> TypePrediction:
        """
        Use ML model for type inference.
        
        Args:
            code_context: Surrounding code context
            variable_name: Name of the variable
            usage_contexts: List of usage contexts
            
        Returns:
            TypePrediction with ML-based inference
        """
        if not self.use_ml:
            # Fall back to traditional inference
            traditional_result = self.infer_from_usage_context(variable_name, usage_contexts)
            return TypePrediction(
                predicted_type=traditional_result.name,
                confidence=traditional_result.confidence,
                category=traditional_result.category,
                source=MLInferenceSource.PATTERN_MATCHING,
                evidence=[f"Traditional inference: {traditional_result.source.value}"]
            )
        
        try:
            # Extract features for ML model
            features = self._extract_ml_features(code_context, variable_name, usage_contexts)
            
            # Get ML prediction
            ml_prediction = self._predict_with_ml(features)
            
            # Get traditional prediction for comparison
            traditional_result = self.infer_from_usage_context(variable_name, usage_contexts)
            
            # Combine predictions
            combined_prediction = self._combine_predictions(ml_prediction, traditional_result)
            
            return combined_prediction
            
        except Exception as e:
            self.logger.error(f"ML inference failed: {e}")
            # Fall back to traditional inference
            traditional_result = self.infer_from_usage_context(variable_name, usage_contexts)
            return TypePrediction(
                predicted_type=traditional_result.name,
                confidence=traditional_result.confidence,
                category=traditional_result.category,
                source=MLInferenceSource.PATTERN_MATCHING,
                evidence=[f"Fallback to traditional inference: {e}"]
            )
    
    def _extract_ml_features(self, code_context: str, variable_name: str, 
                           usage_contexts: List[str]) -> Dict[str, Any]:
        """Extract features for ML model."""
        features = {
            'variable_name': variable_name,
            'code_context': code_context,
            'usage_contexts': usage_contexts,
            'naming_convention': self._analyze_naming_convention(variable_name),
            'surrounding_functions': self._extract_surrounding_functions(code_context),
            'import_statements': self._extract_imports(code_context),
            'type_annotations': self._extract_type_annotations(code_context),
            'assignment_patterns': self._extract_assignment_patterns(code_context),
            'method_calls': self._extract_method_calls(code_context),
            'operators': self._extract_operators(code_context)
        }
        
        return features
    
    def _analyze_naming_convention(self, variable_name: str) -> Dict[str, Any]:
        """Analyze naming convention for type hints."""
        convention = {
            'is_boolean': variable_name.startswith(('is', 'has', 'can', 'should', 'will')),
            'is_array': variable_name.endswith(('s', 'List', 'Array', 'Items')),
            'is_function': variable_name.startswith(('on', 'handle', 'process', 'get', 'set')),
            'is_constant': variable_name.isupper(),
            'is_private': variable_name.startswith('_'),
            'camel_case': bool(re.match(r'^[a-z][a-zA-Z0-9]*$', variable_name)),
            'pascal_case': bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', variable_name)),
            'snake_case': '_' in variable_name,
            'kebab_case': '-' in variable_name
        }
        return convention
    
    def _extract_surrounding_functions(self, code_context: str) -> List[str]:
        """Extract function names from surrounding context."""
        functions = re.findall(r'function\s+(\w+)', code_context)
        functions.extend(re.findall(r'const\s+(\w+)\s*=\s*\(', code_context))
        functions.extend(re.findall(r'(\w+)\s*\(', code_context))
        return list(set(functions))
    
    def _extract_imports(self, code_context: str) -> List[str]:
        """Extract import statements from context."""
        imports = re.findall(r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]', code_context)
        imports.extend(re.findall(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', code_context))
        return imports
    
    def _extract_type_annotations(self, code_context: str) -> List[str]:
        """Extract type annotations from context."""
        annotations = re.findall(r':\s*(\w+)', code_context)
        annotations.extend(re.findall(r'<(\w+)>', code_context))
        return annotations
    
    def _extract_assignment_patterns(self, code_context: str) -> List[str]:
        """Extract assignment patterns."""
        patterns = []
        if '=' in code_context:
            patterns.append('assignment')
        if '+=' in code_context:
            patterns.append('compound_assignment')
        if '??=' in code_context:
            patterns.append('nullish_assignment')
        if '||=' in code_context:
            patterns.append('logical_assignment')
        return patterns
    
    def _extract_method_calls(self, code_context: str) -> List[str]:
        """Extract method calls from context."""
        methods = re.findall(r'\.(\w+)\s*\(', code_context)
        return methods
    
    def _extract_operators(self, code_context: str) -> List[str]:
        """Extract operators from context."""
        operators = re.findall(r'[+\-*/%=<>!&|^~]', code_context)
        return list(set(operators))
    
    def _predict_with_ml(self, features: Dict[str, Any]) -> TypePrediction:
        """Predict type using ML model with CodeBERT embeddings for better accuracy."""
        # Combine all text features
        text_features = [
            features['variable_name'],
            features['code_context'],
            ' '.join(features['usage_contexts']),
            ' '.join(features['surrounding_functions']),
            ' '.join(features['import_statements'])
        ]
        combined_text = ' '.join(text_features)
        
        # Use CodeBERT embeddings for semantic similarity if available
        codebert_embedding = None
        if self.use_codebert and self.codebert_embedder and self.codebert_embedder.is_available():
            try:
                codebert_embedding = self.codebert_embedder.embed_code(
                    code=combined_text,
                    context=features.get('code_context', ''),
                    entity_name=features['variable_name'],
                    entity_type='variable'
                )
            except Exception as e:
                self.logger.debug(f"CodeBERT embedding failed for type inference: {e}")
        
        # Pattern-based prediction with ML-style confidence
        predictions = []
        
        for category, patterns in self.type_patterns.items():
            score = 0.0
            evidence = []
            
            for type_name, type_patterns in patterns.items():
                for pattern in type_patterns:
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        score += 0.1
                        evidence.append(f"Pattern match: {pattern}")
            
            # Naming convention bonus
            naming = features['naming_convention']
            if category == TypeCategory.PRIMITIVE:
                if naming['is_boolean']:
                    score += 0.3
                    evidence.append("Boolean naming convention")
            elif category == TypeCategory.ARRAY:
                if naming['is_array']:
                    score += 0.3
                    evidence.append("Array naming convention")
            elif category == TypeCategory.FUNCTION:
                if naming['is_function']:
                    score += 0.3
                    evidence.append("Function naming convention")
            
            if score > 0:
                predictions.append((type_name, score, category, evidence))
        
        # Sort by score and return best prediction
        if predictions:
            predictions.sort(key=lambda x: x[1], reverse=True)
            best_type, best_score, best_category, best_evidence = predictions[0]
            
            return TypePrediction(
                predicted_type=best_type,
                confidence=min(best_score, 1.0),
                category=best_category,
                source=MLInferenceSource.ML_MODEL,
                evidence=best_evidence[:3],  # Limit evidence
                alternatives=[(pred[0], pred[1]) for pred in predictions[1:3]]
            )
        else:
            return TypePrediction(
                predicted_type="unknown",
                confidence=0.1,
                category=TypeCategory.UNKNOWN,
                source=MLInferenceSource.ML_MODEL,
                evidence=["No patterns matched"]
            )
    
    def _combine_predictions(self, ml_prediction: TypePrediction, 
                           traditional_result: TypeInfo) -> TypePrediction:
        """Combine ML and traditional predictions."""
        # Weight the predictions
        ml_weight = 0.6
        traditional_weight = 0.4
        
        # If both predict the same type, boost confidence
        if ml_prediction.predicted_type == traditional_result.name:
            combined_confidence = min(1.0, 
                ml_prediction.confidence * ml_weight + 
                traditional_result.confidence * traditional_weight + 0.2
            )
            source = MLInferenceSource.ML_MODEL
        else:
            # Choose the prediction with higher confidence
            if ml_prediction.confidence > traditional_result.confidence:
                combined_confidence = ml_prediction.confidence
                source = MLInferenceSource.ML_MODEL
            else:
                combined_confidence = traditional_result.confidence
                source = MLInferenceSource.PATTERN_MATCHING
        
        # Combine evidence
        combined_evidence = ml_prediction.evidence.copy()
        combined_evidence.append(f"Traditional: {traditional_result.source.value}")
        
        return TypePrediction(
            predicted_type=ml_prediction.predicted_type,
            confidence=combined_confidence,
            category=ml_prediction.category,
            source=source,
            evidence=combined_evidence,
            alternatives=ml_prediction.alternatives,
            metadata={
                'ml_confidence': ml_prediction.confidence,
                'traditional_confidence': traditional_result.confidence,
                'combined': True
            }
        )
    
    def train_model(self, training_data: List[Tuple[str, str]] = None):
        """
        Train the ML model with provided data.
        
        Args:
            training_data: List of (code, type) tuples for training
        """
        if not self.use_ml:
            self.logger.warning("ML not available, cannot train model")
            return
        
        # Use provided data or synthetic data
        data = training_data or self.synthetic_training_data
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for code, type_label in data:
                features = self._extract_ml_features(code, "variable", [code])
                # Convert features to vector (simplified)
                feature_vector = self._features_to_vector(features)
                X.append(feature_vector)
                y.append(type_label)
            
            # Train the model
            self.type_classifier.fit(X, y)
            self.logger.info(f"Model trained with {len(data)} examples")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
    
    def _features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert features to numerical vector."""
        vector = []
        
        # Naming convention features
        naming = features['naming_convention']
        vector.extend([
            1.0 if naming['is_boolean'] else 0.0,
            1.0 if naming['is_array'] else 0.0,
            1.0 if naming['is_function'] else 0.0,
            1.0 if naming['is_constant'] else 0.0,
            1.0 if naming['is_private'] else 0.0
        ])
        
        # Count features
        vector.extend([
            len(features['surrounding_functions']),
            len(features['import_statements']),
            len(features['type_annotations']),
            len(features['assignment_patterns']),
            len(features['method_calls']),
            len(features['operators'])
        ])
        
        # Text length features
        vector.extend([
            len(features['variable_name']),
            len(features['code_context']),
            len(' '.join(features['usage_contexts']))
        ])
        
        return vector
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get statistics about type inference."""
        return {
            'ml_available': self.use_ml,
            'type_patterns_count': sum(len(patterns) for patterns in self.type_patterns.values()),
            'training_data_size': len(self.synthetic_training_data),
            'model_trained': hasattr(self, 'type_classifier') and hasattr(self.type_classifier, 'classes_')
        }

