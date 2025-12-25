"""
Code Embedding Engine for Semantic Analysis

This module provides semantic embeddings for code entities using state-of-the-art
NLP models. It enables semantic similarity detection, clustering, and intelligent
relationship discovery between code elements.

Classes:
    CodeEmbeddingEngine: Main engine for generating code embeddings
    SemanticCluster: Represents a cluster of semantically similar entities
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import pickle
import os

# ML imports with fallback handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML dependencies not available: {e}")
    ML_AVAILABLE = False


@dataclass
class SemanticCluster:
    """Represents a cluster of semantically similar code entities."""
    cluster_id: str
    entities: List[str] = field(default_factory=list)
    centroid_embedding: Optional[np.ndarray] = None
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    cluster_type: str = "semantic"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeEmbeddingEngine:
    """
    Engine for generating semantic embeddings of code entities.
    
    Uses state-of-the-art models to understand code semantics and enable
    intelligent relationship discovery between functions, classes, and modules.
    """
    
    def __init__(self, cache_dir: str = "cache/embeddings", use_cache: bool = True):
        """
        Initialize the embedding engine.
        
        Args:
            cache_dir: Directory to cache embeddings
            use_cache: Whether to use cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models if ML is available
        if ML_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("ML dependencies not available. Using fallback mode.")
            self._initialize_fallback()
    
    def _initialize_models(self):
        """Initialize ML models for embedding generation."""
        try:
            # Use a lightweight model for code understanding
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            
            # For semantic similarity
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback mode without ML dependencies."""
        self.tokenizer = None
        self.model = None
        self.semantic_model = None
        self.logger.info("Running in fallback mode")
    
    def embed_function(self, code: str, context: str = "", 
                      function_name: str = "") -> np.ndarray:
        """
        Generate embedding for function code with context.
        
        Args:
            code: Function code
            context: Surrounding context (imports, class, etc.)
            function_name: Name of the function
            
        Returns:
            Embedding vector as numpy array
        """
        if not ML_AVAILABLE or not self.model:
            return self._fallback_embedding(code, context, function_name)
        
        # Check cache first
        cache_key = self._get_cache_key(code, context, function_name)
        if self.use_cache:
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            # Combine code + context + function name for better understanding
            full_text = f"{function_name}\n{context}\n{code}"
            
            # Tokenize and truncate if necessary
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
            # Cache the result
            if self.use_cache:
                self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return self._fallback_embedding(code, context, function_name)
    
    def embed_class(self, class_code: str, class_name: str = "", 
                   methods: List[str] = None) -> np.ndarray:
        """
        Generate embedding for class code.
        
        Args:
            class_code: Class definition code
            class_name: Name of the class
            methods: List of method names in the class
            
        Returns:
            Embedding vector as numpy array
        """
        if not ML_AVAILABLE or not self.model:
            return self._fallback_embedding(class_code, f"class {class_name}", class_name)
        
        # Combine class info
        methods_text = "\n".join(methods) if methods else ""
        full_text = f"class {class_name}\n{methods_text}\n{class_code}"
        
        return self.embed_function(full_text, "", class_name)
    
    def embed_module(self, module_code: str, module_name: str = "", 
                    exports: List[str] = None) -> np.ndarray:
        """
        Generate embedding for module code.
        
        Args:
            module_code: Module code
            module_name: Name of the module
            exports: List of exported symbols
            
        Returns:
            Embedding vector as numpy array
        """
        if not ML_AVAILABLE or not self.model:
            return self._fallback_embedding(module_code, f"module {module_name}", module_name)
        
        # Combine module info
        exports_text = "\n".join(exports) if exports else ""
        full_text = f"module {module_name}\nexports: {exports_text}\n{module_code}"
        
        return self.embed_function(full_text, "", module_name)
    
    def compute_semantic_similarity(self, embedding1: np.ndarray, 
                                  embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Ensure embeddings are 2D for sklearn
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_similar_functions(self, target_embedding: np.ndarray, 
                             function_embeddings: Dict[str, np.ndarray], 
                             threshold: float = 0.7, 
                             max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Find functions similar to target.
        
        Args:
            target_embedding: Embedding of target function
            function_embeddings: Dictionary of function_id -> embedding
            threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of (function_id, similarity_score) tuples
        """
        if target_embedding is None:
            return []
        
        similarities = []
        for func_id, embedding in function_embeddings.items():
            if embedding is not None:
                similarity = self.compute_semantic_similarity(target_embedding, embedding)
                if similarity >= threshold:
                    similarities.append((func_id, similarity))
        
        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def cluster_entities(self, entity_embeddings: Dict[str, np.ndarray], 
                        eps: float = 0.3, min_samples: int = 2) -> List[SemanticCluster]:
        """
        Cluster entities based on semantic similarity.
        
        Args:
            entity_embeddings: Dictionary of entity_id -> embedding
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            List of SemanticCluster objects
        """
        if not entity_embeddings:
            return []
        
        # Filter out None embeddings
        valid_embeddings = {k: v for k, v in entity_embeddings.items() if v is not None}
        if not valid_embeddings:
            return []
        
        try:
            # Prepare data for clustering
            entity_ids = list(valid_embeddings.keys())
            embeddings_matrix = np.array([valid_embeddings[eid] for eid in entity_ids])
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings_matrix)
            
            # Create cluster objects
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                # Get entities in this cluster
                cluster_entities = [entity_ids[i] for i, l in enumerate(cluster_labels) if l == label]
                
                # Calculate centroid
                cluster_embeddings = embeddings_matrix[cluster_labels == label]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate similarity scores
                similarity_scores = {}
                for entity_id in cluster_entities:
                    entity_embedding = valid_embeddings[entity_id]
                    similarity = self.compute_semantic_similarity(centroid, entity_embedding)
                    similarity_scores[entity_id] = similarity
                
                # Create cluster
                cluster = SemanticCluster(
                    cluster_id=f"cluster_{label}",
                    entities=cluster_entities,
                    centroid_embedding=centroid,
                    similarity_scores=similarity_scores,
                    confidence=np.mean(list(similarity_scores.values()))
                )
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering entities: {e}")
            return []
    
    def _fallback_embedding(self, code: str, context: str, name: str) -> np.ndarray:
        """
        Generate fallback embedding when ML models are not available.
        
        Uses simple text features as a basic embedding.
        """
        # Simple feature-based embedding
        text = f"{name} {context} {code}".lower()
        
        # Basic features
        features = [
            len(text),  # Length
            text.count('function'),  # Function keyword count
            text.count('class'),  # Class keyword count
            text.count('import'),  # Import count
            text.count('return'),  # Return count
            text.count('async'),  # Async count
            text.count('await'),  # Await count
            text.count('('),  # Parentheses count
            text.count('{'),  # Brace count
            text.count(';'),  # Semicolon count
        ]
        
        # Normalize features
        features = np.array(features, dtype=float)
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
        
        return features
    
    def _get_cache_key(self, code: str, context: str, name: str) -> str:
        """Generate cache key for embedding."""
        content = f"{name}|{context}|{code}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"cache_size": 0, "cache_files": 0}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_size": total_size,
            "cache_files": len(cache_files),
            "cache_dir": str(self.cache_dir)
        }

