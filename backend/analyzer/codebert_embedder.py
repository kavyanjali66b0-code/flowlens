"""
CodeBERT Embedding Service

This module provides a specialized embedding service using CodeBERT for code-specific
semantic understanding. CodeBERT is trained specifically on code and provides better
accuracy for code analysis tasks compared to generic text transformers.

Key features:
- Batch processing for efficiency
- Intelligent caching with persistent storage
- Smart truncation for long code sequences
- Fallback handling for robustness
"""

import logging
import hashlib
import os
import pickle
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np

# ML imports with fallback handling
# Note: We catch both ImportError and OSError because torch may fail to load
# due to DLL issues on Windows even if the package is installed
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    ML_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    logging.warning(f"ML dependencies not available (torch/transformers may have DLL issues): {e}")
    ML_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModel = None


class CodeBERTEmbedder:
    """
    CodeBERT-based embedding service optimized for code analysis.
    
    Uses microsoft/codebert-base for code-specific semantic understanding.
    Provides better accuracy for code relationships compared to generic text models.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/codebert_embeddings",
        use_cache: bool = True,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize CodeBERT embedder.
        
        Args:
            cache_dir: Directory to cache embeddings
            use_cache: Whether to use cached embeddings
            batch_size: Batch size for processing
            max_length: Maximum token length (CodeBERT limit is 512)
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components (lazy loaded)
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._model_loaded = False
        
        # In-memory cache for current session
        self._session_cache: Dict[str, np.ndarray] = {}
        
        self.logger.info(f"CodeBERTEmbedder initialized (cache={use_cache}, batch_size={batch_size})")
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None and ML_AVAILABLE and AutoTokenizer is not None:
            try:
                self.logger.info("Loading CodeBERT tokenizer...")
                self._tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.logger.info("CodeBERT tokenizer loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load CodeBERT tokenizer: {e}")
                # Don't raise - return None to indicate failure
                return None
        return self._tokenizer
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None and ML_AVAILABLE and AutoModel is not None:
            try:
                self.logger.info("Loading CodeBERT model...")
                self._model = AutoModel.from_pretrained("microsoft/codebert-base")
                self._model.eval()  # Set to evaluation mode
                self._model_loaded = True
                self.logger.info("CodeBERT model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load CodeBERT model: {e}")
                # Don't raise - return None to indicate failure
                return None
        return self._model
    
    def _get_cache_key(self, code: str, context: str = "", entity_name: str = "") -> str:
        """Generate cache key for embedding."""
        combined = f"{entity_name}|{context}|{code}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        # Check session cache first
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]
        
        # Check persistent cache
        if self.use_cache:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    self._session_cache[cache_key] = embedding
                    return embedding
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache."""
        # Save to session cache
        self._session_cache[cache_key] = embedding
        
        # Save to persistent cache
        if self.use_cache:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")
    
    def _truncate_code_intelligently(self, code: str, max_tokens: int) -> str:
        """
        Intelligently truncate code to fit within token limit.
        
        Prioritizes:
        1. Function/component signature
        2. Docstring/comments
        3. Key logic (first part of function body)
        """
        if not code:
            return code
        
        # Split by lines
        lines = code.split('\n')
        
        # If short enough, return as is
        if len(lines) <= max_tokens // 10:  # Rough estimate
            return code
        
        # Priority: signature, docstring, first part of body
        signature = lines[0] if lines else ""
        docstring_lines = []
        body_lines = []
        
        in_docstring = False
        for line in lines[1:]:
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                docstring_lines.append(line)
            elif in_docstring:
                docstring_lines.append(line)
            else:
                body_lines.append(line)
        
        # Combine: signature + docstring + first part of body
        important_lines = [signature] + docstring_lines + body_lines[:max_tokens // 10]
        truncated = '\n'.join(important_lines)
        
        return truncated
    
    def embed_code(
        self,
        code: str,
        context: str = "",
        entity_name: str = "",
        entity_type: str = "function"
    ) -> np.ndarray:
        """
        Generate embedding for a code snippet.
        
        Args:
            code: Code snippet to embed
            context: Surrounding context (imports, class definition, etc.)
            entity_name: Name of the entity (function/class/component name)
            entity_type: Type of entity (function, class, component, module)
            
        Returns:
            Embedding vector as numpy array
        """
        if not ML_AVAILABLE:
            return self._fallback_embedding(code, context, entity_name)
        
        # Check cache
        cache_key = self._get_cache_key(code, context, entity_name)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Format code for CodeBERT
            formatted_code = self._format_code_for_codebert(code, context, entity_name, entity_type)
            
            # Truncate if necessary
            if len(formatted_code) > 1000:  # Rough estimate
                formatted_code = self._truncate_code_intelligently(formatted_code, self.max_length)
            
            # Check if tokenizer and model are available
            tokenizer = self.tokenizer
            model = self.model
            if tokenizer is None or model is None:
                self.logger.warning("CodeBERT tokenizer or model not available, using fallback")
                return self._fallback_embedding(code, context, entity_name)
            
            # Tokenize
            inputs = tokenizer(
                formatted_code,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding (first token) - mean pooling for better results
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                # Also use mean pooling of all tokens for richer representation
                mean_embedding = outputs.last_hidden_state.mean(dim=1)
                # Combine both (simple average)
                embedding = ((cls_embedding + mean_embedding) / 2).numpy().flatten()
            
            # Cache the result
            self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating CodeBERT embedding: {e}")
            return self._fallback_embedding(code, context, entity_name)
    
    def _format_code_for_codebert(
        self,
        code: str,
        context: str,
        entity_name: str,
        entity_type: str
    ) -> str:
        """
        Format code optimally for CodeBERT understanding.
        
        CodeBERT works best with well-structured code that includes:
        - Clear entity names
        - Contextual information
        - Code structure
        """
        parts = []
        
        # Add entity name and type as context
        if entity_name:
            parts.append(f"{entity_type} {entity_name}")
        
        # Add surrounding context
        if context:
            parts.append(f"Context: {context}")
        
        # Add the actual code
        parts.append(code)
        
        return "\n".join(parts)
    
    def batch_embed(
        self,
        code_snippets: List[Tuple[str, str, str, str]],
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Batch embed multiple code snippets for efficiency.
        
        Args:
            code_snippets: List of tuples (code, context, entity_name, entity_type)
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not ML_AVAILABLE:
            return [self._fallback_embedding(code, ctx, name) for code, ctx, name, _ in code_snippets]
        
        embeddings = []
        uncached_indices = []
        uncached_snippets = []
        
        # Check cache for all snippets
        for i, (code, context, entity_name, entity_type) in enumerate(code_snippets):
            cache_key = self._get_cache_key(code, context, entity_name)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_snippets.append((code, context, entity_name, entity_type))
        
        # Process uncached snippets in batches
        if uncached_snippets:
            # Check if tokenizer and model are available
            tokenizer = self.tokenizer
            model = self.model
            if tokenizer is None or model is None:
                self.logger.warning("CodeBERT tokenizer or model not available, using fallback for batch")
                # Use fallback for all uncached snippets
                for i, original_idx in enumerate(uncached_indices):
                    code, context, entity_name, entity_type = uncached_snippets[i]
                    embeddings[original_idx] = self._fallback_embedding(code, context, entity_name)
                return embeddings
            
            self.logger.info(f"Processing {len(uncached_snippets)} uncached embeddings in batches of {self.batch_size}")
            
            for batch_start in range(0, len(uncached_snippets), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(uncached_snippets))
                batch = uncached_snippets[batch_start:batch_end]
                
                # Format all snippets in batch
                formatted_texts = [
                    self._format_code_for_codebert(code, ctx, name, etype)
                    for code, ctx, name, etype in batch
                ]
                
                # Tokenize batch
                inputs = tokenizer(
                    formatted_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                )
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Combine CLS and mean pooling
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    mean_embeddings = outputs.last_hidden_state.mean(dim=1)
                    batch_embeddings = ((cls_embeddings + mean_embeddings) / 2).numpy()
                
                # Store embeddings and cache them
                for i, (code, context, entity_name, entity_type) in enumerate(batch):
                    embedding = batch_embeddings[i]
                    cache_key = self._get_cache_key(code, context, entity_name)
                    self._save_to_cache(cache_key, embedding)
                    
                    # Update embeddings list
                    original_idx = uncached_indices[batch_start + i]
                    embeddings[original_idx] = embedding
        
        return embeddings
    
    def _fallback_embedding(self, code: str, context: str, entity_name: str) -> np.ndarray:
        """Generate a simple fallback embedding when ML is unavailable."""
        # Simple hash-based embedding (not semantic, but consistent)
        combined = f"{entity_name}|{context}|{code}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        # Convert to 768-dim vector (CodeBERT's dimension)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.rand(768).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def clear_cache(self):
        """Clear session cache."""
        self._session_cache.clear()
        self.logger.info("Session cache cleared")
    
    def is_available(self) -> bool:
        """Check if CodeBERT is available."""
        if not ML_AVAILABLE:
            return False
        try:
            tokenizer = self.tokenizer
            model = self.model
            # Both must be available (not None)
            return tokenizer is not None and model is not None
        except Exception:
            return False

