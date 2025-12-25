"""
Enhanced Semantic Analyzer with NLP/Embeddings

This module provides advanced semantic analysis capabilities using:
1. Code embeddings for semantic similarity
2. NLP-based function call resolution
3. Intelligent variable name analysis
4. Semantic clustering for component grouping
5. Cross-file semantic understanding

Key improvements over basic semantic analysis:
- Uses embeddings to find semantically similar code
- Better call graph resolution using semantic similarity
- Identifies implicit relationships (similar purpose, data flow)
- Reduces false positives through confidence scoring
"""

import logging
import hashlib
import os
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from .models import Node, Edge, EdgeType, NodeType
from .symbol_table import SymbolTable
from .semantic_analyzer import SemanticAnalyzer

# Conditional import for CodeBERT
try:
    from .codebert_embedder import CodeBERTEmbedder
    CODEBERT_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    logging.warning(f"CodeBERT not available (may have DLL issues): {e}")
    CODEBERT_AVAILABLE = False
    CodeBERTEmbedder = None

# Fallback to sentence transformers if CodeBERT not available
# Note: We catch OSError and RuntimeError because sentence_transformers
# depends on torch, which may fail to load due to DLL issues on Windows
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    logging.warning(f"SentenceTransformer not available (may have torch DLL issues): {e}")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class SemanticEmbedding:
    """Represents an embedding for a code entity."""
    entity_id: str
    entity_name: str
    entity_type: NodeType
    embedding: np.ndarray
    context: str  # Source code context
    file_path: str
    confidence: float = 1.0


@dataclass
class SemanticMatch:
    """Represents a semantic similarity match between entities."""
    source_id: str
    target_id: str
    similarity_score: float
    match_type: str  # 'call', 'similar_purpose', 'data_flow', 'dependency'
    confidence: float
    explanation: str


class EnhancedSemanticAnalyzer:
    """
    Enhanced semantic analyzer using NLP and embeddings.
    
    Features:
    1. Code embeddings for semantic similarity
    2. Intelligent call resolution
    3. Semantic clustering
    4. Cross-file understanding
    5. Confidence-based edge creation
    
    Usage:
        analyzer = EnhancedSemanticAnalyzer(nodes, file_symbols, symbol_table)
        edges = analyzer.analyze()
    """
    
    def __init__(
        self,
        nodes: List[Node],
        file_symbols: Dict[str, Dict],
        symbol_table: Optional[SymbolTable] = None,
        enable_embeddings: bool = True,
        enable_clustering: bool = True,
        similarity_threshold: float = 0.75,
        use_codebert: Optional[bool] = None,
        batch_size: int = 32
    ):
        """
        Initialize enhanced semantic analyzer.
        
        Args:
            nodes: Parsed nodes
            file_symbols: Symbol information per file
            symbol_table: Optional symbol table
            enable_embeddings: Enable embedding-based analysis
            enable_clustering: Enable semantic clustering
            similarity_threshold: Minimum similarity for edge creation (0.0-1.0)
            use_codebert: Use CodeBERT instead of sentence transformer (default: True if available)
            batch_size: Batch size for CodeBERT processing
        """
        self.nodes = nodes
        self.file_symbols = file_symbols
        self.symbol_table = symbol_table
        self.enable_embeddings = enable_embeddings
        self.enable_clustering = enable_clustering
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Determine which model to use
        if use_codebert is None:
            # Check environment variable or default to CodeBERT if available
            use_codebert = os.environ.get('FLOWLENS_USE_CODEBERT', 'true').lower() == 'true'
            use_codebert = use_codebert and CODEBERT_AVAILABLE
        
        self.use_codebert = use_codebert and CODEBERT_AVAILABLE
        
        # Keep base semantic analyzer for fallback
        self.base_analyzer = SemanticAnalyzer(nodes, file_symbols, symbol_table)
        self.edges: List[Edge] = []
        
        # Embedding storage
        self.embeddings: Dict[str, SemanticEmbedding] = {}
        self.embedding_matrix: Optional[np.ndarray] = None
        self.entity_id_to_index: Dict[str, int] = {}
        
        # Embedding model (lazy load)
        self._codebert_embedder: Optional[CodeBERTEmbedder] = None
        self._sentence_transformer: Optional[SentenceTransformer] = None
        self._model_loaded = False
        
        # Caching
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        model_type = "CodeBERT" if self.use_codebert else "SentenceTransformer"
        logging.info(f"EnhancedSemanticAnalyzer initialized (embeddings={enable_embeddings}, model={model_type})")
    
    @property
    def codebert_embedder(self) -> Optional[CodeBERTEmbedder]:
        """Lazy load CodeBERT embedder."""
        if self._codebert_embedder is None and self.use_codebert and self.enable_embeddings:
            try:
                cache_dir = os.environ.get('FLOWLENS_CODEBERT_CACHE_DIR', 'cache/codebert_embeddings')
                use_cache = os.environ.get('FLOWLENS_CODEBERT_CACHE_ENABLED', 'true').lower() == 'true'
                batch_size = int(os.environ.get('FLOWLENS_CODEBERT_BATCH_SIZE', str(self.batch_size)))
                
                logging.info("Initializing CodeBERT embedder...")
                self._codebert_embedder = CodeBERTEmbedder(
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    batch_size=batch_size
                )
                
                if self._codebert_embedder.is_available():
                    logging.info("CodeBERT embedder ready")
                else:
                    logging.warning("CodeBERT not available, falling back to sentence transformer")
                    self.use_codebert = False
                    self._codebert_embedder = None
            except Exception as e:
                logging.error(f"Failed to initialize CodeBERT: {e}")
                self.use_codebert = False
                self._codebert_embedder = None
        return self._codebert_embedder
    
    @property
    def sentence_transformer(self) -> Optional[SentenceTransformer]:
        """Lazy load sentence transformer as fallback."""
        if self._sentence_transformer is None and not self.use_codebert and self.enable_embeddings:
            if not SENTENCE_TRANSFORMER_AVAILABLE:
                logging.warning("SentenceTransformer not available")
                return None
            try:
                model_name = os.environ.get('FLOWLENS_SENTENCE_MODEL', 'all-MiniLM-L6-v2')
                logging.info(f"Loading sentence transformer model: {model_name}")
                self._sentence_transformer = SentenceTransformer(model_name)
                logging.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load sentence transformer: {e}")
                self.enable_embeddings = False
        return self._sentence_transformer
    
    def analyze(self) -> List[Edge]:
        """
        Perform enhanced semantic analysis.
        
        Returns:
            List of edges with high-confidence semantic relationships
        """
        logging.info("Starting enhanced semantic analysis")
        
        # Step 1: Run base semantic analysis for traditional edges
        base_edges = self.base_analyzer.analyze()
        logging.info(f"Base analysis: {len(base_edges)} edges")
        
        # Step 2: Generate embeddings for all entities
        if self.enable_embeddings:
            self._generate_embeddings()
            logging.info(f"Generated {len(self.embeddings)} embeddings")
        
        # Step 3: Enhanced call resolution using semantic similarity
        semantic_call_edges = self._resolve_semantic_calls()
        logging.info(f"Semantic call resolution: {len(semantic_call_edges)} edges")
        
        # Step 4: Discover implicit relationships through similarity
        similarity_edges = self._discover_similarity_relationships()
        logging.info(f"Similarity analysis: {len(similarity_edges)} edges")
        
        # Step 5: Semantic clustering for component grouping
        if self.enable_clustering:
            cluster_edges = self._perform_semantic_clustering()
            logging.info(f"Clustering: {len(cluster_edges)} edges")
        else:
            cluster_edges = []
        
        # Step 6: Cross-file semantic understanding
        cross_file_edges = self._analyze_cross_file_semantics()
        logging.info(f"Cross-file analysis: {len(cross_file_edges)} edges")
        
        # Combine all edges, removing duplicates and low-confidence edges
        all_edges = self._merge_and_filter_edges(
            base_edges,
            semantic_call_edges,
            similarity_edges,
            cluster_edges,
            cross_file_edges
        )
        
        logging.info(f"Enhanced semantic analysis complete: {len(all_edges)} total edges")
        return all_edges
    
    def _generate_embeddings(self):
        """Generate embeddings for all code entities using CodeBERT or sentence transformer."""
        if not self.enable_embeddings:
            return
        
        # Use CodeBERT if available, otherwise fall back to sentence transformer
        if self.use_codebert:
            embedder = self.codebert_embedder
            if embedder is None or not embedder.is_available():
                logging.warning("CodeBERT not available, falling back to sentence transformer")
                self.use_codebert = False
                embedder = None
        
        if not self.use_codebert:
            embedder = self.sentence_transformer
            if embedder is None:
                logging.warning("No embedding model available")
                return
        
        logging.info(f"Generating code embeddings using {'CodeBERT' if self.use_codebert else 'SentenceTransformer'}...")
        
        # Prepare code snippets for embedding
        code_snippets = []
        entity_ids = []
        cached_embeddings = {}
        
        for node in self.nodes:
            # Get code context for this node
            code, context = self._get_code_context(node)
            entity_name = node.name
            entity_type = node.type.value
            
            # Check cache
            cache_key = self._get_embedding_cache_key(code, context, entity_name)
            if cache_key in self._embedding_cache:
                cached_embeddings[node.id] = self._embedding_cache[cache_key]
            else:
                code_snippets.append((code, context, entity_name, entity_type))
                entity_ids.append(node.id)
        
        # Generate embeddings
        if self.use_codebert and self.codebert_embedder:
            # Use CodeBERT batch embedding
            if code_snippets:
                try:
                    new_embeddings = self.codebert_embedder.batch_embed(
                        code_snippets,
                        show_progress=len(code_snippets) > 50
                    )
                    
                    # Store embeddings
                    for i, (code, context, entity_name, entity_type) in enumerate(code_snippets):
                        entity_id = entity_ids[i]
                        embedding = new_embeddings[i]
                        node = next(n for n in self.nodes if n.id == entity_id)
                        
                        # Store in cache
                        cache_key = self._get_embedding_cache_key(code, context, entity_name)
                        self._embedding_cache[cache_key] = embedding
                        
                        # Store in embeddings dict
                        self.embeddings[entity_id] = SemanticEmbedding(
                            entity_id=entity_id,
                            entity_name=node.name,
                            entity_type=node.type,
                            embedding=embedding,
                            context=context,
                            file_path=node.file
                        )
                except Exception as e:
                    logging.error(f"Failed to generate CodeBERT embeddings: {e}")
                    self.use_codebert = False
                    # Fallback to sentence transformer
                    return self._generate_embeddings()
        else:
            # Use sentence transformer
            if code_snippets:
                try:
                    # Create semantic text for sentence transformer
                    texts_to_embed = [
                        self._create_semantic_text_for_sentence_transformer(code, context, name, etype)
                        for code, context, name, etype in code_snippets
                    ]
                    
                    embeddings = self.sentence_transformer.encode(
                        texts_to_embed,
                        batch_size=self.batch_size,
                        show_progress_bar=len(texts_to_embed) > 100
                    )
                    
                    # Store embeddings
                    for i, (code, context, entity_name, entity_type) in enumerate(code_snippets):
                        entity_id = entity_ids[i]
                        embedding = embeddings[i]
                        node = next(n for n in self.nodes if n.id == entity_id)
                        
                        # Store in cache
                        cache_key = self._get_embedding_cache_key(code, context, entity_name)
                        self._embedding_cache[cache_key] = embedding
                        
                        # Store in embeddings dict
                        self.embeddings[entity_id] = SemanticEmbedding(
                            entity_id=entity_id,
                            entity_name=node.name,
                            entity_type=node.type,
                            embedding=embedding,
                            context=context,
                            file_path=node.file
                        )
                except Exception as e:
                    logging.error(f"Failed to generate sentence transformer embeddings: {e}")
                    self.enable_embeddings = False
                    return
        
        # Add cached embeddings
        for entity_id, embedding in cached_embeddings.items():
            node = next(n for n in self.nodes if n.id == entity_id)
            self.embeddings[entity_id] = SemanticEmbedding(
                entity_id=entity_id,
                entity_name=node.name,
                entity_type=node.type,
                embedding=embedding,
                context="",
                file_path=node.file
            )
        
        # Build embedding matrix for fast similarity computation
        if self.embeddings:
            self._build_embedding_matrix()
            logging.info(f"Generated {len(self.embeddings)} embeddings")
    
    def _get_code_context(self, node: Node) -> Tuple[str, str]:
        """
        Get code and context for a node.
        
        Returns:
            Tuple of (code, context) where:
            - code: The actual code content (if available)
            - context: Surrounding context (file path, imports, etc.)
        """
        # Try to get code from file_symbols
        file_path = node.file
        code = ""
        context_parts = []
        
        # Get file context
        file_path_obj = Path(file_path)
        context_parts.append(f"file: {file_path}")
        
        # Try to get code from file_symbols
        if file_path in self.file_symbols:
            file_info = self.file_symbols[file_path]
            # Some plugins might store code snippets
            if 'code' in file_info:
                code = file_info['code']
            elif 'snippet' in file_info:
                code = file_info['snippet']
            # Get imports for context
            if 'imports' in file_info:
                imports = file_info['imports']
                if isinstance(imports, list):
                    context_parts.append(f"imports: {', '.join(str(i) for i in imports[:5])}")
        
        # If no code available, create a minimal representation
        if not code:
            # Use node name and type as code representation
            code = f"{node.type.value} {node.name}"
        
        # Add metadata context
        if node.metadata:
            if hasattr(node.metadata, 'line_range') and node.metadata.line_range:
                start, end = node.metadata.line_range
                context_parts.append(f"lines: {start}-{end}")
            elif isinstance(node.metadata, dict):
                if 'line_range' in node.metadata:
                    start, end = node.metadata['line_range']
                    context_parts.append(f"lines: {start}-{end}")
                if 'type_info' in node.metadata:
                    type_info = node.metadata['type_info']
                    if isinstance(type_info, dict) and 'name' in type_info:
                        context_parts.append(f"type: {type_info['name']}")
        
        context = " | ".join(context_parts)
        return code, context
    
    def _get_embedding_cache_key(self, code: str, context: str, entity_name: str) -> str:
        """Generate cache key for embedding."""
        combined = f"{entity_name}|{context}|{code}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _create_semantic_text_for_sentence_transformer(
        self,
        code: str,
        context: str,
        entity_name: str,
        entity_type: str
    ) -> str:
        """Create semantic text representation for sentence transformer (backward compatibility)."""
        # Split entity name
        name_tokens = self._split_identifier(entity_name)
        
        # Combine into semantic text
        parts = [
            f"name: {' '.join(name_tokens)}",
            f"type: {entity_type}",
            f"context: {context}",
            f"code: {code[:200]}"  # Truncate code for sentence transformer
        ]
        
        return " | ".join(parts)
    
    def _create_semantic_text(self, node: Node) -> str:
        """
        Create a semantic text representation of a code entity optimized for CodeBERT.
        
        For CodeBERT, we want:
        - Function/component signature
        - Docstring/comments
        - Key logic snippets
        - Props interface (for components)
        - Hooks used (for components)
        - Exports (for modules)
        """
        code, context = self._get_code_context(node)
        
        # For CodeBERT, format code with proper structure
        if self.use_codebert:
            return self._format_code_for_codebert(node, code, context)
        else:
            # Fallback to original format for sentence transformer
            name_tokens = self._split_identifier(node.name)
            file_path = Path(node.file)
            path_tokens = [p for p in file_path.parts if p not in {'.', '..'} and p != file_path.suffix]
            
            semantic_parts = [
                f"name: {' '.join(name_tokens)}",
                f"type: {node.type.value}",
                f"context: {' '.join(path_tokens[-3:])}",
                f"code: {code[:200]}"
            ]
            
            # Add metadata hints if available
            if node.metadata:
                if hasattr(node.metadata, 'to_dict'):
                    metadata = node.metadata.to_dict()
                else:
                    metadata = node.metadata or {}
                
                if 'type_info' in metadata and metadata['type_info']:
                    type_name = metadata['type_info'].get('name', '')
                    if type_name:
                        semantic_parts.append(f"returns: {type_name}")
                
                if 'hook_info' in metadata and metadata['hook_info']:
                    hook_type = metadata['hook_info'].get('hook_type', '')
                    if hook_type:
                        semantic_parts.append(f"hook: {hook_type}")
            
            return " | ".join(semantic_parts)
    
    def _format_code_for_codebert(self, node: Node, code: str, context: str) -> str:
        """
        Format code optimally for CodeBERT understanding.
        
        CodeBERT works best with:
        - Well-structured code
        - Clear entity names
        - Contextual information
        - Code structure preserved
        """
        parts = []
        
        # Entity signature
        entity_type = node.type.value
        entity_name = node.name
        parts.append(f"{entity_type} {entity_name}")
        
        # Add context (file path, imports)
        if context:
            parts.append(f"// Context: {context}")
        
        # Add the actual code
        if code and len(code) > 10:  # Only if we have substantial code
            # For CodeBERT, preserve code structure
            parts.append(code)
        else:
            # Fallback: use entity name as code
            parts.append(f"// {entity_name}")
        
        # Add metadata hints as comments
        if node.metadata:
            if hasattr(node.metadata, 'to_dict'):
                metadata = node.metadata.to_dict()
            else:
                metadata = node.metadata or {}
            
            if 'type_info' in metadata and metadata['type_info']:
                type_info = metadata['type_info']
                if isinstance(type_info, dict) and 'name' in type_info:
                    parts.append(f"// Returns: {type_info['name']}")
            
            if 'hook_info' in metadata and metadata['hook_info']:
                hook_info = metadata['hook_info']
                if isinstance(hook_info, dict):
                    hook_name = hook_info.get('hook_name', '')
                    hook_type = hook_info.get('hook_type', '')
                    if hook_name or hook_type:
                        parts.append(f"// Hook: {hook_name or hook_type}")
        
        return "\n".join(parts)
    
    def _split_identifier(self, name: str) -> List[str]:
        """Split camelCase or snake_case identifier into tokens."""
        import re
        
        # Handle snake_case
        if '_' in name:
            return name.split('_')
        
        # Handle camelCase
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
        return [t.lower() for t in tokens if t]
    
    def _build_embedding_matrix(self):
        """Build matrix for fast similarity computations."""
        if not self.embeddings:
            return
        
        entity_ids = list(self.embeddings.keys())
        embeddings_list = [self.embeddings[eid].embedding for eid in entity_ids]
        
        self.embedding_matrix = np.vstack(embeddings_list)
        self.entity_id_to_index = {eid: i for i, eid in enumerate(entity_ids)}
        
        logging.debug(f"Built embedding matrix: {self.embedding_matrix.shape}")
    
    def _resolve_semantic_calls(self) -> List[Edge]:
        """
        Resolve function calls using semantic similarity.
        
        This improves upon naive name-matching by considering:
        - Semantic similarity of function names
        - Context similarity (file paths, surrounding code)
        - Type compatibility
        """
        edges = []
        
        if not self.embeddings:
            return edges
        
        # Find all function/component nodes
        callables = [n for n in self.nodes if n.type in {
            NodeType.FUNCTION,
            NodeType.COMPONENT,
            NodeType.CLASS
        }]
        
        # For each callable, find potential call targets using similarity
        for source_node in callables:
            if source_node.id not in self.embeddings:
                continue
            
            source_emb = self.embeddings[source_node.id]
            
            # Find similar callables (potential call targets)
            matches = self._find_similar_entities(
                source_emb,
                target_types={NodeType.FUNCTION, NodeType.COMPONENT},
                min_similarity=self.similarity_threshold,
                exclude_self=True,
                max_results=10
            )
            
            for match in matches:
                # Only create CALLS edge if:
                # 1. High similarity (>threshold)
                # 2. Not in the same file (avoid noise)
                # 3. Target is a different entity
                
                target_node = next((n for n in self.nodes if n.id == match.target_id), None)
                if not target_node:
                    continue
                
                # Skip same-file calls (handled by base analyzer better)
                if source_node.file == target_node.file:
                    continue
                
                # Check if there's evidence of import relationship
                has_import = self._check_import_relationship(source_node.file, target_node.file)
                
                # Create edge with confidence based on similarity + import evidence
                confidence = match.similarity_score
                if has_import:
                    confidence = min(1.0, confidence + 0.1)
                
                if confidence >= self.similarity_threshold:
                    edge = Edge(
                        source=source_node.id,
                        target=target_node.id,
                        type=EdgeType.CALLS,
                        metadata={
                            'semantic_similarity': match.similarity_score,
                            'confidence': confidence,
                            'method': 'embedding',
                            'has_import': has_import,
                            'explanation': f"Semantic similarity: {match.similarity_score:.2f}"
                        }
                    )
                    edges.append(edge)
        
        return edges
    
    def _discover_similarity_relationships(self) -> List[Edge]:
        """
        Discover implicit relationships through semantic similarity.
        
        Finds:
        - Components with similar purposes
        - Functions that operate on similar data
        - Related utility functions
        """
        edges = []
        
        if not self.embeddings:
            return edges
        
        # For each entity, find highly similar entities
        for entity_id, embedding in self.embeddings.items():
            matches = self._find_similar_entities(
                embedding,
                target_types={NodeType.FUNCTION, NodeType.COMPONENT, NodeType.CLASS},
                min_similarity=0.85,  # High threshold for "similar purpose"
                exclude_self=True,
                max_results=5
            )
            
            for match in matches:
                # Create SIMILAR_PURPOSE edge
                edge = Edge(
                    source=entity_id,
                    target=match.target_id,
                    type=EdgeType.DEPENDS_ON,  # Use existing type, add metadata
                    metadata={
                        'semantic_similarity': match.similarity_score,
                        'relationship_type': 'similar_purpose',
                        'confidence': match.similarity_score,
                        'method': 'embedding',
                        'explanation': f"Semantically similar: {match.similarity_score:.2f}"
                    }
                )
                edges.append(edge)
        
        return edges
    
    def _perform_semantic_clustering(self) -> List[Edge]:
        """
        Cluster semantically similar components/functions.
        
        Creates edges representing cluster membership.
        """
        edges = []
        
        if not self.embeddings or self.embedding_matrix is None:
            return edges
        
        try:
            # Perform DBSCAN clustering on embeddings
            clustering = DBSCAN(
                eps=0.3,  # Similarity threshold
                min_samples=2,  # Minimum cluster size
                metric='cosine'
            )
            
            labels = clustering.fit_predict(self.embedding_matrix)
            
            # Group entities by cluster
            clusters: Dict[int, List[str]] = {}
            for entity_id, idx in self.entity_id_to_index.items():
                label = labels[idx]
                if label != -1:  # -1 is noise
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(entity_id)
            
            # Create edges within each cluster
            for cluster_id, entity_ids in clusters.items():
                if len(entity_ids) < 2:
                    continue
                
                # Create edges between cluster members
                for i, source_id in enumerate(entity_ids):
                    for target_id in entity_ids[i+1:]:
                        edge = Edge(
                            source=source_id,
                            target=target_id,
                            type=EdgeType.DEPENDS_ON,
                            metadata={
                                'relationship_type': 'semantic_cluster',
                                'cluster_id': cluster_id,
                                'cluster_size': len(entity_ids),
                                'confidence': 0.8,
                                'method': 'clustering',
                                'explanation': f"Part of semantic cluster {cluster_id}"
                            }
                        )
                        edges.append(edge)
            
            logging.info(f"Found {len(clusters)} semantic clusters")
            
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
        
        return edges
    
    def _analyze_cross_file_semantics(self) -> List[Edge]:
        """
        Analyze semantic relationships across files.
        
        Identifies:
        - Cross-file data flows
        - Shared concepts/patterns
        - Module boundaries
        """
        edges = []
        
        # Group nodes by file
        files_to_nodes: Dict[str, List[Node]] = {}
        for node in self.nodes:
            if node.file not in files_to_nodes:
                files_to_nodes[node.file] = []
            files_to_nodes[node.file].append(node)
        
        # For each pair of files, check semantic similarity
        file_list = list(files_to_nodes.keys())
        for i, file1 in enumerate(file_list):
            for file2 in file_list[i+1:]:
                # Calculate file-level similarity
                similarity = self._calculate_file_similarity(file1, file2)
                
                if similarity > 0.7:  # Files are semantically related
                    # Create edges between representative nodes
                    nodes1 = files_to_nodes[file1]
                    nodes2 = files_to_nodes[file2]
                    
                    # Find most representative nodes (e.g., exported components)
                    rep1 = self._find_representative_nodes(nodes1, max_count=3)
                    rep2 = self._find_representative_nodes(nodes2, max_count=3)
                    
                    for n1 in rep1:
                        for n2 in rep2:
                            edge = Edge(
                                source=n1.id,
                                target=n2.id,
                                type=EdgeType.DEPENDS_ON,
                                metadata={
                                    'relationship_type': 'cross_file_semantic',
                                    'file_similarity': similarity,
                                    'confidence': similarity,
                                    'method': 'cross_file_analysis',
                                    'explanation': f"Related files (similarity: {similarity:.2f})"
                                }
                            )
                            edges.append(edge)
        
        return edges
    
    def _find_similar_entities(
        self,
        source: SemanticEmbedding,
        target_types: Set[NodeType],
        min_similarity: float,
        exclude_self: bool = True,
        max_results: int = 10
    ) -> List[SemanticMatch]:
        """Find entities similar to the source embedding."""
        if self.embedding_matrix is None:
            return []
        
        # Calculate similarities
        source_embedding = source.embedding.reshape(1, -1)
        similarities = cosine_similarity(source_embedding, self.embedding_matrix)[0]
        
        # Create matches
        matches = []
        for entity_id, idx in self.entity_id_to_index.items():
            # Skip self
            if exclude_self and entity_id == source.entity_id:
                continue
            
            # Check type filter
            target_emb = self.embeddings[entity_id]
            if target_emb.entity_type not in target_types:
                continue
            
            # Check similarity threshold
            sim = similarities[idx]
            if sim < min_similarity:
                continue
            
            matches.append(SemanticMatch(
                source_id=source.entity_id,
                target_id=entity_id,
                similarity_score=float(sim),
                match_type='semantic',
                confidence=float(sim),
                explanation=f"Cosine similarity: {sim:.3f}"
            ))
        
        # Sort by similarity and limit results
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:max_results]
    
    def _check_import_relationship(self, source_file: str, target_file: str) -> bool:
        """Check if source file imports from target file."""
        if source_file not in self.file_symbols:
            return False
        
        symbols = self.file_symbols[source_file]
        imports = symbols.get('imports', [])
        
        # Check if target file path appears in imports
        for imp in imports:
            if target_file in imp or Path(target_file).stem in imp:
                return True
        
        return False
    
    def _calculate_file_similarity(self, file1: str, file2: str) -> float:
        """Calculate semantic similarity between two files."""
        # Get embeddings for all entities in each file
        embs1 = [emb for emb in self.embeddings.values() if emb.file_path == file1]
        embs2 = [emb for emb in self.embeddings.values() if emb.file_path == file2]
        
        if not embs1 or not embs2:
            return 0.0
        
        # Calculate average similarity
        similarities = []
        for e1 in embs1:
            for e2 in embs2:
                sim = cosine_similarity(
                    e1.embedding.reshape(1, -1),
                    e2.embedding.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _find_representative_nodes(self, nodes: List[Node], max_count: int = 3) -> List[Node]:
        """Find most representative nodes (e.g., exported functions/components)."""
        # Prioritize: COMPONENT > FUNCTION > CLASS
        priority_map = {
            NodeType.COMPONENT: 3,
            NodeType.FUNCTION: 2,
            NodeType.CLASS: 1
        }
        
        sorted_nodes = sorted(
            nodes,
            key=lambda n: priority_map.get(n.type, 0),
            reverse=True
        )
        
        return sorted_nodes[:max_count]
    
    def _merge_and_filter_edges(self, *edge_lists: List[Edge]) -> List[Edge]:
        """
        Merge multiple edge lists, removing duplicates and filtering by confidence.
        """
        # Use set for deduplication
        seen_edges: Set[Tuple[str, str, str]] = set()
        merged_edges: List[Edge] = []
        
        for edge_list in edge_lists:
            for edge in edge_list:
                edge_key = (edge.source, edge.target, edge.type.value)
                
                # Skip duplicates
                if edge_key in seen_edges:
                    continue
                
                # Filter by confidence if available
                if edge.metadata and 'confidence' in edge.metadata:
                    if edge.metadata['confidence'] < 0.5:
                        continue
                
                seen_edges.add(edge_key)
                merged_edges.append(edge)
        
        logging.info(f"Merged {sum(len(el) for el in edge_lists)} edges into {len(merged_edges)} unique edges")
        return merged_edges
