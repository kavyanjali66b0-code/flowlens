"""
M3.0 Phase 4.6: Performance Optimization

Optimizations for semantic analysis and query performance:
1. Entity Indexing - O(1) lookups by name, type, kind
2. Query Result Caching - Avoid re-computing identical queries
3. Performance Profiling - Measure and track performance metrics
4. Memory Optimization - Efficient data structures
"""

import time
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass, field

from .unified_semantic_analyzer import SemanticEntity, SemanticResult


@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis and queries."""
    
    # Analysis metrics
    total_analysis_time_ms: float = 0.0
    parse_time_ms: float = 0.0
    data_flow_time_ms: float = 0.0
    di_resolution_time_ms: float = 0.0
    type_inference_time_ms: float = 0.0
    correlation_time_ms: float = 0.0
    
    # Query metrics
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time_ms: float = 0.0
    
    # Memory metrics
    entity_count: int = 0
    index_size_bytes: int = 0
    cache_size_bytes: int = 0
    
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'analysis': {
                'total_ms': self.total_analysis_time_ms,
                'parse_ms': self.parse_time_ms,
                'data_flow_ms': self.data_flow_time_ms,
                'di_resolution_ms': self.di_resolution_time_ms,
                'type_inference_ms': self.type_inference_time_ms,
                'correlation_ms': self.correlation_time_ms,
            },
            'queries': {
                'total': self.total_queries,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'avg_time_ms': self.avg_query_time_ms,
                'hit_rate_pct': self.cache_hit_rate(),
            },
            'memory': {
                'entities': self.entity_count,
                'index_bytes': self.index_size_bytes,
                'cache_bytes': self.cache_size_bytes,
            }
        }


class EntityIndex:
    """
    Optimized index for fast entity lookups.
    
    Provides O(1) average-case lookups by:
    - name (exact match)
    - type
    - kind
    - confidence range
    """
    
    def __init__(self):
        """Initialize empty indexes."""
        # Primary indexes
        self._by_name: Dict[str, List[SemanticEntity]] = defaultdict(list)
        self._by_type: Dict[str, List[SemanticEntity]] = defaultdict(list)
        self._by_kind: Dict[str, List[SemanticEntity]] = defaultdict(list)
        
        # Secondary indexes
        self._by_confidence: Dict[str, List[SemanticEntity]] = defaultdict(list)  # bucketed
        self._all_entities: List[SemanticEntity] = []
        
        # Dependency indexes
        self._callers: Dict[str, Set[str]] = defaultdict(set)  # who calls this
        self._callees: Dict[str, Set[str]] = defaultdict(set)  # who this calls
        
        # Stats
        self._index_size_bytes: int = 0
    
    def build(self, entities: List[SemanticEntity]) -> None:
        """
        Build indexes from entity list.
        
        Args:
            entities: List of semantic entities to index
        """
        # Clear existing indexes
        self.clear()
        
        # Store all entities
        self._all_entities = entities
        
        # Build indexes
        for entity in entities:
            # Index by name (lowercase for case-insensitive search)
            self._by_name[entity.name.lower()].append(entity)
            
            # Index by type
            if entity.type_info:
                type_name = self._get_type_name(entity.type_info)
                self._by_type[type_name.lower()].append(entity)
            
            # Index by kind
            self._by_kind[entity.kind].append(entity)
            
            # Index by confidence bucket (0.0-0.2, 0.2-0.4, ...)
            confidence_bucket = self._get_confidence_bucket(entity.confidence)
            self._by_confidence[confidence_bucket].append(entity)
            
            # Index call relationships
            for callee in entity.calls:
                self._callees[entity.name].add(callee)
                self._callers[callee].add(entity.name)
        
        # Calculate index size (rough estimate)
        self._index_size_bytes = (
            len(self._all_entities) * 1000 +  # ~1KB per entity
            len(self._by_name) * 100 +
            len(self._by_type) * 100 +
            len(self._by_kind) * 100
        )
    
    def find_by_name_exact(self, name: str) -> List[SemanticEntity]:
        """O(1) exact name lookup (case-insensitive)."""
        return self._by_name.get(name.lower(), [])
    
    def find_by_type_exact(self, type_name: str) -> List[SemanticEntity]:
        """O(1) exact type lookup."""
        return self._by_type.get(type_name.lower(), [])
    
    def find_by_kind(self, kind: str) -> List[SemanticEntity]:
        """O(1) kind lookup."""
        return self._by_kind.get(kind, [])
    
    def find_by_confidence_range(self, min_conf: float, max_conf: float) -> List[SemanticEntity]:
        """Find entities within confidence range."""
        results = []
        
        # Get relevant confidence buckets
        min_bucket = self._get_confidence_bucket(min_conf)
        max_bucket = self._get_confidence_bucket(max_conf)
        
        # Collect from all relevant buckets
        bucket_keys = [f"{i/10:.1f}-{(i+2)/10:.1f}" 
                       for i in range(0, 10, 2)]
        
        for bucket in bucket_keys:
            bucket_min = float(bucket.split('-')[0])
            bucket_max = float(bucket.split('-')[1])
            
            # Check if bucket overlaps with range
            if bucket_max >= min_conf and bucket_min <= max_conf:
                for entity in self._by_confidence[bucket]:
                    if min_conf <= entity.confidence <= max_conf:
                        results.append(entity)
        
        return results
    
    def get_callers(self, entity_name: str) -> Set[str]:
        """Get entities that call this entity."""
        return self._callers.get(entity_name, set())
    
    def get_callees(self, entity_name: str) -> Set[str]:
        """Get entities that this entity calls."""
        return self._callees.get(entity_name, set())
    
    def get_all_entities(self) -> List[SemanticEntity]:
        """Get all indexed entities."""
        return self._all_entities
    
    def get_size_bytes(self) -> int:
        """Get estimated index size in bytes."""
        return self._index_size_bytes
    
    def clear(self) -> None:
        """Clear all indexes."""
        self._by_name.clear()
        self._by_type.clear()
        self._by_kind.clear()
        self._by_confidence.clear()
        self._callers.clear()
        self._callees.clear()
        self._all_entities = []
        self._index_size_bytes = 0
    
    def _get_type_name(self, type_info: Any) -> str:
        """Extract type name from TypeInfo object."""
        if hasattr(type_info, 'name'):
            return type_info.name
        return str(type_info)
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for indexing."""
        # Buckets: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        bucket_index = min(int(confidence * 5) * 2, 8)
        bucket_min = bucket_index / 10
        bucket_max = (bucket_index + 2) / 10
        return f"{bucket_min:.1f}-{bucket_max:.1f}"


class QueryCache:
    """
    LRU cache for query results.
    
    Caches query results to avoid re-computation.
    Uses query signature (method + params) as key.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
        """
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._cache_size_bytes = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            key: Query signature
            
        Returns:
            Cached result or None if not found
        """
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        self._misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Store query result.
        
        Args:
            key: Query signature
            value: Query result to cache
        """
        # If key exists, update and move to end
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            self._cache[key] = value
            return
        
        # If cache is full, evict oldest
        if len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        # Add new entry
        self._cache[key] = value
        self._access_order.append(key)
        
        # Update size estimate
        self._cache_size_bytes = len(self._cache) * 1000  # ~1KB per entry
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
        self._cache_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate_pct': hit_rate,
            'size_bytes': self._cache_size_bytes,
        }
    
    def get_size_bytes(self) -> int:
        """Get estimated cache size in bytes."""
        return self._cache_size_bytes


class PerformanceProfiler:
    """
    Performance profiler for tracking execution time.
    
    Measures time spent in different analysis phases and queries.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of operation to time
        """
        self._start_times[operation] = time.time()
    
    def end(self, operation: str) -> float:
        """
        End timing an operation.
        
        Args:
            operation: Name of operation
            
        Returns:
            Time elapsed in milliseconds
        """
        if operation not in self._start_times:
            return 0.0
        
        elapsed = (time.time() - self._start_times[operation]) * 1000
        self._timings[operation].append(elapsed)
        del self._start_times[operation]
        
        return elapsed
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Name of operation
            
        Returns:
            Dict with min, max, avg, total times
        """
        times = self._timings.get(operation, [])
        
        if not times:
            return {
                'count': 0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'avg_ms': 0.0,
                'total_ms': 0.0,
            }
        
        return {
            'count': len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': sum(times) / len(times),
            'total_ms': sum(times),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            operation: self.get_stats(operation)
            for operation in self._timings.keys()
        }
    
    def clear(self) -> None:
        """Clear all timing data."""
        self._timings.clear()
        self._start_times.clear()


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Combines entity indexing, query caching, and profiling
    to optimize semantic analysis performance.
    """
    
    def __init__(
        self,
        enable_indexing: bool = True,
        enable_caching: bool = True,
        enable_profiling: bool = False,
        cache_size: int = 100
    ):
        """
        Initialize performance optimizer.
        
        Args:
            enable_indexing: Enable entity indexing
            enable_caching: Enable query caching
            enable_profiling: Enable performance profiling
            cache_size: Maximum cache size
        """
        self.enable_indexing = enable_indexing
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        
        # Components
        self.index = EntityIndex() if enable_indexing else None
        self.cache = QueryCache(max_size=cache_size) if enable_caching else None
        self.profiler = PerformanceProfiler() if enable_profiling else None
        
        # Metrics
        self.metrics = PerformanceMetrics()
    
    def optimize_result(self, result: SemanticResult) -> None:
        """
        Optimize semantic result with indexing.
        
        Args:
            result: Semantic result to optimize
        """
        if self.enable_indexing and self.index:
            if self.enable_profiling and self.profiler:
                self.profiler.start('indexing')
            
            self.index.build(result.semantic_entities)
            
            if self.enable_profiling and self.profiler:
                self.profiler.end('indexing')
            
            # Update metrics
            self.metrics.entity_count = len(result.semantic_entities)
            self.metrics.index_size_bytes = self.index.get_size_bytes()
    
    def get_cached_query(self, query_key: str) -> Optional[Any]:
        """
        Get cached query result.
        
        Args:
            query_key: Query signature
            
        Returns:
            Cached result or None
        """
        if not self.enable_caching or not self.cache:
            return None
        
        result = self.cache.get(query_key)
        
        if result is not None:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        self.metrics.total_queries += 1
        
        return result
    
    def cache_query_result(self, query_key: str, result: Any) -> None:
        """
        Cache query result.
        
        Args:
            query_key: Query signature
            result: Query result to cache
        """
        if self.enable_caching and self.cache:
            self.cache.put(query_key, result)
            self.metrics.cache_size_bytes = self.cache.get_size_bytes()
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        if self.enable_profiling and self.profiler:
            self.profiler.start(operation)
    
    def end_timing(self, operation: str) -> float:
        """End timing an operation and return elapsed time."""
        if self.enable_profiling and self.profiler:
            return self.profiler.end(operation)
        return 0.0
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Update cache stats
        if self.cache:
            cache_stats = self.cache.get_stats()
            self.metrics.cache_hits = cache_stats['hits']
            self.metrics.cache_misses = cache_stats['misses']
        
        return self.metrics
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        if self.enable_profiling and self.profiler:
            return self.profiler.get_all_stats()
        return {}
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        if self.cache:
            self.cache.clear()
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics()
        if self.profiler:
            self.profiler.clear()


# Utility function for generating query keys
def make_query_key(method: str, **kwargs) -> str:
    """
    Generate a cache key for a query.
    
    Args:
        method: Query method name
        **kwargs: Query parameters
        
    Returns:
        String key for caching
    """
    # Sort kwargs for consistent keys
    params = sorted(kwargs.items())
    param_str = ",".join(f"{k}={v}" for k, v in params)
    return f"{method}({param_str})"
