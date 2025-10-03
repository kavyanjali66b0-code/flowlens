"""
M3.0 Phase 4.4: Semantic Query Engine

Provides a powerful query interface for searching and exploring semantic entities
across Data Flow, DI Resolution, and Type Inference analyzers.

Features:
- Entity search by name, type, kind
- Relationship queries (dependencies, call graphs)
- Data flow tracing
- Type-based filtering
- Confidence-based filtering
- Performance optimization (M3.0 Phase 4.6)
"""

import re
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass

from .unified_semantic_analyzer import (
    UnifiedSemanticAnalyzer,
    SemanticResult,
    SemanticEntity,
    Location
)
from .type_inference import TypeInfo, TypeCategory
from .performance_optimizer import PerformanceOptimizer, make_query_key


@dataclass
class QueryResult:
    """Result from a semantic query."""
    entities: List[SemanticEntity]
    total_count: int
    query_time_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticQueryEngine:
    """
    Advanced query engine for semantic code analysis.
    
    Provides rich query capabilities over analyzed code:
    - Find entities by name, type, or kind
    - Trace dependencies and call relationships
    - Filter by confidence scores
    - Explore data flow chains
    
    M3.0 Phase 4.6: Enhanced with performance optimization:
    - Entity indexing for O(1) lookups
    - Query result caching
    - Performance profiling
    """
    
    def __init__(
        self,
        analyzer: UnifiedSemanticAnalyzer = None,
        enable_optimization: bool = True,
        enable_caching: bool = True,
        enable_profiling: bool = False
    ):
        """
        Initialize query engine.
        
        Args:
            analyzer: UnifiedSemanticAnalyzer instance (optional, will create if not provided)
            enable_optimization: Enable entity indexing for faster lookups
            enable_caching: Enable query result caching
            enable_profiling: Enable performance profiling
        """
        self.analyzer = analyzer or UnifiedSemanticAnalyzer()
        self._cached_result: Optional[SemanticResult] = None
        
        # M3.0 Phase 4.6: Performance optimizer
        self.optimizer = PerformanceOptimizer(
            enable_indexing=enable_optimization,
            enable_caching=enable_caching,
            enable_profiling=enable_profiling
        )
    
    def _ensure_index(self):
        """Ensure index is built for current cached result."""
        if (self._cached_result and 
            self._cached_result.semantic_entities and 
            self.optimizer.index and
            len(self.optimizer.index.get_all_entities()) == 0):
            # Index not built yet, build it now
            self.optimizer.optimize_result(self._cached_result)
    
    # === Core Query Methods ===
    
    def find_by_name(
        self,
        name_pattern: str,
        exact_match: bool = False,
        case_sensitive: bool = False
    ) -> List[SemanticEntity]:
        """
        Find entities by name pattern.
        
        Args:
            name_pattern: Name or regex pattern to search for
            exact_match: If True, require exact name match
            case_sensitive: If True, case-sensitive search
        
        Returns:
            List of matching entities
        
        Example:
            >>> engine.find_by_name("user", exact_match=False)
            [SemanticEntity(name="userService"), SemanticEntity(name="getUser"), ...]
        """
        # M3.0 Phase 4.6: Profile query time
        self.optimizer.start_timing("find_by_name")
        
        # M3.0 Phase 4.6: Check cache first
        query_key = make_query_key("find_by_name", name_pattern=name_pattern, 
                                   exact_match=exact_match, case_sensitive=case_sensitive)
        cached = self.optimizer.get_cached_query(query_key)
        if cached is not None:
            self.optimizer.end_timing("find_by_name")
            return cached
        
        if not self._cached_result or not self._cached_result.semantic_entities:
            self.optimizer.end_timing("find_by_name")
            return []
        
        # M3.0 Phase 4.6: Ensure index is built
        self._ensure_index()
        
        # M3.0 Phase 4.6: Use index for exact matches (index is always case-insensitive)
        if exact_match and self.optimizer.index and not case_sensitive:
            results = self.optimizer.index.find_by_name_exact(name_pattern)
        elif exact_match and case_sensitive:
            # Case-sensitive requires linear search
            entities = self._cached_result.semantic_entities
            results = [e for e in entities if e.name == name_pattern]
        else:
            # Pattern matching (fallback to linear search)
            entities = self._cached_result.semantic_entities
            
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(name_pattern, flags)
                results = [e for e in entities if pattern.search(e.name)]
            except re.error:
                # Invalid regex, fall back to substring match
                if case_sensitive:
                    results = [e for e in entities if name_pattern in e.name]
                else:
                    pattern_lower = name_pattern.lower()
                    results = [e for e in entities if pattern_lower in e.name.lower()]
        
        # M3.0 Phase 4.6: Cache results
        self.optimizer.cache_query_result(query_key, results)
        self.optimizer.end_timing("find_by_name")
        
        return results
    
    def find_by_type(
        self,
        type_name: str,
        exact_match: bool = True
    ) -> List[SemanticEntity]:
        """
        Find entities with a specific type.
        
        Args:
            type_name: Type name to search for
            exact_match: If True, require exact type match
        
        Returns:
            List of entities with matching type
        
        Example:
            >>> engine.find_by_type("string")
            [SemanticEntity(name="userName", type_info=TypeInfo(name="string")), ...]
        """
        # M3.0 Phase 4.6: Profile query time
        self.optimizer.start_timing("find_by_type")
        
        # M3.0 Phase 4.6: Check cache first
        query_key = make_query_key("find_by_type", type_name=type_name, exact_match=exact_match)
        cached = self.optimizer.get_cached_query(query_key)
        if cached is not None:
            self.optimizer.end_timing("find_by_type")
            return cached
        
        if not self._cached_result or not self._cached_result.semantic_entities:
            self.optimizer.end_timing("find_by_type")
            return []
        
        # M3.0 Phase 4.6: Ensure index is built
        self._ensure_index()
        
        # M3.0 Phase 4.6: Use index for exact matches
        if exact_match and self.optimizer.index:
            results = self.optimizer.index.find_by_type_exact(type_name)
        else:
            # Fallback to linear search for partial matches
            entities = []
            for entity in self._cached_result.semantic_entities:
                if not entity.type_info:
                    continue
                
                entity_type_name = self._get_type_name(entity.type_info)
                if not entity_type_name:
                    continue
                
                if type_name.lower() in entity_type_name.lower():
                    entities.append(entity)
            
            results = entities
        
        # M3.0 Phase 4.6: Cache results
        self.optimizer.cache_query_result(query_key, results)
        self.optimizer.end_timing("find_by_type")
        
        return results
    
    def find_by_kind(self, kind: str) -> List[SemanticEntity]:
        """
        Find entities by kind (variable, function, service, etc.).
        
        Args:
            kind: Entity kind to filter by
        
        Returns:
            List of entities with matching kind
        
        Example:
            >>> engine.find_by_kind("function")
            [SemanticEntity(name="processData", kind="function"), ...]
        """
        # M3.0 Phase 4.6: Profile query time
        self.optimizer.start_timing("find_by_kind")
        
        # M3.0 Phase 4.6: Check cache first
        query_key = make_query_key("find_by_kind", kind=kind)
        cached = self.optimizer.get_cached_query(query_key)
        if cached is not None:
            self.optimizer.end_timing("find_by_kind")
            return cached
        
        if not self._cached_result or not self._cached_result.semantic_entities:
            self.optimizer.end_timing("find_by_kind")
            return []
        
        # M3.0 Phase 4.6: Ensure index is built
        self._ensure_index()
        
        # M3.0 Phase 4.6: Use index for lookups
        if self.optimizer.index:
            results = self.optimizer.index.find_by_kind(kind)
        else:
            # Fallback to linear search
            results = [e for e in self._cached_result.semantic_entities if e.kind == kind]
        
        # M3.0 Phase 4.6: Cache results
        self.optimizer.cache_query_result(query_key, results)
        self.optimizer.end_timing("find_by_kind")
        
        return results
    
    def find_dependencies(
        self,
        entity_name: str,
        include_transitive: bool = False
    ) -> List[SemanticEntity]:
        """
        Find all dependencies of an entity.
        
        Args:
            entity_name: Name of entity to find dependencies for
            include_transitive: If True, include indirect dependencies
        
        Returns:
            List of entities that the specified entity depends on
        
        Example:
            >>> engine.find_dependencies("UserController")
            [SemanticEntity(name="UserService"), SemanticEntity(name="Logger"), ...]
        """
        if not self._cached_result or not self._cached_result.semantic_entities:
            return []
        
        # Find the entity
        entity = next(
            (e for e in self._cached_result.semantic_entities if e.name == entity_name),
            None
        )
        if not entity:
            return []
        
        # Get direct dependencies (things this entity calls)
        dep_names = entity.calls
        dependencies = [
            e for e in self._cached_result.semantic_entities
            if e.name in dep_names
        ]
        
        if include_transitive:
            # Recursively find dependencies
            visited = {entity_name}
            seen_entities = {e.name for e in dependencies}  # Track added entities
            to_visit = list(dep_names)
            
            while to_visit:
                current_name = to_visit.pop(0)
                if current_name in visited:
                    continue
                visited.add(current_name)
                
                current = next(
                    (e for e in self._cached_result.semantic_entities if e.name == current_name),
                    None
                )
                if current:
                    # Add children of current to visit list
                    to_visit.extend(c for c in current.calls if c not in visited)
                    # Only add to results if not already there
                    if current.name not in seen_entities:
                        dependencies.append(current)
                        seen_entities.add(current.name)
        
        return dependencies
    
    def find_dependents(
        self,
        entity_name: str,
        include_transitive: bool = False
    ) -> List[SemanticEntity]:
        """
        Find all entities that depend on the specified entity.
        
        Args:
            entity_name: Name of entity to find dependents for
            include_transitive: If True, include indirect dependents
        
        Returns:
            List of entities that depend on the specified entity
        
        Example:
            >>> engine.find_dependents("UserService")
            [SemanticEntity(name="UserController"), SemanticEntity(name="AuthService"), ...]
        """
        if not self._cached_result or not self._cached_result.semantic_entities:
            return []
        
        # Find the entity
        entity = next(
            (e for e in self._cached_result.semantic_entities if e.name == entity_name),
            None
        )
        if not entity:
            return []
        
        # Get direct dependents (things that call this entity)
        dependent_names = entity.called_by
        dependents = [
            e for e in self._cached_result.semantic_entities
            if e.name in dependent_names
        ]
        
        if include_transitive:
            # Recursively find dependents
            visited = {entity_name}
            seen_entities = {e.name for e in dependents}  # Track added entities
            to_visit = list(dependent_names)
            
            while to_visit:
                current_name = to_visit.pop(0)
                if current_name in visited:
                    continue
                visited.add(current_name)
                
                current = next(
                    (e for e in self._cached_result.semantic_entities if e.name == current_name),
                    None
                )
                if current:
                    # Add parents of current to visit list
                    to_visit.extend(c for c in current.called_by if c not in visited)
                    # Only add to results if not already there
                    if current.name not in seen_entities:
                        dependents.append(current)
                        seen_entities.add(current.name)
        
        return dependents
    
    def trace_data_flow(
        self,
        variable_name: str,
        max_depth: int = 10
    ) -> List[SemanticEntity]:
        """
        Trace data flow from a variable.
        
        Args:
            variable_name: Name of variable to trace
            max_depth: Maximum depth to trace (prevents infinite loops)
        
        Returns:
            List of entities in the data flow chain
        
        Example:
            >>> engine.trace_data_flow("userData")
            [SemanticEntity(name="userData"), SemanticEntity(name="processData"), ...]
        """
        if not self._cached_result or not self._cached_result.semantic_entities:
            return []
        
        # Find the starting entity
        entity = next(
            (e for e in self._cached_result.semantic_entities if e.name == variable_name),
            None
        )
        if not entity:
            return []
        
        # Trace through calls (data flows through function calls)
        flow_chain = [entity]
        visited = {variable_name}
        current = entity
        depth = 0
        
        while depth < max_depth and current.calls:
            # Follow first call (simplified - could be made more sophisticated)
            next_name = current.calls[0]
            if next_name in visited:
                break
            
            next_entity = next(
                (e for e in self._cached_result.semantic_entities if e.name == next_name),
                None
            )
            if not next_entity:
                break
            
            flow_chain.append(next_entity)
            visited.add(next_name)
            current = next_entity
            depth += 1
        
        return flow_chain
    
    def find_related_entities(
        self,
        entity_name: str,
        relation_types: List[str] = None
    ) -> Dict[str, List[SemanticEntity]]:
        """
        Find all entities related to the specified entity.
        
        Args:
            entity_name: Name of entity to find relations for
            relation_types: List of relation types to include
                           (calls, called_by, same_type, dependencies)
        
        Returns:
            Dictionary mapping relation types to lists of entities
        
        Example:
            >>> engine.find_related_entities("UserService")
            {
                "calls": [SemanticEntity(name="Database"), ...],
                "called_by": [SemanticEntity(name="UserController"), ...],
                "same_type": [SemanticEntity(name="AdminService"), ...]
            }
        """
        if not self._cached_result or not self._cached_result.semantic_entities:
            return {}
        
        if relation_types is None:
            relation_types = ["calls", "called_by", "same_type"]
        
        entity = next(
            (e for e in self._cached_result.semantic_entities if e.name == entity_name),
            None
        )
        if not entity:
            return {}
        
        related = {}
        
        if "calls" in relation_types:
            related["calls"] = [
                e for e in self._cached_result.semantic_entities
                if e.name in entity.calls
            ]
        
        if "called_by" in relation_types:
            related["called_by"] = [
                e for e in self._cached_result.semantic_entities
                if e.name in entity.called_by
            ]
        
        if "same_type" in relation_types and entity.type_info:
            entity_type = self._get_type_name(entity.type_info)
            if entity_type:
                related["same_type"] = [
                    e for e in self._cached_result.semantic_entities
                    if e.name != entity_name and e.type_info and 
                    self._get_type_name(e.type_info) == entity_type
                ]
        
        if "dependencies" in relation_types and entity.di_info:
            # Entities with DI info that are related
            related["dependencies"] = self.find_dependencies(entity_name)
        
        return related
    
    def filter_by_confidence(
        self,
        entities: List[SemanticEntity] = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0
    ) -> List[SemanticEntity]:
        """
        Filter entities by confidence score.
        
        Args:
            entities: List of entities to filter (None = all entities)
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
        
        Returns:
            Filtered list of entities
        
        Example:
            >>> engine.filter_by_confidence(min_confidence=0.9)  # High confidence only
            [SemanticEntity(name="userService", confidence=0.95), ...]
        """
        if entities is None:
            if not self._cached_result or not self._cached_result.semantic_entities:
                return []
            entities = self._cached_result.semantic_entities
        
        return [
            e for e in entities
            if min_confidence <= e.confidence <= max_confidence
        ]
    
    # === Query Building Interface ===
    
    def query(self, code: str, language: str = "javascript") -> 'QueryBuilder':
        """
        Start a query builder for the given code.
        
        Args:
            code: Source code to analyze
            language: Programming language
        
        Returns:
            QueryBuilder instance for chaining queries
        
        Example:
            >>> results = engine.query(code) \\
            ...     .with_name("user") \\
            ...     .with_min_confidence(0.8) \\
            ...     .execute()
        """
        # M3.0 Phase 4.6: Profile analysis time
        self.optimizer.start_timing("analysis")
        
        # Analyze the code
        result = self.analyzer.analyze(code, language=language)
        self._cached_result = result
        
        # M3.0 Phase 4.6: Optimize result (build indexes)
        self.optimizer.optimize_result(result)
        
        analysis_time = self.optimizer.end_timing("analysis")
        
        return QueryBuilder(self, result)
    
    # === Helper Methods ===
    
    def _get_type_name(self, type_info: TypeInfo) -> Optional[str]:
        """Extract a simple type name from TypeInfo."""
        if hasattr(type_info, 'name') and type_info.name:
            return type_info.name
        if hasattr(type_info, 'primitive_type') and type_info.primitive_type:
            return str(type_info.primitive_type)
        return None
    
    # === M3.0 Phase 4.6: Performance Metrics ===
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the query engine.
        
        Returns:
            Dictionary with performance metrics including:
            - analysis: Analysis time stats
            - queries: Query count, cache hits/misses
            - memory: Entity count, index size
            - profiles: Individual operation timings
        
        Example:
            >>> metrics = engine.get_performance_metrics()
            >>> print(f"Cache hit rate: {metrics['queries']['hit_rate_pct']}%")
        """
        metrics = self.optimizer.get_metrics()
        
        return {
            'analysis': {
                'total_ms': metrics.total_analysis_time_ms,
            },
            'queries': {
                'total': metrics.total_queries,
                'cache_hits': metrics.cache_hits,
                'cache_misses': metrics.cache_misses,
                'hit_rate_pct': metrics.cache_hit_rate(),
            },
            'memory': {
                'entity_count': metrics.entity_count,
                'index_size_bytes': metrics.index_size_bytes,
            },
            'profiles': self.optimizer.get_profile_stats() if self.optimizer.profiler else {},
        }
    
    def clear_cache(self):
        """
        Clear query cache.
        
        Useful for freeing memory or forcing fresh query execution.
        """
        if self.optimizer.cache:
            self.optimizer.cache.clear()
    
    def reset_performance_metrics(self):
        """
        Reset all performance tracking data.
        
        Clears cache, indexes, and profiling stats.
        """
        if self.optimizer.cache:
            self.optimizer.cache.clear()
        if self.optimizer.index:
            self.optimizer.index.clear()
        if self.optimizer.profiler:
            self.optimizer.profiler.clear()


class QueryBuilder:
    """
    Fluent interface for building complex queries.
    
    Allows chaining of query conditions:
        engine.query(code)
            .with_name("user")
            .with_type("string")
            .with_min_confidence(0.8)
            .execute()
    """
    
    def __init__(self, engine: SemanticQueryEngine, result: SemanticResult):
        """
        Initialize query builder.
        
        Args:
            engine: SemanticQueryEngine instance
            result: SemanticResult from analysis
        """
        self.engine = engine
        self.result = result
        self._filters: List[Callable[[SemanticEntity], bool]] = []
    
    def with_name(self, name_pattern: str, exact_match: bool = False) -> 'QueryBuilder':
        """Add name filter."""
        if exact_match:
            self._filters.append(lambda e: e.name == name_pattern)
        else:
            pattern = re.compile(name_pattern, re.IGNORECASE)
            self._filters.append(lambda e: pattern.search(e.name) is not None)
        return self
    
    def with_type(self, type_name: str) -> 'QueryBuilder':
        """Add type filter."""
        def type_filter(e: SemanticEntity) -> bool:
            if not e.type_info:
                return False
            entity_type = self.engine._get_type_name(e.type_info)
            return entity_type == type_name if entity_type else False
        
        self._filters.append(type_filter)
        return self
    
    def with_kind(self, kind: str) -> 'QueryBuilder':
        """Add kind filter."""
        self._filters.append(lambda e: e.kind == kind)
        return self
    
    def with_min_confidence(self, min_conf: float) -> 'QueryBuilder':
        """Add minimum confidence filter."""
        self._filters.append(lambda e: e.confidence >= min_conf)
        return self
    
    def with_calls(self, target: str) -> 'QueryBuilder':
        """Filter entities that call the specified target."""
        self._filters.append(lambda e: target in e.calls)
        return self
    
    def with_called_by(self, caller: str) -> 'QueryBuilder':
        """Filter entities called by the specified caller."""
        self._filters.append(lambda e: caller in e.called_by)
        return self
    
    def execute(self) -> QueryResult:
        """
        Execute the query and return results.
        
        Returns:
            QueryResult with filtered entities
        """
        import time
        start_time = time.time()
        
        entities = self.result.semantic_entities or []
        
        # Apply all filters
        for filter_func in self._filters:
            entities = [e for e in entities if filter_func(e)]
        
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QueryResult(
            entities=entities,
            total_count=len(entities),
            query_time_ms=query_time,
            metadata={"filters_applied": len(self._filters)}
        )
