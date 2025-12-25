"""
M3.0 Phase 4: Unified Semantic Analyzer

Orchestrates Data Flow Analyzer, DI Resolver, and Type Inference Engine
to provide comprehensive semantic code analysis.

This is the culmination of M3.0, bringing together three powerful analyzers:
- Phase 1: Data Flow Analyzer (52/52 tests)
- Phase 2: DI Resolver (56/56 tests)  
- Phase 3: Type Inference Engine (63/64 tests)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from enum import Enum

# Import our three analyzers
from .data_flow_analyzer import DataFlowAnalyzer
from .data_flow import DataFlowGraph, DataFlowNode
from .di_resolver import DIResolver, Dependency
from .type_inference import TypeInferenceEngine, TypeInfo
from .parser import LanguageParser
from .symbol_table import SymbolTable


@dataclass
class Location:
    """Location in source code."""
    file: str
    line: int
    column: int = 0
    
    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class AnalysisError:
    """Error encountered during analysis."""
    analyzer: str  # "data_flow", "di", "type_inference", "correlation"
    error_type: str  # "parse_error", "runtime_error", "timeout"
    message: str
    location: Optional[Location] = None
    severity: str = "error"  # "error", "warning", "info"
    
    def __str__(self):
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.upper()}] {self.analyzer}: {self.message}{loc}"


@dataclass
class SemanticResult:
    """
    Unified result from all analyzers.
    Contains data from individual analyzers plus correlated insights.
    """
    
    # === Data Flow Analysis Results ===
    data_flow_graph: Optional[DataFlowGraph] = None
    """Complete data flow graph from DataFlowAnalyzer"""
    
    # === DI Resolution Results ===
    dependencies: Dict[str, List[Dependency]] = field(default_factory=dict)
    """Dependencies from DIResolver"""
    
    # === Type Inference Results ===
    type_map: Dict[str, TypeInfo] = field(default_factory=dict)
    type_errors: List[str] = field(default_factory=list)
    
    # === Correlated Insights (computed by SemanticAnalyzer) ===
    typed_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    """Dependencies enriched with type information"""
    
    typed_data_flow: List[Dict[str, Any]] = field(default_factory=list)
    """Data flow edges with type information"""
    
    semantic_entities: List['SemanticEntity'] = field(default_factory=list)
    """Unified entities with data from all analyzers"""
    
    # === Metadata ===
    file_path: Optional[str] = None
    language: str = "javascript"
    analysis_time_ms: float = 0.0
    errors: List[AnalysisError] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if analysis encountered any errors."""
        return len([e for e in self.errors if e.severity == "error"]) > 0
    
    def get_warnings(self) -> List[AnalysisError]:
        """Get all warnings."""
        return [e for e in self.errors if e.severity == "warning"]
    
    def get_errors(self) -> List[AnalysisError]:
        """Get all errors."""
        return [e for e in self.errors if e.severity == "error"]


@dataclass
class SemanticEntity:
    """
    Unified entity with data from all analyzers.
    Represents a single code entity (variable, function, service, etc.)
    with correlated information from multiple analysis passes.
    """
    
    # === Identity ===
    name: str
    kind: str  # "variable", "function", "service", "parameter", "class"
    location: Optional[Location] = None
    
    # === Data Flow Info ===
    data_flow: Optional[Dict[str, Any]] = None
    # Example:
    # {
    #     "defined_at": Location(...),
    #     "used_at": [Location(...), Location(...)],
    #     "scope": "function",
    #     "flows_to": ["otherVar"],
    #     "flows_from": ["sourceVar"]
    # }
    
    # === DI Info (if applicable) ===
    di_info: Optional[Dict[str, Any]] = None
    # Example:
    # {
    #     "is_service": True,
    #     "service_type": "singleton",
    #     "provides": "UserService",
    #     "dependencies": ["Database", "Logger"],
    #     "injected_by": ["AuthController"]
    # }
    
    # === Type Info ===
    type_info: Optional[TypeInfo] = None
    
    # === Relationships ===
    calls: List[str] = field(default_factory=list)  # Functions this entity calls
    called_by: List[str] = field(default_factory=list)  # Who calls this entity
    
    # === Metadata ===
    confidence: float = 1.0  # Confidence in correlation (0-1)
    analysis_notes: List[str] = field(default_factory=list)  # Issues, warnings


@dataclass
class CorrelationKey:
    """
    Key for matching entities across analyzers.
    Used to identify the same entity in different analyzer results.
    """
    name: str
    kind: str
    scope: Optional[str] = None
    
    def __hash__(self):
        return hash((self.name, self.kind, self.scope))
    
    def __eq__(self, other):
        if not isinstance(other, CorrelationKey):
            return False
        return (self.name == other.name and 
                self.kind == other.kind and 
                self.scope == other.scope)


class UnifiedSemanticAnalyzer:
    """
    Unified semantic analyzer integrating data flow, DI, and type inference.
    
    This is the culmination of M3.0 Phase 4, orchestrating three analyzers:
    1. DataFlowAnalyzer (Phase 1)
    2. DIResolver (Phase 2)
    3. TypeInferenceEngine (Phase 3)
    
    Features:
    - Shared AST parsing (parse once, analyze thrice)
    - Result caching for performance
    - Graceful degradation (if one analyzer fails, others continue)
    - Cross-analyzer correlation for unified insights
    """
    
    def __init__(
        self,
        enable_data_flow: bool = True,
        enable_di_resolution: bool = True,
        enable_type_inference: bool = True,
        cache_results: bool = True,
        timeout_seconds: int = 30
    ):
        """
        Initialize unified semantic analyzer.
        
        Args:
            enable_data_flow: Enable data flow analysis
            enable_di_resolution: Enable DI resolution
            enable_type_inference: Enable type inference
            cache_results: Enable result caching
            timeout_seconds: Timeout for analysis operations
        """
        self.enable_data_flow = enable_data_flow
        self.enable_di_resolution = enable_di_resolution
        self.enable_type_inference = enable_type_inference
        self.cache_results = cache_results
        self.timeout_seconds = timeout_seconds
        
        # Lazy initialization of analyzers (only create when needed)
        self._data_flow_analyzer = None
        self._di_resolver = None
        self._type_engine = None
        self._parser = None
        
        # Result cache: code hash -> SemanticResult
        self._cache: Dict[int, SemanticResult] = {}
        
        self.logger = logging.getLogger(__name__)
    
    # === Lazy Analyzer Initialization ===
    
    @property
    def data_flow_analyzer(self) -> DataFlowAnalyzer:
        """Lazy initialization of DataFlowAnalyzer."""
        if self._data_flow_analyzer is None:
            self._data_flow_analyzer = DataFlowAnalyzer()
            self.logger.debug("Initialized DataFlowAnalyzer")
        return self._data_flow_analyzer
    
    @property
    def di_resolver(self) -> DIResolver:
        """Lazy initialization of DIResolver."""
        if self._di_resolver is None:
            self._di_resolver = DIResolver()
            self.logger.debug("Initialized DIResolver")
        return self._di_resolver
    
    @property
    def type_engine(self) -> TypeInferenceEngine:
        """Lazy initialization of TypeInferenceEngine."""
        if self._type_engine is None:
            self._type_engine = TypeInferenceEngine()
            self.logger.debug("Initialized TypeInferenceEngine")
        return self._type_engine
    
    @property
    def parser(self) -> LanguageParser:
        """Lazy initialization of LanguageParser."""
        if self._parser is None:
            self._parser = LanguageParser()
            self.logger.debug("Initialized LanguageParser")
        return self._parser
    
    # === Main Analysis Methods ===
    
    def analyze(
        self,
        code: str,
        file_path: Optional[str] = None,
        language: str = "javascript"
    ) -> SemanticResult:
        """
        Perform complete semantic analysis on code.
        
        This is the main entry point. It:
        1. Parses code once (shared AST)
        2. Runs enabled analyzers
        3. Correlates results
        4. Returns unified SemanticResult
        
        Args:
            code: Source code to analyze
            file_path: Optional file path for context
            language: Programming language (default: javascript)
        
        Returns:
            SemanticResult with data from all analyzers
        """
        start_time = time.time()
        
        # Check cache
        cache_key = hash((code, language))
        if self.cache_results and cache_key in self._cache:
            self.logger.info(f"Cache hit for analysis")
            cached_result = self._cache[cache_key]
            cached_result.file_path = file_path  # Update path
            return cached_result
        
        # Initialize result
        result = SemanticResult(
            file_path=file_path,
            language=language
        )
        
        # Phase 1: Parse code (shared by all analyzers)
        tree = self._parse_code(code, language)
        if tree is None:
            result.errors.append(AnalysisError(
                analyzer="parser",
                error_type="parse_error",
                message="Failed to parse code",
                severity="error"
            ))
            result.analysis_time_ms = (time.time() - start_time) * 1000
            return result
        
        root_node = tree.root_node
        
        # Phase 2: Run enabled analyzers (with error handling)
        if self.enable_data_flow:
            self._run_data_flow_analysis(root_node, code, result)
        
        if self.enable_di_resolution:
            self._run_di_resolution(root_node, code, result)
        
        if self.enable_type_inference:
            self._run_type_inference(root_node, code, result)
        
        # Phase 3: Correlate results across analyzers
        try:
            self.correlate_results(result)
        except Exception as e:
            self.logger.exception("Correlation failed")
            result.errors.append(AnalysisError(
                analyzer="correlation",
                error_type="runtime_error",
                message=f"Correlation failed: {str(e)}",
                severity="warning"  # Warning, not error (analysis still useful)
            ))
        
        # Finalize
        result.analysis_time_ms = (time.time() - start_time) * 1000
        
        # Cache result
        if self.cache_results:
            self._cache[cache_key] = result
        
        self.logger.info(
            f"Analysis complete in {result.analysis_time_ms:.1f}ms "
            f"({len(result.errors)} errors, {len(result.get_warnings())} warnings)"
        )
        
        return result
    
    def analyze_file(self, file_path: Path) -> SemanticResult:
        """
        Analyze a file from disk.
        
        Args:
            file_path: Path to file to analyze
        
        Returns:
            SemanticResult for the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Infer language from extension
            language = self._infer_language(file_path)
            
            return self.analyze(code, str(file_path), language)
        except Exception as e:
            self.logger.exception(f"Failed to analyze file {file_path}")
            result = SemanticResult(file_path=str(file_path))
            result.errors.append(AnalysisError(
                analyzer="file_io",
                error_type="runtime_error",
                message=f"Failed to read file: {str(e)}",
                severity="error"
            ))
            return result
    
    def analyze_project(
        self,
        root_path: Path,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, SemanticResult]:
        """
        Analyze an entire project.
        
        Args:
            root_path: Root directory of project
            file_patterns: File patterns to include (e.g., ["*.js", "*.ts"])
        
        Returns:
            Dict mapping file paths to SemanticResults
        """
        if file_patterns is None:
            file_patterns = ["*.js", "*.jsx", "*.ts", "*.tsx"]
        
        results = {}
        
        for pattern in file_patterns:
            for file_path in root_path.rglob(pattern):
                if file_path.is_file():
                    self.logger.info(f"Analyzing {file_path}")
                    results[str(file_path)] = self.analyze_file(file_path)
        
        return results
    
    # === Analyzer Execution (with error handling) ===
    
    def _run_data_flow_analysis(
        self,
        root_node: Any,
        code: str,
        result: SemanticResult
    ) -> None:
        """Run data flow analysis with error handling."""
        try:
            self.logger.debug("Running data flow analysis")
            
            # Data flow analyzer requires SymbolTable and node list
            # For now, create minimal dependencies
            symbol_table = SymbolTable()
            nodes = [root_node]  # Simplified - would need proper node extraction
            
            analyzer = DataFlowAnalyzer(symbol_table, nodes, result.file_path or "unknown")
            data_flow_graph = analyzer.analyze()
            
            result.data_flow_graph = data_flow_graph
            
            self.logger.debug(
                f"Data flow: {len(data_flow_graph.nodes)} nodes, "
                f"{len(data_flow_graph.edges)} edges"
            )
            
        except Exception as e:
            self.logger.exception("Data flow analysis failed")
            result.errors.append(AnalysisError(
                analyzer="data_flow",
                error_type="runtime_error",
                message=f"Data flow analysis failed: {str(e)}",
                severity="error"
            ))
    
    def _run_di_resolution(
        self,
        root_node: Any,
        code: str,
        result: SemanticResult
    ) -> None:
        """Run DI resolution with error handling."""
        try:
            self.logger.debug("Running DI resolution")
            
            # DI resolver requires SymbolTable and node list
            symbol_table = SymbolTable()
            nodes = [root_node]
            
            resolver = DIResolver(symbol_table, nodes, result.file_path or "unknown")
            dependencies = resolver.analyze_dependencies()
            
            result.dependencies = dependencies
            
            self.logger.debug(f"DI: {len(dependencies)} components analyzed")
            
        except Exception as e:
            self.logger.exception("DI resolution failed")
            result.errors.append(AnalysisError(
                analyzer="di",
                error_type="runtime_error",
                message=f"DI resolution failed: {str(e)}",
                severity="error"
            ))
    
    def _run_type_inference(
        self,
        root_node: Any,
        code: str,
        result: SemanticResult
    ) -> None:
        """Run type inference with error handling."""
        try:
            self.logger.debug("Running type inference")
            
            # TypeInferenceEngine accepts AST node directly
            self.type_engine.infer_types(root_node, code)
            
            result.type_map = self.type_engine.type_map.copy()
            
            self.logger.debug(f"Type inference: {len(result.type_map)} types inferred")
            
        except Exception as e:
            self.logger.exception("Type inference failed")
            result.errors.append(AnalysisError(
                analyzer="type_inference",
                error_type="runtime_error",
                message=f"Type inference failed: {str(e)}",
                severity="error"
            ))
    
    # === Correlation Logic ===
    
    def correlate_results(self, result: SemanticResult) -> None:
        """
        Correlate data across analyzers to create unified insights.
        
        This creates:
        - typed_dependencies: Dependencies with type information
        - typed_data_flow: Data flow edges with type information
        - semantic_entities: Unified entities with all analyzer data
        
        Args:
            result: SemanticResult to enrich with correlations
        """
        self.logger.debug("Correlating results across analyzers")
        
        # Correlation 1: Dependencies + Types
        self._correlate_dependencies_with_types(result)
        
        # Correlation 2: Data Flow + Types
        self._correlate_data_flow_with_types(result)
        
        # Correlation 3: Build unified semantic entities
        self._build_semantic_entities(result)
        
        self.logger.debug(
            f"Correlation complete: "
            f"{len(result.typed_dependencies)} typed dependencies, "
            f"{len(result.typed_data_flow)} typed flows, "
            f"{len(result.semantic_entities)} entities"
        )
    
    def _correlate_dependencies_with_types(self, result: SemanticResult) -> None:
        """Correlate dependencies with type information."""
        for component, deps in result.dependencies.items():
            for dep in deps:
                # Try to find type info for provider
                provider_type = result.type_map.get(dep.provider)
                consumer_type = result.type_map.get(dep.consumer)
                
                if provider_type or consumer_type:
                    result.typed_dependencies.append({
                        'dependency': dep,
                        'provider_type': provider_type,
                        'consumer_type': consumer_type
                    })
    
    def _correlate_data_flow_with_types(self, result: SemanticResult) -> None:
        """Correlate data flow with type information."""
        if not result.data_flow_graph:
            return
        
        for node in result.data_flow_graph.nodes:
            var_name = node.variable_name
            type_info = result.type_map.get(var_name)
            
            if type_info:
                result.typed_data_flow.append({
                    'node': node,
                    'variable': var_name,
                    'type_info': type_info
                })
    
    def _build_semantic_entities(self, result: SemanticResult) -> None:
        """Build unified semantic entities from all analyzer data."""
        entities: Dict[CorrelationKey, SemanticEntity] = {}
        
        # Step 1: Index data flow nodes
        if result.data_flow_graph:
            for node in result.data_flow_graph.nodes:
                var_name = node.variable_name
                key = CorrelationKey(var_name, "variable")
                entities[key] = SemanticEntity(
                    name=var_name,
                    kind="variable",
                    data_flow={'node': node}
                )
        
        # Step 2: Enrich with DI info
        for component, deps in result.dependencies.items():
            key = CorrelationKey(component, "variable")
            di_info = {'dependencies': deps}
            
            if key in entities:
                entities[key].di_info = di_info
                entities[key].kind = "component"
            else:
                entities[key] = SemanticEntity(
                    name=component,
                    kind="component",
                    di_info=di_info
                )
        
        # Step 3: Enrich with type info
        for identifier, type_info in result.type_map.items():
            key = CorrelationKey(identifier, "variable")
            if key in entities:
                entities[key].type_info = type_info
            else:
                # Try function
                key = CorrelationKey(identifier, "function")
                if key not in entities:
                    entities[key] = SemanticEntity(
                        name=identifier,
                        kind="function",
                        type_info=type_info
                    )
                else:
                    entities[key].type_info = type_info
        
        # Step 4: Compute relationships (NEW in Phase 4.3)
        self._compute_relationships(entities, result)
        
        # Step 5: Calculate confidence scores (NEW in Phase 4.3)
        self._calculate_confidence_scores(entities)
        
        result.semantic_entities = list(entities.values())
    
    # === Relationship Computation (Phase 4.3) ===
    
    def _compute_relationships(
        self,
        entities: Dict[CorrelationKey, SemanticEntity],
        result: SemanticResult
    ) -> None:
        """
        Compute call relationships between entities.
        
        This analyzes:
        - Function calls in data flow
        - DI dependencies (component A depends on B)
        - Type relationships (functions returning specific types)
        """
        self.logger.debug("Computing entity relationships")
        
        # Build call relationships from data flow
        if result.data_flow_graph:
            self._compute_call_relationships_from_data_flow(
                entities,
                result.data_flow_graph
            )
        
        # Build dependency relationships from DI
        if result.dependencies:
            self._compute_dependency_relationships(entities, result.dependencies)
        
        # Build type-based relationships
        if result.type_map:
            self._compute_type_relationships(entities, result.type_map)
    
    def _compute_call_relationships_from_data_flow(
        self,
        entities: Dict[CorrelationKey, SemanticEntity],
        graph: DataFlowGraph
    ) -> None:
        """Extract call relationships from data flow graph."""
        for edge in graph.edges:
            # Find source and target nodes from the nodes dict
            source_node = graph.nodes.get(edge.source)
            target_node = graph.nodes.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # Check if this is a function call edge
            if edge.flow_type.value in ['CALL', 'CALLS']:
                source_key = CorrelationKey(source_node.variable_name, "variable")
                target_key = CorrelationKey(target_node.variable_name, "function")
                
                # Add to calls/called_by
                if source_key in entities:
                    if target_node.variable_name not in entities[source_key].calls:
                        entities[source_key].calls.append(target_node.variable_name)
                
                if target_key in entities:
                    if source_node.variable_name not in entities[target_key].called_by:
                        entities[target_key].called_by.append(source_node.variable_name)
    
    def _compute_dependency_relationships(
        self,
        entities: Dict[CorrelationKey, SemanticEntity],
        dependencies: Dict[str, List[Dependency]]
    ) -> None:
        """Extract dependency relationships from DI analysis."""
        for consumer, deps in dependencies.items():
            consumer_key = CorrelationKey(consumer, "variable")
            
            if consumer_key not in entities:
                continue
            
            for dep in deps:
                # Consumer depends on (calls) provider
                provider_name = dep.provider
                
                if provider_name not in entities[consumer_key].calls:
                    entities[consumer_key].calls.append(provider_name)
                
                # Provider is called by consumer
                provider_key = CorrelationKey(provider_name, "variable")
                if provider_key in entities:
                    if consumer not in entities[provider_key].called_by:
                        entities[provider_key].called_by.append(consumer)
    
    def _compute_type_relationships(
        self,
        entities: Dict[CorrelationKey, SemanticEntity],
        type_map: Dict[str, TypeInfo]
    ) -> None:
        """
        Extract type-based relationships.
        
        For example, if function A returns type T, and variable B is of type T,
        we can infer B might be assigned from A.
        """
        # Group entities by type
        entities_by_type: Dict[str, List[str]] = {}
        
        for key, entity in entities.items():
            if entity.type_info:
                type_name = self._get_type_name(entity.type_info)
                if type_name:
                    if type_name not in entities_by_type:
                        entities_by_type[type_name] = []
                    entities_by_type[type_name].append(entity.name)
        
        # Note relationships in analysis notes
        for type_name, entity_names in entities_by_type.items():
            if len(entity_names) > 1:
                for name in entity_names:
                    key = CorrelationKey(name, "variable")
                    if key in entities:
                        note = f"Shares type '{type_name}' with: {', '.join([n for n in entity_names if n != name])}"
                        entities[key].analysis_notes.append(note)
    
    def _get_type_name(self, type_info: TypeInfo) -> Optional[str]:
        """Extract a simple type name from TypeInfo."""
        if hasattr(type_info, 'name') and type_info.name:
            return type_info.name
        if hasattr(type_info, 'primitive_type') and type_info.primitive_type:
            return str(type_info.primitive_type)
        return None
    
    # === Confidence Scoring (Phase 4.3) ===
    
    def _calculate_confidence_scores(
        self,
        entities: Dict[CorrelationKey, SemanticEntity]
    ) -> None:
        """
        Calculate confidence scores for entity correlations.
        
        Confidence is based on:
        - How many analyzers contributed data (more = higher confidence)
        - Data quality (e.g., explicit types vs inferred)
        - Relationship consistency
        """
        self.logger.debug("Calculating confidence scores")
        
        for key, entity in entities.items():
            score = 0.0
            total_weight = 0.0
            
            # Data flow contribution (weight: 0.3)
            if entity.data_flow is not None:
                score += 0.3
            total_weight += 0.3
            
            # DI contribution (weight: 0.3)
            if entity.di_info is not None:
                score += 0.3
            total_weight += 0.3
            
            # Type inference contribution (weight: 0.4 - most important)
            if entity.type_info is not None:
                # Higher confidence for explicit types (source="explicit" or "annotation")
                if hasattr(entity.type_info, 'source') and entity.type_info.source in ['explicit', 'annotation']:
                    score += 0.4
                else:
                    score += 0.3  # Slightly lower for inferred types
            total_weight += 0.4
            
            # Relationship bonus (weight: bonus up to 0.1)
            relationship_count = len(entity.calls) + len(entity.called_by)
            if relationship_count > 0:
                # More relationships = more confidence (capped at 0.1)
                relationship_bonus = min(0.1, relationship_count * 0.02)
                score += relationship_bonus
                total_weight += 0.1
            
            # Normalize to 0-1 range
            entity.confidence = score / total_weight if total_weight > 0 else 0.0
            
            # Add confidence note
            if entity.confidence >= 0.9:
                entity.analysis_notes.append("High confidence correlation")
            elif entity.confidence >= 0.7:
                entity.analysis_notes.append("Medium confidence correlation")
            elif entity.confidence >= 0.5:
                entity.analysis_notes.append("Low confidence correlation")
            else:
                entity.analysis_notes.append("Very low confidence - needs review")
    
    # === Helper Methods ===
    
    def _extract_data_flow_info(self, node: DataFlowNode) -> Dict[str, Any]:
        """Extract data flow info from DataFlowNode."""
        return {
            'node_id': node.id,
            'node_type': node.node_type,
            'variable': node.variable_name
        }
    
    def _extract_di_info(self, deps: List[Dependency]) -> Dict[str, Any]:
        """Extract DI info from dependencies."""
        return {
            'dependencies': [
                {
                    'provider': dep.provider,
                    'injection_type': dep.injection_type
                }
                for dep in deps
            ]
        }
    
    # === Helper Methods ===
    
    def _parse_code(self, code: str, language: str) -> Optional[Any]:
        """Parse code using tree-sitter."""
        try:
            tree = self.parser.parse(code, language)
            return tree
        except Exception as e:
            self.logger.exception(f"Failed to parse code: {e}")
            return None
    
    def _infer_language(self, file_path: Path) -> str:
        """Infer language from file extension."""
        ext = file_path.suffix.lower()
        mapping = {
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.py': 'python',
            '.java': 'java',
        }
        return mapping.get(ext, 'javascript')
    
    def clear_cache(self) -> None:
        """Clear result cache."""
        self._cache.clear()
        self.logger.info("Result cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'enabled': self.cache_results
        }
