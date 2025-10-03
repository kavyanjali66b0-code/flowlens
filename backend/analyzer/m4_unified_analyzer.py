"""
M4.0 Unified Semantic Analyzer - Complete Integration

This module integrates all M4.0 components into a single, cohesive semantic analysis system:
- TypeInferenceEngine: Rich type inference with React support
- DataFlowTracker: Comprehensive data flow tracking
- HookDetector: React hooks detection and analysis
- DIAnalyzer: Dependency injection pattern analysis
- ConfidenceScorer: Unified confidence scoring

Provides a complete semantic analysis pipeline for JavaScript/TypeScript codebases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import json

from analyzer.type_inference_engine import (
    TypeInferenceEngine,
    TypeInfo,
    TypeCategory,
    InferenceSource
)
from analyzer.data_flow_tracker import (
    DataFlowTracker,
    DataFlowEdge,
    FlowType,
    FlowDirection
)
from analyzer.hook_detector import (
    HookDetector,
    HookCall,
    HookType,
    HookCategory
)
from analyzer.di_analyzer import (
    DIAnalyzer,
    DIEntity,
    Dependency,
    DIType
)
from analyzer.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceScore,
    ConfidenceFactor,
    EntityCategory,
    ConfidenceLevel
)


class AnalysisMode(Enum):
    """Analysis modes for different use cases"""
    FAST = "fast"           # Quick analysis, basic features
    STANDARD = "standard"   # Standard analysis with all features
    DEEP = "deep"           # Deep analysis with maximum detail


@dataclass
class EntityAnalysis:
    """Complete analysis result for a single entity"""
    entity_id: str
    entity_name: str
    entity_type: str  # "variable", "function", "component", "hook", etc.
    
    # Type inference
    type_info: Optional[TypeInfo] = None
    
    # Data flow
    data_flows: List[DataFlowEdge] = field(default_factory=list)
    
    # Hooks (if applicable)
    hooks_used: List[HookCall] = field(default_factory=list)
    
    # Dependencies (if applicable)
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Confidence
    confidence_score: Optional[ConfidenceScore] = None
    
    # Metadata
    location: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'type_info': self.type_info.to_dict() if self.type_info else None,
            'data_flows': [
                {
                    'source': flow.source_id,
                    'target': flow.target_id,
                    'flow_type': flow.flow_type.value,
                    'direction': flow.direction.value if flow.direction else None
                }
                for flow in self.data_flows
            ],
            'hooks_used': [
                {
                    'hook_name': hook.hook_name,
                    'hook_type': hook.hook_type.value,
                    'dependencies': hook.dependencies,
                    'is_stable': hook.is_stable
                }
                for hook in self.hooks_used
            ],
            'dependencies': [
                {
                    'name': dep.name,
                    'di_type': dep.di_type.value,
                    'source': dep.source,
                    'optional': dep.optional
                }
                for dep in self.dependencies
            ],
            'confidence_score': self.confidence_score.to_dict() if self.confidence_score else None,
            'location': self.location,
            'metadata': self.metadata
        }


@dataclass
class FileAnalysis:
    """Complete analysis result for a file"""
    file_id: str
    file_path: str
    entities: List[EntityAnalysis] = field(default_factory=list)
    file_confidence: Optional[ConfidenceScore] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_id': self.file_id,
            'file_path': self.file_path,
            'entities': [entity.to_dict() for entity in self.entities],
            'file_confidence': self.file_confidence.to_dict() if self.file_confidence else None,
            'statistics': self.statistics
        }


@dataclass
class AnalysisReport:
    """Complete analysis report"""
    files: List[FileAnalysis] = field(default_factory=list)
    global_statistics: Dict[str, Any] = field(default_factory=dict)
    analysis_mode: AnalysisMode = AnalysisMode.STANDARD
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'files': [file.to_dict() for file in self.files],
            'global_statistics': self.global_statistics,
            'analysis_mode': self.analysis_mode.value,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=2)


class M4UnifiedAnalyzer:
    """
    M4.0 Unified Semantic Analyzer - Complete Integration
    
    Integrates all M4.0 components:
    - TypeInferenceEngine
    - DataFlowTracker
    - HookDetector
    - DIAnalyzer
    - ConfidenceScorer
    
    Provides a complete semantic analysis pipeline.
    """
    
    def __init__(self, mode: AnalysisMode = AnalysisMode.STANDARD):
        self.mode = mode
        
        # Initialize all M4.0 components
        self.type_engine = TypeInferenceEngine()
        self.flow_tracker = DataFlowTracker()
        self.hook_detector = HookDetector()
        self.di_analyzer = DIAnalyzer()
        self.confidence_scorer = ConfidenceScorer()
        
        # Analysis results
        self.entities: Dict[str, EntityAnalysis] = {}
        self.files: Dict[str, FileAnalysis] = {}
    
    def analyze_variable(
        self,
        var_id: str,
        var_name: str,
        ast_node: Optional[Any] = None,
        literal_value: Optional[Any] = None,
        location: Optional[Dict[str, Any]] = None
    ) -> EntityAnalysis:
        """
        Analyze a variable with all M4.0 components.
        
        Args:
            var_id: Unique variable identifier
            var_name: Variable name
            ast_node: AST node (if available)
            literal_value: Literal value (if available)
            location: Source location
            
        Returns:
            Complete EntityAnalysis
        """
        entity = EntityAnalysis(
            entity_id=var_id,
            entity_name=var_name,
            entity_type="variable",
            location=location
        )
        
        # Type inference
        if literal_value is not None:
            entity.type_info = self.type_engine.infer_from_literal(literal_value)
        elif ast_node is not None:
            entity.type_info = self.type_engine.infer_from_ast_node(ast_node)
        
        # Data flow (get flows for this variable)
        flows_dict = self.flow_tracker.get_flows_for_node(var_id)
        if flows_dict:
            entity.data_flows = flows_dict.get('outgoing', []) + flows_dict.get('incoming', [])
        
        # Confidence scoring
        if entity.type_info:
            confidence_factors = {
                ConfidenceFactor.TYPE_INFERENCE: entity.type_info.confidence,
                ConfidenceFactor.EVIDENCE: min(1.0, len(entity.type_info.sources) * 0.3)
            }
            
            if entity.data_flows:
                flow_completeness = min(1.0, len(entity.data_flows) * 0.2)
                confidence_factors[ConfidenceFactor.DATA_FLOW] = flow_completeness
            
            entity.confidence_score = self.confidence_scorer.score_entity(
                entity_id=var_id,
                entity_name=var_name,
                entity_category=EntityCategory.VARIABLE,
                factors=confidence_factors,
                metadata={'has_type': True, 'has_flows': len(entity.data_flows) > 0}
            )
        
        self.entities[var_id] = entity
        return entity
    
    def analyze_react_state(
        self,
        state_id: str,
        state_name: str,
        initial_value: Optional[Any] = None,
        component_id: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None
    ) -> EntityAnalysis:
        """
        Analyze React state variable (useState).
        
        Args:
            state_id: State variable identifier
            state_name: State variable name
            initial_value: Initial state value
            component_id: Parent component ID
            location: Source location
            
        Returns:
            Complete EntityAnalysis
        """
        entity = EntityAnalysis(
            entity_id=state_id,
            entity_name=state_name,
            entity_type="react_state",
            location=location,
            metadata={'component_id': component_id}
        )
        
        # Type inference (React useState)
        entity.type_info = self.type_engine.infer_react_hook_type(
            'useState',
            [initial_value] if initial_value is not None else []
        )
        
        # Track useState initialization
        self.flow_tracker.track_react_state_init(
            state_id=state_id,
            state_name=state_name,
            setter_name=f"set{state_name[0].upper()}{state_name[1:]}",
            initial_value=initial_value,
            component_id=component_id or "unknown"
        )
        
        # Get data flows
        entity.data_flows = self.flow_tracker.get_react_state_flows(state_name)
        
        # Detect useState hook
        hook_call = self.hook_detector.detect_hook_call(
            hook_name='useState',
            args=[initial_value] if initial_value is not None else [],
            component_id=component_id,
            location=location
        )
        if hook_call:
            entity.hooks_used.append(hook_call)
        
        # Confidence scoring
        confidence_factors = {
            ConfidenceFactor.TYPE_INFERENCE: entity.type_info.confidence if entity.type_info else 0.5,
            ConfidenceFactor.HOOK_PATTERN: 1.0,  # useState is standard hook
            ConfidenceFactor.DATA_FLOW: min(1.0, len(entity.data_flows) * 0.2)
        }
        
        entity.confidence_score = self.confidence_scorer.score_entity(
            entity_id=state_id,
            entity_name=state_name,
            entity_category=EntityCategory.VARIABLE,
            factors=confidence_factors,
            metadata={'is_react_state': True}
        )
        
        self.entities[state_id] = entity
        return entity
    
    def analyze_component(
        self,
        component_id: str,
        component_name: str,
        props: Optional[List[str]] = None,
        hooks: Optional[List[Dict[str, Any]]] = None,
        location: Optional[Dict[str, Any]] = None
    ) -> EntityAnalysis:
        """
        Analyze a React component comprehensively.
        
        Args:
            component_id: Component identifier
            component_name: Component name
            props: List of prop names
            hooks: List of hook usage data
            location: Source location
            
        Returns:
            Complete EntityAnalysis
        """
        entity = EntityAnalysis(
            entity_id=component_id,
            entity_name=component_name,
            entity_type="component",
            location=location
        )
        
        # Register component as DI entity
        self.di_analyzer.register_entity(
            entity_id=component_id,
            name=component_name,
            entity_type="component"
        )
        
        # Detect prop dependencies
        if props:
            for prop_name in props:
                self.di_analyzer.detect_prop_dependency(
                    entity_id=component_id,
                    prop_name=prop_name
                )
        
        # Get dependencies
        entity.dependencies = self.di_analyzer.get_entity_dependencies(component_id)
        
        # Detect hooks
        if hooks:
            for hook_data in hooks:
                hook_call = self.hook_detector.detect_hook_call(
                    hook_name=hook_data.get('name'),
                    args=hook_data.get('args', []),
                    deps=hook_data.get('deps'),
                    component_id=component_id,
                    location=location
                )
                if hook_call:
                    entity.hooks_used.append(hook_call)
        
        # Type inference for component
        entity.type_info = TypeInfo(
            inferred_type="React.Component",
            category=TypeCategory.FUNCTION,
            confidence=0.9 if props else 0.7,
            sources=[InferenceSource.ANNOTATION] if props else [InferenceSource.CONTEXT]
        )
        
        # Confidence scoring
        entity.confidence_score = self.confidence_scorer.score_component(
            component_id=component_id,
            component_name=component_name,
            type_info={'confidence': entity.type_info.confidence},
            hook_count=len(entity.hooks_used),
            di_count=len(entity.dependencies),
            has_props=bool(props),
            usage_count=0  # Would need call graph analysis
        )
        
        self.entities[component_id] = entity
        return entity
    
    def analyze_hook_usage(
        self,
        hook_id: str,
        hook_name: str,
        hook_args: Optional[List[Any]] = None,
        deps: Optional[List[str]] = None,
        component_id: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None
    ) -> EntityAnalysis:
        """
        Analyze hook usage.
        
        Args:
            hook_id: Hook usage identifier
            hook_name: Hook name
            hook_args: Hook arguments
            deps: Dependency array
            component_id: Parent component
            location: Source location
            
        Returns:
            Complete EntityAnalysis
        """
        entity = EntityAnalysis(
            entity_id=hook_id,
            entity_name=hook_name,
            entity_type="hook",
            location=location,
            metadata={'component_id': component_id}
        )
        
        # Detect hook
        hook_call = self.hook_detector.detect_hook_call(
            hook_name=hook_name,
            args=hook_args or [],
            deps=deps,
            component_id=component_id,
            location=location
        )
        
        if hook_call:
            entity.hooks_used.append(hook_call)
            
            # Type inference based on hook type
            if hook_name == 'useState':
                entity.type_info = self.type_engine.infer_react_hook_type('useState', hook_args or [])
            elif hook_name == 'useEffect':
                entity.type_info = self.type_engine.infer_react_hook_type('useEffect', hook_args or [])
            
            # Confidence scoring
            hook_data = {
                'type': hook_call.hook_type.value,
                'dependencies': hook_call.dependencies,
                'is_stable': hook_call.is_stable
            }
            
            entity.confidence_score = self.confidence_scorer.score_from_hook_analysis(
                entity_id=hook_id,
                entity_name=hook_name,
                hook_data=hook_data
            )
        
        self.entities[hook_id] = entity
        return entity
    
    def track_assignment(
        self,
        source_id: str,
        target_id: str,
        scope: str = "global"
    ):
        """Track variable assignment flow"""
        self.flow_tracker.track_assignment(source_id, target_id, scope)
    
    def track_function_call(
        self,
        func_id: str,
        args: List[str],
        return_id: Optional[str] = None,
        scope: str = "global"
    ):
        """Track function call flow"""
        self.flow_tracker.track_function_call(func_id, args, return_id, scope)
    
    def register_context_provider(
        self,
        provider_id: str,
        context_name: str,
        value_id: Optional[str] = None,
        component_id: Optional[str] = None
    ):
        """Register React Context provider"""
        self.di_analyzer.register_context_provider(
            provider_id=provider_id,
            context_name=context_name,
            value_id=value_id,
            component_id=component_id
        )
    
    def analyze_file(
        self,
        file_id: str,
        file_path: str
    ) -> FileAnalysis:
        """
        Analyze a complete file.
        
        Args:
            file_id: File identifier
            file_path: File path
            
        Returns:
            Complete FileAnalysis
        """
        # Get all entities for this file
        file_entities = [
            entity for entity in self.entities.values()
            if entity.location and entity.location.get('file') == file_path
        ]
        
        # Calculate file-level confidence
        entity_scores = [
            entity.confidence_score for entity in file_entities
            if entity.confidence_score is not None
        ]
        
        file_confidence = None
        if entity_scores:
            file_confidence = self.confidence_scorer.score_file(
                file_id=file_id,
                file_name=file_path,
                entity_scores=entity_scores
            )
        
        # Calculate statistics
        statistics = {
            'total_entities': len(file_entities),
            'by_type': {},
            'hooks_used': sum(len(e.hooks_used) for e in file_entities),
            'dependencies': sum(len(e.dependencies) for e in file_entities),
            'data_flows': sum(len(e.data_flows) for e in file_entities)
        }
        
        # Count by type
        for entity in file_entities:
            entity_type = entity.entity_type
            statistics['by_type'][entity_type] = statistics['by_type'].get(entity_type, 0) + 1
        
        file_analysis = FileAnalysis(
            file_id=file_id,
            file_path=file_path,
            entities=file_entities,
            file_confidence=file_confidence,
            statistics=statistics
        )
        
        self.files[file_id] = file_analysis
        return file_analysis
    
    def generate_report(self) -> AnalysisReport:
        """
        Generate complete analysis report.
        
        Returns:
            Complete AnalysisReport
        """
        import datetime
        
        # Calculate global statistics
        all_entities = list(self.entities.values())
        
        global_stats = {
            'total_entities': len(all_entities),
            'total_files': len(self.files),
            'by_type': {},
            'confidence_breakdown': {
                'very_high': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'very_low': 0
            },
            'total_hooks': sum(len(e.hooks_used) for e in all_entities),
            'total_dependencies': sum(len(e.dependencies) for e in all_entities),
            'total_data_flows': sum(len(e.data_flows) for e in all_entities),
            'average_confidence': 0.0
        }
        
        # Count by type
        for entity in all_entities:
            entity_type = entity.entity_type
            global_stats['by_type'][entity_type] = global_stats['by_type'].get(entity_type, 0) + 1
        
        # Confidence breakdown
        confidence_scores = [e.confidence_score for e in all_entities if e.confidence_score]
        if confidence_scores:
            global_stats['average_confidence'] = sum(
                s.overall_score for s in confidence_scores
            ) / len(confidence_scores)
            
            for score in confidence_scores:
                level = score.confidence_level.value
                global_stats['confidence_breakdown'][level] += 1
        
        report = AnalysisReport(
            files=list(self.files.values()),
            global_statistics=global_stats,
            analysis_mode=self.mode,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        return report
    
    def get_entity(self, entity_id: str) -> Optional[EntityAnalysis]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    def get_file(self, file_id: str) -> Optional[FileAnalysis]:
        """Get file analysis by ID"""
        return self.files.get(file_id)
    
    def get_all_entities(self) -> List[EntityAnalysis]:
        """Get all entities"""
        return list(self.entities.values())
    
    def get_entities_by_type(self, entity_type: str) -> List[EntityAnalysis]:
        """Get entities filtered by type"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def get_low_confidence_entities(self, threshold: float = 0.5) -> List[EntityAnalysis]:
        """Get entities with low confidence"""
        return [
            entity for entity in self.entities.values()
            if entity.confidence_score and entity.confidence_score.overall_score < threshold
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'type_engine': self.type_engine.get_statistics(),
            'flow_tracker': self.flow_tracker.get_statistics(),
            'hook_detector': self.hook_detector.get_statistics(),
            'di_analyzer': self.di_analyzer.get_statistics(),
            'confidence_scorer': self.confidence_scorer.get_statistics()
        }
    
    def clear(self):
        """Clear all analysis data"""
        self.type_engine.clear()
        self.flow_tracker.clear()
        self.hook_detector.clear()
        self.di_analyzer.clear()
        self.confidence_scorer.clear()
        self.entities.clear()
        self.files.clear()
