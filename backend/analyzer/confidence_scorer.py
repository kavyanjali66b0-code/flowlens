"""
Confidence Scoring System for Semantic Analysis

This module provides unified confidence scoring across all M4.0 components:
- Type inference confidence (from TypeInferenceEngine)
- Data flow completeness (from DataFlowTracker)
- Hook usage patterns (from HookDetector)
- DI completeness (from DIAnalyzer)
- Code annotation presence
- Evidence count

Confidence scores range from 0.0 (no confidence) to 1.0 (full confidence).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import json


class ConfidenceFactor(Enum):
    """Factors that contribute to confidence scoring"""
    TYPE_INFERENCE = "type_inference"
    DATA_FLOW = "data_flow"
    HOOK_PATTERN = "hook_pattern"
    DI_ANALYSIS = "di_analysis"
    ANNOTATION = "annotation"
    EVIDENCE = "evidence"
    USAGE = "usage"
    CONSISTENCY = "consistency"


class EntityCategory(Enum):
    """Categories of entities that can be scored"""
    VARIABLE = "variable"
    FUNCTION = "function"
    COMPONENT = "component"
    CLASS = "class"
    HOOK = "hook"
    DEPENDENCY = "dependency"
    FILE = "file"
    MODULE = "module"


class ConfidenceLevel(Enum):
    """Human-readable confidence levels"""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class FactorScore:
    """Individual factor contribution to confidence"""
    factor: ConfidenceFactor
    score: float  # 0.0 - 1.0
    weight: float  # How much this factor contributes
    reason: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ConfidenceScore:
    """Complete confidence assessment for an entity"""
    entity_id: str
    entity_name: str
    entity_category: EntityCategory
    overall_score: float  # 0.0 - 1.0
    confidence_level: ConfidenceLevel
    factor_scores: List[FactorScore] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_category': self.entity_category.value,
            'overall_score': round(self.overall_score, 3),
            'confidence_level': self.confidence_level.value,
            'factor_scores': [
                {
                    'factor': fs.factor.value,
                    'score': round(fs.score, 3),
                    'weight': round(fs.weight, 3),
                    'reason': fs.reason,
                    'evidence': fs.evidence
                }
                for fs in self.factor_scores
            ],
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class ConfidenceScorer:
    """
    Unified confidence scoring system for semantic analysis.
    
    Calculates confidence scores based on multiple factors:
    - Type inference quality
    - Data flow completeness
    - Hook pattern analysis
    - Dependency injection analysis
    - Code annotations
    - Evidence strength
    """
    
    def __init__(self):
        self.scores: Dict[str, ConfidenceScore] = {}
        self.default_weights = {
            ConfidenceFactor.TYPE_INFERENCE: 0.25,
            ConfidenceFactor.DATA_FLOW: 0.20,
            ConfidenceFactor.HOOK_PATTERN: 0.15,
            ConfidenceFactor.DI_ANALYSIS: 0.15,
            ConfidenceFactor.ANNOTATION: 0.10,
            ConfidenceFactor.EVIDENCE: 0.10,
            ConfidenceFactor.USAGE: 0.03,
            ConfidenceFactor.CONSISTENCY: 0.02
        }
    
    def score_entity(
        self,
        entity_id: str,
        entity_name: str,
        entity_category: EntityCategory,
        factors: Dict[ConfidenceFactor, float],
        evidence: Optional[Dict[ConfidenceFactor, List[str]]] = None,
        custom_weights: Optional[Dict[ConfidenceFactor, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """
        Calculate confidence score for an entity.
        
        Args:
            entity_id: Unique identifier
            entity_name: Human-readable name
            entity_category: Category of entity
            factors: Factor scores (0.0-1.0)
            evidence: Supporting evidence per factor
            custom_weights: Override default weights
            metadata: Additional context
            
        Returns:
            ConfidenceScore with overall score and breakdown
        """
        weights = custom_weights if custom_weights else self.default_weights
        evidence = evidence or {}
        metadata = metadata or {}
        
        # Calculate factor scores
        factor_scores = []
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, score in factors.items():
            weight = weights.get(factor, 0.0)
            if weight > 0:
                # Clamp score to [0.0, 1.0]
                clamped_score = max(0.0, min(1.0, score))
                
                factor_score = FactorScore(
                    factor=factor,
                    score=clamped_score,
                    weight=weight,
                    reason=self._get_factor_reason(factor, clamped_score),
                    evidence=evidence.get(factor, [])
                )
                factor_scores.append(factor_score)
                
                weighted_sum += clamped_score * weight
                total_weight += weight
        
        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            entity_category, factor_scores, overall_score
        )
        
        # Create confidence score
        confidence_score = ConfidenceScore(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_category=entity_category,
            overall_score=overall_score,
            confidence_level=confidence_level,
            factor_scores=factor_scores,
            recommendations=recommendations,
            metadata=metadata
        )
        
        self.scores[entity_id] = confidence_score
        return confidence_score
    
    def score_from_type_inference(
        self,
        entity_id: str,
        entity_name: str,
        type_info: Dict[str, Any]
    ) -> ConfidenceScore:
        """
        Score entity based on type inference data.
        
        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            type_info: Type information with 'confidence' and 'sources'
            
        Returns:
            ConfidenceScore
        """
        type_confidence = type_info.get('confidence', 0.5)
        sources = type_info.get('sources', [])
        has_annotation = type_info.get('has_annotation', False)
        
        factors = {
            ConfidenceFactor.TYPE_INFERENCE: type_confidence,
            ConfidenceFactor.ANNOTATION: 1.0 if has_annotation else 0.0,
            ConfidenceFactor.EVIDENCE: min(1.0, len(sources) * 0.3)
        }
        
        evidence = {
            ConfidenceFactor.TYPE_INFERENCE: [f"Inferred type: {type_info.get('type', 'unknown')}"],
            ConfidenceFactor.EVIDENCE: [f"Sources: {', '.join(sources[:3])}"]
        }
        
        if has_annotation:
            evidence[ConfidenceFactor.ANNOTATION] = ["Type annotation present"]
        
        return self.score_entity(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_category=EntityCategory.VARIABLE,
            factors=factors,
            evidence=evidence,
            metadata={'type_info': type_info}
        )
    
    def score_from_data_flow(
        self,
        entity_id: str,
        entity_name: str,
        flow_data: Dict[str, Any]
    ) -> ConfidenceScore:
        """
        Score entity based on data flow tracking.
        
        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            flow_data: Data flow information with 'flows' and 'completeness'
            
        Returns:
            ConfidenceScore
        """
        flows = flow_data.get('flows', [])
        incoming = len([f for f in flows if f.get('direction') == 'incoming'])
        outgoing = len([f for f in flows if f.get('direction') == 'outgoing'])
        
        # Flow completeness: both sources and targets identified
        flow_completeness = min(1.0, (incoming + outgoing) * 0.2)
        
        # Consistency: flows are well-defined
        consistency = flow_data.get('consistency', 0.5)
        
        factors = {
            ConfidenceFactor.DATA_FLOW: flow_completeness,
            ConfidenceFactor.CONSISTENCY: consistency,
            ConfidenceFactor.EVIDENCE: min(1.0, len(flows) * 0.25)
        }
        
        evidence = {
            ConfidenceFactor.DATA_FLOW: [
                f"{incoming} incoming flows, {outgoing} outgoing flows"
            ],
            ConfidenceFactor.EVIDENCE: [f"Total flows: {len(flows)}"]
        }
        
        return self.score_entity(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_category=EntityCategory.VARIABLE,
            factors=factors,
            evidence=evidence,
            metadata={'flow_data': flow_data}
        )
    
    def score_from_hook_analysis(
        self,
        entity_id: str,
        entity_name: str,
        hook_data: Dict[str, Any]
    ) -> ConfidenceScore:
        """
        Score entity based on React hook analysis.
        
        Args:
            entity_id: Entity identifier
            entity_name: Entity name (hook name)
            hook_data: Hook information with 'type', 'dependencies', 'stability'
            
        Returns:
            ConfidenceScore
        """
        hook_type = hook_data.get('type', 'unknown')
        dependencies = hook_data.get('dependencies', [])
        is_stable = hook_data.get('is_stable', False)
        
        # Pattern score: known hook type
        pattern_score = 1.0 if hook_type != 'CUSTOM' else 0.7
        
        # Dependency completeness
        dep_score = 1.0 if dependencies or is_stable else 0.5
        
        # Consistency: stable dependencies
        consistency = 1.0 if is_stable else 0.6
        
        factors = {
            ConfidenceFactor.HOOK_PATTERN: pattern_score,
            ConfidenceFactor.EVIDENCE: dep_score,
            ConfidenceFactor.CONSISTENCY: consistency
        }
        
        evidence = {
            ConfidenceFactor.HOOK_PATTERN: [f"Hook type: {hook_type}"],
            ConfidenceFactor.EVIDENCE: [
                f"Dependencies: {', '.join(dependencies) if dependencies else 'none'}"
            ]
        }
        
        return self.score_entity(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_category=EntityCategory.HOOK,
            factors=factors,
            evidence=evidence,
            metadata={'hook_data': hook_data}
        )
    
    def score_from_di_analysis(
        self,
        entity_id: str,
        entity_name: str,
        di_data: Dict[str, Any]
    ) -> ConfidenceScore:
        """
        Score entity based on dependency injection analysis.
        
        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            di_data: DI information with 'dependencies', 'type', 'optional'
            
        Returns:
            ConfidenceScore
        """
        dependencies = di_data.get('dependencies', [])
        di_type = di_data.get('type', 'unknown')
        is_optional = di_data.get('optional', False)
        
        # DI completeness: dependencies identified
        di_completeness = min(1.0, len(dependencies) * 0.3)
        
        # Type clarity
        type_score = 1.0 if di_type in ['PROP', 'CONTEXT', 'HOOK'] else 0.6
        
        # Consistency: required deps have sources
        consistency = 0.8 if not is_optional else 1.0
        
        factors = {
            ConfidenceFactor.DI_ANALYSIS: di_completeness,
            ConfidenceFactor.TYPE_INFERENCE: type_score,
            ConfidenceFactor.CONSISTENCY: consistency,
            ConfidenceFactor.EVIDENCE: min(1.0, len(dependencies) * 0.25)
        }
        
        evidence = {
            ConfidenceFactor.DI_ANALYSIS: [f"DI type: {di_type}"],
            ConfidenceFactor.EVIDENCE: [f"{len(dependencies)} dependencies"]
        }
        
        return self.score_entity(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_category=EntityCategory.DEPENDENCY,
            factors=factors,
            evidence=evidence,
            metadata={'di_data': di_data}
        )
    
    def score_component(
        self,
        component_id: str,
        component_name: str,
        type_info: Optional[Dict[str, Any]] = None,
        hook_count: int = 0,
        di_count: int = 0,
        has_props: bool = False,
        usage_count: int = 0
    ) -> ConfidenceScore:
        """
        Score a React component comprehensively.
        
        Args:
            component_id: Component identifier
            component_name: Component name
            type_info: Type information (props, return type)
            hook_count: Number of hooks used
            di_count: Number of dependencies
            has_props: Whether props are defined
            usage_count: Number of times component is used
            
        Returns:
            ConfidenceScore
        """
        # Type inference: based on props and return type
        type_score = 0.5
        if type_info:
            type_score = type_info.get('confidence', 0.5)
            if has_props:
                type_score = min(1.0, type_score + 0.2)
        
        # Hook pattern: higher confidence with more hooks
        hook_score = min(1.0, hook_count * 0.25)
        
        # DI analysis: dependencies tracked
        di_score = min(1.0, di_count * 0.2)
        
        # Usage: component is used in codebase
        usage_score = min(1.0, usage_count * 0.3)
        
        factors = {
            ConfidenceFactor.TYPE_INFERENCE: type_score,
            ConfidenceFactor.HOOK_PATTERN: hook_score,
            ConfidenceFactor.DI_ANALYSIS: di_score,
            ConfidenceFactor.USAGE: usage_score,
            ConfidenceFactor.EVIDENCE: min(1.0, (hook_count + di_count) * 0.15)
        }
        
        evidence = {
            ConfidenceFactor.HOOK_PATTERN: [f"{hook_count} hooks"],
            ConfidenceFactor.DI_ANALYSIS: [f"{di_count} dependencies"],
            ConfidenceFactor.USAGE: [f"Used {usage_count} times"]
        }
        
        return self.score_entity(
            entity_id=component_id,
            entity_name=component_name,
            entity_category=EntityCategory.COMPONENT,
            factors=factors,
            evidence=evidence,
            metadata={
                'hook_count': hook_count,
                'di_count': di_count,
                'has_props': has_props,
                'usage_count': usage_count
            }
        )
    
    def score_file(
        self,
        file_id: str,
        file_name: str,
        entity_scores: List[ConfidenceScore]
    ) -> ConfidenceScore:
        """
        Aggregate confidence score for a file.
        
        Args:
            file_id: File identifier
            file_name: File name
            entity_scores: Scores of entities in file
            
        Returns:
            ConfidenceScore
        """
        if not entity_scores:
            return self.score_entity(
                entity_id=file_id,
                entity_name=file_name,
                entity_category=EntityCategory.FILE,
                factors={ConfidenceFactor.EVIDENCE: 0.0}
            )
        
        # Aggregate scores by factor
        factor_aggregates: Dict[ConfidenceFactor, List[float]] = {}
        for score in entity_scores:
            for factor_score in score.factor_scores:
                if factor_score.factor not in factor_aggregates:
                    factor_aggregates[factor_score.factor] = []
                factor_aggregates[factor_score.factor].append(factor_score.score)
        
        # Calculate average per factor
        factors = {}
        evidence = {}
        for factor, scores in factor_aggregates.items():
            avg_score = sum(scores) / len(scores)
            factors[factor] = avg_score
            evidence[factor] = [
                f"{len(scores)} entities, avg score: {avg_score:.2f}"
            ]
        
        # Add file-level evidence
        factors[ConfidenceFactor.EVIDENCE] = min(1.0, len(entity_scores) * 0.1)
        evidence[ConfidenceFactor.EVIDENCE] = [f"{len(entity_scores)} entities analyzed"]
        
        return self.score_entity(
            entity_id=file_id,
            entity_name=file_name,
            entity_category=EntityCategory.FILE,
            factors=factors,
            evidence=evidence,
            metadata={'entity_count': len(entity_scores)}
        )
    
    def get_score(self, entity_id: str) -> Optional[ConfidenceScore]:
        """Get confidence score by entity ID"""
        return self.scores.get(entity_id)
    
    def get_all_scores(self) -> List[ConfidenceScore]:
        """Get all confidence scores"""
        return list(self.scores.values())
    
    def get_scores_by_category(self, category: EntityCategory) -> List[ConfidenceScore]:
        """Get scores filtered by entity category"""
        return [
            score for score in self.scores.values()
            if score.entity_category == category
        ]
    
    def get_scores_by_level(self, level: ConfidenceLevel) -> List[ConfidenceScore]:
        """Get scores filtered by confidence level"""
        return [
            score for score in self.scores.values()
            if score.confidence_level == level
        ]
    
    def get_low_confidence_entities(self, threshold: float = 0.5) -> List[ConfidenceScore]:
        """
        Get entities with low confidence scores.
        
        Args:
            threshold: Score threshold (default 0.5)
            
        Returns:
            List of low-confidence entities
        """
        return [
            score for score in self.scores.values()
            if score.overall_score < threshold
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scoring statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.scores:
            return {
                'total_entities': 0,
                'average_score': 0.0,
                'by_category': {},
                'by_level': {},
                'low_confidence': 0
            }
        
        scores = list(self.scores.values())
        total = len(scores)
        avg_score = sum(s.overall_score for s in scores) / total
        
        # By category
        by_category = {}
        for category in EntityCategory:
            cat_scores = [s for s in scores if s.entity_category == category]
            if cat_scores:
                by_category[category.value] = {
                    'count': len(cat_scores),
                    'average_score': sum(s.overall_score for s in cat_scores) / len(cat_scores)
                }
        
        # By level
        by_level = {}
        for level in ConfidenceLevel:
            level_scores = [s for s in scores if s.confidence_level == level]
            by_level[level.value] = len(level_scores)
        
        # Low confidence
        low_confidence = len([s for s in scores if s.overall_score < 0.5])
        
        return {
            'total_entities': total,
            'average_score': round(avg_score, 3),
            'by_category': by_category,
            'by_level': by_level,
            'low_confidence': low_confidence
        }
    
    def export_to_json(self) -> str:
        """Export all scores to JSON"""
        return json.dumps(
            {
                'scores': [score.to_dict() for score in self.scores.values()],
                'statistics': self.get_statistics()
            },
            indent=2
        )
    
    def clear(self):
        """Clear all scores"""
        self.scores.clear()
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level"""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def _get_factor_reason(self, factor: ConfidenceFactor, score: float) -> str:
        """Generate human-readable reason for factor score"""
        level = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
        
        reasons = {
            ConfidenceFactor.TYPE_INFERENCE: f"Type inference confidence is {level}",
            ConfidenceFactor.DATA_FLOW: f"Data flow completeness is {level}",
            ConfidenceFactor.HOOK_PATTERN: f"Hook pattern recognition is {level}",
            ConfidenceFactor.DI_ANALYSIS: f"Dependency injection analysis is {level}",
            ConfidenceFactor.ANNOTATION: f"Type annotation presence is {level}",
            ConfidenceFactor.EVIDENCE: f"Supporting evidence strength is {level}",
            ConfidenceFactor.USAGE: f"Usage tracking is {level}",
            ConfidenceFactor.CONSISTENCY: f"Pattern consistency is {level}"
        }
        
        return reasons.get(factor, f"{factor.value} score is {level}")
    
    def _generate_recommendations(
        self,
        category: EntityCategory,
        factor_scores: List[FactorScore],
        overall_score: float
    ) -> List[str]:
        """Generate recommendations for improving confidence"""
        recommendations = []
        
        # Low overall score
        if overall_score < 0.5:
            recommendations.append("Overall confidence is low. Consider adding more context.")
        
        # Check each factor
        for factor_score in factor_scores:
            if factor_score.score < 0.5:
                if factor_score.factor == ConfidenceFactor.TYPE_INFERENCE:
                    recommendations.append("Add type annotations to improve type inference")
                elif factor_score.factor == ConfidenceFactor.DATA_FLOW:
                    recommendations.append("Track data flow sources and targets more completely")
                elif factor_score.factor == ConfidenceFactor.HOOK_PATTERN:
                    recommendations.append("Follow React hooks naming conventions (use* prefix)")
                elif factor_score.factor == ConfidenceFactor.DI_ANALYSIS:
                    recommendations.append("Explicitly declare dependencies (props, context)")
                elif factor_score.factor == ConfidenceFactor.ANNOTATION:
                    recommendations.append("Add JSDoc or TypeScript annotations")
                elif factor_score.factor == ConfidenceFactor.EVIDENCE:
                    recommendations.append("Provide more context and evidence")
        
        # Category-specific recommendations
        if category == EntityCategory.COMPONENT and overall_score < 0.7:
            recommendations.append("Document component props and usage patterns")
        elif category == EntityCategory.HOOK and overall_score < 0.7:
            recommendations.append("Add dependency arrays to hooks")
        
        return recommendations[:5]  # Limit to top 5 recommendations
