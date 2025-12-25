"""
Type Inference Engine for M4.0

This module provides advanced type inference capabilities including:
- Literal type inference (0 → number, "text" → string)
- Function signature analysis
- React hook type inference (useState, useEffect, etc.)
- Generic type parameter handling
- Confidence scoring for type inferences
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field


class TypeCategory(Enum):
    """Categories of inferred types."""
    PRIMITIVE = "primitive"
    OBJECT = "object"
    ARRAY = "array"
    FUNCTION = "function"
    CLASS = "class"
    INTERFACE = "interface"
    UNION = "union"
    INTERSECTION = "intersection"
    GENERIC = "generic"
    TUPLE = "tuple"
    REACT_COMPONENT = "react_component"
    REACT_HOOK = "react_hook"
    UNKNOWN = "unknown"


class InferenceSource(Enum):
    """Source of type inference."""
    LITERAL = "literal"  # Inferred from literal value
    SIGNATURE = "signature"  # From function/method signature
    ANNOTATION = "annotation"  # From type annotation
    USAGE = "usage"  # From usage context
    RETURN = "return"  # From return statement
    PARAMETER = "parameter"  # From parameter usage
    HOOK = "hook"  # From React hook
    ASSIGNMENT = "assignment"  # From variable assignment
    IMPLICIT = "implicit"  # Implicit inference


@dataclass
class TypeInfo:
    """Detailed type information."""
    name: str
    category: TypeCategory
    confidence: float  # 0.0 to 1.0
    source: InferenceSource
    
    # Additional type details
    base_type: Optional[str] = None  # For arrays, promises, etc.
    generic_params: List[str] = field(default_factory=list)
    return_type: Optional[str] = None  # For functions
    param_types: List[str] = field(default_factory=list)  # For functions
    return_tuple: List[str] = field(default_factory=list)  # For tuple returns (useState)
    nullable: bool = False
    optional: bool = False
    
    # React-specific
    is_hook: bool = False
    hook_type: Optional[str] = None  # useState, useEffect, etc.
    
    # Metadata
    inferred_at: Optional[str] = None  # File location
    context: Optional[str] = None  # Context information

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category.value,
            'confidence': self.confidence,
            'source': self.source.value,
            'base_type': self.base_type,
            'generic_params': self.generic_params,
            'return_type': self.return_type,
            'param_types': self.param_types,
            'return_tuple': self.return_tuple,
            'nullable': self.nullable,
            'optional': self.optional,
            'is_hook': self.is_hook,
            'hook_type': self.hook_type,
            'inferred_at': self.inferred_at,
            'context': self.context
        }


class TypeInferenceEngine:
    """
    Advanced type inference engine for semantic analysis.
    
    Features:
    - Literal type inference
    - Function signature analysis
    - React hook type inference
    - Generic type handling
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize the type inference engine."""
        self.type_cache: Dict[str, TypeInfo] = {}
        self.inference_rules: List[Any] = []
        
        # React hook type signatures
        self.react_hook_signatures = {
            'useState': {
                'return_type': 'tuple',
                'generic_params': ['T'],
                'return_tuple': ['T', 'Dispatch<SetStateAction<T>>']
            },
            'useEffect': {
                'return_type': 'void',
                'param_types': ['() => void | (() => void)', 'any[]']
            },
            'useContext': {
                'return_type': 'T',
                'generic_params': ['T']
            },
            'useRef': {
                'return_type': 'MutableRefObject<T>',
                'generic_params': ['T']
            },
            'useMemo': {
                'return_type': 'T',
                'generic_params': ['T'],
                'param_types': ['() => T', 'any[]']
            },
            'useCallback': {
                'return_type': 'T',
                'generic_params': ['T'],
                'param_types': ['T', 'any[]']
            },
            'useReducer': {
                'return_type': 'tuple',
                'generic_params': ['R', 'I'],
                'return_tuple': ['R', 'Dispatch<A>']
            }
        }
        
        logging.info("TypeInferenceEngine initialized")
    
    def infer_from_literal(self, value: Any) -> TypeInfo:
        """
        Infer type from a literal value.
        
        Args:
            value: The literal value
            
        Returns:
            TypeInfo with inferred type
        """
        if isinstance(value, bool):
            return TypeInfo(
                name='boolean',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL
            )
        
        if isinstance(value, int):
            return TypeInfo(
                name='number',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL
            )
        
        if isinstance(value, float):
            return TypeInfo(
                name='number',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL
            )
        
        if isinstance(value, str):
            return TypeInfo(
                name='string',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL
            )
        
        if value is None:
            return TypeInfo(
                name='null',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL,
                nullable=True
            )
        
        if isinstance(value, list):
            return TypeInfo(
                name='array',
                category=TypeCategory.ARRAY,
                confidence=1.0,
                source=InferenceSource.LITERAL,
                base_type='any'
            )
        
        if isinstance(value, dict):
            return TypeInfo(
                name='object',
                category=TypeCategory.OBJECT,
                confidence=1.0,
                source=InferenceSource.LITERAL
            )
        
        return TypeInfo(
            name='unknown',
            category=TypeCategory.UNKNOWN,
            confidence=0.0,
            source=InferenceSource.IMPLICIT
        )
    
    def infer_from_ast_literal(self, literal_type: str, value: str) -> TypeInfo:
        """
        Infer type from AST literal node.
        
        Args:
            literal_type: Type of literal from AST (number, string, etc.)
            value: String representation of the value
            
        Returns:
            TypeInfo with inferred type
        """
        if literal_type in ['number', 'numeric_literal']:
            return TypeInfo(
                name='number',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL,
                context=f"literal: {value}"
            )
        
        if literal_type in ['string', 'string_literal', 'template_string']:
            return TypeInfo(
                name='string',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL,
                context=f"literal: {value[:50]}"
            )
        
        if literal_type in ['true', 'false', 'boolean']:
            return TypeInfo(
                name='boolean',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL,
                context=f"literal: {value}"
            )
        
        if literal_type in ['null', 'undefined']:
            return TypeInfo(
                name='null' if literal_type == 'null' else 'undefined',
                category=TypeCategory.PRIMITIVE,
                confidence=1.0,
                source=InferenceSource.LITERAL,
                nullable=True
            )
        
        if literal_type in ['array', 'array_expression']:
            return TypeInfo(
                name='array',
                category=TypeCategory.ARRAY,
                confidence=0.9,
                source=InferenceSource.LITERAL,
                base_type='any'
            )
        
        if literal_type in ['object', 'object_expression']:
            return TypeInfo(
                name='object',
                category=TypeCategory.OBJECT,
                confidence=0.9,
                source=InferenceSource.LITERAL
            )
        
        return TypeInfo(
            name='unknown',
            category=TypeCategory.UNKNOWN,
            confidence=0.3,
            source=InferenceSource.IMPLICIT
        )
    
    def infer_react_hook_type(self, hook_name: str, args: List[Any]) -> TypeInfo:
        """
        Infer type from React hook usage.
        
        Args:
            hook_name: Name of the hook (useState, useEffect, etc.)
            args: Arguments passed to the hook
            
        Returns:
            TypeInfo with inferred hook type
        """
        if hook_name not in self.react_hook_signatures:
            return TypeInfo(
                name='unknown',
                category=TypeCategory.REACT_HOOK,
                confidence=0.5,
                source=InferenceSource.HOOK,
                is_hook=True,
                hook_type=hook_name
            )
        
        hook_sig = self.react_hook_signatures[hook_name]
        
        # Special handling for useState
        if hook_name == 'useState' and args:
            # Infer state type from initial value
            initial_value = args[0]
            state_type = self.infer_from_literal(initial_value)
            
            return TypeInfo(
                name=f"[{state_type.name}, Dispatch<SetStateAction<{state_type.name}>>]",
                category=TypeCategory.REACT_HOOK,
                confidence=0.95,
                source=InferenceSource.HOOK,
                base_type='tuple',
                generic_params=[state_type.name],
                return_tuple=[state_type.name, f"Dispatch<SetStateAction<{state_type.name}>>"],
                is_hook=True,
                hook_type='useState',
                context=f"useState({initial_value})"
            )
        
        # Generic hook type
        return TypeInfo(
            name=hook_sig.get('return_type', 'unknown'),
            category=TypeCategory.REACT_HOOK,
            confidence=0.85,
            source=InferenceSource.HOOK,
            generic_params=hook_sig.get('generic_params', []),
            param_types=hook_sig.get('param_types', []),
            is_hook=True,
            hook_type=hook_name
        )
    
    def infer_function_type(
        self,
        params: List[Dict[str, Any]],
        return_expr: Optional[Any] = None,
        is_arrow: bool = False,
        is_async: bool = False
    ) -> TypeInfo:
        """
        Infer function type from signature and body.
        
        Args:
            params: Function parameters
            return_expr: Return expression (if any)
            is_arrow: Whether it's an arrow function
            is_async: Whether it's an async function
            
        Returns:
            TypeInfo with inferred function type
        """
        param_types = []
        for param in params:
            param_name = param.get('name', 'unknown')
            param_type = param.get('type', 'any')
            param_types.append(f"{param_name}: {param_type}")
        
        # Infer return type
        return_type = 'void'
        confidence = 0.7
        
        if return_expr:
            # Try to infer from return expression
            if isinstance(return_expr, dict):
                expr_type = return_expr.get('type', 'unknown')
                if expr_type in ['number', 'string', 'boolean']:
                    return_type = expr_type
                    confidence = 0.9
                elif expr_type == 'jsx_element':
                    return_type = 'JSX.Element'
                    confidence = 0.95
        
        if is_async:
            return_type = f"Promise<{return_type}>"
        
        func_signature = f"({', '.join(param_types)}) => {return_type}"
        
        return TypeInfo(
            name=func_signature,
            category=TypeCategory.FUNCTION,
            confidence=confidence,
            source=InferenceSource.SIGNATURE,
            return_type=return_type,
            param_types=[p.split(': ')[1] if ': ' in p else 'any' for p in param_types],
            context='arrow_function' if is_arrow else 'function_declaration'
        )
    
    def infer_from_assignment(
        self,
        var_name: str,
        value_type: str,
        is_const: bool = False
    ) -> TypeInfo:
        """
        Infer variable type from assignment.
        
        Args:
            var_name: Variable name
            value_type: Type of assigned value
            is_const: Whether it's a const declaration
            
        Returns:
            TypeInfo with inferred type
        """
        confidence = 0.85 if is_const else 0.7
        
        return TypeInfo(
            name=value_type,
            category=self._categorize_type_name(value_type),
            confidence=confidence,
            source=InferenceSource.ASSIGNMENT,
            context=f"{'const' if is_const else 'let/var'} {var_name}"
        )
    
    def infer_jsx_element_type(self, component_name: str) -> TypeInfo:
        """
        Infer type for JSX element.
        
        Args:
            component_name: Name of the component
            
        Returns:
            TypeInfo for JSX element
        """
        return TypeInfo(
            name='JSX.Element',
            category=TypeCategory.REACT_COMPONENT,
            confidence=0.95,
            source=InferenceSource.SIGNATURE,
            context=f"<{component_name} />"
        )
    
    def infer_from_usage_context(
        self,
        entity_name: str,
        usage_contexts: List[str]
    ) -> TypeInfo:
        """
        Infer type from usage contexts.
        
        Args:
            entity_name: Name of the entity
            usage_contexts: List of usage contexts
            
        Returns:
            TypeInfo with inferred type
        """
        # Analyze usage patterns
        is_called = any('call' in ctx for ctx in usage_contexts)
        is_accessed = any('property_access' in ctx for ctx in usage_contexts)
        is_indexed = any('index' in ctx for ctx in usage_contexts)
        
        if is_called:
            return TypeInfo(
                name='function',
                category=TypeCategory.FUNCTION,
                confidence=0.75,
                source=InferenceSource.USAGE,
                context='called as function'
            )
        
        if is_accessed:
            return TypeInfo(
                name='object',
                category=TypeCategory.OBJECT,
                confidence=0.70,
                source=InferenceSource.USAGE,
                context='property access'
            )
        
        if is_indexed:
            return TypeInfo(
                name='array',
                category=TypeCategory.ARRAY,
                confidence=0.70,
                source=InferenceSource.USAGE,
                base_type='any',
                context='indexed access'
            )
        
        return TypeInfo(
            name='unknown',
            category=TypeCategory.UNKNOWN,
            confidence=0.3,
            source=InferenceSource.IMPLICIT
        )
    
    def calculate_confidence(
        self,
        type_info: TypeInfo,
        evidence_count: int = 1,
        has_annotation: bool = False
    ) -> float:
        """
        Calculate confidence score for type inference.
        
        Args:
            type_info: The inferred type info
            evidence_count: Number of pieces of evidence
            has_annotation: Whether there's a type annotation
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = type_info.confidence
        
        # Boost for annotations
        if has_annotation:
            base_confidence = min(1.0, base_confidence + 0.2)
        
        # Boost for multiple evidence
        evidence_boost = min(0.15, evidence_count * 0.05)
        base_confidence = min(1.0, base_confidence + evidence_boost)
        
        # Penalty for unknown types
        if type_info.category == TypeCategory.UNKNOWN:
            base_confidence *= 0.5
        
        return round(base_confidence, 2)
    
    def merge_type_inferences(
        self,
        type_infos: List[TypeInfo]
    ) -> TypeInfo:
        """
        Merge multiple type inferences into one.
        
        Args:
            type_infos: List of type inferences to merge
            
        Returns:
            Merged TypeInfo
        """
        if not type_infos:
            return TypeInfo(
                name='unknown',
                category=TypeCategory.UNKNOWN,
                confidence=0.0,
                source=InferenceSource.IMPLICIT
            )
        
        if len(type_infos) == 1:
            return type_infos[0]
        
        # Group by type name
        type_groups: Dict[str, List[TypeInfo]] = {}
        for ti in type_infos:
            if ti.name not in type_groups:
                type_groups[ti.name] = []
            type_groups[ti.name].append(ti)
        
        # Find most confident type
        best_type = max(type_groups.items(), key=lambda x: sum(ti.confidence for ti in x[1]))
        best_type_name, best_type_infos = best_type
        
        # Calculate merged confidence
        avg_confidence = sum(ti.confidence for ti in best_type_infos) / len(best_type_infos)
        merged_confidence = min(1.0, avg_confidence + (len(best_type_infos) - 1) * 0.05)
        
        # Use the most detailed type info
        base_info = max(best_type_infos, key=lambda ti: ti.confidence)
        base_info.confidence = merged_confidence
        base_info.context = f"merged from {len(type_infos)} inferences"
        
        return base_info
    
    def _categorize_type_name(self, type_name: str) -> TypeCategory:
        """Categorize a type name."""
        type_lower = type_name.lower()
        
        if type_lower in ['number', 'string', 'boolean', 'null', 'undefined']:
            return TypeCategory.PRIMITIVE
        if 'array' in type_lower or type_name.endswith('[]'):
            return TypeCategory.ARRAY
        if 'function' in type_lower or '=>' in type_name:
            return TypeCategory.FUNCTION
        if 'jsx' in type_lower or 'element' in type_lower:
            return TypeCategory.REACT_COMPONENT
        if type_name.startswith('use'):
            return TypeCategory.REACT_HOOK
        if 'object' in type_lower:
            return TypeCategory.OBJECT
        
        return TypeCategory.UNKNOWN
    
    def clear_cache(self):
        """Clear the type cache."""
        self.type_cache.clear()
        logging.debug("Type inference cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about type inferences."""
        return {
            'cached_types': len(self.type_cache),
            'inference_rules': len(self.inference_rules)
        }


# Helper functions

def infer_type_from_node(node: Dict[str, Any], engine: TypeInferenceEngine) -> TypeInfo:
    """
    Infer type from an AST node.
    
    Args:
        node: AST node
        engine: TypeInferenceEngine instance
        
    Returns:
        TypeInfo with inferred type
    """
    node_type = node.get('type', 'unknown')
    
    # Literal inference
    if 'literal' in node_type.lower() or node_type in ['number', 'string', 'boolean', 'true', 'false', 'null']:
        value = node.get('value', '')
        return engine.infer_from_ast_literal(node_type, str(value))
    
    # Function inference
    if 'function' in node_type.lower():
        params = node.get('parameters', [])
        return_expr = node.get('return_expression')
        is_arrow = 'arrow' in node_type.lower()
        is_async = node.get('async', False)
        return engine.infer_function_type(params, return_expr, is_arrow, is_async)
    
    # JSX element
    if 'jsx' in node_type.lower():
        component_name = node.get('name', 'Component')
        return engine.infer_jsx_element_type(component_name)
    
    # Assignment
    if 'assignment' in node_type.lower() or 'declaration' in node_type.lower():
        var_name = node.get('name', 'var')
        value_type = node.get('value_type', 'any')
        is_const = node.get('kind') == 'const'
        return engine.infer_from_assignment(var_name, value_type, is_const)
    
    # Unknown
    return TypeInfo(
        name='unknown',
        category=TypeCategory.UNKNOWN,
        confidence=0.3,
        source=InferenceSource.IMPLICIT,
        context=f"node_type: {node_type}"
    )
