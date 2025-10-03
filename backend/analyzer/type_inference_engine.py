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
    
    # ==================== Advanced Type Inference (M3.4.2) ====================
    
    def infer_generic_type(
        self, 
        base_type: str, 
        type_params: List[str],
        constraints: Optional[Dict[str, str]] = None
    ) -> TypeInfo:
        """
        Infer generic types with type parameters and constraints.
        
        Examples:
            Array<T> where T extends number
            Promise<User>
            Map<K, V> where K extends string
            
        Args:
            base_type: Base generic type (Array, Promise, Map, etc.)
            type_params: Type parameter names (T, K, V, etc.)
            constraints: Optional type constraints for parameters
            
        Returns:
            TypeInfo with generic type information
        """
        # Build generic type name
        params_str = ", ".join(type_params)
        type_name = f"{base_type}<{params_str}>"
        
        # Add constraints if present
        constraint_info = []
        if constraints:
            for param, constraint in constraints.items():
                constraint_info.append(f"{param} extends {constraint}")
        
        return TypeInfo(
            name=type_name,
            category=TypeCategory.GENERIC,
            confidence=0.9,
            source=InferenceSource.SIGNATURE,
            base_type=base_type,
            generic_params=type_params,
            context=f"constraints: {', '.join(constraint_info)}" if constraint_info else None
        )
    
    def infer_conditional_type(
        self,
        condition_type: str,
        true_type: str,
        false_type: str
    ) -> TypeInfo:
        """
        Infer conditional types (TypeScript ternary types).
        
        Examples:
            T extends string ? string[] : T[]
            IsArray<T> = T extends Array<any> ? true : false
            
        Args:
            condition_type: The condition being checked
            true_type: Type when condition is true
            false_type: Type when condition is false
            
        Returns:
            TypeInfo with conditional type
        """
        type_name = f"{condition_type} ? {true_type} : {false_type}"
        
        return TypeInfo(
            name=type_name,
            category=TypeCategory.GENERIC,
            confidence=0.8,
            source=InferenceSource.SIGNATURE,
            context=f"conditional: {condition_type}"
        )
    
    def infer_mapped_type(
        self,
        key_type: str,
        value_type: str,
        modifiers: Optional[List[str]] = None
    ) -> TypeInfo:
        """
        Infer mapped types (TypeScript mapped types).
        
        Examples:
            { [K in keyof T]: T[K] }
            { readonly [P in Keys]: Type }
            { [P in Keys]?: Type }
            
        Args:
            key_type: Type of keys (keyof T, string, etc.)
            value_type: Type of values
            modifiers: Optional modifiers (readonly, optional)
            
        Returns:
            TypeInfo with mapped type
        """
        modifiers_str = " ".join(modifiers) if modifiers else ""
        type_name = f"{{ {modifiers_str} [K in {key_type}]: {value_type} }}"
        
        return TypeInfo(
            name=type_name,
            category=TypeCategory.OBJECT,
            confidence=0.85,
            source=InferenceSource.SIGNATURE,
            context=f"mapped_type: key={key_type}, value={value_type}"
        )
    
    def infer_utility_type(
        self,
        utility: str,
        target_type: str,
        additional_params: Optional[List[str]] = None
    ) -> TypeInfo:
        """
        Infer TypeScript utility types.
        
        Supported utilities:
            Partial<T>, Required<T>, Readonly<T>
            Pick<T, K>, Omit<T, K>, Exclude<T, U>, Extract<T, U>
            NonNullable<T>, ReturnType<T>, Parameters<T>
            
        Args:
            utility: Utility type name (Partial, Pick, etc.)
            target_type: Target type
            additional_params: Additional parameters for utilities like Pick, Omit
            
        Returns:
            TypeInfo with utility type
        """
        if additional_params:
            params_str = ", ".join([target_type] + additional_params)
            type_name = f"{utility}<{params_str}>"
        else:
            type_name = f"{utility}<{target_type}>"
        
        # Determine category based on utility
        category = TypeCategory.OBJECT
        if utility in ['ReturnType', 'Parameters']:
            category = TypeCategory.FUNCTION
        elif utility in ['NonNullable', 'Extract', 'Exclude']:
            category = TypeCategory.UNION
        
        return TypeInfo(
            name=type_name,
            category=category,
            confidence=0.95,
            source=InferenceSource.SIGNATURE,
            base_type=target_type,
            context=f"utility: {utility}"
        )
    
    def resolve_type_constraint(
        self,
        param_type: str,
        constraint: str,
        actual_type: str
    ) -> Tuple[bool, float]:
        """
        Check if an actual type satisfies a type constraint.
        
        Examples:
            T extends number → is actual type a number?
            K extends keyof T → is K a valid key of T?
            
        Args:
            param_type: Type parameter name (T, K, etc.)
            constraint: Constraint type (number, string, keyof T, etc.)
            actual_type: Actual type to check
            
        Returns:
            Tuple of (satisfies_constraint, confidence)
        """
        # Simple primitive constraint checking
        primitive_map = {
            'number': ['number', 'int', 'float', 'double'],
            'string': ['string', 'text'],
            'boolean': ['boolean', 'bool'],
            'object': ['object', 'Object'],
            'any': ['any', 'unknown']
        }
        
        if constraint in primitive_map:
            satisfies = actual_type in primitive_map[constraint]
            return (satisfies, 1.0 if satisfies else 0.5)
        
        # Array constraint
        if constraint.startswith('Array<') or constraint.endswith('[]'):
            satisfies = 'array' in actual_type.lower() or actual_type.endswith('[]')
            return (satisfies, 0.9 if satisfies else 0.4)
        
        # Object constraint (keyof, etc.)
        if 'keyof' in constraint:
            # Would need actual object structure to validate
            return (True, 0.6)  # Medium confidence
        
        # Generic constraint
        if '<' in constraint:
            # Check if actual_type is compatible generic
            satisfies = '<' in actual_type
            return (satisfies, 0.7 if satisfies else 0.5)
        
        # Default: assume compatible with low confidence
        return (True, 0.5)
    
    def infer_variance(
        self,
        type_param: str,
        position: str
    ) -> str:
        """
        Infer type variance (covariant, contravariant, invariant).
        
        Examples:
            Covariant: Array<T> - can use subtypes
            Contravariant: Function parameters - can use supertypes
            Invariant: Mutable structures - must be exact type
            
        Args:
            type_param: Type parameter name
            position: Usage position ('return', 'parameter', 'property')
            
        Returns:
            Variance type: 'covariant', 'contravariant', 'invariant'
        """
        if position == 'return':
            return 'covariant'  # Return types are covariant
        elif position == 'parameter':
            return 'contravariant'  # Parameter types are contravariant
        else:
            return 'invariant'  # Properties are invariant
    
    def infer_union_intersection(
        self,
        types: List[str],
        operator: str
    ) -> TypeInfo:
        """
        Infer union or intersection types.
        
        Examples:
            Union: string | number | boolean
            Intersection: Person & Employee
            
        Args:
            types: List of types to combine
            operator: '|' for union, '&' for intersection
            
        Returns:
            TypeInfo with combined type
        """
        type_name = f" {operator} ".join(types)
        category = TypeCategory.UNION if operator == '|' else TypeCategory.INTERSECTION
        
        return TypeInfo(
            name=type_name,
            category=category,
            confidence=0.95,
            source=InferenceSource.SIGNATURE,
            generic_params=types,
            context=f"combined: {len(types)} types"
        )
    
    def infer_discriminated_union(
        self,
        discriminator: str,
        variants: Dict[str, Dict[str, str]]
    ) -> TypeInfo:
        """
        Infer discriminated union types (tagged unions).
        
        Examples:
            type Shape = 
                | { kind: 'circle', radius: number }
                | { kind: 'square', size: number }
                
        Args:
            discriminator: Discriminator property name
            variants: Dict mapping discriminant values to type shapes
            
        Returns:
            TypeInfo with discriminated union
        """
        variant_names = list(variants.keys())
        type_name = " | ".join([f"{{ {discriminator}: '{v}', ... }}" for v in variant_names])
        
        return TypeInfo(
            name=type_name,
            category=TypeCategory.UNION,
            confidence=0.9,
            source=InferenceSource.SIGNATURE,
            context=f"discriminated: {discriminator} with {len(variants)} variants"
        )
    
    # ==================== Type-Based Relationships (M3.4.3) ====================
    
    def check_type_compatibility(
        self,
        source_type: str,
        target_type: str,
        strict: bool = False
    ) -> Tuple[bool, float, str]:
        """
        Check if source type is compatible with target type.
        
        Examples:
            'number' → 'any' (compatible, widening)
            'string' → 'number' (incompatible)
            'Dog' → 'Animal' (compatible if Dog extends Animal)
            
        Args:
            source_type: Source type
            target_type: Target type
            strict: Whether to use strict type checking
            
        Returns:
            Tuple of (is_compatible, confidence, reason)
        """
        # Any type accepts everything
        if target_type == 'any' or target_type == 'unknown':
            return (True, 1.0, "target is any/unknown")
        
        # Exact match
        if source_type == target_type:
            return (True, 1.0, "exact match")
        
        # Null/undefined compatibility
        if source_type in ['null', 'undefined']:
            if strict:
                return (False, 0.9, "strict mode rejects null/undefined")
            return (True, 0.7, "nullable in non-strict mode")
        
        # Number compatibility
        if target_type == 'number':
            if source_type in ['int', 'float', 'double', 'bigint']:
                return (True, 0.95, "numeric type compatible")
            return (False, 0.9, "not a number type")
        
        # String compatibility
        if target_type == 'string':
            if source_type in ['string', 'text', 'String']:
                return (True, 1.0, "string type compatible")
            return (False, 0.9, "not a string type")
        
        # Array compatibility
        if target_type.startswith('Array<') or target_type.endswith('[]'):
            if source_type.startswith('Array<') or source_type.endswith('[]'):
                # Could check element types here
                return (True, 0.85, "array types compatible")
            return (False, 0.9, "not an array type")
        
        # Object compatibility
        if target_type == 'object' or target_type == 'Object':
            if source_type not in ['number', 'string', 'boolean', 'null', 'undefined']:
                return (True, 0.7, "object-like type")
            return (False, 0.8, "primitive type, not object")
        
        # Union type compatibility
        if ' | ' in target_type:
            union_types = [t.strip() for t in target_type.split('|')]
            for union_member in union_types:
                is_compat, conf, _ = self.check_type_compatibility(source_type, union_member, strict)
                if is_compat:
                    return (True, conf * 0.9, f"compatible with union member: {union_member}")
            return (False, 0.7, "not compatible with any union member")
        
        # Generic type compatibility
        if '<' in source_type and '<' in target_type:
            source_base = source_type.split('<')[0]
            target_base = target_type.split('<')[0]
            if source_base == target_base:
                return (True, 0.8, f"same generic base: {source_base}")
        
        # Unknown compatibility - could be subtypes
        return (False, 0.5, "types may be incompatible (needs type hierarchy)")
    
    def create_type_edge(
        self,
        from_node: str,
        to_node: str,
        relation_type: str,
        type_info: Optional[TypeInfo] = None
    ) -> Dict[str, Any]:
        """
        Create a type-based edge between nodes.
        
        Edge types:
            - type_assignment: Variable receives typed value
            - type_parameter: Function parameter type
            - type_return: Function return type
            - type_extends: Class/interface extension
            - type_implements: Interface implementation
            - type_constraint: Generic type constraint
            
        Args:
            from_node: Source node ID
            to_node: Target node ID
            relation_type: Type of relationship
            type_info: Optional TypeInfo with additional details
            
        Returns:
            Edge dictionary
        """
        edge = {
            'from': from_node,
            'to': to_node,
            'type': relation_type,
            'metadata': {
                'analysis_type': 'type_inference',
                'edge_category': 'type_relationship'
            }
        }
        
        if type_info:
            edge['type_details'] = {
                'type_name': type_info.name,
                'type_category': type_info.category.value,
                'confidence': type_info.confidence,
                'source': type_info.source.value
            }
        
        return edge
    
    def extract_type_hierarchy(
        self,
        nodes: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Extract type hierarchy from AST nodes.
        
        Builds inheritance and implementation relationships:
            - Class extends relationships
            - Interface extends relationships
            - Class implements relationships
            
        Args:
            nodes: List of AST nodes
            
        Returns:
            Dict mapping type names to their parent types
        """
        hierarchy = {}
        
        for node in nodes:
            node_type = node.get('type', '')
            
            # Class declarations
            if node_type == 'class_declaration':
                class_name = node.get('name', 'Unknown')
                parents = []
                
                # Extends clause
                if 'heritage_clause' in node or 'extends' in node:
                    extends = node.get('extends') or node.get('heritage_clause', {}).get('extends', [])
                    if isinstance(extends, list):
                        parents.extend(extends)
                    elif extends:
                        parents.append(extends)
                
                # Implements clause
                if 'implements' in node:
                    implements = node.get('implements', [])
                    if isinstance(implements, list):
                        parents.extend(implements)
                    elif implements:
                        parents.append(implements)
                
                if parents:
                    hierarchy[class_name] = parents
            
            # Interface declarations
            elif node_type == 'interface_declaration':
                interface_name = node.get('name', 'Unknown')
                extends = node.get('extends', [])
                
                if extends:
                    if isinstance(extends, list):
                        hierarchy[interface_name] = extends
                    else:
                        hierarchy[interface_name] = [extends]
        
        return hierarchy
    
    def build_type_relationships(
        self,
        nodes: List[Dict[str, Any]],
        existing_edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build type-based edges from AST nodes.
        
        Creates edges for:
            - Variable assignments with types
            - Function parameters with types
            - Function return types
            - Class/interface inheritance
            - Generic type constraints
            
        Args:
            nodes: List of AST nodes
            existing_edges: Existing edges to augment
            
        Returns:
            List of new type-based edges
        """
        new_edges = []
        
        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', '')
            
            # Variable declarations with types
            if node_type in ['variable_declaration', 'lexical_declaration']:
                declarations = node.get('declarations', [node])
                for decl in declarations:
                    var_name = decl.get('name', 'var')
                    type_annotation = decl.get('type_annotation')
                    
                    if type_annotation:
                        type_info = TypeInfo(
                            name=type_annotation,
                            category=TypeCategory.UNKNOWN,
                            confidence=0.95,
                            source=InferenceSource.ANNOTATION
                        )
                        edge = self.create_type_edge(
                            from_node=node_id,
                            to_node=f"type:{type_annotation}",
                            relation_type='type_annotation',
                            type_info=type_info
                        )
                        new_edges.append(edge)
            
            # Function parameters
            elif node_type in ['function_declaration', 'method_definition', 'arrow_function']:
                func_name = node.get('name', 'anonymous')
                params = node.get('parameters', [])
                
                for param in params:
                    param_name = param.get('name', 'param') if isinstance(param, dict) else param
                    param_type = param.get('type_annotation') if isinstance(param, dict) else None
                    
                    if param_type:
                        type_info = TypeInfo(
                            name=param_type,
                            category=TypeCategory.UNKNOWN,
                            confidence=0.95,
                            source=InferenceSource.PARAMETER
                        )
                        edge = self.create_type_edge(
                            from_node=f"{node_id}:{param_name}",
                            to_node=f"type:{param_type}",
                            relation_type='type_parameter',
                            type_info=type_info
                        )
                        new_edges.append(edge)
                
                # Return type
                return_type = node.get('return_type')
                if return_type:
                    type_info = TypeInfo(
                        name=return_type,
                        category=TypeCategory.UNKNOWN,
                        confidence=0.95,
                        source=InferenceSource.RETURN
                    )
                    edge = self.create_type_edge(
                        from_node=node_id,
                        to_node=f"type:{return_type}",
                        relation_type='type_return',
                        type_info=type_info
                    )
                    new_edges.append(edge)
            
            # Class/Interface inheritance
            elif node_type in ['class_declaration', 'interface_declaration']:
                class_name = node.get('name', 'Unknown')
                
                # Extends
                extends = node.get('extends')
                if extends:
                    extends_list = [extends] if isinstance(extends, str) else extends
                    for parent in extends_list:
                        edge = self.create_type_edge(
                            from_node=f"type:{class_name}",
                            to_node=f"type:{parent}",
                            relation_type='type_extends',
                            type_info=None
                        )
                        new_edges.append(edge)
                
                # Implements
                implements = node.get('implements', [])
                if implements:
                    implements_list = [implements] if isinstance(implements, str) else implements
                    for interface in implements_list:
                        edge = self.create_type_edge(
                            from_node=f"type:{class_name}",
                            to_node=f"type:{interface}",
                            relation_type='type_implements',
                            type_info=None
                        )
                        new_edges.append(edge)
        
        return new_edges
    
    def analyze_type_flow(
        self,
        source_node: Dict[str, Any],
        target_node: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze type flow between two nodes.
        
        Checks:
            - Is type information flowing from source to target?
            - Are the types compatible?
            - What is the confidence level?
            
        Args:
            source_node: Source AST node
            target_node: Target AST node
            
        Returns:
            Type flow analysis dict or None
        """
        source_type = source_node.get('inferred_type')
        target_type = target_node.get('expected_type')
        
        if not source_type or not target_type:
            return None
        
        is_compatible, confidence, reason = self.check_type_compatibility(
            source_type, target_type
        )
        
        return {
            'from': source_node.get('id'),
            'to': target_node.get('id'),
            'source_type': source_type,
            'target_type': target_type,
            'compatible': is_compatible,
            'confidence': confidence,
            'reason': reason,
            'flow_type': 'type_propagation'
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
