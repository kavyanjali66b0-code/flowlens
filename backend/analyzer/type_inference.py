"""
Type Inference Engine

Infers types from literals, operations, function returns, and propagates
type information through the code.

Features:
- Literal type inference (string, number, boolean, null, array, object)
- Operation type inference (arithmetic, logical, comparison)
- Function return type inference
- Type propagation through assignments and calls
- Type compatibility checking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Set, Any, Union, Tuple
from tree_sitter import Node


class TypeCategory(Enum):
    """Categories of inferred types."""
    UNKNOWN = "unknown"
    PRIMITIVE = "primitive"      # string, number, boolean, null, undefined
    ARRAY = "array"              # Array types
    OBJECT = "object"            # Object types
    FUNCTION = "function"        # Function types
    CLASS = "class"              # Class types
    UNION = "union"              # Union types (string | number)
    GENERIC = "generic"          # Generic types (Array<T>)


class PrimitiveType(Enum):
    """Primitive JavaScript/TypeScript types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"
    UNDEFINED = "undefined"
    SYMBOL = "symbol"
    BIGINT = "bigint"


@dataclass
class TypeInfo:
    """Represents inferred type information."""
    category: TypeCategory
    name: str                                    # Type name (e.g., "string", "Array", "UserService")
    primitive_type: Optional[PrimitiveType] = None
    element_type: Optional['TypeInfo'] = None    # For arrays: Array<element_type>
    properties: Dict[str, 'TypeInfo'] = field(default_factory=dict)  # For objects
    parameters: List['TypeInfo'] = field(default_factory=list)       # For functions
    return_type: Optional['TypeInfo'] = None     # For functions
    union_types: List['TypeInfo'] = field(default_factory=list)     # For union types
    nullable: bool = False
    confidence: float = 1.0                      # Confidence score (0.0-1.0)
    source: str = "inferred"                     # Source of type info
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of type."""
        if self.category == TypeCategory.PRIMITIVE:
            suffix = " | null" if self.nullable else ""
            return f"{self.name}{suffix}"
        elif self.category == TypeCategory.ARRAY:
            element = str(self.element_type) if self.element_type else "unknown"
            return f"Array<{element}>"
        elif self.category == TypeCategory.UNION:
            types = " | ".join(str(t) for t in self.union_types)
            return f"({types})"
        elif self.category == TypeCategory.FUNCTION:
            params = ", ".join(str(p) for p in self.parameters)
            ret = str(self.return_type) if self.return_type else "unknown"
            return f"({params}) => {ret}"
        return self.name
    
    def is_compatible_with(self, other: 'TypeInfo') -> bool:
        """Check if this type is compatible with another type."""
        # Same type
        if self.category == other.category and self.name == other.name:
            # Check nullable compatibility
            # Non-nullable is NOT compatible with nullable (can't assign null to non-nullable)
            if other.nullable and not self.nullable:
                return False
            return True
        
        # Any type is compatible with everything
        if self.name == "any" or other.name == "any":
            return True
        
        # Unknown types are compatible with everything
        if self.category == TypeCategory.UNKNOWN or other.category == TypeCategory.UNKNOWN:
            return True
        
        # Array compatibility - check element types
        if self.category == TypeCategory.ARRAY and other.category == TypeCategory.ARRAY:
            if self.element_type and other.element_type:
                return self.element_type.is_compatible_with(other.element_type)
            return True  # Unknown element types are compatible
        
        # Union type compatibility
        if other.category == TypeCategory.UNION:
            return any(self.is_compatible_with(t) for t in other.union_types)
        
        if self.category == TypeCategory.UNION:
            return all(t.is_compatible_with(other) for t in self.union_types)
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'category': self.category.value,
            'name': self.name,
            'primitive_type': self.primitive_type.value if self.primitive_type else None,
            'nullable': self.nullable,
            'confidence': self.confidence,
            'source': self.source,
            'element_type': self.element_type.to_dict() if self.element_type else None,
            'properties': {k: v.to_dict() for k, v in self.properties.items()},
            'parameters': [p.to_dict() for p in self.parameters],
            'return_type': self.return_type.to_dict() if self.return_type else None,
            'union_types': [t.to_dict() for t in self.union_types],
            'metadata': self.metadata
        }


class TypeInferenceEngine:
    """
    Infers types from JavaScript/TypeScript code.
    
    Features:
    - Infer types from literal values
    - Infer types from operations
    - Infer function return types
    - Propagate types through assignments
    """
    
    def __init__(self, nodes: List[Node], file_path: str, source_code: bytes):
        """
        Initialize type inference engine.
        
        Args:
            nodes: List of AST nodes to analyze
            file_path: Path to the file being analyzed
            source_code: Source code as bytes
        """
        self.nodes = nodes
        self.file_path = file_path
        self.source_code = source_code
        self.type_map: Dict[str, TypeInfo] = {}  # variable -> TypeInfo
        self.inferred_types: Dict[Node, TypeInfo] = {}  # node -> TypeInfo
    
    def infer_all_types(self) -> Dict[str, TypeInfo]:
        """
        Infer types for all nodes.
        
        Returns:
            Dictionary mapping variable names to their inferred types
        """
        for node in self.nodes:
            self._infer_node_type(node)
        
        return self.type_map
    
    def _infer_node_type(self, node: Node) -> Optional[TypeInfo]:
        """
        Infer type for a single node.
        
        Args:
            node: AST node to infer type for
            
        Returns:
            Inferred type information or None
        """
        # Check cache
        if node in self.inferred_types:
            return self.inferred_types[node]
        
        type_info = None
        
        # Literal values
        if node.type == 'string':
            type_info = self._infer_string_literal(node)
        elif node.type == 'number':
            type_info = self._infer_number_literal(node)
        elif node.type in ('true', 'false'):
            type_info = self._infer_boolean_literal(node)
        elif node.type == 'null':
            type_info = self._infer_null_literal(node)
        elif node.type == 'undefined':
            type_info = self._infer_undefined_literal(node)
        elif node.type == 'array':
            type_info = self._infer_array_literal(node)
        elif node.type == 'object':
            type_info = self._infer_object_literal(node)
        
        # Variable declarations
        elif node.type in ('lexical_declaration', 'variable_declaration'):
            type_info = self._infer_variable_declaration(node)
        
        # Expressions
        elif node.type == 'binary_expression':
            type_info = self._infer_binary_expression(node)
        elif node.type == 'unary_expression':
            type_info = self._infer_unary_expression(node)
        elif node.type == 'call_expression':
            type_info = self._infer_call_expression(node)
        
        # Functions
        elif node.type in ('function_declaration', 'arrow_function', 'function'):
            type_info = self._infer_function_type(node)
        
        # Identifiers - lookup in type map
        elif node.type == 'identifier':
            name = self._get_text(node)
            type_info = self.type_map.get(name)
        
        # Cache and return
        if type_info:
            self.inferred_types[node] = type_info
        
        # Recurse into children
        for child in node.children:
            self._infer_node_type(child)
        
        return type_info
    
    def _infer_string_literal(self, node: Node) -> TypeInfo:
        """Infer type from string literal."""
        return TypeInfo(
            category=TypeCategory.PRIMITIVE,
            name="string",
            primitive_type=PrimitiveType.STRING,
            confidence=1.0,
            source="literal"
        )
    
    def _infer_number_literal(self, node: Node) -> TypeInfo:
        """Infer type from number literal."""
        return TypeInfo(
            category=TypeCategory.PRIMITIVE,
            name="number",
            primitive_type=PrimitiveType.NUMBER,
            confidence=1.0,
            source="literal"
        )
    
    def _infer_boolean_literal(self, node: Node) -> TypeInfo:
        """Infer type from boolean literal."""
        return TypeInfo(
            category=TypeCategory.PRIMITIVE,
            name="boolean",
            primitive_type=PrimitiveType.BOOLEAN,
            confidence=1.0,
            source="literal"
        )
    
    def _infer_null_literal(self, node: Node) -> TypeInfo:
        """Infer type from null literal."""
        return TypeInfo(
            category=TypeCategory.PRIMITIVE,
            name="null",
            primitive_type=PrimitiveType.NULL,
            nullable=True,
            confidence=1.0,
            source="literal"
        )
    
    def _infer_undefined_literal(self, node: Node) -> TypeInfo:
        """Infer type from undefined literal."""
        return TypeInfo(
            category=TypeCategory.PRIMITIVE,
            name="undefined",
            primitive_type=PrimitiveType.UNDEFINED,
            nullable=True,
            confidence=1.0,
            source="literal"
        )
    
    def _infer_array_literal(self, node: Node) -> TypeInfo:
        """Infer type from array literal."""
        # Find element types
        element_types: Set[str] = set()
        elements = [child for child in node.children if child.type != '[' and child.type != ']' and child.type != ',']
        
        for element in elements:
            element_type = self._infer_node_type(element)
            if element_type:
                element_types.add(element_type.name)
        
        # Determine common element type
        if len(element_types) == 0:
            element_type_info = TypeInfo(
                category=TypeCategory.UNKNOWN,
                name="unknown"
            )
        elif len(element_types) == 1:
            # Homogeneous array
            element_name = list(element_types)[0]
            element_type_info = TypeInfo(
                category=TypeCategory.PRIMITIVE if element_name in [e.value for e in PrimitiveType] else TypeCategory.UNKNOWN,
                name=element_name
            )
        else:
            # Heterogeneous array - create union type
            union_types = [
                TypeInfo(category=TypeCategory.PRIMITIVE, name=t)
                for t in element_types
            ]
            element_type_info = TypeInfo(
                category=TypeCategory.UNION,
                name=" | ".join(element_types),
                union_types=union_types
            )
        
        return TypeInfo(
            category=TypeCategory.ARRAY,
            name="Array",
            element_type=element_type_info,
            confidence=0.9,
            source="literal"
        )
    
    def _infer_object_literal(self, node: Node) -> TypeInfo:
        """Infer type from object literal."""
        properties: Dict[str, TypeInfo] = {}
        
        # Find property definitions
        for child in node.children:
            if child.type == 'pair':
                key_node = self._find_child(child, 'property_identifier')
                if not key_node:
                    key_node = self._find_child(child, 'string')
                
                value_node = None
                for c in child.children:
                    if c.type not in ('property_identifier', 'string', ':', ','):
                        value_node = c
                        break
                
                if key_node and value_node:
                    key = self._get_text(key_node).strip('"\'')
                    value_type = self._infer_node_type(value_node)
                    if value_type:
                        properties[key] = value_type
        
        return TypeInfo(
            category=TypeCategory.OBJECT,
            name="object",
            properties=properties,
            confidence=0.9,
            source="literal"
        )
    
    def _infer_variable_declaration(self, node: Node) -> Optional[TypeInfo]:
        """Infer type from variable declaration."""
        # Find variable declarator
        declarator = self._find_child(node, 'variable_declarator')
        if not declarator:
            return None
        
        # Get variable name
        name_node = self._find_child(declarator, 'identifier')
        if not name_node:
            return None
        
        var_name = self._get_text(name_node)
        
        # Get initializer (value) - it's after the '=' sign
        value_node = None
        found_equals = False
        for child in declarator.children:
            if child.type == '=':
                found_equals = True
            elif found_equals and child.type not in (';', 'type_annotation'):
                value_node = child
                break
        
        if value_node:
            type_info = self._infer_node_type(value_node)
            if type_info:
                self.type_map[var_name] = type_info
                return type_info
        
        return None
    
    def _infer_binary_expression(self, node: Node) -> Optional[TypeInfo]:
        """Infer type from binary expression."""
        # Get operator
        operator = None
        left = None
        right = None
        
        for child in node.children:
            if child.type == 'identifier' and not left:
                left = child
            elif child.type == 'identifier' and left:
                right = child
            elif child.type not in ('(', ')') and not child.type.endswith('expression'):
                if not left:
                    left = child
                elif not right:
                    if self._get_text(child) not in ('+', '-', '*', '/', '%', '==', '===', '!=', '!==', '<', '>', '<=', '>=', '&&', '||'):
                        right = child
                    else:
                        operator = self._get_text(child)
        
        # Handle case where operator is between left and right
        if not operator:
            for i, child in enumerate(node.children):
                if i > 0 and i < len(node.children) - 1:
                    text = self._get_text(child)
                    if text in ('+', '-', '*', '/', '%', '==', '===', '!=', '!==', '<', '>', '<=', '>=', '&&', '||'):
                        operator = text
                        break
        
        if not operator or not left or not right:
            return None
        
        left_type = self._infer_node_type(left)
        right_type = self._infer_node_type(right)
        
        # Arithmetic operators: +, -, *, /, %
        if operator in ('+', '-', '*', '/', '%'):
            # Special case: + with strings is concatenation
            if operator == '+':
                if left_type and left_type.name == "string" or right_type and right_type.name == "string":
                    return TypeInfo(
                        category=TypeCategory.PRIMITIVE,
                        name="string",
                        primitive_type=PrimitiveType.STRING,
                        confidence=0.9,
                        source="operation"
                    )
            # Otherwise, arithmetic returns number
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="number",
                primitive_type=PrimitiveType.NUMBER,
                confidence=0.9,
                source="operation"
            )
        
        # Comparison operators: ==, ===, !=, !==, <, >, <=, >=
        if operator in ('==', '===', '!=', '!==', '<', '>', '<=', '>='):
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="boolean",
                primitive_type=PrimitiveType.BOOLEAN,
                confidence=0.95,
                source="operation"
            )
        
        # Logical operators: &&, ||
        if operator in ('&&', '||'):
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="boolean",
                primitive_type=PrimitiveType.BOOLEAN,
                confidence=0.9,
                source="operation"
            )
        
        return None
    
    def _infer_unary_expression(self, node: Node) -> Optional[TypeInfo]:
        """Infer type from unary expression."""
        # Get operator
        operator = None
        for child in node.children:
            text = self._get_text(child)
            if text in ('!', '~', '+', '-', 'typeof', 'void', 'delete'):
                operator = text
                break
        
        if not operator:
            return None
        
        # ! returns boolean
        if operator == '!':
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="boolean",
                primitive_type=PrimitiveType.BOOLEAN,
                confidence=1.0,
                source="operation"
            )
        
        # typeof returns string
        if operator == 'typeof':
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="string",
                primitive_type=PrimitiveType.STRING,
                confidence=1.0,
                source="operation"
            )
        
        # Unary + and - return number
        if operator in ('+', '-', '~'):
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="number",
                primitive_type=PrimitiveType.NUMBER,
                confidence=0.95,
                source="operation"
            )
        
        # void returns undefined
        if operator == 'void':
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="undefined",
                primitive_type=PrimitiveType.UNDEFINED,
                confidence=1.0,
                source="operation"
            )
        
        return None
    
    def _infer_call_expression(self, node: Node) -> Optional[TypeInfo]:
        """Infer type from call expression."""
        # For now, return unknown - will be enhanced in later phases
        return TypeInfo(
            category=TypeCategory.UNKNOWN,
            name="unknown",
            confidence=0.5,
            source="call"
        )
    
    def _infer_function_type(self, node: Node) -> Optional[TypeInfo]:
        """Infer function type from declaration."""
        # Get function name if it's a function_declaration
        func_name = None
        if node.type == 'function_declaration':
            for child in node.children:
                if child.type == 'identifier':
                    func_name = self._get_text(child)
                    break
        
        # Get parameters
        param_types: List[TypeInfo] = []
        params_node = self._find_child(node, 'formal_parameters')
        if params_node:
            for param in params_node.children:
                if param.type == 'identifier':
                    # Parameter without type annotation - unknown
                    param_types.append(TypeInfo(
                        category=TypeCategory.UNKNOWN,
                        name="unknown"
                    ))
        
        # Get return type from return statements
        return_type = self._infer_return_type(node)
        
        type_info = TypeInfo(
            category=TypeCategory.FUNCTION,
            name="function",
            parameters=param_types,
            return_type=return_type,
            confidence=0.8,
            source="declaration"
        )
        
        # Add to type map if it has a name
        if func_name:
            self.type_map[func_name] = type_info
        
        return type_info
    
    def _infer_return_type(self, function_node: Node) -> TypeInfo:
        """Infer return type from function body's return statements."""
        # Find function body
        body_node = None
        
        # For arrow functions with expression body (no braces)
        if function_node.type == 'arrow_function':
            # Check if it has a statement_block or is an expression
            for child in function_node.children:
                if child.type == 'statement_block':
                    body_node = child
                    break
                elif child.type not in ('formal_parameters', '=>', '(', ')'):
                    # This is an expression body - implicit return
                    return_expr_type = self._infer_node_type(child)
                    if return_expr_type:
                        return return_expr_type
                    else:
                        return TypeInfo(
                            category=TypeCategory.UNKNOWN,
                            name="unknown",
                            confidence=0.5,
                            source="implicit_return"
                        )
        else:
            # Regular function or method
            body_node = self._find_child(function_node, 'statement_block')
        
        if not body_node:
            # No body found - return undefined
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="undefined",
                primitive_type=PrimitiveType.UNDEFINED,
                confidence=0.9,
                source="no_body"
            )
        
        # Find all return statements in the body
        return_types: List[TypeInfo] = []
        self._collect_return_types(body_node, return_types)
        
        # No return statements - returns undefined
        if not return_types:
            return TypeInfo(
                category=TypeCategory.PRIMITIVE,
                name="undefined",
                primitive_type=PrimitiveType.UNDEFINED,
                confidence=0.95,
                source="no_return"
            )
        
        # Single return type
        if len(return_types) == 1:
            return return_types[0]
        
        # Multiple returns - check if all same type
        first_type = return_types[0]
        all_same = all(
            rt.category == first_type.category and 
            rt.name == first_type.name 
            for rt in return_types[1:]
        )
        
        if all_same:
            # All returns have same type
            return first_type
        
        # Different types - create union
        return TypeInfo(
            category=TypeCategory.UNION,
            name=" | ".join(rt.name for rt in return_types),
            union_types=return_types,
            confidence=0.85,
            source="multiple_returns"
        )
    
    def _collect_return_types(self, node: Node, return_types: List[TypeInfo]) -> None:
        """Recursively collect return types from all return statements."""
        if node.type == 'return_statement':
            # Find the expression being returned
            return_expr = None
            for child in node.children:
                if child.type not in ('return', ';'):
                    return_expr = child
                    break
            
            if return_expr:
                # Infer type of return expression
                expr_type = self._infer_node_type(return_expr)
                if expr_type:
                    return_types.append(expr_type)
            else:
                # Empty return - returns undefined
                return_types.append(TypeInfo(
                    category=TypeCategory.PRIMITIVE,
                    name="undefined",
                    primitive_type=PrimitiveType.UNDEFINED,
                    confidence=1.0,
                    source="empty_return"
                ))
            return
        
        # Recurse into children, but skip nested functions
        if node.type not in ('function_declaration', 'function', 'arrow_function', 'method_definition'):
            for child in node.children:
                self._collect_return_types(child, return_types)
    
    def get_type_for_variable(self, var_name: str) -> Optional[TypeInfo]:
        """Get inferred type for a variable."""
        return self.type_map.get(var_name)
    
    def get_type_for_node(self, node: Node) -> Optional[TypeInfo]:
        """Get inferred type for a node."""
        return self.inferred_types.get(node)
    
    def get_all_types(self) -> Dict[str, TypeInfo]:
        """Get all inferred types."""
        return self.type_map.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get type inference statistics."""
        type_categories = {}
        for type_info in self.type_map.values():
            category = type_info.category.value
            type_categories[category] = type_categories.get(category, 0) + 1
        
        return {
            'total_types_inferred': len(self.type_map),
            'total_nodes_typed': len(self.inferred_types),
            'type_categories': type_categories,
            'average_confidence': sum(t.confidence for t in self.type_map.values()) / len(self.type_map) if self.type_map else 0.0
        }
    
    # Helper methods
    
    def _find_child(self, node: Node, child_type: str) -> Optional[Node]:
        """Find first child of given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def _get_text(self, node: Node) -> str:
        """Get text content of a node."""
        return self.source_code[node.start_byte:node.end_byte].decode('utf-8')
