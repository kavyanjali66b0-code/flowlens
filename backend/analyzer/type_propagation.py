"""
Type Propagation Module

Tracks type flow through:
- Variable assignments (forward propagation)
- Conditional expressions (backward propagation/type narrowing)
- Function parameter inference
- Control flow analysis

Example:
    Forward: x = 42; y = x  → y is number
    Backward: if (x > 0) → x is number
"""

from typing import Dict, Optional, List, Set, Tuple
from dataclasses import dataclass
from tree_sitter import Node
from analyzer.type_inference import TypeInfo, TypeCategory, PrimitiveType


@dataclass
class PropagationContext:
    """Context for type propagation."""
    variable_types: Dict[str, TypeInfo]  # Current known types
    narrowed_types: Dict[str, TypeInfo]  # Types narrowed by conditions
    scope_stack: List[Dict[str, TypeInfo]]  # Scope hierarchy


class TypePropagator:
    """
    Propagates types through variable assignments and control flow.
    
    Handles:
    - Forward propagation: x = 42; y = x
    - Backward propagation: if (x > 0) → x is number
    - Assignment chains
    - Conditional type narrowing
    """
    
    def __init__(
        self,
        nodes: List[Node],
        file_path: str,
        source_code: bytes,
        initial_types: Optional[Dict[str, TypeInfo]] = None
    ):
        """
        Initialize type propagator.
        
        Args:
            nodes: AST nodes to analyze
            file_path: Path to source file
            source_code: Source code as bytes
            initial_types: Initial type information from inference
        """
        self.nodes = nodes
        self.file_path = file_path
        self.source_code = source_code
        self.variable_types: Dict[str, TypeInfo] = initial_types or {}
        self.propagated_types: Dict[str, TypeInfo] = {}
        self.assignment_graph: Dict[str, List[str]] = {}  # var -> vars it's assigned to
    
    def propagate_all(self) -> Dict[str, TypeInfo]:
        """
        Propagate types through all nodes.
        
        Returns:
            Dictionary of variable names to their propagated types
        """
        # First pass: build assignment graph
        for node in self.nodes:
            self._build_assignment_graph(node)
        
        # Second pass: propagate types forward
        self._propagate_forward()
        
        # Merge with initial types
        result = self.variable_types.copy()
        result.update(self.propagated_types)
        
        return result
    
    def _build_assignment_graph(self, node: Node) -> None:
        """Build graph of variable assignments."""
        if node.type in ('lexical_declaration', 'variable_declaration'):
            self._process_variable_declaration(node)
        elif node.type == 'assignment_expression':
            self._process_assignment(node)
        
        # Recurse into children
        for child in node.children:
            self._build_assignment_graph(child)
    
    def _process_variable_declaration(self, node: Node) -> None:
        """Process variable declaration for propagation."""
        declarator = self._find_child(node, 'variable_declarator')
        if not declarator:
            return
        
        # Get variable name
        name_node = self._find_child(declarator, 'identifier')
        if not name_node:
            return
        
        var_name = self._get_text(name_node)
        
        # Check if initialized from another variable
        value_node = None
        for child in declarator.children:
            if child.type not in ('identifier', '=', 'type_annotation'):
                value_node = child
                break
        
        if value_node and value_node.type == 'identifier':
            # Assignment from another variable: const y = x;
            source_var = self._get_text(value_node)
            if source_var in self.variable_types:
                # Record propagation
                if source_var not in self.assignment_graph:
                    self.assignment_graph[source_var] = []
                self.assignment_graph[source_var].append(var_name)
    
    def _process_assignment(self, node: Node) -> None:
        """Process assignment expression."""
        # Find left and right sides
        left = None
        right = None
        
        for i, child in enumerate(node.children):
            if child.type == 'identifier' and not left:
                left = child
            elif child.type == '=':
                # Next non-operator child is the right side
                if i + 1 < len(node.children):
                    right = node.children[i + 1]
                break
        
        if left and right and right.type == 'identifier':
            left_var = self._get_text(left)
            right_var = self._get_text(right)
            
            if right_var in self.variable_types:
                # Record propagation
                if right_var not in self.assignment_graph:
                    self.assignment_graph[right_var] = []
                self.assignment_graph[right_var].append(left_var)
    
    def _propagate_forward(self) -> None:
        """Propagate types forward through assignment graph."""
        for source_var, target_vars in self.assignment_graph.items():
            if source_var in self.variable_types:
                source_type = self.variable_types[source_var]
                for target_var in target_vars:
                    # Propagate type to target
                    self.propagated_types[target_var] = TypeInfo(
                        category=source_type.category,
                        name=source_type.name,
                        primitive_type=source_type.primitive_type,
                        element_type=source_type.element_type,
                        properties=source_type.properties,
                        parameters=source_type.parameters,
                        return_type=source_type.return_type,
                        union_types=source_type.union_types,
                        nullable=source_type.nullable,
                        confidence=source_type.confidence * 0.95,  # Slightly lower confidence
                        source="propagated",
                        metadata={'propagated_from': source_var}
                    )
    
    def narrow_type_in_condition(
        self,
        node: Node,
        condition_node: Node
    ) -> Dict[str, TypeInfo]:
        """
        Narrow types based on conditional expression.
        
        Args:
            node: Node containing the condition
            condition_node: The condition expression itself
            
        Returns:
            Dictionary of narrowed types for variables
        """
        narrowed = {}
        
        # Check for comparison with number
        if condition_node.type == 'binary_expression':
            narrowed.update(self._narrow_from_comparison(condition_node))
        
        # Check for typeof checks
        if condition_node.type == 'binary_expression':
            narrowed.update(self._narrow_from_typeof(condition_node))
        
        # Check for truthiness checks
        if condition_node.type == 'identifier':
            var_name = self._get_text(condition_node)
            # if (x) suggests x is truthy (not null/undefined/false)
            narrowed[var_name] = TypeInfo(
                category=TypeCategory.UNKNOWN,
                name="truthy",
                confidence=0.7,
                source="truthiness_check"
            )
        
        return narrowed
    
    def _narrow_from_comparison(self, node: Node) -> Dict[str, TypeInfo]:
        """Narrow types from comparison operations."""
        narrowed = {}
        
        # Find operator and operands
        operator = None
        left = None
        right = None
        
        for child in node.children:
            if child.type == 'identifier' and not left:
                left = child
            elif child.type in ('>', '<', '>=', '<=', '==', '===', '!=', '!=='):
                operator = self._get_text(child)
            elif child.type in ('number', 'identifier') and left and not right:
                right = child
        
        # If comparing with number, variable must be number
        if left and right and operator in ('>', '<', '>=', '<='):
            if right.type == 'number':
                var_name = self._get_text(left)
                narrowed[var_name] = TypeInfo(
                    category=TypeCategory.PRIMITIVE,
                    name="number",
                    primitive_type=PrimitiveType.NUMBER,
                    confidence=0.9,
                    source="comparison_narrowing"
                )
            elif left.type == 'number' and right.type == 'identifier':
                var_name = self._get_text(right)
                narrowed[var_name] = TypeInfo(
                    category=TypeCategory.PRIMITIVE,
                    name="number",
                    primitive_type=PrimitiveType.NUMBER,
                    confidence=0.9,
                    source="comparison_narrowing"
                )
        
        return narrowed
    
    def _narrow_from_typeof(self, node: Node) -> Dict[str, TypeInfo]:
        """Narrow types from typeof checks."""
        narrowed = {}
        
        # Look for pattern: typeof x === "string"
        left = None
        right = None
        operator = None
        
        for i, child in enumerate(node.children):
            if child.type == 'unary_expression':
                # Check if it's typeof
                typeof_op = self._find_child(child, 'typeof')
                if typeof_op or 'typeof' in self._get_text(child):
                    left = child
            elif child.type in ('==', '===', '!=', '!=='):
                operator = self._get_text(child)
            elif child.type == 'string' and not right:
                right = child
        
        if left and right and operator in ('==', '==='):
            # Extract variable from typeof expression
            for child in left.children:
                if child.type == 'identifier':
                    var_name = self._get_text(child)
                    type_str = self._get_text(right).strip('"').strip("'")
                    
                    # Map typeof string to TypeInfo
                    if type_str == 'string':
                        narrowed[var_name] = TypeInfo(
                            category=TypeCategory.PRIMITIVE,
                            name="string",
                            primitive_type=PrimitiveType.STRING,
                            confidence=0.95,
                            source="typeof_narrowing"
                        )
                    elif type_str == 'number':
                        narrowed[var_name] = TypeInfo(
                            category=TypeCategory.PRIMITIVE,
                            name="number",
                            primitive_type=PrimitiveType.NUMBER,
                            confidence=0.95,
                            source="typeof_narrowing"
                        )
                    elif type_str == 'boolean':
                        narrowed[var_name] = TypeInfo(
                            category=TypeCategory.PRIMITIVE,
                            name="boolean",
                            primitive_type=PrimitiveType.BOOLEAN,
                            confidence=0.95,
                            source="typeof_narrowing"
                        )
                    break
        
        return narrowed
    
    def get_type_at_location(
        self,
        var_name: str,
        node: Node
    ) -> Optional[TypeInfo]:
        """
        Get type of variable at specific location in code.
        
        Args:
            var_name: Variable name
            node: AST node location
            
        Returns:
            Type at that location, considering narrowing
        """
        # Check if in propagated types
        if var_name in self.propagated_types:
            return self.propagated_types[var_name]
        
        # Fall back to initial types
        return self.variable_types.get(var_name)
    
    def get_all_propagated_types(self) -> Dict[str, TypeInfo]:
        """Get all types after propagation."""
        result = self.variable_types.copy()
        result.update(self.propagated_types)
        return result
    
    def get_assignment_chains(self) -> Dict[str, List[str]]:
        """Get assignment graph for debugging."""
        return self.assignment_graph.copy()
    
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
