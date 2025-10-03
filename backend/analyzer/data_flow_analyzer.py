"""
Data Flow Analyzer

Analyzes code to build data flow graphs showing how data moves and transforms
through the codebase. Tracks variable assignments, function calls, returns,
object mutations, and other data transformations.

Classes:
    DataFlowAnalyzer: Main analyzer that builds data flow graphs
"""

from typing import List, Dict, Optional, Set, Any
from tree_sitter import Node
from analyzer.data_flow import (
    DataFlowGraph,
    DataFlowNode,
    DataFlowNodeType,
    DataFlowType
)
from analyzer.symbol_table import SymbolTable


class DataFlowAnalyzer:
    """
    Analyzes data flow through code.
    
    This analyzer traverses AST nodes and builds a comprehensive data flow graph
    showing how data moves between variables, functions, and operations.
    """
    
    def __init__(self, symbol_table: SymbolTable, nodes: List[Node], 
                 file_path: str = "unknown"):
        """
        Initialize data flow analyzer.
        
        Args:
            symbol_table: Symbol table with cross-file symbol information
            nodes: List of AST nodes to analyze
            file_path: Path to the file being analyzed
        """
        self.symbol_table = symbol_table
        self.nodes = nodes
        self.file_path = file_path
        self.flow_graph = DataFlowGraph()
        
        # Track variable definitions for flow analysis
        self._variable_definitions: Dict[str, str] = {}  # var_name -> node_id
        
        # Node ID counter for unique identifiers
        self._next_node_id = 0
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        node_id = f"node_{self._next_node_id}"
        self._next_node_id += 1
        return node_id
    
    def analyze(self) -> DataFlowGraph:
        """
        Build complete data flow graph.
        
        Returns:
            DataFlowGraph with all detected flows
        """
        # Analyze in order of data flow
        self._analyze_imports()
        self._analyze_parameters()
        self._analyze_declarations()
        self._analyze_assignments()
        self._analyze_function_calls()
        self._analyze_returns()
        self._analyze_object_flows()
        
        return self.flow_graph
    
    def _analyze_imports(self) -> None:
        """Track imports as data sources."""
        for node in self.nodes:
            if node.type == 'import_statement':
                self._process_import(node)
    
    def _process_import(self, node: Node) -> None:
        """Process an import statement."""
        # Handle: import { x, y } from 'module'
        import_clause = self._find_child(node, 'import_clause')
        if not import_clause:
            return
        
        named_imports = self._find_child(import_clause, 'named_imports')
        if named_imports:
            for child in named_imports.children:
                if child.type == 'import_specifier':
                    name_node = self._find_child(child, 'identifier')
                    if name_node:
                        var_name = self._get_text(name_node)
                        node_id = self._generate_node_id()
                        
                        df_node = DataFlowNode(
                            id=node_id,
                            variable_name=var_name,
                            source_location=(self.file_path, node.start_point[0]),
                            node_type=DataFlowNodeType.IMPORT
                        )
                        self.flow_graph.add_node(df_node)
                        self._variable_definitions[var_name] = node_id
    
    def _analyze_parameters(self) -> None:
        """Track function parameters."""
        for node in self.nodes:
            if node.type in ('function_declaration', 'arrow_function', 
                           'function_expression', 'method_definition'):
                self._process_function_parameters(node)
    
    def _process_function_parameters(self, node: Node) -> None:
        """Process function parameters."""
        params = self._find_child(node, 'formal_parameters')
        if not params:
            return
        
        for param in params.children:
            if param.type == 'identifier':
                var_name = self._get_text(param)
                node_id = self._generate_node_id()
                
                df_node = DataFlowNode(
                    id=node_id,
                    variable_name=var_name,
                    source_location=(self.file_path, param.start_point[0]),
                    node_type=DataFlowNodeType.PARAMETER,
                    scope=self._get_function_name(node)
                )
                self.flow_graph.add_node(df_node)
                self._variable_definitions[var_name] = node_id
            elif param.type == 'required_parameter':
                # TypeScript: param: Type
                pattern = self._find_child(param, 'identifier')
                if pattern:
                    var_name = self._get_text(pattern)
                    node_id = self._generate_node_id()
                    
                    df_node = DataFlowNode(
                        id=node_id,
                        variable_name=var_name,
                        source_location=(self.file_path, param.start_point[0]),
                        node_type=DataFlowNodeType.PARAMETER,
                        scope=self._get_function_name(node)
                    )
                    self.flow_graph.add_node(df_node)
                    self._variable_definitions[var_name] = node_id
    
    def _analyze_declarations(self) -> None:
        """Track variable declarations."""
        for node in self.nodes:
            if node.type == 'variable_declaration':
                self._process_variable_declaration(node)
    
    def _process_variable_declaration(self, node: Node) -> None:
        """Process variable declaration (const, let, var)."""
        for child in node.children:
            if child.type == 'variable_declarator':
                name_node = child.child_by_field_name('name')
                value_node = child.child_by_field_name('value')
                
                if name_node:
                    var_name = self._get_text(name_node)
                    node_id = self._generate_node_id()
                    
                    df_node = DataFlowNode(
                        id=node_id,
                        variable_name=var_name,
                        source_location=(self.file_path, name_node.start_point[0]),
                        node_type=DataFlowNodeType.DECLARATION
                    )
                    self.flow_graph.add_node(df_node)
                    self._variable_definitions[var_name] = node_id
                    
                    # If there's an initializer, create flow edge
                    if value_node:
                        self._process_value_flow(value_node, node_id)
    
    def _analyze_assignments(self) -> None:
        """Track variable assignments and updates."""
        for node in self.nodes:
            if node.type == 'assignment_expression':
                self._process_assignment(node)
    
    def _process_assignment(self, node: Node) -> None:
        """Process assignment expression."""
        left = node.child_by_field_name('left')
        right = node.child_by_field_name('right')
        
        if not (left and right):
            return
        
        # Create node for assignment target
        var_name = self._get_text(left)
        node_id = self._generate_node_id()
        
        df_node = DataFlowNode(
            id=node_id,
            variable_name=var_name,
            source_location=(self.file_path, left.start_point[0]),
            node_type=DataFlowNodeType.ASSIGNMENT
        )
        self.flow_graph.add_node(df_node)
        self._variable_definitions[var_name] = node_id
        
        # Track flow from right side
        self._process_value_flow(right, node_id)
    
    def _analyze_function_calls(self) -> None:
        """Track data flow through function calls."""
        for node in self.nodes:
            if node.type == 'call_expression':
                self._process_function_call(node)
    
    def _process_function_call(self, node: Node) -> None:
        """Process function call and argument flows."""
        function = node.child_by_field_name('function')
        arguments = node.child_by_field_name('arguments')
        
        if not (function and arguments):
            return
        
        func_name = self._get_text(function)
        
        # Process each argument
        for arg in arguments.children:
            if arg.type in ('identifier', 'member_expression'):
                arg_name = self._get_text(arg)
                
                # If argument is a known variable, create argument node
                if arg_name in self._variable_definitions:
                    arg_node_id = self._generate_node_id()
                    
                    df_node = DataFlowNode(
                        id=arg_node_id,
                        variable_name=f"{func_name}:arg",
                        source_location=(self.file_path, arg.start_point[0]),
                        node_type=DataFlowNodeType.ARGUMENT,
                        metadata={'function': func_name, 'argument': arg_name}
                    )
                    self.flow_graph.add_node(df_node)
                    
                    # Create flow from variable to argument
                    source_id = self._variable_definitions[arg_name]
                    source_node = self.flow_graph.get_node(source_id)
                    if source_node:
                        self.flow_graph.add_flow(
                            source_node,
                            df_node,
                            DataFlowType.PARAMETER,
                            operation=func_name
                        )
    
    def _analyze_returns(self) -> None:
        """Track return value flows."""
        for node in self.nodes:
            if node.type == 'return_statement':
                self._process_return(node)
    
    def _process_return(self, node: Node) -> None:
        """Process return statement."""
        # Get the returned expression
        for child in node.children:
            if child.type != 'return':
                return_node_id = self._generate_node_id()
                
                df_node = DataFlowNode(
                    id=return_node_id,
                    variable_name="<return>",
                    source_location=(self.file_path, node.start_point[0]),
                    node_type=DataFlowNodeType.RETURN
                )
                self.flow_graph.add_node(df_node)
                
                # Track flow from returned value
                self._process_value_flow(child, return_node_id)
                break
    
    def _analyze_object_flows(self) -> None:
        """Track data flow through object properties."""
        for node in self.nodes:
            if node.type == 'member_expression':
                self._process_member_expression(node)
    
    def _process_member_expression(self, node: Node) -> None:
        """Process object property access."""
        obj = node.child_by_field_name('object')
        prop = node.child_by_field_name('property')
        
        if not (obj and prop):
            return
        
        obj_name = self._get_text(obj)
        prop_name = self._get_text(prop)
        
        # If object is a known variable, create property access node
        if obj_name in self._variable_definitions:
            prop_node_id = self._generate_node_id()
            
            df_node = DataFlowNode(
                id=prop_node_id,
                variable_name=f"{obj_name}.{prop_name}",
                source_location=(self.file_path, node.start_point[0]),
                node_type=DataFlowNodeType.PROPERTY,
                metadata={'object': obj_name, 'property': prop_name}
            )
            self.flow_graph.add_node(df_node)
            
            # Create flow from object to property
            source_id = self._variable_definitions[obj_name]
            source_node = self.flow_graph.get_node(source_id)
            if source_node:
                self.flow_graph.add_flow(
                    source_node,
                    df_node,
                    DataFlowType.PROPERTY_ACCESS
                )
    
    def _process_value_flow(self, value_node: Node, target_id: str) -> None:
        """
        Process data flow from a value expression to a target node.
        
        Args:
            value_node: AST node representing the value
            target_id: ID of the target data flow node
        """
        target = self.flow_graph.get_node(target_id)
        if not target:
            return
        
        # Handle different value types
        if value_node.type == 'identifier':
            # Direct assignment from variable
            var_name = self._get_text(value_node)
            if var_name in self._variable_definitions:
                source_id = self._variable_definitions[var_name]
                source_node = self.flow_graph.get_node(source_id)
                if source_node:
                    self.flow_graph.add_flow(
                        source_node,
                        target,
                        DataFlowType.DIRECT
                    )
        
        elif value_node.type == 'call_expression':
            # Assignment from function call result
            func = value_node.child_by_field_name('function')
            if func:
                func_name = self._get_text(func)
                # Create temporary node for call result
                call_result_id = self._generate_node_id()
                call_result = DataFlowNode(
                    id=call_result_id,
                    variable_name=f"{func_name}:result",
                    source_location=(self.file_path, value_node.start_point[0]),
                    node_type=DataFlowNodeType.OPERATION,
                    metadata={'operation': 'call', 'function': func_name}
                )
                self.flow_graph.add_node(call_result)
                self.flow_graph.add_flow(
                    call_result,
                    target,
                    DataFlowType.TRANSFORM,
                    operation=func_name
                )
        
        elif value_node.type in ('number', 'string', 'true', 'false', 'null'):
            # Literal value
            literal_id = self._generate_node_id()
            literal_node = DataFlowNode(
                id=literal_id,
                variable_name=self._get_text(value_node),
                source_location=(self.file_path, value_node.start_point[0]),
                node_type=DataFlowNodeType.LITERAL
            )
            self.flow_graph.add_node(literal_node)
            self.flow_graph.add_flow(
                literal_node,
                target,
                DataFlowType.DIRECT
            )
        
        elif value_node.type == 'member_expression':
            # Property access
            obj = value_node.child_by_field_name('object')
            if obj:
                obj_name = self._get_text(obj)
                if obj_name in self._variable_definitions:
                    source_id = self._variable_definitions[obj_name]
                    source_node = self.flow_graph.get_node(source_id)
                    if source_node:
                        self.flow_graph.add_flow(
                            source_node,
                            target,
                            DataFlowType.PROPERTY_ACCESS
                        )
        
        elif value_node.type == 'binary_expression':
            # Binary operation (e.g., a + b)
            self._process_binary_expression(value_node, target)
        
        elif value_node.type == 'await_expression':
            # Async await
            arg = value_node.child_by_field_name('argument') or value_node.children[-1]
            if arg:
                self._process_value_flow(arg, target_id)
    
    def _process_binary_expression(self, node: Node, target: DataFlowNode) -> None:
        """Process binary expression and create flow edges."""
        left = node.child_by_field_name('left')
        right = node.child_by_field_name('right')
        operator = node.child_by_field_name('operator')
        
        op_text = self._get_text(operator) if operator else '?'
        
        # Create flows from both operands to target
        for operand in (left, right):
            if operand and operand.type == 'identifier':
                var_name = self._get_text(operand)
                if var_name in self._variable_definitions:
                    source_id = self._variable_definitions[var_name]
                    source_node = self.flow_graph.get_node(source_id)
                    if source_node:
                        self.flow_graph.add_flow(
                            source_node,
                            target,
                            DataFlowType.TRANSFORM,
                            operation=op_text
                        )
    
    # Helper methods
    
    def _find_child(self, node: Node, child_type: str) -> Optional[Node]:
        """Find first child of given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def _get_text(self, node: Node) -> str:
        """Get text content of node."""
        if hasattr(node, 'text'):
            return node.text.decode('utf8') if isinstance(node.text, bytes) else node.text
        return ""
    
    def _get_function_name(self, node: Node) -> Optional[str]:
        """Extract function name from function node."""
        if node.type == 'function_declaration':
            name = node.child_by_field_name('name')
            return self._get_text(name) if name else None
        elif node.type == 'method_definition':
            name = node.child_by_field_name('name')
            return self._get_text(name) if name else None
        return None
