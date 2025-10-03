"""
Data Flow Tracker for M4.0 Phase 2

This module tracks how data flows through a JavaScript/TypeScript codebase:
- Variable assignments and reassignments
- Function parameter passing
- Return value propagation
- React state flow (useState → state variable → setState calls)
- Property access and mutations
- Array/object destructuring

The tracker creates data flow edges that show how values move through the code,
enabling visualization of data dependencies and state management patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of data flow relationships"""
    ASSIGNMENT = "assignment"              # x = y
    PARAMETER = "parameter"                # function(x) - x receives arg
    RETURN = "return"                      # return x - x flows to caller
    STATE_INIT = "state_init"              # useState(x) - initial state
    STATE_UPDATE = "state_update"          # setState(x) - state update
    STATE_READ = "state_read"              # using state variable
    DESTRUCTURE = "destructure"            # const {x} = obj or const [x] = arr
    PROPERTY_ACCESS = "property_access"    # obj.x
    ARRAY_ACCESS = "array_access"          # arr[i]
    CALL_ARGUMENT = "call_argument"        # func(x) - x flows to function
    MUTATION = "mutation"                  # obj.x = y or arr[i] = y
    SPREAD = "spread"                      # {...obj} or [...arr]
    CALLBACK = "callback"                  # setTimeout(() => x)


class FlowDirection(Enum):
    """Direction of data flow"""
    FORWARD = "forward"    # Data flows from source to target
    BACKWARD = "backward"  # Data flows from target to source (rare)
    BIDIRECTIONAL = "bidirectional"  # Data flows both ways


@dataclass
class DataFlowEdge:
    """Represents a single data flow relationship"""
    source_id: str                    # Entity ID where data originates
    target_id: str                    # Entity ID where data flows to
    flow_type: FlowType               # Type of flow relationship
    direction: FlowDirection          # Direction of flow
    
    # Flow metadata
    confidence: float = 1.0           # Confidence in this flow (0.0-1.0)
    line_number: Optional[int] = None # Line where flow occurs
    column: Optional[int] = None      # Column where flow occurs
    
    # Context information
    variable_name: Optional[str] = None     # Variable involved in flow
    property_name: Optional[str] = None     # Property name for property access
    is_conditional: bool = False            # Flow happens conditionally
    loop_context: Optional[str] = None      # Flow inside loop (for/while/map/etc)
    
    # React-specific
    is_react_state: bool = False            # Flow involves React state
    hook_name: Optional[str] = None         # Hook name (useState, useEffect, etc)
    state_variable: Optional[str] = None    # Name of state variable
    setter_function: Optional[str] = None   # Name of setter function
    
    # Additional context
    context: Optional[str] = None           # Human-readable description
    ast_node_type: Optional[str] = None     # AST node type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'flow_type': self.flow_type.value,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'line_number': self.line_number,
            'column': self.column,
            'variable_name': self.variable_name,
            'property_name': self.property_name,
            'is_conditional': self.is_conditional,
            'loop_context': self.loop_context,
            'is_react_state': self.is_react_state,
            'hook_name': self.hook_name,
            'state_variable': self.state_variable,
            'setter_function': self.setter_function,
            'context': self.context,
            'ast_node_type': self.ast_node_type
        }


@dataclass
class FlowNode:
    """Represents a node in the data flow graph"""
    node_id: str                           # Unique identifier
    node_type: str                         # Type of node (variable, function, etc)
    name: str                              # Name of the entity
    
    # Connections
    incoming_flows: List[DataFlowEdge] = field(default_factory=list)
    outgoing_flows: List[DataFlowEdge] = field(default_factory=list)
    
    # Metadata
    scope: Optional[str] = None            # Scope where node exists
    defined_at: Optional[int] = None       # Line where defined
    data_type: Optional[str] = None        # Inferred data type
    
    def add_incoming(self, edge: DataFlowEdge):
        """Add incoming flow edge"""
        self.incoming_flows.append(edge)
    
    def add_outgoing(self, edge: DataFlowEdge):
        """Add outgoing flow edge"""
        self.outgoing_flows.append(edge)
    
    def get_flow_summary(self) -> Dict[str, int]:
        """Get summary of flows"""
        return {
            'incoming_count': len(self.incoming_flows),
            'outgoing_count': len(self.outgoing_flows),
            'total_flows': len(self.incoming_flows) + len(self.outgoing_flows)
        }


class DataFlowTracker:
    """
    Main class for tracking data flow through JavaScript/TypeScript code.
    
    This tracker analyzes AST nodes and creates a data flow graph showing how
    values move through variables, functions, and React state.
    """
    
    def __init__(self):
        """Initialize the data flow tracker"""
        self.flows: List[DataFlowEdge] = []
        self.nodes: Dict[str, FlowNode] = {}
        
        # Tracking state
        self.variable_definitions: Dict[str, str] = {}  # var_name -> node_id
        self.function_parameters: Dict[str, List[str]] = {}  # func_id -> [param_ids]
        self.react_state_map: Dict[str, Dict[str, str]] = {}  # state_var -> {setter, hook_call}
        
        # Scope tracking
        self.current_scope: List[str] = []
        self.scope_variables: Dict[str, Set[str]] = {}  # scope -> {var_names}
        
        # Statistics
        self.stats = {
            'total_flows': 0,
            'assignment_flows': 0,
            'state_flows': 0,
            'function_flows': 0,
            'property_flows': 0
        }
    
    def enter_scope(self, scope_name: str):
        """Enter a new scope (function, block, etc)"""
        self.current_scope.append(scope_name)
        scope_key = '::'.join(self.current_scope)
        if scope_key not in self.scope_variables:
            self.scope_variables[scope_key] = set()
        logger.debug(f"Entered scope: {scope_key}")
    
    def exit_scope(self):
        """Exit current scope"""
        if self.current_scope:
            exited = self.current_scope.pop()
            logger.debug(f"Exited scope: {exited}")
    
    def get_current_scope(self) -> str:
        """Get current scope identifier"""
        return '::'.join(self.current_scope) if self.current_scope else 'global'
    
    def register_node(self, node_id: str, node_type: str, name: str, 
                     defined_at: Optional[int] = None, 
                     data_type: Optional[str] = None) -> FlowNode:
        """Register a node in the flow graph"""
        if node_id not in self.nodes:
            node = FlowNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                scope=self.get_current_scope(),
                defined_at=defined_at,
                data_type=data_type
            )
            self.nodes[node_id] = node
            logger.debug(f"Registered node: {node_id} ({node_type}: {name})")
        return self.nodes[node_id]
    
    def create_node(self, node_id: str, node_type: str, name: str, 
                   scope: Optional[str] = None, 
                   defined_at: Optional[int] = None, 
                   data_type: Optional[str] = None) -> FlowNode:
        """Create a node in the flow graph (alias for register_node with scope support)"""
        if node_id not in self.nodes:
            node = FlowNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                scope=scope or self.get_current_scope(),
                defined_at=defined_at,
                data_type=data_type
            )
            self.nodes[node_id] = node
            logger.debug(f"Created node: {node_id} ({node_type}: {name})")
        return self.nodes[node_id]
    
    def track_assignment(self, 
                        target_id: str,
                        source_id: str,
                        variable_name: str,
                        line_number: Optional[int] = None,
                        is_const: bool = False,
                        is_destructure: bool = False) -> DataFlowEdge:
        """
        Track variable assignment: target = source
        
        Examples:
        - const x = 5
        - let count = 0
        - const [a, b] = arr
        """
        flow_type = FlowType.DESTRUCTURE if is_destructure else FlowType.ASSIGNMENT
        confidence = 1.0 if is_const else 0.9  # const is more certain
        
        edge = DataFlowEdge(
            source_id=source_id,
            target_id=target_id,
            flow_type=flow_type,
            direction=FlowDirection.FORWARD,
            confidence=confidence,
            line_number=line_number,
            variable_name=variable_name,
            context=f"{'const' if is_const else 'let/var'} {variable_name} = ..."
        )
        
        self._add_flow(edge)
        self.variable_definitions[variable_name] = target_id
        self.stats['assignment_flows'] += 1
        
        return edge
    
    def track_react_state_init(self,
                               hook_call_id: str,
                               state_var_id: str,
                               setter_id: str,
                               state_name: str,
                               setter_name: str,
                               initial_value_id: Optional[str] = None,
                               line_number: Optional[int] = None) -> List[DataFlowEdge]:
        """
        Track React useState initialization
        
        Example: const [count, setCount] = useState(0)
        Creates flows:
        1. useState(0) -> count (state initialization)
        2. Hook call -> setCount (setter creation)
        """
        edges = []
        
        # Flow 1: useState(initial) -> state variable
        if initial_value_id:
            state_init_edge = DataFlowEdge(
                source_id=initial_value_id,
                target_id=state_var_id,
                flow_type=FlowType.STATE_INIT,
                direction=FlowDirection.FORWARD,
                confidence=0.95,
                line_number=line_number,
                variable_name=state_name,
                is_react_state=True,
                hook_name='useState',
                state_variable=state_name,
                setter_function=setter_name,
                context=f"const [{state_name}, {setter_name}] = useState(...)"
            )
            edges.append(state_init_edge)
            self._add_flow(state_init_edge)
        
        # Flow 2: Hook call -> setter function
        setter_edge = DataFlowEdge(
            source_id=hook_call_id,
            target_id=setter_id,
            flow_type=FlowType.STATE_INIT,
            direction=FlowDirection.FORWARD,
            confidence=1.0,
            line_number=line_number,
            variable_name=setter_name,
            is_react_state=True,
            hook_name='useState',
            state_variable=state_name,
            setter_function=setter_name,
            context=f"Setter function {setter_name} for state {state_name}"
        )
        edges.append(setter_edge)
        self._add_flow(setter_edge)
        
        # Register state mapping
        self.react_state_map[state_name] = {
            'setter': setter_name,
            'hook_call': hook_call_id,
            'state_id': state_var_id,
            'setter_id': setter_id
        }
        
        self.stats['state_flows'] += 2
        logger.debug(f"Tracked React state: {state_name}, {setter_name}")
        
        return edges
    
    def track_state_update(self,
                          setter_call_id: str,
                          state_var_id: str,
                          new_value_id: Optional[str],
                          state_name: str,
                          setter_name: str,
                          line_number: Optional[int] = None,
                          is_functional_update: bool = False) -> DataFlowEdge:
        """
        Track React state update: setCount(newValue) or setCount(prev => prev + 1)
        
        Examples:
        - setCount(5) - direct update
        - setCount(count + 1) - using current state
        - setCount(prev => prev + 1) - functional update
        """
        confidence = 0.95 if is_functional_update else 0.9
        
        # Create flow from new value to state variable
        source = new_value_id if new_value_id else setter_call_id
        
        edge = DataFlowEdge(
            source_id=source,
            target_id=state_var_id,
            flow_type=FlowType.STATE_UPDATE,
            direction=FlowDirection.FORWARD,
            confidence=confidence,
            line_number=line_number,
            variable_name=state_name,
            is_react_state=True,
            hook_name='useState',
            state_variable=state_name,
            setter_function=setter_name,
            context=f"{setter_name}({'functional' if is_functional_update else 'direct'} update)"
        )
        
        self._add_flow(edge)
        self.stats['state_flows'] += 1
        
        return edge
    
    def track_state_read(self,
                        state_var_id: str,
                        usage_id: str,
                        state_name: str,
                        line_number: Optional[int] = None,
                        usage_context: Optional[str] = None) -> DataFlowEdge:
        """
        Track reading React state variable
        
        Example: <div>{count}</div> - reading count state
        """
        edge = DataFlowEdge(
            source_id=state_var_id,
            target_id=usage_id,
            flow_type=FlowType.STATE_READ,
            direction=FlowDirection.FORWARD,
            confidence=1.0,
            line_number=line_number,
            variable_name=state_name,
            is_react_state=True,
            state_variable=state_name,
            context=usage_context or f"Reading state variable {state_name}"
        )
        
        self._add_flow(edge)
        self.stats['state_flows'] += 1
        
        return edge
    
    def track_function_call(self,
                           function_id: str,
                           call_site_id: str,
                           arguments: List[Tuple[str, str]],  # [(arg_id, param_name)]
                           return_value_id: Optional[str] = None,
                           line_number: Optional[int] = None) -> List[DataFlowEdge]:
        """
        Track function call data flow
        
        Creates flows:
        1. Each argument -> corresponding parameter
        2. Function return -> call site (if return value used)
        """
        edges = []
        
        # Flow 1: Arguments -> Parameters
        for i, (arg_id, param_name) in enumerate(arguments):
            param_edge = DataFlowEdge(
                source_id=arg_id,
                target_id=function_id,
                flow_type=FlowType.CALL_ARGUMENT,
                direction=FlowDirection.FORWARD,
                confidence=0.9,
                line_number=line_number,
                variable_name=param_name,
                context=f"Argument {i} ({param_name}) passed to function"
            )
            edges.append(param_edge)
            self._add_flow(param_edge)
        
        # Flow 2: Return value -> Call site
        if return_value_id:
            return_edge = DataFlowEdge(
                source_id=function_id,
                target_id=return_value_id,
                flow_type=FlowType.RETURN,
                direction=FlowDirection.FORWARD,
                confidence=0.85,
                line_number=line_number,
                context="Function return value"
            )
            edges.append(return_edge)
            self._add_flow(return_edge)
        
        self.stats['function_flows'] += len(edges)
        
        return edges
    
    def track_property_access(self,
                             object_id: str,
                             property_name: str,
                             access_id: str,
                             line_number: Optional[int] = None,
                             is_mutation: bool = False) -> DataFlowEdge:
        """
        Track property access: obj.property
        
        Examples:
        - const x = obj.prop (read)
        - obj.prop = 5 (mutation)
        """
        flow_type = FlowType.MUTATION if is_mutation else FlowType.PROPERTY_ACCESS
        direction = FlowDirection.FORWARD if not is_mutation else FlowDirection.BIDIRECTIONAL
        
        edge = DataFlowEdge(
            source_id=object_id,
            target_id=access_id,
            flow_type=flow_type,
            direction=direction,
            confidence=0.9,
            line_number=line_number,
            property_name=property_name,
            context=f"Property access: {property_name} {'(mutation)' if is_mutation else '(read)'}"
        )
        
        self._add_flow(edge)
        self.stats['property_flows'] += 1
        
        return edge
    
    def track_array_access(self,
                          array_id: str,
                          index_id: str,
                          access_id: str,
                          line_number: Optional[int] = None,
                          is_mutation: bool = False) -> DataFlowEdge:
        """
        Track array access: arr[index]
        
        Examples:
        - const x = arr[0] (read)
        - arr[0] = 5 (mutation)
        """
        flow_type = FlowType.MUTATION if is_mutation else FlowType.ARRAY_ACCESS
        direction = FlowDirection.FORWARD if not is_mutation else FlowDirection.BIDIRECTIONAL
        
        edge = DataFlowEdge(
            source_id=array_id,
            target_id=access_id,
            flow_type=flow_type,
            direction=direction,
            confidence=0.85,
            line_number=line_number,
            context=f"Array access {'(mutation)' if is_mutation else '(read)'}"
        )
        
        self._add_flow(edge)
        self.stats['property_flows'] += 1
        
        return edge
    
    def track_destructuring(self,
                           source_id: str,
                           targets: List[Tuple[str, str]],  # [(target_id, property_name)]
                           line_number: Optional[int] = None,
                           is_array: bool = False) -> List[DataFlowEdge]:
        """
        Track destructuring assignment
        
        Examples:
        - const {x, y} = obj (object destructuring)
        - const [a, b] = arr (array destructuring)
        - const [count, setCount] = useState(0) (React hook destructuring)
        """
        edges = []
        
        for target_id, prop_name in targets:
            edge = DataFlowEdge(
                source_id=source_id,
                target_id=target_id,
                flow_type=FlowType.DESTRUCTURE,
                direction=FlowDirection.FORWARD,
                confidence=0.95,
                line_number=line_number,
                variable_name=prop_name,
                property_name=None if is_array else prop_name,
                context=f"{'Array' if is_array else 'Object'} destructuring: {prop_name}"
            )
            edges.append(edge)
            self._add_flow(edge)
        
        self.stats['assignment_flows'] += len(edges)
        
        return edges
    
    def get_flows_for_node(self, node_id: str) -> Dict[str, List[DataFlowEdge]]:
        """Get all flows related to a specific node"""
        if node_id not in self.nodes:
            return {'incoming': [], 'outgoing': []}
        
        node = self.nodes[node_id]
        return {
            'incoming': node.incoming_flows,
            'outgoing': node.outgoing_flows
        }
    
    def get_react_state_flows(self, state_name: str) -> List[DataFlowEdge]:
        """Get all flows related to a specific React state variable"""
        flows = []
        for edge in self.flows:
            if edge.is_react_state and edge.state_variable == state_name:
                flows.append(edge)
        return flows
    
    def get_variable_flow_chain(self, variable_name: str) -> List[DataFlowEdge]:
        """Get the complete flow chain for a variable"""
        if variable_name not in self.variable_definitions:
            return []
        
        node_id = self.variable_definitions[variable_name]
        visited = set()
        chain = []
        
        def traverse(nid: str):
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            
            node = self.nodes[nid]
            chain.extend(node.incoming_flows)
            chain.extend(node.outgoing_flows)
            
            # Continue traversing
            for edge in node.outgoing_flows:
                traverse(edge.target_id)
        
        traverse(node_id)
        return chain
    
    def _add_flow(self, edge: DataFlowEdge):
        """Internal method to add a flow edge"""
        self.flows.append(edge)
        self.stats['total_flows'] += 1
        
        # Update nodes
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].add_outgoing(edge)
        
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].add_incoming(edge)
        
        logger.debug(f"Added flow: {edge.flow_type.value} from {edge.source_id} to {edge.target_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        flow_type_counts = {}
        for flow in self.flows:
            ft = flow.flow_type.value
            flow_type_counts[ft] = flow_type_counts.get(ft, 0) + 1
        
        return {
            **self.stats,
            'total_nodes': len(self.nodes),
            'total_scopes': len(self.scope_variables),
            'react_states': len(self.react_state_map),
            'flow_type_breakdown': flow_type_counts,
            'average_flows_per_node': len(self.flows) / len(self.nodes) if self.nodes else 0
        }
    
    def export_flows(self) -> List[Dict[str, Any]]:
        """Export all flows as dictionaries"""
        return [flow.to_dict() for flow in self.flows]
    
    def clear(self):
        """Clear all tracked data"""
        self.flows.clear()
        self.nodes.clear()
        self.variable_definitions.clear()
        self.function_parameters.clear()
        self.react_state_map.clear()
        self.current_scope.clear()
        self.scope_variables.clear()
        self.stats = {
            'total_flows': 0,
            'assignment_flows': 0,
            'state_flows': 0,
            'function_flows': 0,
            'property_flows': 0
        }
        logger.info("Data flow tracker cleared")
