"""
Call Graph Builder for M3.2.1

Builds a call graph tracking function and method invocations:
- Function calls (direct, indirect)
- Method invocations (object.method, class.staticMethod)
- Constructor calls (new ClassName)
- Higher-order functions (callbacks, promises, async/await)

The call graph shows:
- Who calls who (caller -> callee relationships)
- Call sites (where calls happen)
- Call context (parameters, return values)
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CallType(Enum):
    """Types of function calls"""
    FUNCTION_CALL = "function_call"           # Regular function call
    METHOD_CALL = "method_call"               # obj.method()
    STATIC_CALL = "static_call"               # Class.staticMethod()
    CONSTRUCTOR_CALL = "constructor_call"     # new ClassName()
    CALLBACK = "callback"                     # Callback function
    ASYNC_CALL = "async_call"                 # await func()
    PROMISE_THEN = "promise_then"             # promise.then()
    PROMISE_CATCH = "promise_catch"           # promise.catch()
    EVENT_HANDLER = "event_handler"           # addEventListener, onClick
    HOF_CALL = "hof_call"                     # Higher-order function


@dataclass
class CallSite:
    """Represents a location where a call happens"""
    call_id: str                              # Unique call site ID
    caller_id: str                            # ID of the calling function
    callee_id: str                            # ID of the called function
    call_type: CallType                       # Type of call
    
    # Location
    file_path: Optional[str] = None           # File containing the call
    line_number: Optional[int] = None         # Line number
    
    # Context
    arguments: List[str] = field(default_factory=list)  # Argument IDs
    return_value_id: Optional[str] = None     # Return value ID
    
    # Metadata
    is_async: bool = False                    # Is it an async call?
    is_conditional: bool = False              # Is it inside conditional?
    confidence: float = 1.0                   # Confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'call_id': self.call_id,
            'caller_id': self.caller_id,
            'callee_id': self.callee_id,
            'call_type': self.call_type.value,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'arguments': self.arguments,
            'return_value_id': self.return_value_id,
            'is_async': self.is_async,
            'is_conditional': self.is_conditional,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class CallNode:
    """Represents a callable entity in the call graph"""
    node_id: str                              # Unique node ID
    name: str                                 # Function/method name
    node_type: str                            # Type: function, method, constructor
    
    # Relationships
    callers: Set[str] = field(default_factory=set)      # IDs of functions that call this
    callees: Set[str] = field(default_factory=set)      # IDs of functions this calls
    call_sites: List[CallSite] = field(default_factory=list)  # Where this is called
    
    # Metadata
    file_path: Optional[str] = None           # File where defined
    line_number: Optional[int] = None         # Line where defined
    is_async: bool = False                    # Is async function?
    is_generator: bool = False                # Is generator?
    parameters: List[str] = field(default_factory=list)  # Parameter names
    return_type: Optional[str] = None         # Inferred return type
    
    # Statistics
    call_count: int = 0                       # How many times called
    caller_count: int = 0                     # How many callers
    callee_count: int = 0                     # How many callees
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_caller(self, caller_id: str):
        """Add a caller"""
        if caller_id not in self.callers:
            self.callers.add(caller_id)
            self.caller_count += 1
    
    def add_callee(self, callee_id: str):
        """Add a callee"""
        if callee_id not in self.callees:
            self.callees.add(callee_id)
            self.callee_count += 1
    
    def add_call_site(self, call_site: CallSite):
        """Add a call site"""
        self.call_sites.append(call_site)
        self.call_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'node_id': self.node_id,
            'name': self.name,
            'node_type': self.node_type,
            'callers': list(self.callers),
            'callees': list(self.callees),
            'call_sites_count': len(self.call_sites),
            'file_path': self.file_path,
            'line_number': self.line_number,
            'is_async': self.is_async,
            'is_generator': self.is_generator,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'call_count': self.call_count,
            'caller_count': self.caller_count,
            'callee_count': self.callee_count,
            'metadata': self.metadata
        }


class CallGraph:
    """
    Main class for building and managing the call graph.
    
    The call graph tracks:
    1. Function definitions and declarations
    2. Function calls and invocations
    3. Caller-callee relationships
    4. Call sites and context
    """
    
    def __init__(self):
        """Initialize the call graph"""
        self.nodes: Dict[str, CallNode] = {}
        self.call_sites: List[CallSite] = []
        
        # Index for quick lookups
        self.name_to_nodes: Dict[str, List[str]] = {}  # name -> [node_ids]
        self.file_to_nodes: Dict[str, List[str]] = {}  # file -> [node_ids]
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_calls': 0,
            'function_calls': 0,
            'method_calls': 0,
            'constructor_calls': 0,
            'async_calls': 0,
            'callback_calls': 0
        }
        
        logger.info("CallGraph initialized")
    
    def add_function(
        self,
        node_id: str,
        name: str,
        node_type: str = 'function',
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        is_async: bool = False,
        is_generator: bool = False,
        parameters: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CallNode:
        """
        Add a function/method to the call graph.
        
        Args:
            node_id: Unique identifier for the node
            name: Function/method name
            node_type: Type of callable (function, method, constructor)
            file_path: File where defined
            line_number: Line number where defined
            is_async: Whether it's an async function
            is_generator: Whether it's a generator
            parameters: List of parameter names
            metadata: Additional metadata
            
        Returns:
            The created CallNode
        """
        if node_id in self.nodes:
            logger.warning(f"Function {node_id} already exists in call graph")
            return self.nodes[node_id]
        
        node = CallNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            file_path=file_path,
            line_number=line_number,
            is_async=is_async,
            is_generator=is_generator,
            parameters=parameters or [],
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self.stats['total_nodes'] += 1
        
        # Index by name
        if name not in self.name_to_nodes:
            self.name_to_nodes[name] = []
        self.name_to_nodes[name].append(node_id)
        
        # Index by file
        if file_path:
            if file_path not in self.file_to_nodes:
                self.file_to_nodes[file_path] = []
            self.file_to_nodes[file_path].append(node_id)
        
        logger.debug(f"Added function {name} ({node_id}) to call graph")
        return node
    
    def add_call(
        self,
        call_id: str,
        caller_id: str,
        callee_id: str,
        call_type: CallType = CallType.FUNCTION_CALL,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        arguments: Optional[List[str]] = None,
        return_value_id: Optional[str] = None,
        is_async: bool = False,
        is_conditional: bool = False,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CallSite:
        """
        Add a call to the call graph.
        
        Args:
            call_id: Unique identifier for the call
            caller_id: ID of the calling function
            callee_id: ID of the called function
            call_type: Type of call
            file_path: File containing the call
            line_number: Line number of the call
            arguments: Argument IDs
            return_value_id: Return value ID
            is_async: Is it an async call?
            is_conditional: Is it inside conditional?
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            The created CallSite
        """
        call_site = CallSite(
            call_id=call_id,
            caller_id=caller_id,
            callee_id=callee_id,
            call_type=call_type,
            file_path=file_path,
            line_number=line_number,
            arguments=arguments or [],
            return_value_id=return_value_id,
            is_async=is_async,
            is_conditional=is_conditional,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.call_sites.append(call_site)
        self.stats['total_calls'] += 1
        
        # Update call type statistics
        if call_type == CallType.FUNCTION_CALL:
            self.stats['function_calls'] += 1
        elif call_type == CallType.METHOD_CALL:
            self.stats['method_calls'] += 1
        elif call_type == CallType.CONSTRUCTOR_CALL:
            self.stats['constructor_calls'] += 1
        elif call_type == CallType.CALLBACK:
            self.stats['callback_calls'] += 1
        
        if is_async:
            self.stats['async_calls'] += 1
        
        # Update caller and callee relationships
        if caller_id in self.nodes:
            self.nodes[caller_id].add_callee(callee_id)
        
        if callee_id in self.nodes:
            self.nodes[callee_id].add_caller(caller_id)
            self.nodes[callee_id].add_call_site(call_site)
        
        logger.debug(f"Added call {caller_id} -> {callee_id} ({call_type.value})")
        return call_site
    
    def get_callers(self, node_id: str) -> List[CallNode]:
        """Get all functions that call this function"""
        if node_id not in self.nodes:
            return []
        
        caller_ids = self.nodes[node_id].callers
        return [self.nodes[cid] for cid in caller_ids if cid in self.nodes]
    
    def get_callees(self, node_id: str) -> List[CallNode]:
        """Get all functions that this function calls"""
        if node_id not in self.nodes:
            return []
        
        callee_ids = self.nodes[node_id].callees
        return [self.nodes[cid] for cid in callee_ids if cid in self.nodes]
    
    def get_call_chain(self, start_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        Get call chains starting from a function.
        
        Args:
            start_id: Starting function ID
            max_depth: Maximum chain depth
            
        Returns:
            List of call chains (each chain is a list of node IDs)
        """
        if start_id not in self.nodes:
            return []
        
        chains = []
        visited = set()
        
        def dfs(node_id: str, chain: List[str], depth: int):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            chain.append(node_id)
            
            if node_id in self.nodes:
                callees = self.nodes[node_id].callees
                if not callees:
                    chains.append(chain.copy())
                else:
                    for callee_id in callees:
                        dfs(callee_id, chain.copy(), depth + 1)
            
            visited.remove(node_id)
        
        dfs(start_id, [], 0)
        return chains
    
    def find_recursive_calls(self) -> List[List[str]]:
        """Find recursive call patterns"""
        recursive = []
        
        for node_id, node in self.nodes.items():
            # Direct recursion
            if node_id in node.callees:
                recursive.append([node_id, node_id])
            
            # Mutual recursion (simple case)
            for callee_id in node.callees:
                if callee_id in self.nodes:
                    if node_id in self.nodes[callee_id].callees:
                        recursive.append([node_id, callee_id, node_id])
        
        return recursive
    
    def get_entry_points(self) -> List[CallNode]:
        """Get functions that are never called (potential entry points)"""
        return [node for node in self.nodes.values() if not node.callers]
    
    def get_leaf_functions(self) -> List[CallNode]:
        """Get functions that don't call anything"""
        return [node for node in self.nodes.values() if not node.callees]
    
    def get_most_called_functions(self, limit: int = 10) -> List[CallNode]:
        """Get the most frequently called functions"""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.call_count,
            reverse=True
        )
        return sorted_nodes[:limit]
    
    def get_functions_by_name(self, name: str) -> List[CallNode]:
        """Get all functions with a given name"""
        node_ids = self.name_to_nodes.get(name, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_functions_in_file(self, file_path: str) -> List[CallNode]:
        """Get all functions defined in a file"""
        node_ids = self.file_to_nodes.get(file_path, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get call graph statistics"""
        return {
            **self.stats,
            'entry_points': len(self.get_entry_points()),
            'leaf_functions': len(self.get_leaf_functions()),
            'average_callees_per_function': (
                sum(n.callee_count for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0
            ),
            'average_callers_per_function': (
                sum(n.caller_count for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0
            )
        }
    
    def export_nodes(self) -> List[Dict[str, Any]]:
        """Export all nodes as dictionaries"""
        return [node.to_dict() for node in self.nodes.values()]
    
    def export_call_sites(self) -> List[Dict[str, Any]]:
        """Export all call sites as dictionaries"""
        return [site.to_dict() for site in self.call_sites]
    
    def clear(self):
        """Clear the call graph"""
        self.nodes.clear()
        self.call_sites.clear()
        self.name_to_nodes.clear()
        self.file_to_nodes.clear()
        self.stats = {
            'total_nodes': 0,
            'total_calls': 0,
            'function_calls': 0,
            'method_calls': 0,
            'constructor_calls': 0,
            'async_calls': 0,
            'callback_calls': 0
        }
        logger.info("Call graph cleared")
