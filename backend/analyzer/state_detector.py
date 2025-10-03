"""
State Pattern Detector for M3.1.3

Detects state management patterns in JavaScript/TypeScript codebases:
- React Hooks (useState, useReducer, useContext)
- Zustand stores
- Redux (actions, reducers, selectors)
- MobX observables
- Recoil atoms/selectors
- Context API patterns

Tracks state initialization, updates, and mutations.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StateManagementType(Enum):
    """Types of state management patterns"""
    REACT_HOOK = "react_hook"              # useState, useReducer, useContext
    ZUSTAND = "zustand"                     # Zustand store
    REDUX = "redux"                         # Redux store/actions/reducers
    MOBX = "mobx"                           # MobX observables
    RECOIL = "recoil"                       # Recoil atoms/selectors
    CONTEXT_API = "context_api"             # React Context
    LOCAL_STATE = "local_state"             # Component local state (class)
    CUSTOM = "custom"                       # Custom state management


class StateOperationType(Enum):
    """Types of state operations"""
    INIT = "initialization"                 # State initialization
    READ = "read"                           # Reading state
    UPDATE = "update"                       # Updating state (immutable)
    MUTATION = "mutation"                   # Mutating state (mutable)
    SUBSCRIBE = "subscribe"                 # Subscribing to state changes
    DISPATCH = "dispatch"                   # Dispatching actions


@dataclass
class StatePattern:
    """Represents a detected state management pattern"""
    pattern_id: str                         # Unique identifier
    pattern_type: StateManagementType       # Type of state management
    node_id: str                            # Associated AST node ID
    
    # State info
    state_name: Optional[str] = None        # Name of state variable
    store_name: Optional[str] = None        # Name of store (for Zustand/Redux)
    
    # Metadata
    file_path: Optional[str] = None         # File where pattern is defined
    line_number: Optional[int] = None       # Line number
    
    # Operations
    operations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relations
    dependencies: List[str] = field(default_factory=list)  # Other state it depends on
    subscribers: List[str] = field(default_factory=list)   # Components using this state
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'node_id': self.node_id,
            'state_name': self.state_name,
            'store_name': self.store_name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'operations_count': len(self.operations),
            'dependencies': self.dependencies,
            'subscribers': self.subscribers,
            'metadata': self.metadata
        }


class StateDetector:
    """
    Main class for detecting state management patterns.
    
    Analyzes AST nodes and metadata to identify:
    1. React Hook patterns (useState, useReducer, etc.)
    2. State library patterns (Zustand, Redux, MobX, Recoil)
    3. State operations (init, read, update, mutation)
    4. State dependencies and subscribers
    """
    
    def __init__(self):
        """Initialize the state detector"""
        self.patterns: Dict[str, StatePattern] = {}
        self.state_operations: List[Dict[str, Any]] = []
        
        # Track state relationships
        self.state_to_consumers: Dict[str, Set[str]] = {}  # state -> components using it
        self.component_to_state: Dict[str, Set[str]] = {}  # component -> states it uses
        
        # Statistics
        self.stats = {
            'total_patterns': 0,
            'react_hooks': 0,
            'zustand_stores': 0,
            'redux_stores': 0,
            'mobx_observables': 0,
            'recoil_atoms': 0,
            'context_providers': 0,
            'total_operations': 0
        }
        
        logger.info("StateDetector initialized")
    
    def detect_patterns(self, nodes: List[Any], file_symbols: Dict[str, Any]) -> List[StatePattern]:
        """
        Detect state management patterns in nodes.
        
        Args:
            nodes: List of AST nodes
            file_symbols: Symbol information per file
            
        Returns:
            List of detected state patterns
        """
        logger.info(f"Detecting state patterns in {len(nodes)} nodes")
        
        for node in nodes:
            try:
                # Get node metadata
                metadata = getattr(node, 'metadata', {}) or {}
                
                # Detect React Hooks
                if 'detected_hooks' in metadata:
                    self._detect_react_hooks(node, metadata)
                
                # Detect Zustand stores
                if self._is_zustand_store(node, metadata):
                    self._detect_zustand_store(node, metadata)
                
                # Detect Redux patterns
                if self._is_redux_pattern(node, metadata):
                    self._detect_redux_pattern(node, metadata)
                
                # Detect MobX observables
                if self._is_mobx_observable(node, metadata):
                    self._detect_mobx_observable(node, metadata)
                
                # Detect Recoil atoms/selectors
                if self._is_recoil_pattern(node, metadata):
                    self._detect_recoil_pattern(node, metadata)
                
                # Detect Context API
                if self._is_context_pattern(node, metadata):
                    self._detect_context_pattern(node, metadata)
                
            except Exception as e:
                logger.warning(f"Error detecting state pattern in node {node.id}: {e}")
        
        logger.info(f"State pattern detection complete: {len(self.patterns)} patterns found")
        return list(self.patterns.values())
    
    def _detect_react_hooks(self, node, metadata):
        """Detect React Hook patterns"""
        hooks = metadata.get('detected_hooks', [])
        
        for hook in hooks:
            if isinstance(hook, dict):
                hook_name = hook.get('name', '')
                
                # useState
                if hook_name == 'useState':
                    pattern = StatePattern(
                        pattern_id=f"hook_{node.id}_{hook.get('state_var', 'state')}",
                        pattern_type=StateManagementType.REACT_HOOK,
                        node_id=node.id,
                        state_name=hook.get('state_var', 'state'),
                        file_path=getattr(node, 'file_path', None),
                        line_number=getattr(node, 'line_number', None),
                        metadata={
                            'hook_type': 'useState',
                            'setter': hook.get('setter_var', 'setState'),
                            'initial_value': hook.get('initial_value', None)
                        }
                    )
                    
                    self.patterns[pattern.pattern_id] = pattern
                    self.stats['react_hooks'] += 1
                    self.stats['total_patterns'] += 1
                
                # useReducer
                elif hook_name == 'useReducer':
                    pattern = StatePattern(
                        pattern_id=f"hook_{node.id}_reducer",
                        pattern_type=StateManagementType.REACT_HOOK,
                        node_id=node.id,
                        state_name=hook.get('state_var', 'state'),
                        file_path=getattr(node, 'file_path', None),
                        line_number=getattr(node, 'line_number', None),
                        metadata={
                            'hook_type': 'useReducer',
                            'dispatch': hook.get('dispatch_var', 'dispatch'),
                            'reducer': hook.get('reducer', None)
                        }
                    )
                    
                    self.patterns[pattern.pattern_id] = pattern
                    self.stats['react_hooks'] += 1
                    self.stats['total_patterns'] += 1
                
                # useContext
                elif hook_name == 'useContext':
                    pattern = StatePattern(
                        pattern_id=f"hook_{node.id}_context",
                        pattern_type=StateManagementType.CONTEXT_API,
                        node_id=node.id,
                        state_name=hook.get('context_var', 'context'),
                        file_path=getattr(node, 'file_path', None),
                        line_number=getattr(node, 'line_number', None),
                        metadata={
                            'hook_type': 'useContext',
                            'context_name': hook.get('context', None)
                        }
                    )
                    
                    self.patterns[pattern.pattern_id] = pattern
                    self.stats['context_providers'] += 1
                    self.stats['total_patterns'] += 1
    
    def _is_zustand_store(self, node, metadata) -> bool:
        """Check if node represents a Zustand store"""
        # Look for create() from zustand
        name = node.name if hasattr(node, 'name') else ''
        code_snippet = metadata.get('code_snippet', '')
        
        return ('create' in name and 'zustand' in code_snippet.lower()) or \
               'create(' in code_snippet and 'set' in code_snippet and 'get' in code_snippet
    
    def _detect_zustand_store(self, node, metadata):
        """Detect Zustand store pattern"""
        pattern = StatePattern(
            pattern_id=f"zustand_{node.id}",
            pattern_type=StateManagementType.ZUSTAND,
            node_id=node.id,
            store_name=node.name if hasattr(node, 'name') else 'store',
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'line_number', None),
            metadata={
                'library': 'zustand'
            }
        )
        
        self.patterns[pattern.pattern_id] = pattern
        self.stats['zustand_stores'] += 1
        self.stats['total_patterns'] += 1
    
    def _is_redux_pattern(self, node, metadata) -> bool:
        """Check if node represents a Redux pattern"""
        name = node.name if hasattr(node, 'name') else ''
        code_snippet = metadata.get('code_snippet', '')
        
        return ('reducer' in name.lower() or 
                'action' in name.lower() or
                'createSlice' in code_snippet or
                'configureStore' in code_snippet)
    
    def _detect_redux_pattern(self, node, metadata):
        """Detect Redux pattern"""
        name = node.name if hasattr(node, 'name') else ''
        
        pattern_type = 'unknown'
        if 'reducer' in name.lower():
            pattern_type = 'reducer'
        elif 'action' in name.lower():
            pattern_type = 'action'
        elif 'selector' in name.lower():
            pattern_type = 'selector'
        
        pattern = StatePattern(
            pattern_id=f"redux_{node.id}",
            pattern_type=StateManagementType.REDUX,
            node_id=node.id,
            store_name=name,
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'line_number', None),
            metadata={
                'library': 'redux',
                'redux_type': pattern_type
            }
        )
        
        self.patterns[pattern.pattern_id] = pattern
        self.stats['redux_stores'] += 1
        self.stats['total_patterns'] += 1
    
    def _is_mobx_observable(self, node, metadata) -> bool:
        """Check if node represents a MobX observable"""
        code_snippet = metadata.get('code_snippet', '')
        return 'observable' in code_snippet or 'makeObservable' in code_snippet
    
    def _detect_mobx_observable(self, node, metadata):
        """Detect MobX observable pattern"""
        pattern = StatePattern(
            pattern_id=f"mobx_{node.id}",
            pattern_type=StateManagementType.MOBX,
            node_id=node.id,
            store_name=node.name if hasattr(node, 'name') else 'observable',
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'line_number', None),
            metadata={
                'library': 'mobx'
            }
        )
        
        self.patterns[pattern.pattern_id] = pattern
        self.stats['mobx_observables'] += 1
        self.stats['total_patterns'] += 1
    
    def _is_recoil_pattern(self, node, metadata) -> bool:
        """Check if node represents a Recoil atom/selector"""
        code_snippet = metadata.get('code_snippet', '')
        return 'atom(' in code_snippet or 'selector(' in code_snippet
    
    def _detect_recoil_pattern(self, node, metadata):
        """Detect Recoil atom/selector pattern"""
        code_snippet = metadata.get('code_snippet', '')
        pattern_type = 'atom' if 'atom(' in code_snippet else 'selector'
        
        pattern = StatePattern(
            pattern_id=f"recoil_{node.id}",
            pattern_type=StateManagementType.RECOIL,
            node_id=node.id,
            state_name=node.name if hasattr(node, 'name') else 'state',
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'line_number', None),
            metadata={
                'library': 'recoil',
                'recoil_type': pattern_type
            }
        )
        
        self.patterns[pattern.pattern_id] = pattern
        self.stats['recoil_atoms'] += 1
        self.stats['total_patterns'] += 1
    
    def _is_context_pattern(self, node, metadata) -> bool:
        """Check if node represents a Context Provider"""
        name = node.name if hasattr(node, 'name') else ''
        code_snippet = metadata.get('code_snippet', '')
        
        return ('createContext' in code_snippet or 
                'Context' in name and 'Provider' in code_snippet)
    
    def _detect_context_pattern(self, node, metadata):
        """Detect React Context pattern"""
        pattern = StatePattern(
            pattern_id=f"context_{node.id}",
            pattern_type=StateManagementType.CONTEXT_API,
            node_id=node.id,
            state_name=node.name if hasattr(node, 'name') else 'Context',
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'line_number', None),
            metadata={
                'library': 'react'
            }
        )
        
        self.patterns[pattern.pattern_id] = pattern
        self.stats['context_providers'] += 1
        self.stats['total_patterns'] += 1
    
    def track_state_operation(
        self,
        pattern_id: str,
        operation_type: StateOperationType,
        node_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Track a state operation"""
        if pattern_id not in self.patterns:
            logger.warning(f"Pattern {pattern_id} not found for operation tracking")
            return
        
        operation = {
            'type': operation_type.value,
            'node_id': node_id,
            'details': details or {}
        }
        
        self.patterns[pattern_id].operations.append(operation)
        self.state_operations.append(operation)
        self.stats['total_operations'] += 1
    
    def get_state_for_component(self, component_id: str) -> List[StatePattern]:
        """Get all state patterns used by a component"""
        if component_id not in self.component_to_state:
            return []
        
        state_ids = self.component_to_state[component_id]
        return [self.patterns[sid] for sid in state_ids if sid in self.patterns]
    
    def get_consumers_for_state(self, pattern_id: str) -> List[str]:
        """Get all components that consume a state pattern"""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return []
        
        return pattern.subscribers
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            **self.stats,
            'patterns_by_type': {
                'react_hooks': self.stats['react_hooks'],
                'zustand': self.stats['zustand_stores'],
                'redux': self.stats['redux_stores'],
                'mobx': self.stats['mobx_observables'],
                'recoil': self.stats['recoil_atoms'],
                'context': self.stats['context_providers']
            },
            'total_components_with_state': len(self.component_to_state),
            'average_operations_per_pattern': (
                self.stats['total_operations'] / self.stats['total_patterns']
                if self.stats['total_patterns'] > 0 else 0
            )
        }
    
    def export_patterns(self) -> List[Dict[str, Any]]:
        """Export all patterns as dictionaries"""
        return [pattern.to_dict() for pattern in self.patterns.values()]
    
    def clear(self):
        """Clear all detected patterns"""
        self.patterns.clear()
        self.state_operations.clear()
        self.state_to_consumers.clear()
        self.component_to_state.clear()
        self.stats = {
            'total_patterns': 0,
            'react_hooks': 0,
            'zustand_stores': 0,
            'redux_stores': 0,
            'mobx_observables': 0,
            'recoil_atoms': 0,
            'context_providers': 0,
            'total_operations': 0
        }
        logger.info("State detector cleared")
