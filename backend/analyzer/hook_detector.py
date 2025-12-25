"""
Hook Detector for M4.0 Phase 3

This module detects and analyzes React hooks in JavaScript/TypeScript code:
- Identifies hook calls (useState, useEffect, useContext, etc.)
- Tracks hook dependencies and dependency arrays
- Detects custom hooks
- Analyzes hook rules compliance
- Tracks hook call order and hierarchy

The detector provides metadata about hooks that can be used for:
- Dependency injection analysis
- Data flow visualization
- Code quality checks
- React-specific optimizations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of React hooks"""
    # State hooks
    USE_STATE = "useState"
    USE_REDUCER = "useReducer"
    
    # Effect hooks
    USE_EFFECT = "useEffect"
    USE_LAYOUT_EFFECT = "useLayoutEffect"
    USE_INSERTION_EFFECT = "useInsertionEffect"
    
    # Context hooks
    USE_CONTEXT = "useContext"
    
    # Ref hooks
    USE_REF = "useRef"
    USE_IMPERATIVE_HANDLE = "useImperativeHandle"
    
    # Performance hooks
    USE_MEMO = "useMemo"
    USE_CALLBACK = "useCallback"
    USE_TRANSITION = "useTransition"
    USE_DEFERRED_VALUE = "useDeferredValue"
    
    # Other hooks
    USE_DEBUG_VALUE = "useDebugValue"
    USE_ID = "useId"
    USE_SYNC_EXTERNAL_STORE = "useSyncExternalStore"
    
    # Custom hook (any function starting with 'use')
    CUSTOM = "custom"


class HookCategory(Enum):
    """Categories of hooks"""
    STATE = "state"              # useState, useReducer
    EFFECT = "effect"            # useEffect, useLayoutEffect
    CONTEXT = "context"          # useContext
    REF = "ref"                  # useRef, useImperativeHandle
    PERFORMANCE = "performance"  # useMemo, useCallback
    CUSTOM = "custom"            # Custom hooks


@dataclass
class HookDependency:
    """Represents a dependency in a hook's dependency array"""
    name: str                          # Dependency name
    type: str                          # Dependency type (variable, property, function)
    is_stable: bool = False            # Is dependency stable (won't change)?
    source_node_id: Optional[str] = None  # Node ID of dependency source
    line_number: Optional[int] = None  # Line where dependency appears
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.type,
            'is_stable': self.is_stable,
            'source_node_id': self.source_node_id,
            'line_number': self.line_number
        }


@dataclass
class HookCall:
    """Represents a single hook call"""
    hook_id: str                       # Unique identifier for this hook call
    hook_type: HookType                # Type of hook
    hook_category: HookCategory        # Category of hook
    hook_name: str                     # Name of the hook (useState, useEffect, etc.)
    
    # Location
    line_number: Optional[int] = None
    column: Optional[int] = None
    component_name: Optional[str] = None  # Component where hook is called
    
    # Arguments
    arguments: List[str] = field(default_factory=list)  # Hook arguments
    argument_types: List[str] = field(default_factory=list)  # Argument types
    
    # Dependencies
    dependencies: List[HookDependency] = field(default_factory=list)
    has_dependency_array: bool = False
    dependency_array_empty: bool = False
    
    # State-specific (useState, useReducer)
    state_variable: Optional[str] = None    # State variable name
    setter_function: Optional[str] = None   # Setter function name
    initial_value: Optional[str] = None     # Initial value
    
    # Effect-specific (useEffect, useLayoutEffect)
    has_cleanup: bool = False              # Does effect have cleanup?
    cleanup_function: Optional[str] = None # Cleanup function name
    effect_type: Optional[str] = None      # 'mount', 'update', 'unmount', 'always'
    
    # Ref-specific (useRef)
    ref_initial_value: Optional[str] = None
    
    # Memo/Callback-specific
    memoized_value: Optional[str] = None   # What's being memoized
    
    # Custom hook-specific
    is_custom: bool = False
    custom_hook_name: Optional[str] = None
    return_values: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 1.0                # Confidence in detection
    context: Optional[str] = None          # Additional context
    ast_node_type: Optional[str] = None    # AST node type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'hook_id': self.hook_id,
            'hook_type': self.hook_type.value,
            'hook_category': self.hook_category.value,
            'hook_name': self.hook_name,
            'line_number': self.line_number,
            'column': self.column,
            'component_name': self.component_name,
            'arguments': self.arguments,
            'argument_types': self.argument_types,
            'dependencies': [d.to_dict() for d in self.dependencies],
            'has_dependency_array': self.has_dependency_array,
            'dependency_array_empty': self.dependency_array_empty,
            'state_variable': self.state_variable,
            'setter_function': self.setter_function,
            'initial_value': self.initial_value,
            'has_cleanup': self.has_cleanup,
            'cleanup_function': self.cleanup_function,
            'effect_type': self.effect_type,
            'ref_initial_value': self.ref_initial_value,
            'memoized_value': self.memoized_value,
            'is_custom': self.is_custom,
            'custom_hook_name': self.custom_hook_name,
            'return_values': self.return_values,
            'confidence': self.confidence,
            'context': self.context,
            'ast_node_type': self.ast_node_type
        }


class HookDetector:
    """
    Detects and analyzes React hooks in JavaScript/TypeScript code.
    
    This detector identifies hook calls, extracts their dependencies,
    and provides metadata for dependency injection analysis.
    """
    
    def __init__(self):
        """Initialize the hook detector"""
        self.hooks: List[HookCall] = []
        self.hook_map: Dict[str, HookCall] = {}  # hook_id -> HookCall
        
        # Component tracking
        self.current_component: Optional[str] = None
        self.component_hooks: Dict[str, List[str]] = {}  # component -> [hook_ids]
        
        # Hook order tracking (for rules validation)
        self.hook_call_order: List[str] = []  # Ordered list of hook_ids
        
        # Custom hooks registry
        self.custom_hooks: Set[str] = set()  # Set of custom hook names
        
        # Statistics
        self.stats = {
            'total_hooks': 0,
            'state_hooks': 0,
            'effect_hooks': 0,
            'custom_hooks': 0,
            'hooks_with_deps': 0,
            'hooks_without_deps': 0
        }
        
        # Known React hooks
        self.known_hooks: Dict[str, Tuple[HookType, HookCategory]] = {
            'useState': (HookType.USE_STATE, HookCategory.STATE),
            'useReducer': (HookType.USE_REDUCER, HookCategory.STATE),
            'useEffect': (HookType.USE_EFFECT, HookCategory.EFFECT),
            'useLayoutEffect': (HookType.USE_LAYOUT_EFFECT, HookCategory.EFFECT),
            'useInsertionEffect': (HookType.USE_INSERTION_EFFECT, HookCategory.EFFECT),
            'useContext': (HookType.USE_CONTEXT, HookCategory.CONTEXT),
            'useRef': (HookType.USE_REF, HookCategory.REF),
            'useImperativeHandle': (HookType.USE_IMPERATIVE_HANDLE, HookCategory.REF),
            'useMemo': (HookType.USE_MEMO, HookCategory.PERFORMANCE),
            'useCallback': (HookType.USE_CALLBACK, HookCategory.PERFORMANCE),
            'useTransition': (HookType.USE_TRANSITION, HookCategory.PERFORMANCE),
            'useDeferredValue': (HookType.USE_DEFERRED_VALUE, HookCategory.PERFORMANCE),
            'useDebugValue': (HookType.USE_DEBUG_VALUE, HookCategory.CUSTOM),
            'useId': (HookType.USE_ID, HookCategory.CUSTOM),
            'useSyncExternalStore': (HookType.USE_SYNC_EXTERNAL_STORE, HookCategory.CUSTOM),
        }
    
    def enter_component(self, component_name: str):
        """Enter a component scope"""
        self.current_component = component_name
        if component_name not in self.component_hooks:
            self.component_hooks[component_name] = []
        logger.debug(f"Entered component: {component_name}")
    
    def exit_component(self):
        """Exit current component scope"""
        logger.debug(f"Exited component: {self.current_component}")
        self.current_component = None
    
    def is_hook_name(self, name: str) -> bool:
        """Check if a name follows hook naming convention (starts with 'use')"""
        return name.startswith('use') and len(name) > 3 and name[3].isupper()
    
    def detect_hook_call(self,
                        hook_id: str,
                        hook_name: str,
                        arguments: Optional[List[str]] = None,
                        argument_types: Optional[List[str]] = None,
                        line_number: Optional[int] = None,
                        column: Optional[int] = None,
                        ast_node_type: Optional[str] = None) -> Optional[HookCall]:
        """
        Detect a hook call and create HookCall object
        
        Args:
            hook_id: Unique identifier for this hook call
            hook_name: Name of the hook (useState, useEffect, etc.)
            arguments: List of argument strings
            argument_types: List of argument type strings
            line_number: Line where hook is called
            column: Column where hook is called
            ast_node_type: AST node type
        
        Returns:
            HookCall object if hook detected, None otherwise
        """
        if not self.is_hook_name(hook_name):
            return None
        
        # Determine hook type and category
        if hook_name in self.known_hooks:
            hook_type, hook_category = self.known_hooks[hook_name]
            is_custom = False
        else:
            # Custom hook
            hook_type = HookType.CUSTOM
            hook_category = HookCategory.CUSTOM
            is_custom = True
            self.custom_hooks.add(hook_name)
        
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=hook_type,
            hook_category=hook_category,
            hook_name=hook_name,
            line_number=line_number,
            column=column,
            component_name=self.current_component,
            arguments=arguments or [],
            argument_types=argument_types or [],
            is_custom=is_custom,
            custom_hook_name=hook_name if is_custom else None,
            ast_node_type=ast_node_type
        )
        
        self._add_hook(hook_call)
        return hook_call
    
    def detect_use_state(self,
                        hook_id: str,
                        state_variable: str,
                        setter_function: str,
                        initial_value: Optional[str] = None,
                        line_number: Optional[int] = None) -> HookCall:
        """
        Detect useState hook call
        
        Example: const [count, setCount] = useState(0)
        """
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=HookType.USE_STATE,
            hook_category=HookCategory.STATE,
            hook_name='useState',
            line_number=line_number,
            component_name=self.current_component,
            state_variable=state_variable,
            setter_function=setter_function,
            initial_value=initial_value,
            arguments=[initial_value] if initial_value else [],
            context=f"const [{state_variable}, {setter_function}] = useState({initial_value or ''})"
        )
        
        self._add_hook(hook_call)
        self.stats['state_hooks'] += 1
        
        return hook_call
    
    def detect_use_effect(self,
                         hook_id: str,
                         dependencies: Optional[List[HookDependency]] = None,
                         has_cleanup: bool = False,
                         cleanup_function: Optional[str] = None,
                         line_number: Optional[int] = None) -> HookCall:
        """
        Detect useEffect hook call
        
        Examples:
        - useEffect(() => { ... }) - No deps (always runs)
        - useEffect(() => { ... }, []) - Empty deps (mount only)
        - useEffect(() => { ... }, [count]) - With deps
        """
        deps = dependencies or []
        has_deps_array = dependencies is not None
        deps_empty = has_deps_array and len(deps) == 0
        
        # Determine effect type
        if not has_deps_array:
            effect_type = 'always'  # Runs on every render
        elif deps_empty:
            effect_type = 'mount'   # Runs only on mount
        else:
            effect_type = 'update'  # Runs on deps change
        
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=HookType.USE_EFFECT,
            hook_category=HookCategory.EFFECT,
            hook_name='useEffect',
            line_number=line_number,
            component_name=self.current_component,
            dependencies=deps,
            has_dependency_array=has_deps_array,
            dependency_array_empty=deps_empty,
            has_cleanup=has_cleanup,
            cleanup_function=cleanup_function,
            effect_type=effect_type,
            context=f"useEffect with {effect_type} behavior"
        )
        
        self._add_hook(hook_call)
        self.stats['effect_hooks'] += 1
        
        if has_deps_array:
            self.stats['hooks_with_deps'] += 1
        else:
            self.stats['hooks_without_deps'] += 1
        
        return hook_call
    
    def detect_use_context(self,
                          hook_id: str,
                          context_name: str,
                          line_number: Optional[int] = None) -> HookCall:
        """
        Detect useContext hook call
        
        Example: const theme = useContext(ThemeContext)
        """
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=HookType.USE_CONTEXT,
            hook_category=HookCategory.CONTEXT,
            hook_name='useContext',
            line_number=line_number,
            component_name=self.current_component,
            arguments=[context_name],
            context=f"useContext({context_name})"
        )
        
        self._add_hook(hook_call)
        return hook_call
    
    def detect_use_ref(self,
                      hook_id: str,
                      initial_value: Optional[str] = None,
                      line_number: Optional[int] = None) -> HookCall:
        """
        Detect useRef hook call
        
        Example: const ref = useRef(null)
        """
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=HookType.USE_REF,
            hook_category=HookCategory.REF,
            hook_name='useRef',
            line_number=line_number,
            component_name=self.current_component,
            ref_initial_value=initial_value,
            arguments=[initial_value] if initial_value else [],
            context=f"useRef({initial_value or ''})"
        )
        
        self._add_hook(hook_call)
        return hook_call
    
    def detect_use_memo(self,
                       hook_id: str,
                       memoized_value: str,
                       dependencies: Optional[List[HookDependency]] = None,
                       line_number: Optional[int] = None) -> HookCall:
        """
        Detect useMemo hook call
        
        Example: const value = useMemo(() => computeExpensive(a, b), [a, b])
        """
        deps = dependencies or []
        
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=HookType.USE_MEMO,
            hook_category=HookCategory.PERFORMANCE,
            hook_name='useMemo',
            line_number=line_number,
            component_name=self.current_component,
            memoized_value=memoized_value,
            dependencies=deps,
            has_dependency_array=dependencies is not None,
            dependency_array_empty=len(deps) == 0 if dependencies else False,
            context=f"useMemo with {len(deps)} dependencies"
        )
        
        self._add_hook(hook_call)
        
        if dependencies is not None:
            self.stats['hooks_with_deps'] += 1
        
        return hook_call
    
    def detect_use_callback(self,
                           hook_id: str,
                           callback_name: str,
                           dependencies: Optional[List[HookDependency]] = None,
                           line_number: Optional[int] = None) -> HookCall:
        """
        Detect useCallback hook call
        
        Example: const handleClick = useCallback(() => { ... }, [count])
        """
        deps = dependencies or []
        
        hook_call = HookCall(
            hook_id=hook_id,
            hook_type=HookType.USE_CALLBACK,
            hook_category=HookCategory.PERFORMANCE,
            hook_name='useCallback',
            line_number=line_number,
            component_name=self.current_component,
            memoized_value=callback_name,
            dependencies=deps,
            has_dependency_array=dependencies is not None,
            dependency_array_empty=len(deps) == 0 if dependencies else False,
            context=f"useCallback({callback_name}) with {len(deps)} dependencies"
        )
        
        self._add_hook(hook_call)
        
        if dependencies is not None:
            self.stats['hooks_with_deps'] += 1
        
        return hook_call
    
    def add_dependency(self,
                      hook_id: str,
                      dep_name: str,
                      dep_type: str = 'variable',
                      is_stable: bool = False,
                      source_node_id: Optional[str] = None) -> bool:
        """
        Add a dependency to a hook's dependency array
        
        Args:
            hook_id: ID of the hook
            dep_name: Name of the dependency
            dep_type: Type of dependency (variable, property, function)
            is_stable: Is dependency stable?
            source_node_id: Node ID of dependency source
        
        Returns:
            True if dependency added, False if hook not found
        """
        if hook_id not in self.hook_map:
            return False
        
        hook = self.hook_map[hook_id]
        dependency = HookDependency(
            name=dep_name,
            type=dep_type,
            is_stable=is_stable,
            source_node_id=source_node_id,
            line_number=hook.line_number
        )
        
        hook.dependencies.append(dependency)
        hook.has_dependency_array = True
        hook.dependency_array_empty = False
        
        logger.debug(f"Added dependency '{dep_name}' to hook {hook_id}")
        return True
    
    def get_hook(self, hook_id: str) -> Optional[HookCall]:
        """Get a hook by ID"""
        return self.hook_map.get(hook_id)
    
    def get_hooks_by_component(self, component_name: str) -> List[HookCall]:
        """Get all hooks for a specific component"""
        if component_name not in self.component_hooks:
            return []
        
        hook_ids = self.component_hooks[component_name]
        return [self.hook_map[hid] for hid in hook_ids if hid in self.hook_map]
    
    def get_hooks_by_type(self, hook_type: HookType) -> List[HookCall]:
        """Get all hooks of a specific type"""
        return [h for h in self.hooks if h.hook_type == hook_type]
    
    def get_hooks_by_category(self, category: HookCategory) -> List[HookCall]:
        """Get all hooks in a specific category"""
        return [h for h in self.hooks if h.hook_category == category]
    
    def get_state_hooks(self) -> List[HookCall]:
        """Get all state hooks (useState, useReducer)"""
        return self.get_hooks_by_category(HookCategory.STATE)
    
    def get_effect_hooks(self) -> List[HookCall]:
        """Get all effect hooks (useEffect, useLayoutEffect, etc.)"""
        return self.get_hooks_by_category(HookCategory.EFFECT)
    
    def get_custom_hooks(self) -> List[HookCall]:
        """Get all custom hooks"""
        return [h for h in self.hooks if h.is_custom]
    
    def get_hooks_with_dependencies(self) -> List[HookCall]:
        """Get all hooks that have dependency arrays"""
        return [h for h in self.hooks if h.has_dependency_array]
    
    def get_hook_dependencies(self, hook_id: str) -> List[HookDependency]:
        """Get dependencies for a specific hook"""
        hook = self.get_hook(hook_id)
        return hook.dependencies if hook else []
    
    def _add_hook(self, hook_call: HookCall):
        """Internal method to add a hook"""
        self.hooks.append(hook_call)
        self.hook_map[hook_call.hook_id] = hook_call
        self.hook_call_order.append(hook_call.hook_id)
        self.stats['total_hooks'] += 1
        
        if hook_call.is_custom:
            self.stats['custom_hooks'] += 1
        
        # Track by component
        if self.current_component:
            if self.current_component not in self.component_hooks:
                self.component_hooks[self.current_component] = []
            self.component_hooks[self.current_component].append(hook_call.hook_id)
        
        logger.debug(f"Detected hook: {hook_call.hook_name} (ID: {hook_call.hook_id})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        hook_type_counts = {}
        for hook in self.hooks:
            ht = hook.hook_type.value
            hook_type_counts[ht] = hook_type_counts.get(ht, 0) + 1
        
        return {
            **self.stats,
            'total_components': len(self.component_hooks),
            'total_custom_hook_types': len(self.custom_hooks),
            'hook_type_breakdown': hook_type_counts,
            'average_hooks_per_component': (
                len(self.hooks) / len(self.component_hooks) 
                if self.component_hooks else 0
            )
        }
    
    def export_hooks(self) -> List[Dict[str, Any]]:
        """Export all hooks as dictionaries"""
        return [hook.to_dict() for hook in self.hooks]
    
    def clear(self):
        """Clear all detected hooks"""
        self.hooks.clear()
        self.hook_map.clear()
        self.current_component = None
        self.component_hooks.clear()
        self.hook_call_order.clear()
        self.custom_hooks.clear()
        self.stats = {
            'total_hooks': 0,
            'state_hooks': 0,
            'effect_hooks': 0,
            'custom_hooks': 0,
            'hooks_with_deps': 0,
            'hooks_without_deps': 0
        }
        logger.info("Hook detector cleared")
