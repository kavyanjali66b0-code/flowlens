"""
Dependency Injection Resolver

Detects and analyzes dependency injection patterns in JavaScript/TypeScript code.
Supports React hooks, constructor injection, property injection, and other DI patterns.

Classes:
    DIType: Enum for types of dependency injection
    Dependency: Represents a dependency relationship
    DIResolver: Main resolver that detects DI patterns
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Tuple
from tree_sitter import Node
from analyzer.symbol_table import SymbolTable


class DIType(Enum):
    """Types of dependency injection patterns."""
    CONSTRUCTOR = auto()      # Constructor injection
    PROPERTY = auto()         # Property injection
    SETTER = auto()           # Setter injection
    HOOK = auto()             # React/Vue hooks
    PROVIDER = auto()         # Provider pattern
    SERVICE_LOCATOR = auto()  # Service locator pattern
    MODULE_IMPORT = auto()    # Import as dependency
    CONTEXT = auto()          # Context injection (React/Angular)
    DECORATOR = auto()        # Decorator-based injection
    FACTORY = auto()          # Factory pattern


@dataclass
class Dependency:
    """
    Represents a dependency relationship between components.
    
    Attributes:
        consumer: Component/service that needs the dependency
        provider: Component/service providing the dependency
        injection_type: Type of dependency injection used
        location: (file_path, line_number) where dependency is declared
        metadata: Additional information about the dependency
    """
    consumer: str
    provider: str
    injection_type: DIType
    location: Tuple[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Make dependency hashable."""
        return hash((self.consumer, self.provider, self.injection_type.name))
    
    def __eq__(self, other: object) -> bool:
        """Compare dependencies."""
        if not isinstance(other, Dependency):
            return False
        return (self.consumer == other.consumer and
                self.provider == other.provider and
                self.injection_type == other.injection_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'consumer': self.consumer,
            'provider': self.provider,
            'injection_type': self.injection_type.name,
            'location': {
                'file': self.location[0],
                'line': self.location[1]
            },
            'metadata': self.metadata
        }


class DIResolver:
    """
    Resolves dependency injection patterns in code.
    
    Detects various DI patterns including constructor injection, hooks,
    property injection, and provider patterns. Builds a comprehensive
    map of dependencies between components.
    """
    
    def __init__(self, symbol_table: SymbolTable, nodes: List[Node],
                 file_path: str = "unknown"):
        """
        Initialize DI resolver.
        
        Args:
            symbol_table: Symbol table with cross-file information
            nodes: List of AST nodes to analyze
            file_path: Path to the file being analyzed
        """
        self.symbol_table = symbol_table
        self.nodes = nodes
        self.file_path = file_path
        self.dependencies: Dict[str, List[Dependency]] = {}
        
        # Track current component being analyzed
        self._current_component: Optional[str] = None
        
        # Known React/Vue hooks patterns
        self._hook_patterns = {
            'useState', 'useEffect', 'useContext', 'useReducer',
            'useCallback', 'useMemo', 'useRef', 'useImperativeHandle',
            'useLayoutEffect', 'useDebugValue', 'useDeferredValue',
            'useTransition', 'useId', 'useSyncExternalStore'
        }
    
    def analyze_dependencies(self) -> Dict[str, List[Dependency]]:
        """
        Find all DI relationships in the code.
        
        Returns:
            Dictionary mapping component names to their dependencies
        """
        # Detect various DI patterns
        self._detect_module_imports()
        self._detect_constructor_injection()
        self._detect_hook_usage()
        self._detect_context_usage()
        self._detect_property_injection()
        self._detect_provider_patterns()
        self._detect_decorator_injection()
        
        return self.dependencies
    
    def _add_dependency(self, consumer: str, provider: str,
                       injection_type: DIType, location: Tuple[str, int],
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a dependency to the dependencies map.
        
        Args:
            consumer: Component needing the dependency
            provider: Component providing the dependency
            injection_type: Type of injection
            location: Source location
            metadata: Optional additional information
        """
        dependency = Dependency(
            consumer=consumer,
            provider=provider,
            injection_type=injection_type,
            location=location,
            metadata=metadata or {}
        )
        
        if consumer not in self.dependencies:
            self.dependencies[consumer] = []
        
        # Avoid duplicates
        if dependency not in self.dependencies[consumer]:
            self.dependencies[consumer].append(dependency)
    
    def _detect_module_imports(self) -> None:
        """Detect module imports as dependencies."""
        for node in self.nodes:
            if node.type == 'import_statement':
                self._process_import_dependency(node)
    
    def _process_import_dependency(self, node: Node) -> None:
        """Process import statement as dependency."""
        # Get the module source
        source = None
        for child in node.children:
            if child.type == 'string':
                source = self._get_text(child).strip('"\'')
                break
        
        if not source:
            return
        
        # Get imported names
        import_clause = self._find_child(node, 'import_clause')
        if not import_clause:
            return
        
        # Named imports: import { X, Y } from 'module'
        named_imports = self._find_child(import_clause, 'named_imports')
        if named_imports:
            for child in named_imports.children:
                if child.type == 'import_specifier':
                    name_node = self._find_child(child, 'identifier')
                    if name_node:
                        imported_name = self._get_text(name_node)
                        self._add_dependency(
                            consumer=self.file_path,
                            provider=f"{source}.{imported_name}",
                            injection_type=DIType.MODULE_IMPORT,
                            location=(self.file_path, node.start_point[0]),
                            metadata={'module': source, 'import': imported_name}
                        )
        
        # Default import: import X from 'module'
        default_import = self._find_child(import_clause, 'identifier')
        if default_import:
            imported_name = self._get_text(default_import)
            self._add_dependency(
                consumer=self.file_path,
                provider=source,
                injection_type=DIType.MODULE_IMPORT,
                location=(self.file_path, node.start_point[0]),
                metadata={'module': source, 'import': imported_name, 'default': True}
            )
    
    def _detect_constructor_injection(self) -> None:
        """Detect constructor injection pattern."""
        for node in self.nodes:
            if node.type in ('class_declaration', 'class'):
                self._process_class_constructor(node)
    
    def _process_class_constructor(self, node: Node) -> None:
        """Process class constructor for injection."""
        # Get class name
        class_name_node = node.child_by_field_name('name')
        if not class_name_node:
            return
        
        class_name = self._get_text(class_name_node)
        
        # Find constructor
        body = node.child_by_field_name('body') or self._find_child(node, 'class_body')
        if not body:
            return
        
        for member in body.children:
            if member.type == 'method_definition':
                name = member.child_by_field_name('name')
                if name and self._get_text(name) == 'constructor':
                    self._process_constructor_params(class_name, member)
    
    def _process_constructor_params(self, class_name: str, constructor: Node) -> None:
        """Process constructor parameters as dependencies."""
        params = constructor.child_by_field_name('parameters')
        if not params:
            return
        
        for param in params.children:
            param_type = None
            param_name = None
            
            # TypeScript: constructor(private service: ServiceType)
            if param.type == 'required_parameter':
                # Get parameter name
                pattern = self._find_child(param, 'identifier')
                if pattern:
                    param_name = self._get_text(pattern)
                
                # Get type annotation
                type_annotation = self._find_child(param, 'type_annotation')
                if type_annotation:
                    type_node = self._find_child(type_annotation, 'type_identifier')
                    if type_node:
                        param_type = self._get_text(type_node)
            
            # Regular JavaScript: constructor(service)
            elif param.type == 'identifier':
                param_name = self._get_text(param)
            
            if param_name:
                provider = param_type if param_type else param_name
                self._add_dependency(
                    consumer=class_name,
                    provider=provider,
                    injection_type=DIType.CONSTRUCTOR,
                    location=(self.file_path, param.start_point[0]),
                    metadata={'parameter': param_name, 'type': param_type}
                )
    
    def _detect_hook_usage(self) -> None:
        """Detect React/Vue hook usage."""
        for node in self.nodes:
            # Function components can use hooks
            if node.type in ('function_declaration', 'arrow_function',
                           'function_expression'):
                self._process_function_hooks(node)
    
    def _process_function_hooks(self, node: Node) -> None:
        """Process hooks used in a function component."""
        # Get component name
        component_name = None
        if node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            if name_node:
                component_name = self._get_text(name_node)
        
        if not component_name:
            component_name = self.file_path
        
        # Find hook calls in function body
        self._find_hook_calls(node, component_name)
    
    def _find_hook_calls(self, node: Node, component_name: str) -> None:
        """Recursively find hook calls in AST."""
        if node.type == 'call_expression':
            func = node.child_by_field_name('function')
            if func:
                func_name = self._get_text(func)
                
                # Check if it's a known hook or custom hook (starts with 'use')
                if func_name in self._hook_patterns or func_name.startswith('use'):
                    self._add_dependency(
                        consumer=component_name,
                        provider=func_name,
                        injection_type=DIType.HOOK,
                        location=(self.file_path, node.start_point[0]),
                        metadata={'hook': func_name, 'custom': func_name not in self._hook_patterns}
                    )
        
        # Recurse into children
        for child in node.children:
            self._find_hook_calls(child, component_name)
    
    def _detect_context_usage(self) -> None:
        """Detect Context API usage (React useContext, Angular providers)."""
        for node in self.nodes:
            self._find_use_context_calls(node)
    
    def _find_use_context_calls(self, node: Node) -> None:
        """Recursively find useContext calls."""
        if node.type == 'call_expression':
            func = node.child_by_field_name('function')
            if func and self._get_text(func) == 'useContext':
                self._process_use_context(node)
        
        # Recurse into children
        for child in node.children:
            self._find_use_context_calls(child)
    
    def _process_use_context(self, node: Node) -> None:
        """Process useContext call."""
        # Get the context argument
        args = node.child_by_field_name('arguments')
        if not args or len(args.children) == 0:
            return
        
        context_arg = args.children[0]
        context_name = self._get_text(context_arg)
        
        # Find the containing component
        component_name = self._find_containing_component(node)
        
        self._add_dependency(
            consumer=component_name or self.file_path,
            provider=context_name,
            injection_type=DIType.CONTEXT,
            location=(self.file_path, node.start_point[0]),
            metadata={'context': context_name}
        )
    
    def _detect_property_injection(self) -> None:
        """Detect property injection pattern."""
        for node in self.nodes:
            if node.type == 'assignment_expression':
                self._process_property_assignment(node)
    
    def _process_property_assignment(self, node: Node) -> None:
        """Check if assignment is property injection."""
        left = node.child_by_field_name('left')
        right = node.child_by_field_name('right')
        
        if not (left and right):
            return
        
        # Check for this.property = service pattern
        if left.type == 'member_expression':
            obj = left.child_by_field_name('object')
            prop = left.child_by_field_name('property')
            
            if obj and prop and self._get_text(obj) == 'this':
                prop_name = self._get_text(prop)
                
                # If right side is a service/dependency
                if right.type in ('identifier', 'new_expression'):
                    service_name = self._get_text(right)
                    component_name = self._find_containing_class(node)
                    
                    self._add_dependency(
                        consumer=component_name or self.file_path,
                        provider=service_name,
                        injection_type=DIType.PROPERTY,
                        location=(self.file_path, node.start_point[0]),
                        metadata={'property': prop_name}
                    )
    
    def _detect_provider_patterns(self) -> None:
        """Detect provider/container patterns."""
        for node in self.nodes:
            # React Provider: <Context.Provider value={...}>
            if node.type == 'jsx_element':
                self._process_jsx_provider(node)
            
            # Service container: container.get('ServiceName')
            elif node.type == 'call_expression':
                func = node.child_by_field_name('function')
                if func and func.type == 'member_expression':
                    method = func.child_by_field_name('property')
                    if method and self._get_text(method) in ('get', 'resolve', 'make'):
                        self._process_service_locator(node)
    
    def _process_jsx_provider(self, node: Node) -> None:
        """Process JSX Provider component."""
        opening = self._find_child(node, 'jsx_opening_element')
        if not opening:
            return
        
        name = opening.child_by_field_name('name')
        if not name:
            return
        
        element_name = self._get_text(name)
        
        # Check if it's a Provider (ends with .Provider or is named Provider)
        if 'Provider' in element_name:
            component_name = self._find_containing_component(node)
            
            self._add_dependency(
                consumer=component_name or self.file_path,
                provider=element_name,
                injection_type=DIType.PROVIDER,
                location=(self.file_path, node.start_point[0]),
                metadata={'provider': element_name}
            )
    
    def _process_service_locator(self, node: Node) -> None:
        """Process service locator pattern."""
        # Get service name from argument
        args = node.child_by_field_name('arguments')
        if not args or len(args.children) == 0:
            return
        
        service_arg = args.children[0]
        if service_arg.type == 'string':
            service_name = self._get_text(service_arg).strip('"\'')
            component_name = self._find_containing_component(node) or self._find_containing_class(node)
            
            self._add_dependency(
                consumer=component_name or self.file_path,
                provider=service_name,
                injection_type=DIType.SERVICE_LOCATOR,
                location=(self.file_path, node.start_point[0]),
                metadata={'locator_call': True}
            )
    
    def _detect_decorator_injection(self) -> None:
        """Detect decorator-based injection (Angular, NestJS)."""
        for node in self.nodes:
            if node.type == 'decorator':
                self._process_decorator(node)
    
    def _process_decorator(self, node: Node) -> None:
        """Process decorator for injection metadata."""
        # Get decorator name
        call = self._find_child(node, 'call_expression')
        if call:
            func = call.child_by_field_name('function')
            if func:
                decorator_name = self._get_text(func)
                
                # Check for Angular @Inject decorator
                if decorator_name == 'Inject':
                    args = call.child_by_field_name('arguments')
                    if args and len(args.children) > 0:
                        service_name = self._get_text(args.children[0]).strip('"\'')
                        
                        # Find the decorated parameter/property
                        parent = node.parent
                        if parent:
                            component_name = self._find_containing_class(parent)
                            
                            self._add_dependency(
                                consumer=component_name or self.file_path,
                                provider=service_name,
                                injection_type=DIType.DECORATOR,
                                location=(self.file_path, node.start_point[0]),
                                metadata={'decorator': decorator_name}
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
    
    def _find_containing_component(self, node: Node) -> Optional[str]:
        """Find the name of the containing React component."""
        current = node
        while current:
            if current.type in ('function_declaration', 'arrow_function'):
                if current.type == 'function_declaration':
                    name = current.child_by_field_name('name')
                    if name:
                        func_name = self._get_text(name)
                        # React components start with uppercase
                        if func_name and func_name[0].isupper():
                            return func_name
            current = current.parent if hasattr(current, 'parent') else None
        return None
    
    def _find_containing_class(self, node: Node) -> Optional[str]:
        """Find the name of the containing class."""
        current = node
        while current:
            if current.type in ('class_declaration', 'class'):
                name = current.child_by_field_name('name')
                if name:
                    return self._get_text(name)
            current = current.parent if hasattr(current, 'parent') else None
        return None
    
    def get_dependencies_for(self, component: str) -> List[Dependency]:
        """
        Get all dependencies for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            List of dependencies
        """
        return self.dependencies.get(component, [])
    
    def get_all_consumers(self, provider: str) -> List[str]:
        """
        Get all components that depend on a provider.
        
        Args:
            provider: Provider/service name
            
        Returns:
            List of consumer component names
        """
        consumers = []
        for component, deps in self.dependencies.items():
            if any(d.provider == provider for d in deps):
                consumers.append(component)
        return consumers
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about dependencies.
        
        Returns:
            Dictionary with dependency statistics
        """
        total_deps = sum(len(deps) for deps in self.dependencies.values())
        
        # Count by injection type
        type_counts = {}
        for deps in self.dependencies.values():
            for dep in deps:
                type_name = dep.injection_type.name
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            'total_components': len(self.dependencies),
            'total_dependencies': total_deps,
            'by_type': type_counts,
            'components': list(self.dependencies.keys())
        }
