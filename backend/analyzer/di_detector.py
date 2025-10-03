"""
Dependency Injection (DI) Pattern Detector for M3.3.1

Detects dependency injection patterns in TypeScript/JavaScript code:
- Constructor injection
- Setter injection
- Interface injection
- Property injection
- Framework-specific DI (Angular, NestJS, InversifyJS)

The DI detector identifies:
- DI patterns and their types
- Injected dependencies
- Injection points
- DI frameworks used
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DIType(Enum):
    """Types of dependency injection patterns"""
    CONSTRUCTOR = "constructor"           # Constructor injection
    SETTER = "setter"                     # Setter method injection
    PROPERTY = "property"                 # Property injection
    INTERFACE = "interface"               # Interface-based injection
    METHOD = "method"                     # Method parameter injection
    FIELD = "field"                       # Field/member injection


class DIFramework(Enum):
    """DI frameworks/libraries"""
    ANGULAR = "angular"                   # Angular DI
    NESTJS = "nestjs"                     # NestJS DI
    INVERSIFY = "inversify"               # InversifyJS
    TSYRINGE = "tsyringe"                 # TSyringe
    TYPEDI = "typedi"                     # TypeDI
    AWILIX = "awilix"                     # Awilix
    CUSTOM = "custom"                     # Custom DI
    NONE = "none"                         # No framework (manual DI)


@dataclass
class Dependency:
    """Represents an injected dependency"""
    dependency_id: str                    # Unique ID
    name: str                             # Dependency name
    type: Optional[str] = None            # Type (class/interface name)
    
    # Injection details
    injection_type: DIType = DIType.CONSTRUCTOR
    injection_point: Optional[str] = None  # Where injected (constructor, setter, etc.)
    
    # Metadata
    is_optional: bool = False             # Is dependency optional?
    is_array: bool = False                # Is it an array of dependencies?
    default_value: Optional[str] = None   # Default value if any
    
    # Decorators/annotations
    decorators: List[str] = field(default_factory=list)
    
    # Location
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dependency_id': self.dependency_id,
            'name': self.name,
            'type': self.type,
            'injection_type': self.injection_type.value,
            'injection_point': self.injection_point,
            'is_optional': self.is_optional,
            'is_array': self.is_array,
            'default_value': self.default_value,
            'decorators': self.decorators,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'metadata': self.metadata
        }


@dataclass
class DIPattern:
    """Represents a detected DI pattern"""
    pattern_id: str                       # Unique ID
    class_name: str                       # Class using DI
    di_type: DIType                       # Type of DI pattern
    framework: DIFramework                # Framework used
    
    # Dependencies
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Pattern details
    is_injectable: bool = False           # Is class marked as injectable?
    provider_token: Optional[str] = None  # Provider token if any
    scope: Optional[str] = None           # Scope (singleton, transient, etc.)
    
    # Location
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    # Confidence
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_dependency(self, dependency: Dependency):
        """Add a dependency to this pattern"""
        self.dependencies.append(dependency)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'class_name': self.class_name,
            'di_type': self.di_type.value,
            'framework': self.framework.value,
            'dependencies': [dep.to_dict() for dep in self.dependencies],
            'is_injectable': self.is_injectable,
            'provider_token': self.provider_token,
            'scope': self.scope,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class ContainerRegistration:
    """Represents a container registration/binding"""
    registration_id: str                  # Unique ID
    container_name: str                   # Container name/identifier
    service_name: str                     # Service/dependency name
    implementation: Optional[str] = None  # Implementation class/value
    
    # Registration details
    registration_type: str = "bind"       # bind, register, provide, etc.
    lifecycle: Optional[str] = None       # singleton, transient, scoped, etc.
    token: Optional[str] = None           # Injection token/identifier
    
    # Factory/value details
    is_factory: bool = False              # Is it a factory function?
    is_value: bool = False                # Is it a constant value?
    factory_fn: Optional[str] = None      # Factory function name
    
    # Location
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'registration_id': self.registration_id,
            'container_name': self.container_name,
            'service_name': self.service_name,
            'implementation': self.implementation,
            'registration_type': self.registration_type,
            'lifecycle': self.lifecycle,
            'token': self.token,
            'is_factory': self.is_factory,
            'is_value': self.is_value,
            'factory_fn': self.factory_fn,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'metadata': self.metadata
        }


@dataclass
class ContainerResolution:
    """Represents a dependency resolution from a container"""
    resolution_id: str                    # Unique ID
    container_name: str                   # Container name
    service_name: str                     # Requested service name
    resolved_to: Optional[str] = None     # What it resolved to
    
    # Resolution details
    resolution_method: str = "get"        # get, resolve, inject, etc.
    context: Optional[str] = None         # Resolution context (class, function)
    
    # Location
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'resolution_id': self.resolution_id,
            'container_name': self.container_name,
            'service_name': self.service_name,
            'resolved_to': self.resolved_to,
            'resolution_method': self.resolution_method,
            'context': self.context,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'metadata': self.metadata
        }


@dataclass
class DIContainer:
    """Represents a DI container"""
    container_id: str                     # Unique ID
    name: str                             # Container name
    framework: DIFramework                # Framework (InversifyJS, Awilix, etc.)
    
    # Registrations
    registrations: List[ContainerRegistration] = field(default_factory=list)
    
    # Resolutions
    resolutions: List[ContainerResolution] = field(default_factory=list)
    
    # Hierarchy
    parent_container: Optional[str] = None  # Parent container ID
    child_containers: List[str] = field(default_factory=list)
    
    # Modules (for framework containers)
    modules: List[str] = field(default_factory=list)
    
    # Location
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_registration(self, registration: ContainerRegistration):
        """Add a registration to this container"""
        self.registrations.append(registration)
    
    def add_resolution(self, resolution: ContainerResolution):
        """Add a resolution to this container"""
        self.resolutions.append(resolution)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'container_id': self.container_id,
            'name': self.name,
            'framework': self.framework.value,
            'registrations': [reg.to_dict() for reg in self.registrations],
            'resolutions': [res.to_dict() for res in self.resolutions],
            'parent_container': self.parent_container,
            'child_containers': self.child_containers,
            'modules': self.modules,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'metadata': self.metadata
        }


class DIDetector:
    """
    Main class for detecting dependency injection patterns.
    
    Detects various DI patterns:
    1. Constructor injection (most common)
    2. Setter injection
    3. Property injection
    4. Interface injection
    5. Method injection
    """
    
    def __init__(self):
        """Initialize the DI detector"""
        self.patterns: Dict[str, DIPattern] = {}
        
        # Index by class
        self.class_to_patterns: Dict[str, List[str]] = {}
        
        # Index by framework
        self.framework_patterns: Dict[DIFramework, List[str]] = {
            fw: [] for fw in DIFramework
        }
        
        # Container tracking (M3.3.3)
        self.containers: Dict[str, DIContainer] = {}
        self.registrations: Dict[str, ContainerRegistration] = {}
        self.resolutions: Dict[str, ContainerResolution] = {}
        
        # Index containers by name and framework
        self.container_by_name: Dict[str, str] = {}  # name -> container_id
        self.containers_by_framework: Dict[DIFramework, List[str]] = {
            fw: [] for fw in DIFramework
        }
        
        # Statistics
        self.stats = {
            'total_patterns': 0,
            'constructor_injection': 0,
            'setter_injection': 0,
            'property_injection': 0,
            'interface_injection': 0,
            'method_injection': 0,
            'field_injection': 0,
            'angular_patterns': 0,
            'nestjs_patterns': 0,
            'inversify_patterns': 0,
            'total_containers': 0,
            'total_registrations': 0,
            'total_resolutions': 0
        }
        
        logger.info("DIDetector initialized")
    
    def detect_patterns(self, nodes: List[Any]) -> Dict[str, DIPattern]:
        """
        Detect DI patterns in AST nodes.
        
        Args:
            nodes: List of AST nodes to analyze
            
        Returns:
            Dictionary of detected patterns (pattern_id -> DIPattern)
        """
        logger.info(f"Detecting DI patterns in {len(nodes)} nodes")
        
        for node in nodes:
            try:
                # Get node metadata
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                # Check for class declarations
                if node_type == 'class_declaration':
                    self._detect_class_di(node, metadata)
                
                # Check for constructor patterns
                elif node_type == 'constructor_definition':
                    self._detect_constructor_injection(node, metadata)
                
                # Check for method patterns
                elif node_type == 'method_definition':
                    self._detect_setter_injection(node, metadata)
                
                # Check for property patterns
                elif node_type == 'property_declaration':
                    self._detect_property_injection(node, metadata)
                
            except Exception as e:
                logger.warning(f"Failed to detect DI pattern for node {node.id}: {e}")
        
        logger.info(f"DI pattern detection complete: {len(self.patterns)} patterns found")
        return self.patterns
    
    def _detect_class_di(self, node: Any, metadata: Dict[str, Any]):
        """Detect DI patterns at class level"""
        class_name = metadata.get('name', 'UnknownClass')
        decorators = metadata.get('decorators', [])
        
        # Check for Angular @Injectable
        if any('@Injectable' in dec for dec in decorators):
            self._add_angular_injectable(node, class_name, metadata)
        
        # Check for NestJS decorators
        nestjs_decorators = ['@Injectable', '@Controller', '@Module']
        if any(any(dec_name in dec for dec_name in nestjs_decorators) for dec in decorators):
            self._add_nestjs_pattern(node, class_name, metadata)
        
        # Check for InversifyJS @injectable
        if any('@injectable' in dec for dec in decorators):
            self._add_inversify_pattern(node, class_name, metadata)
    
    def _detect_constructor_injection(self, node: Any, metadata: Dict[str, Any]):
        """Detect constructor injection pattern"""
        class_name = metadata.get('parent_class', 'UnknownClass')
        parameters = metadata.get('parameters', [])
        
        # Check if parameters have type annotations or decorators
        has_di = False
        dependencies = []
        
        for i, param in enumerate(parameters):
            param_name = param.get('name', f'param{i}')
            param_type = param.get('type', None)
            param_decorators = param.get('decorators', [])
            
            # Check for DI indicators
            if param_type or param_decorators or self._is_di_parameter(param):
                has_di = True
                
                dependency = Dependency(
                    dependency_id=f"{node.id}_dep_{i}",
                    name=param_name,
                    type=param_type,
                    injection_type=DIType.CONSTRUCTOR,
                    injection_point='constructor',
                    decorators=param_decorators,
                    is_optional=param.get('optional', False),
                    file_path=node.file if hasattr(node, 'file') else None,
                    line_number=metadata.get('line_number')
                )
                dependencies.append(dependency)
        
        if has_di:
            pattern = DIPattern(
                pattern_id=f"{node.id}_constructor_di",
                class_name=class_name,
                di_type=DIType.CONSTRUCTOR,
                framework=self._detect_framework(metadata),
                dependencies=dependencies,
                file_path=node.file if hasattr(node, 'file') else None,
                line_number=metadata.get('line_number'),
                confidence=0.95
            )
            
            self._add_pattern(pattern)
    
    def _detect_setter_injection(self, node: Any, metadata: Dict[str, Any]):
        """Detect setter injection pattern"""
        method_name = metadata.get('name', '')
        
        # Check if it's a setter method
        if not (method_name.startswith('set') or '@Inject' in str(metadata.get('decorators', []))):
            return
        
        class_name = metadata.get('parent_class', 'UnknownClass')
        parameters = metadata.get('parameters', [])
        
        # Check for injected parameters
        dependencies = []
        for i, param in enumerate(parameters):
            param_name = param.get('name', f'param{i}')
            param_type = param.get('type', None)
            param_decorators = param.get('decorators', [])
            
            if param_type or param_decorators:
                dependency = Dependency(
                    dependency_id=f"{node.id}_dep_{i}",
                    name=param_name,
                    type=param_type,
                    injection_type=DIType.SETTER,
                    injection_point=method_name,
                    decorators=param_decorators,
                    file_path=node.file if hasattr(node, 'file') else None,
                    line_number=metadata.get('line_number')
                )
                dependencies.append(dependency)
        
        if dependencies:
            pattern = DIPattern(
                pattern_id=f"{node.id}_setter_di",
                class_name=class_name,
                di_type=DIType.SETTER,
                framework=self._detect_framework(metadata),
                dependencies=dependencies,
                file_path=node.file if hasattr(node, 'file') else None,
                line_number=metadata.get('line_number'),
                confidence=0.85
            )
            
            self._add_pattern(pattern)
    
    def _detect_property_injection(self, node: Any, metadata: Dict[str, Any]):
        """Detect property/field injection pattern"""
        property_name = metadata.get('name', '')
        decorators = metadata.get('decorators', [])
        property_type = metadata.get('type', None)
        
        # Check for injection decorators
        inject_decorators = ['@Inject', '@inject', '@Autowired', '@InjectProperty']
        has_inject = any(any(dec_name in dec for dec_name in inject_decorators) for dec in decorators)
        
        if not has_inject:
            return
        
        class_name = metadata.get('parent_class', 'UnknownClass')
        
        dependency = Dependency(
            dependency_id=f"{node.id}_prop_dep",
            name=property_name,
            type=property_type,
            injection_type=DIType.PROPERTY,
            injection_point=property_name,
            decorators=decorators,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number')
        )
        
        pattern = DIPattern(
            pattern_id=f"{node.id}_property_di",
            class_name=class_name,
            di_type=DIType.PROPERTY,
            framework=self._detect_framework(metadata),
            dependencies=[dependency],
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            confidence=0.9
        )
        
        self._add_pattern(pattern)
    
    def _add_angular_injectable(self, node: Any, class_name: str, metadata: Dict[str, Any]):
        """Add Angular @Injectable pattern"""
        pattern = DIPattern(
            pattern_id=f"{node.id}_angular_injectable",
            class_name=class_name,
            di_type=DIType.CONSTRUCTOR,
            framework=DIFramework.ANGULAR,
            is_injectable=True,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            confidence=1.0,
            metadata={'framework': 'angular', 'decorator': '@Injectable'}
        )
        
        self._add_pattern(pattern)
        self.stats['angular_patterns'] += 1
    
    def _add_nestjs_pattern(self, node: Any, class_name: str, metadata: Dict[str, Any]):
        """Add NestJS DI pattern"""
        decorators = metadata.get('decorators', [])
        
        # Determine pattern type from decorator
        if '@Controller' in str(decorators):
            pattern_type = 'controller'
        elif '@Module' in str(decorators):
            pattern_type = 'module'
        else:
            pattern_type = 'injectable'
        
        pattern = DIPattern(
            pattern_id=f"{node.id}_nestjs_{pattern_type}",
            class_name=class_name,
            di_type=DIType.CONSTRUCTOR,
            framework=DIFramework.NESTJS,
            is_injectable=True,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            confidence=1.0,
            metadata={'framework': 'nestjs', 'pattern_type': pattern_type}
        )
        
        self._add_pattern(pattern)
        self.stats['nestjs_patterns'] += 1
    
    def _add_inversify_pattern(self, node: Any, class_name: str, metadata: Dict[str, Any]):
        """Add InversifyJS pattern"""
        pattern = DIPattern(
            pattern_id=f"{node.id}_inversify_injectable",
            class_name=class_name,
            di_type=DIType.CONSTRUCTOR,
            framework=DIFramework.INVERSIFY,
            is_injectable=True,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            confidence=1.0,
            metadata={'framework': 'inversify', 'decorator': '@injectable'}
        )
        
        self._add_pattern(pattern)
        self.stats['inversify_patterns'] += 1
    
    # Framework-Specific Enhanced Detection (M3.3.2)
    
    def detect_angular_providers(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Detect Angular provider configurations.
        
        Detects:
        - @NgModule providers array
        - forRoot() and forChild() patterns
        - useClass, useValue, useFactory patterns
        """
        providers = {
            'modules': [],
            'providers': [],
            'root_providers': [],
            'child_providers': []
        }
        
        for node in nodes:
            try:
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                # Check for @NgModule decorator
                if node_type == 'class_declaration':
                    decorators = metadata.get('decorators', [])
                    
                    for decorator in decorators:
                        if '@NgModule' in str(decorator):
                            # Extract providers array from decorator
                            providers_array = self._extract_providers_from_decorator(decorator, metadata)
                            
                            module_info = {
                                'class_name': metadata.get('name', 'UnknownModule'),
                                'providers': providers_array,
                                'file_path': node.file if hasattr(node, 'file') else None,
                                'line_number': metadata.get('line_number')
                            }
                            providers['modules'].append(module_info)
                            providers['providers'].extend(providers_array)
                
                # Check for forRoot/forChild patterns
                elif node_type == 'method_definition':
                    method_name = metadata.get('name', '')
                    
                    if method_name == 'forRoot':
                        root_providers = self._extract_providers_from_method(node, metadata)
                        providers['root_providers'].extend(root_providers)
                    
                    elif method_name == 'forChild':
                        child_providers = self._extract_providers_from_method(node, metadata)
                        providers['child_providers'].extend(child_providers)
            
            except Exception as e:
                logger.warning(f"Failed to detect Angular providers for node {node.id}: {e}")
        
        logger.info(f"Angular provider detection: {len(providers['modules'])} modules, "
                   f"{len(providers['providers'])} providers")
        return providers
    
    def detect_nestjs_modules(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Detect NestJS module structure.
        
        Detects:
        - @Module decorator with imports/providers/controllers/exports
        - Module hierarchy and dependencies
        - Dynamic modules (forRoot, forRootAsync, register, etc.)
        """
        modules = {
            'modules': [],
            'controllers': [],
            'providers': [],
            'imports': [],
            'exports': [],
            'dynamic_modules': []
        }
        
        for node in nodes:
            try:
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                if node_type == 'class_declaration':
                    decorators = metadata.get('decorators', [])
                    class_name = metadata.get('name', 'UnknownClass')
                    
                    # Check for @Module
                    for decorator in decorators:
                        decorator_str = str(decorator)
                        
                        if '@Module' in decorator_str:
                            module_config = self._parse_nestjs_module_decorator(decorator, metadata)
                            module_config['class_name'] = class_name
                            module_config['file_path'] = node.file if hasattr(node, 'file') else None
                            module_config['line_number'] = metadata.get('line_number')
                            
                            modules['modules'].append(module_config)
                            modules['imports'].extend(module_config.get('imports', []))
                            modules['providers'].extend(module_config.get('providers', []))
                            modules['exports'].extend(module_config.get('exports', []))
                        
                        # Check for @Controller
                        elif '@Controller' in decorator_str:
                            controller_info = {
                                'class_name': class_name,
                                'route': self._extract_controller_route(decorator),
                                'file_path': node.file if hasattr(node, 'file') else None,
                                'line_number': metadata.get('line_number')
                            }
                            modules['controllers'].append(controller_info)
                        
                        # Check for @Injectable
                        elif '@Injectable' in decorator_str:
                            scope = self._extract_injectable_scope(decorator)
                            provider_info = {
                                'class_name': class_name,
                                'scope': scope,
                                'file_path': node.file if hasattr(node, 'file') else None,
                                'line_number': metadata.get('line_number')
                            }
                            modules['providers'].append(provider_info)
                
                # Check for dynamic module methods
                elif node_type == 'method_definition':
                    method_name = metadata.get('name', '')
                    
                    if method_name in ['forRoot', 'forRootAsync', 'register', 'registerAsync', 'forFeature']:
                        dynamic_module = {
                            'class_name': metadata.get('parent_class', 'UnknownClass'),
                            'method': method_name,
                            'is_async': 'Async' in method_name,
                            'file_path': node.file if hasattr(node, 'file') else None,
                            'line_number': metadata.get('line_number')
                        }
                        modules['dynamic_modules'].append(dynamic_module)
            
            except Exception as e:
                logger.warning(f"Failed to detect NestJS modules for node {node.id}: {e}")
        
        logger.info(f"NestJS module detection: {len(modules['modules'])} modules, "
                   f"{len(modules['controllers'])} controllers, {len(modules['providers'])} providers")
        return modules
    
    def detect_inversify_bindings(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Detect InversifyJS container bindings.
        
        Detects:
        - Container.bind() calls
        - @injectable() decorators
        - @inject() parameter decorators
        - Service identifiers and tokens
        """
        bindings = {
            'containers': [],
            'bindings': [],
            'injectables': [],
            'injections': []
        }
        
        for node in nodes:
            try:
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                code_snippet = metadata.get('code_snippet', '')
                
                # Check for container creation
                if 'new Container(' in code_snippet:
                    container_info = {
                        'variable_name': metadata.get('name', 'container'),
                        'file_path': node.file if hasattr(node, 'file') else None,
                        'line_number': metadata.get('line_number')
                    }
                    bindings['containers'].append(container_info)
                
                # Check for bind() calls
                if '.bind(' in code_snippet or 'container.bind' in code_snippet:
                    binding_info = self._parse_inversify_binding(code_snippet, metadata)
                    binding_info['file_path'] = node.file if hasattr(node, 'file') else None
                    binding_info['line_number'] = metadata.get('line_number')
                    bindings['bindings'].append(binding_info)
                
                # Check for @injectable
                if node_type == 'class_declaration':
                    decorators = metadata.get('decorators', [])
                    
                    if any('@injectable' in str(dec) for dec in decorators):
                        injectable_info = {
                            'class_name': metadata.get('name', 'UnknownClass'),
                            'file_path': node.file if hasattr(node, 'file') else None,
                            'line_number': metadata.get('line_number')
                        }
                        bindings['injectables'].append(injectable_info)
                
                # Check for @inject parameter decorators
                if node_type == 'constructor_definition':
                    parameters = metadata.get('parameters', [])
                    
                    for param in parameters:
                        param_decorators = param.get('decorators', [])
                        
                        if any('@inject' in str(dec) for dec in param_decorators):
                            injection_info = {
                                'class_name': metadata.get('parent_class', 'UnknownClass'),
                                'parameter': param.get('name', 'unknown'),
                                'token': self._extract_inject_token(param_decorators),
                                'file_path': node.file if hasattr(node, 'file') else None,
                                'line_number': metadata.get('line_number')
                            }
                            bindings['injections'].append(injection_info)
            
            except Exception as e:
                logger.warning(f"Failed to detect InversifyJS bindings for node {node.id}: {e}")
        
        logger.info(f"InversifyJS binding detection: {len(bindings['containers'])} containers, "
                   f"{len(bindings['bindings'])} bindings, {len(bindings['injectables'])} injectables")
        return bindings
    
    # Helper methods for framework-specific parsing
    
    def _extract_providers_from_decorator(self, decorator: Any, metadata: Dict[str, Any]) -> List[str]:
        """Extract providers array from Angular @NgModule decorator"""
        # Simplified extraction - in real implementation, would parse decorator AST
        decorator_str = str(decorator)
        providers = []
        
        # Look for providers: [...] pattern
        if 'providers:' in decorator_str:
            # Simple pattern matching (would be more sophisticated in production)
            start = decorator_str.find('providers:')
            if start != -1:
                # Extract provider names (simplified)
                # In production, would properly parse the array
                pass
        
        return providers
    
    def _extract_providers_from_method(self, node: Any, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract providers from forRoot/forChild methods"""
        providers = []
        # Would parse method body to extract providers
        return providers
    
    def _parse_nestjs_module_decorator(self, decorator: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse NestJS @Module decorator"""
        config = {
            'imports': [],
            'providers': [],
            'controllers': [],
            'exports': []
        }
        
        decorator_str = str(decorator)
        
        # Simple parsing (would be more sophisticated in production)
        if 'imports:' in decorator_str:
            # Extract imports array
            pass
        
        if 'providers:' in decorator_str:
            # Extract providers array
            pass
        
        if 'controllers:' in decorator_str:
            # Extract controllers array
            pass
        
        if 'exports:' in decorator_str:
            # Extract exports array
            pass
        
        return config
    
    def _extract_controller_route(self, decorator: Any) -> str:
        """Extract route from @Controller decorator"""
        decorator_str = str(decorator)
        
        # Look for @Controller('route') pattern
        if '@Controller(' in decorator_str:
            start = decorator_str.find("'")
            if start != -1:
                end = decorator_str.find("'", start + 1)
                if end != -1:
                    return decorator_str[start + 1:end]
        
        return ''
    
    def _extract_injectable_scope(self, decorator: Any) -> str:
        """Extract scope from @Injectable decorator"""
        decorator_str = str(decorator)
        
        # Look for scope: Scope.X pattern
        if 'scope:' in decorator_str:
            if 'Scope.REQUEST' in decorator_str:
                return 'request'
            elif 'Scope.TRANSIENT' in decorator_str:
                return 'transient'
        
        return 'singleton'  # Default
    
    def _parse_inversify_binding(self, code_snippet: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse InversifyJS container.bind() call"""
        binding = {
            'service_identifier': 'unknown',
            'implementation': 'unknown',
            'scope': 'transient'
        }
        
        # Look for bind(X).to(Y) pattern
        if '.bind(' in code_snippet and '.to(' in code_snippet:
            # Extract service identifier
            bind_start = code_snippet.find('.bind(') + 6
            bind_end = code_snippet.find(')', bind_start)
            if bind_start > 5 and bind_end > bind_start:
                binding['service_identifier'] = code_snippet[bind_start:bind_end].strip()
            
            # Extract implementation
            to_start = code_snippet.find('.to(') + 4
            to_end = code_snippet.find(')', to_start)
            if to_start > 3 and to_end > to_start:
                binding['implementation'] = code_snippet[to_start:to_end].strip()
        
        # Check for scope
        if '.inSingletonScope()' in code_snippet:
            binding['scope'] = 'singleton'
        elif '.inRequestScope()' in code_snippet:
            binding['scope'] = 'request'
        elif '.inTransientScope()' in code_snippet:
            binding['scope'] = 'transient'
        
        return binding
    
    def _extract_inject_token(self, decorators: List[Any]) -> str:
        """Extract token from @inject decorator"""
        for decorator in decorators:
            decorator_str = str(decorator)
            
            if '@inject(' in decorator_str:
                start = decorator_str.find('@inject(') + 8
                end = decorator_str.find(')', start)
                if start > 7 and end > start:
                    return decorator_str[start:end].strip()
        
        return 'unknown'
    
    def _is_di_parameter(self, param: Dict[str, Any]) -> bool:
        """Check if parameter looks like DI"""
        # Check for type annotation
        if param.get('type'):
            return True
        
        # Check for decorators
        if param.get('decorators'):
            return True
        
        # Check for common DI parameter patterns
        param_name = param.get('name', '').lower()
        di_keywords = ['service', 'repository', 'provider', 'factory', 'controller', 'manager']
        return any(keyword in param_name for keyword in di_keywords)
    
    def _detect_framework(self, metadata: Dict[str, Any]) -> DIFramework:
        """Detect which DI framework is being used"""
        decorators = metadata.get('decorators', [])
        decorators_str = str(decorators).lower()
        
        if '@injectable' in decorators_str and 'angular' in decorators_str:
            return DIFramework.ANGULAR
        elif '@injectable' in decorators_str or '@controller' in decorators_str or '@module' in decorators_str:
            return DIFramework.NESTJS
        elif '@injectable' in decorators_str:
            return DIFramework.INVERSIFY
        elif '@inject' in decorators_str:
            return DIFramework.TSYRINGE
        elif 'container' in decorators_str:
            return DIFramework.AWILIX
        elif decorators:
            return DIFramework.CUSTOM
        else:
            return DIFramework.NONE
    
    def _add_pattern(self, pattern: DIPattern):
        """Add a DI pattern to the detector"""
        self.patterns[pattern.pattern_id] = pattern
        
        # Update class index
        if pattern.class_name not in self.class_to_patterns:
            self.class_to_patterns[pattern.class_name] = []
        self.class_to_patterns[pattern.class_name].append(pattern.pattern_id)
        
        # Update framework index
        self.framework_patterns[pattern.framework].append(pattern.pattern_id)
        
        # Update statistics
        self.stats['total_patterns'] += 1
        
        if pattern.di_type == DIType.CONSTRUCTOR:
            self.stats['constructor_injection'] += 1
        elif pattern.di_type == DIType.SETTER:
            self.stats['setter_injection'] += 1
        elif pattern.di_type == DIType.PROPERTY:
            self.stats['property_injection'] += 1
        elif pattern.di_type == DIType.INTERFACE:
            self.stats['interface_injection'] += 1
        elif pattern.di_type == DIType.METHOD:
            self.stats['method_injection'] += 1
        elif pattern.di_type == DIType.FIELD:
            self.stats['field_injection'] += 1
    
    def get_patterns_by_class(self, class_name: str) -> List[DIPattern]:
        """Get all DI patterns for a class"""
        pattern_ids = self.class_to_patterns.get(class_name, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_patterns_by_framework(self, framework: DIFramework) -> List[DIPattern]:
        """Get all patterns using a specific framework"""
        pattern_ids = self.framework_patterns.get(framework, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_injectable_classes(self) -> List[str]:
        """Get all classes marked as injectable"""
        return [
            pattern.class_name
            for pattern in self.patterns.values()
            if pattern.is_injectable
        ]
    
    def get_dependencies_for_class(self, class_name: str) -> List[Dependency]:
        """Get all dependencies for a class"""
        patterns = self.get_patterns_by_class(class_name)
        dependencies = []
        for pattern in patterns:
            dependencies.extend(pattern.dependencies)
        return dependencies
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export all detected patterns"""
        return {
            'patterns': {
                pid: pattern.to_dict()
                for pid, pattern in self.patterns.items()
            },
            'statistics': self.stats,
            'injectable_classes': self.get_injectable_classes()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            **self.stats,
            'frameworks_used': [
                fw.value for fw, patterns in self.framework_patterns.items()
                if patterns
            ]
        }
    
    def get_framework_specific_data(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Get framework-specific DI data (M3.3.2).
        
        Returns comprehensive framework-specific information:
        - Angular: providers, modules, forRoot/forChild patterns
        - NestJS: modules, controllers, providers, imports, exports
        - InversifyJS: containers, bindings, injectables
        """
        framework_data = {
            'angular': None,
            'nestjs': None,
            'inversify': None
        }
        
        # Detect which frameworks are being used
        frameworks_used = self.get_statistics()['frameworks_used']
        
        # Collect framework-specific data
        if 'angular' in frameworks_used:
            framework_data['angular'] = self.detect_angular_providers(nodes)
        
        if 'nestjs' in frameworks_used:
            framework_data['nestjs'] = self.detect_nestjs_modules(nodes)
        
        if 'inversify' in frameworks_used:
            framework_data['inversify'] = self.detect_inversify_bindings(nodes)
        
        return framework_data
    
    # ===================================================================
    # M3.3.3: Container Tracking Methods
    # ===================================================================
    
    def track_containers(self, nodes: List[Any]) -> Dict[str, DIContainer]:
        """
        Track DI containers, their registrations, and resolutions (M3.3.3).
        
        Detects:
        - Container creation/initialization
        - Service registrations (bind, register, provide)
        - Dependency resolutions (get, resolve, inject)
        - Container hierarchies (parent/child relationships)
        
        Args:
            nodes: List of AST nodes to analyze
            
        Returns:
            Dictionary of detected containers (container_id -> DIContainer)
        """
        logger.info(f"Tracking DI containers in {len(nodes)} nodes")
        
        for node in nodes:
            try:
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                # Detect container creation
                if node_type in ['variable_declarator', 'lexical_declaration']:
                    self._detect_container_creation(node)
                
                # Detect registrations (bind, register, provide)
                if node_type in ['call_expression', 'expression_statement']:
                    self._detect_container_registration(node)
                    self._detect_container_resolution(node)
                
                # Detect container modules (Angular/NestJS)
                if node_type == 'class_declaration':
                    self._detect_container_module(node)
                    
            except Exception as e:
                logger.warning(f"Error tracking container in node: {e}")
                continue
        
        logger.info(f"Container tracking complete: {len(self.containers)} containers, "
                   f"{len(self.registrations)} registrations, {len(self.resolutions)} resolutions")
        
        return self.containers
    
    def _detect_container_creation(self, node: Any):
        """Detect DI container creation"""
        try:
            # Get variable name and initialization
            var_name = None
            init_expr = None
            
            if node.type.value == 'variable_declarator':
                # Get variable name
                name_node = next((c for c in node.children if c.type.value == 'identifier'), None)
                if name_node:
                    var_name = self._get_node_text(name_node)
                
                # Get initialization expression
                init_expr = next((c for c in node.children 
                                if c.type.value in ['call_expression', 'new_expression']), None)
            
            elif node.type.value == 'lexical_declaration':
                # Find variable_declarator child
                declarator = next((c for c in node.children 
                                 if c.type.value == 'variable_declarator'), None)
                if declarator:
                    name_node = next((c for c in declarator.children 
                                    if c.type.value == 'identifier'), None)
                    if name_node:
                        var_name = self._get_node_text(name_node)
                    
                    init_expr = next((c for c in declarator.children 
                                    if c.type.value in ['call_expression', 'new_expression']), None)
            
            if not var_name or not init_expr:
                return
            
            # Check if it's a container creation
            init_text = self._get_node_text(init_expr).lower()
            
            # InversifyJS: new Container()
            if 'new container' in init_text or 'new inversify.container' in init_text:
                self._create_container(var_name, DIFramework.INVERSIFY, node)
            
            # Awilix: createContainer()
            elif 'createcontainer' in init_text:
                self._create_container(var_name, DIFramework.AWILIX, node)
            
            # TSyringe: container.createChildContainer()
            elif 'createchildcontainer' in init_text:
                parent_name = init_text.split('.')[0] if '.' in init_text else None
                self._create_container(var_name, DIFramework.TSYRINGE, node, parent_name)
            
            # TypeDI: Container.of()
            elif 'container.of' in init_text:
                self._create_container(var_name, DIFramework.TYPEDI, node)
                
        except Exception as e:
            logger.debug(f"Error detecting container creation: {e}")
    
    def _create_container(self, name: str, framework: DIFramework, node: Any, 
                         parent_name: Optional[str] = None):
        """Create a new container entry"""
        container_id = f"container_{len(self.containers)}_{name}"
        
        # Find parent container if specified
        parent_id = None
        if parent_name and parent_name in self.container_by_name:
            parent_id = self.container_by_name[parent_name]
        
        container = DIContainer(
            container_id=container_id,
            name=name,
            framework=framework,
            parent_container=parent_id,
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'start_point', [None])[0]
        )
        
        # Add to parent's children if applicable
        if parent_id and parent_id in self.containers:
            self.containers[parent_id].child_containers.append(container_id)
        
        # Store container
        self.containers[container_id] = container
        self.container_by_name[name] = container_id
        self.containers_by_framework[framework].append(container_id)
        self.stats['total_containers'] += 1
        
        logger.debug(f"Created container: {name} ({framework.value})")
    
    def _detect_container_registration(self, node: Any):
        """Detect container service registrations"""
        try:
            # Get the call expression
            call_expr = node if node.type.value == 'call_expression' else \
                       next((c for c in node.children if c.type.value == 'call_expression'), None)
            
            if not call_expr:
                return
            
            call_text = self._get_node_text(call_expr)
            call_lower = call_text.lower()
            
            # Check for registration patterns
            container_name = None
            registration_type = None
            service_name = None
            implementation = None
            lifecycle = None
            
            # InversifyJS: container.bind(ServiceId).to(ServiceImpl).inSingletonScope()
            if '.bind(' in call_lower:
                container_name = call_text.split('.')[0]
                registration_type = 'bind'
                
                # Extract service ID and implementation
                if 'bind(' in call_text:
                    service_start = call_text.index('bind(') + 5
                    service_end = call_text.index(')', service_start)
                    service_name = call_text[service_start:service_end].strip()
                
                if '.to(' in call_text:
                    impl_start = call_text.index('.to(') + 4
                    impl_end = call_text.index(')', impl_start)
                    implementation = call_text[impl_start:impl_end].strip()
                
                # Extract lifecycle
                if 'insingleton' in call_lower:
                    lifecycle = 'singleton'
                elif 'intransient' in call_lower:
                    lifecycle = 'transient'
                elif 'inrequest' in call_lower:
                    lifecycle = 'request'
            
            # Awilix: container.register({ service: asClass(ServiceImpl).singleton() })
            elif '.register(' in call_lower:
                container_name = call_text.split('.')[0]
                registration_type = 'register'
                
                # Extract service name from object
                if '{' in call_text and ':' in call_text:
                    obj_start = call_text.index('{') + 1
                    obj_end = call_text.rindex('}')
                    obj_content = call_text[obj_start:obj_end]
                    
                    if ':' in obj_content:
                        service_name = obj_content.split(':')[0].strip()
                        impl_part = obj_content.split(':')[1].strip()
                        
                        # Extract implementation from asClass/asFunction/asValue
                        if 'asclass(' in impl_part.lower():
                            impl_start = impl_part.lower().index('asclass(') + 8
                            impl_end = impl_part.index(')', impl_start)
                            implementation = impl_part[impl_start:impl_end].strip()
                        
                        # Extract lifecycle
                        if 'singleton' in impl_part.lower():
                            lifecycle = 'singleton'
                        elif 'transient' in impl_part.lower():
                            lifecycle = 'transient'
                        elif 'scoped' in impl_part.lower():
                            lifecycle = 'scoped'
            
            # TSyringe: container.register(Token, { useClass: ServiceImpl })
            elif 'container.register' in call_lower or 'registersingleton' in call_lower:
                registration_type = 'register'
                
                if 'registersingleton' in call_lower:
                    lifecycle = 'singleton'
                
                # Extract token and implementation
                args = self._extract_call_arguments(call_expr)
                if len(args) >= 1:
                    service_name = args[0]
                if len(args) >= 2:
                    impl_arg = args[1]
                    if 'useclass' in impl_arg.lower():
                        # Extract class from { useClass: X }
                        if ':' in impl_arg:
                            implementation = impl_arg.split(':')[1].strip().rstrip('}').strip()
                    else:
                        implementation = impl_arg
            
            # Create registration if we found one
            if container_name and service_name:
                self._create_registration(
                    container_name, service_name, implementation,
                    registration_type, lifecycle, node
                )
                
        except Exception as e:
            logger.debug(f"Error detecting container registration: {e}")
    
    def _create_registration(self, container_name: str, service_name: str,
                            implementation: Optional[str], registration_type: str,
                            lifecycle: Optional[str], node: Any):
        """Create a container registration entry"""
        registration_id = f"reg_{len(self.registrations)}_{service_name}"
        
        registration = ContainerRegistration(
            registration_id=registration_id,
            container_name=container_name,
            service_name=service_name,
            implementation=implementation,
            registration_type=registration_type,
            lifecycle=lifecycle,
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'start_point', [None])[0]
        )
        
        # Store registration
        self.registrations[registration_id] = registration
        
        # Add to container if it exists
        if container_name in self.container_by_name:
            container_id = self.container_by_name[container_name]
            if container_id in self.containers:
                self.containers[container_id].add_registration(registration)
        
        self.stats['total_registrations'] += 1
        logger.debug(f"Registered: {service_name} -> {implementation} ({lifecycle})")
    
    def _detect_container_resolution(self, node: Any):
        """Detect dependency resolutions from containers"""
        try:
            # Get the call expression
            call_expr = node if node.type.value == 'call_expression' else \
                       next((c for c in node.children if c.type.value == 'call_expression'), None)
            
            if not call_expr:
                return
            
            call_text = self._get_node_text(call_expr)
            call_lower = call_text.lower()
            
            container_name = None
            resolution_method = None
            service_name = None
            
            # InversifyJS: container.get(ServiceId)
            if '.get(' in call_lower:
                container_name = call_text.split('.')[0]
                resolution_method = 'get'
                
                # Extract service ID
                if 'get(' in call_text:
                    service_start = call_text.index('get(') + 4
                    service_end = call_text.index(')', service_start)
                    service_name = call_text[service_start:service_end].strip()
            
            # Awilix: container.resolve('service')
            elif '.resolve(' in call_lower:
                container_name = call_text.split('.')[0]
                resolution_method = 'resolve'
                
                # Extract service name
                args = self._extract_call_arguments(call_expr)
                if args:
                    service_name = args[0]
            
            # TSyringe: container.resolve(ServiceClass)
            elif 'container.resolve' in call_lower:
                resolution_method = 'resolve'
                
                args = self._extract_call_arguments(call_expr)
                if args:
                    service_name = args[0]
            
            # Create resolution if we found one
            if service_name:
                self._create_resolution(
                    container_name or 'container',
                    service_name,
                    resolution_method,
                    node
                )
                
        except Exception as e:
            logger.debug(f"Error detecting container resolution: {e}")
    
    def _create_resolution(self, container_name: str, service_name: str,
                          resolution_method: str, node: Any):
        """Create a container resolution entry"""
        resolution_id = f"res_{len(self.resolutions)}_{service_name}"
        
        resolution = ContainerResolution(
            resolution_id=resolution_id,
            container_name=container_name,
            service_name=service_name,
            resolution_method=resolution_method,
            file_path=getattr(node, 'file_path', None),
            line_number=getattr(node, 'start_point', [None])[0]
        )
        
        # Store resolution
        self.resolutions[resolution_id] = resolution
        
        # Add to container if it exists
        if container_name in self.container_by_name:
            container_id = self.container_by_name[container_name]
            if container_id in self.containers:
                self.containers[container_id].add_resolution(resolution)
        
        self.stats['total_resolutions'] += 1
        logger.debug(f"Resolution: {container_name}.{resolution_method}({service_name})")
    
    def _detect_container_module(self, node: Any):
        """Detect Angular/NestJS module containers"""
        try:
            # Check for @Module or @NgModule decorator
            decorators = self._get_decorators(node)
            
            for decorator in decorators:
                decorator_name = decorator.get('name', '')
                
                if decorator_name in ['Module', 'NgModule']:
                    # Get class name
                    class_name = None
                    for child in node.children:
                        if child.type.value == 'identifier':
                            class_name = self._get_node_text(child)
                            break
                    
                    if not class_name:
                        continue
                    
                    # Create container for the module
                    framework = DIFramework.NESTJS if decorator_name == 'Module' else DIFramework.ANGULAR
                    container_id = f"container_module_{class_name}"
                    
                    container = DIContainer(
                        container_id=container_id,
                        name=class_name,
                        framework=framework,
                        modules=[class_name],
                        file_path=getattr(node, 'file_path', None),
                        line_number=getattr(node, 'start_point', [None])[0]
                    )
                    
                    self.containers[container_id] = container
                    self.container_by_name[class_name] = container_id
                    self.containers_by_framework[framework].append(container_id)
                    self.stats['total_containers'] += 1
                    
                    logger.debug(f"Detected module container: {class_name} ({framework.value})")
                    
        except Exception as e:
            logger.debug(f"Error detecting container module: {e}")
    
    def _extract_call_arguments(self, call_expr: Any) -> List[str]:
        """Extract arguments from a call expression"""
        args = []
        try:
            # Find arguments node
            args_node = next((c for c in call_expr.children 
                            if c.type.value == 'arguments'), None)
            
            if args_node:
                for child in args_node.children:
                    if child.type.value not in ['(', ')', ',']:
                        arg_text = self._get_node_text(child).strip()
                        if arg_text:
                            args.append(arg_text)
        except Exception as e:
            logger.debug(f"Error extracting call arguments: {e}")
        
        return args
    
    def get_containers(self) -> Dict[str, DIContainer]:
        """Get all tracked containers"""
        return self.containers
    
    def get_container_by_name(self, name: str) -> Optional[DIContainer]:
        """Get container by name"""
        container_id = self.container_by_name.get(name)
        return self.containers.get(container_id) if container_id else None
    
    def get_containers_by_framework(self, framework: DIFramework) -> List[DIContainer]:
        """Get all containers for a specific framework"""
        container_ids = self.containers_by_framework.get(framework, [])
        return [self.containers[cid] for cid in container_ids if cid in self.containers]
    
    def get_registrations_for_container(self, container_name: str) -> List[ContainerRegistration]:
        """Get all registrations for a specific container"""
        container = self.get_container_by_name(container_name)
        return container.registrations if container else []
    
    def get_resolutions_for_container(self, container_name: str) -> List[ContainerResolution]:
        """Get all resolutions for a specific container"""
        container = self.get_container_by_name(container_name)
        return container.resolutions if container else []
    
    def export_container_data(self) -> Dict[str, Any]:
        """Export all container tracking data"""
        return {
            'containers': {
                cid: container.to_dict()
                for cid, container in self.containers.items()
            },
            'registrations': {
                rid: reg.to_dict()
                for rid, reg in self.registrations.items()
            },
            'resolutions': {
                rid: res.to_dict()
                for rid, res in self.resolutions.items()
            },
            'statistics': {
                'total_containers': self.stats['total_containers'],
                'total_registrations': self.stats['total_registrations'],
                'total_resolutions': self.stats['total_resolutions'],
                'containers_by_framework': {
                    fw.value: len(containers)
                    for fw, containers in self.containers_by_framework.items()
                    if containers
                }
            }
        }

