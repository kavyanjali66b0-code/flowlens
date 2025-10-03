"""
Dependency Injection (DI) Analyzer for M4.0 Phase 3

This module analyzes dependency injection patterns in JavaScript/TypeScript code:
- Detects component prop dependencies
- Tracks service/context injection patterns
- Identifies constructor injection (classes)
- Analyzes function parameter injection
- Detects React Context providers and consumers
- Tracks prop drilling patterns

The analyzer provides metadata about DI patterns for:
- Architecture visualization
- Dependency graphs
- Code quality analysis
- Refactoring suggestions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DIType(Enum):
    """Types of dependency injection"""
    PROP = "prop"                      # React prop injection
    CONTEXT = "context"                # React Context injection
    HOOK = "hook"                      # Custom hook injection
    PARAMETER = "parameter"            # Function parameter injection
    CONSTRUCTOR = "constructor"        # Class constructor injection
    IMPORT = "import"                  # Module import injection
    SERVICE = "service"                # Service injection pattern
    HOC = "hoc"                        # Higher-Order Component injection


class DICategory(Enum):
    """Categories of dependencies"""
    DATA = "data"                      # Data dependencies
    FUNCTION = "function"              # Function dependencies
    SERVICE = "service"                # Service dependencies
    CONFIGURATION = "configuration"    # Configuration dependencies
    COMPONENT = "component"            # Component dependencies
    CONTEXT = "context"                # Context dependencies


@dataclass
class Dependency:
    """Represents a single dependency"""
    name: str                          # Dependency name
    di_type: DIType                    # Type of injection
    di_category: DICategory            # Category of dependency
    
    # Source information
    source_id: Optional[str] = None    # Where dependency comes from
    source_type: Optional[str] = None  # Type of source (component, service, etc.)
    
    # Type information
    data_type: Optional[str] = None    # Data type of dependency
    is_optional: bool = False          # Is dependency optional?
    has_default: bool = False          # Has default value?
    default_value: Optional[str] = None
    
    # Usage information
    line_number: Optional[int] = None
    used_in: List[str] = field(default_factory=list)  # Where dependency is used
    
    # Metadata
    confidence: float = 1.0            # Confidence in detection
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'di_type': self.di_type.value,
            'di_category': self.di_category.value,
            'source_id': self.source_id,
            'source_type': self.source_type,
            'data_type': self.data_type,
            'is_optional': self.is_optional,
            'has_default': self.has_default,
            'default_value': self.default_value,
            'line_number': self.line_number,
            'used_in': self.used_in,
            'confidence': self.confidence,
            'context': self.context
        }


@dataclass
class DIEntity:
    """Represents an entity that has dependencies (component, function, class)"""
    entity_id: str                     # Unique identifier
    entity_name: str                   # Entity name
    entity_type: str                   # Type (component, function, class, etc.)
    
    # Dependencies
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Provides (for Context providers, services)
    provides: List[str] = field(default_factory=list)
    
    # Location
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    
    # Metadata
    is_provider: bool = False          # Is this a provider?
    is_consumer: bool = False          # Is this a consumer?
    
    def add_dependency(self, dependency: Dependency):
        """Add a dependency to this entity"""
        self.dependencies.append(dependency)
        self.is_consumer = True
    
    def add_provides(self, provided: str):
        """Add a provided dependency"""
        if provided not in self.provides:
            self.provides.append(provided)
            self.is_provider = True
    
    def get_dependency_count(self) -> int:
        """Get total number of dependencies"""
        return len(self.dependencies)
    
    def get_dependencies_by_type(self, di_type: DIType) -> List[Dependency]:
        """Get dependencies of a specific type"""
        return [d for d in self.dependencies if d.di_type == di_type]
    
    def get_dependencies_by_category(self, category: DICategory) -> List[Dependency]:
        """Get dependencies of a specific category"""
        return [d for d in self.dependencies if d.di_category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'dependencies': [d.to_dict() for d in self.dependencies],
            'provides': self.provides,
            'line_number': self.line_number,
            'file_path': self.file_path,
            'is_provider': self.is_provider,
            'is_consumer': self.is_consumer,
            'dependency_count': self.get_dependency_count()
        }


class DIAnalyzer:
    """
    Analyzes dependency injection patterns in JavaScript/TypeScript code.
    
    This analyzer identifies how dependencies are injected into components,
    functions, and classes, enabling visualization of dependency graphs.
    """
    
    def __init__(self):
        """Initialize the DI analyzer"""
        self.entities: Dict[str, DIEntity] = {}  # entity_id -> DIEntity
        
        # Tracking
        self.current_entity: Optional[str] = None
        self.context_providers: Dict[str, str] = {}  # context_name -> provider_id
        self.context_consumers: Dict[str, List[str]] = {}  # context_name -> [consumer_ids]
        
        # Statistics
        self.stats = {
            'total_entities': 0,
            'total_dependencies': 0,
            'prop_dependencies': 0,
            'context_dependencies': 0,
            'hook_dependencies': 0,
            'providers': 0,
            'consumers': 0
        }
    
    def register_entity(self,
                       entity_id: str,
                       entity_name: str,
                       entity_type: str,
                       line_number: Optional[int] = None,
                       file_path: Optional[str] = None) -> DIEntity:
        """Register a new entity (component, function, class)"""
        if entity_id not in self.entities:
            entity = DIEntity(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type,
                line_number=line_number,
                file_path=file_path
            )
            self.entities[entity_id] = entity
            self.stats['total_entities'] += 1
            logger.debug(f"Registered DI entity: {entity_name} ({entity_type})")
        
        return self.entities[entity_id]
    
    def enter_entity(self, entity_id: str):
        """Enter an entity scope"""
        self.current_entity = entity_id
        logger.debug(f"Entered entity: {entity_id}")
    
    def exit_entity(self):
        """Exit current entity scope"""
        logger.debug(f"Exited entity: {self.current_entity}")
        self.current_entity = None
    
    def detect_prop_dependency(self,
                              entity_id: str,
                              prop_name: str,
                              prop_type: Optional[str] = None,
                              is_optional: bool = False,
                              has_default: bool = False,
                              default_value: Optional[str] = None,
                              line_number: Optional[int] = None,
                              source_component: Optional[str] = None) -> Dependency:
        """
        Detect React prop dependency
        
        Example: function Counter({ count, setCount }) { ... }
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not registered")
        
        entity = self.entities[entity_id]
        
        # Determine category based on prop type
        if prop_type:
            if 'function' in prop_type.lower() or prop_name.startswith(('on', 'handle')):
                category = DICategory.FUNCTION
            elif 'Context' in prop_type or 'Provider' in prop_type:
                category = DICategory.CONTEXT
            else:
                category = DICategory.DATA
        else:
            # Heuristic based on name
            if prop_name.startswith(('on', 'handle')):
                category = DICategory.FUNCTION
            else:
                category = DICategory.DATA
        
        dependency = Dependency(
            name=prop_name,
            di_type=DIType.PROP,
            di_category=category,
            source_id=source_component,
            source_type='component',
            data_type=prop_type,
            is_optional=is_optional,
            has_default=has_default,
            default_value=default_value,
            line_number=line_number,
            context=f"Prop: {prop_name}" + (f" = {default_value}" if has_default else "")
        )
        
        entity.add_dependency(dependency)
        self.stats['total_dependencies'] += 1
        self.stats['prop_dependencies'] += 1
        
        logger.debug(f"Detected prop dependency: {prop_name} in {entity.entity_name}")
        return dependency
    
    def detect_context_dependency(self,
                                 entity_id: str,
                                 context_name: str,
                                 variable_name: str,
                                 context_type: Optional[str] = None,
                                 line_number: Optional[int] = None,
                                 hook_id: Optional[str] = None) -> Dependency:
        """
        Detect React Context dependency
        
        Example: const theme = useContext(ThemeContext)
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not registered")
        
        entity = self.entities[entity_id]
        
        # Track as context consumer
        if context_name not in self.context_consumers:
            self.context_consumers[context_name] = []
        self.context_consumers[context_name].append(entity_id)
        
        # Find provider if exists
        provider_id = self.context_providers.get(context_name)
        
        dependency = Dependency(
            name=variable_name,
            di_type=DIType.CONTEXT,
            di_category=DICategory.CONTEXT,
            source_id=provider_id,
            source_type='context_provider',
            data_type=context_type,
            line_number=line_number,
            context=f"useContext({context_name})"
        )
        
        entity.add_dependency(dependency)
        self.stats['total_dependencies'] += 1
        self.stats['context_dependencies'] += 1
        self.stats['consumers'] += 1
        
        logger.debug(f"Detected context dependency: {context_name} in {entity.entity_name}")
        return dependency
    
    def detect_hook_dependency(self,
                              entity_id: str,
                              hook_name: str,
                              hook_type: str,
                              return_values: List[str],
                              line_number: Optional[int] = None,
                              hook_id: Optional[str] = None) -> Dependency:
        """
        Detect custom hook dependency
        
        Example: const { data, loading } = useCustomHook()
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not registered")
        
        entity = self.entities[entity_id]
        
        dependency = Dependency(
            name=hook_name,
            di_type=DIType.HOOK,
            di_category=DICategory.SERVICE,
            source_type='custom_hook',
            data_type=hook_type,
            line_number=line_number,
            context=f"Custom hook: {hook_name}"
        )
        
        entity.add_dependency(dependency)
        self.stats['total_dependencies'] += 1
        self.stats['hook_dependencies'] += 1
        
        logger.debug(f"Detected hook dependency: {hook_name} in {entity.entity_name}")
        return dependency
    
    def detect_parameter_dependency(self,
                                   entity_id: str,
                                   param_name: str,
                                   param_type: Optional[str] = None,
                                   has_default: bool = False,
                                   default_value: Optional[str] = None,
                                   line_number: Optional[int] = None) -> Dependency:
        """
        Detect function parameter dependency
        
        Example: function add(a, b = 0) { ... }
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not registered")
        
        entity = self.entities[entity_id]
        
        # Determine category
        if param_type and 'function' in param_type.lower():
            category = DICategory.FUNCTION
        else:
            category = DICategory.DATA
        
        dependency = Dependency(
            name=param_name,
            di_type=DIType.PARAMETER,
            di_category=category,
            data_type=param_type,
            has_default=has_default,
            default_value=default_value,
            line_number=line_number,
            context=f"Parameter: {param_name}"
        )
        
        entity.add_dependency(dependency)
        self.stats['total_dependencies'] += 1
        
        return dependency
    
    def detect_import_dependency(self,
                                entity_id: str,
                                import_name: str,
                                import_source: str,
                                import_type: str = 'default',
                                line_number: Optional[int] = None) -> Dependency:
        """
        Detect module import dependency
        
        Examples:
        - import React from 'react'
        - import { useState } from 'react'
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not registered")
        
        entity = self.entities[entity_id]
        
        # Determine category based on import
        if 'react' in import_source.lower():
            category = DICategory.COMPONENT
        elif import_name.endswith('Service') or import_name.endswith('API'):
            category = DICategory.SERVICE
        elif import_name.endswith('Config') or 'config' in import_source.lower():
            category = DICategory.CONFIGURATION
        else:
            category = DICategory.SERVICE
        
        dependency = Dependency(
            name=import_name,
            di_type=DIType.IMPORT,
            di_category=category,
            source_id=import_source,
            source_type='module',
            line_number=line_number,
            context=f"import {import_name} from '{import_source}'"
        )
        
        entity.add_dependency(dependency)
        self.stats['total_dependencies'] += 1
        
        return dependency
    
    def register_context_provider(self,
                                 provider_id: str,
                                 context_name: str,
                                 provided_value: str,
                                 line_number: Optional[int] = None):
        """
        Register a Context provider
        
        Example: <ThemeContext.Provider value={theme}>
        """
        if provider_id not in self.entities:
            self.register_entity(provider_id, context_name, 'context_provider', line_number)
        
        entity = self.entities[provider_id]
        entity.add_provides(context_name)
        
        self.context_providers[context_name] = provider_id
        self.stats['providers'] += 1
        
        logger.debug(f"Registered context provider: {context_name}")
    
    def get_entity(self, entity_id: str) -> Optional[DIEntity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
    
    def get_entity_dependencies(self, entity_id: str) -> List[Dependency]:
        """Get all dependencies for an entity"""
        entity = self.get_entity(entity_id)
        return entity.dependencies if entity else []
    
    def get_entities_by_type(self, entity_type: str) -> List[DIEntity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def get_components(self) -> List[DIEntity]:
        """Get all component entities"""
        return [e for e in self.entities.values() 
                if e.entity_type in ['component', 'function_component', 'class_component']]
    
    def get_providers(self) -> List[DIEntity]:
        """Get all provider entities"""
        return [e for e in self.entities.values() if e.is_provider]
    
    def get_consumers(self) -> List[DIEntity]:
        """Get all consumer entities"""
        return [e for e in self.entities.values() if e.is_consumer]
    
    def get_context_flow(self, context_name: str) -> Dict[str, Any]:
        """
        Get the complete flow of a context from provider to consumers
        
        Returns:
            Dict with 'provider' and 'consumers' keys
        """
        provider_id = self.context_providers.get(context_name)
        consumer_ids = self.context_consumers.get(context_name, [])
        
        return {
            'context_name': context_name,
            'provider': self.get_entity(provider_id) if provider_id else None,
            'consumers': [self.get_entity(cid) for cid in consumer_ids if cid in self.entities],
            'consumer_count': len(consumer_ids)
        }
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get dependency graph as adjacency list
        
        Returns:
            Dict mapping entity_id -> [dependency_source_ids]
        """
        graph = {}
        for entity_id, entity in self.entities.items():
            deps = []
            for dep in entity.dependencies:
                if dep.source_id:
                    deps.append(dep.source_id)
            graph[entity_id] = deps
        return graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        di_type_counts = {}
        di_category_counts = {}
        
        for entity in self.entities.values():
            for dep in entity.dependencies:
                # Count by DI type
                dt = dep.di_type.value
                di_type_counts[dt] = di_type_counts.get(dt, 0) + 1
                
                # Count by category
                dc = dep.di_category.value
                di_category_counts[dc] = di_category_counts.get(dc, 0) + 1
        
        return {
            **self.stats,
            'total_contexts': len(self.context_providers),
            'di_type_breakdown': di_type_counts,
            'di_category_breakdown': di_category_counts,
            'average_dependencies_per_entity': (
                self.stats['total_dependencies'] / self.stats['total_entities']
                if self.stats['total_entities'] > 0 else 0
            )
        }
    
    def export_entities(self) -> List[Dict[str, Any]]:
        """Export all entities as dictionaries"""
        return [entity.to_dict() for entity in self.entities.values()]
    
    def clear(self):
        """Clear all analyzed data"""
        self.entities.clear()
        self.current_entity = None
        self.context_providers.clear()
        self.context_consumers.clear()
        self.stats = {
            'total_entities': 0,
            'total_dependencies': 0,
            'prop_dependencies': 0,
            'context_dependencies': 0,
            'hook_dependencies': 0,
            'providers': 0,
            'consumers': 0
        }
        logger.info("DI analyzer cleared")
