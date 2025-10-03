"""
External Dependency Parser for M3.5.1

This module provides comprehensive external dependency analysis including:
- package.json parsing (dependencies, devDependencies, peerDependencies)
- node_modules structure analysis
- Import/require statement tracking
- Dependency version analysis
- Dependency tree construction
- Unused dependency detection
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DependencyType(Enum):
    """Type of dependency."""
    PRODUCTION = "production"  # dependencies
    DEVELOPMENT = "development"  # devDependencies
    PEER = "peer"  # peerDependencies
    OPTIONAL = "optional"  # optionalDependencies
    BUNDLED = "bundled"  # bundledDependencies


class ImportType(Enum):
    """Type of import statement."""
    ES6_IMPORT = "es6_import"  # import x from 'y'
    ES6_IMPORT_NAMED = "es6_import_named"  # import { x } from 'y'
    ES6_IMPORT_NAMESPACE = "es6_import_namespace"  # import * as x from 'y'
    ES6_IMPORT_DEFAULT = "es6_import_default"  # import x from 'y'
    COMMONJS_REQUIRE = "commonjs_require"  # require('x')
    DYNAMIC_IMPORT = "dynamic_import"  # import('x')
    AMD_REQUIRE = "amd_require"  # define(['x'], ...)


@dataclass
class DependencyInfo:
    """Information about an external dependency."""
    name: str
    version: str
    type: DependencyType
    
    # Package metadata
    description: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    
    # Usage tracking
    imported_in: List[str] = field(default_factory=list)  # Files importing this
    import_count: int = 0
    import_statements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dependency analysis
    subdependencies: List[str] = field(default_factory=list)
    is_used: bool = False
    is_direct: bool = True  # Direct vs transitive
    
    # Metadata
    installed: bool = False
    install_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'type': self.type.value,
            'description': self.description,
            'license': self.license,
            'homepage': self.homepage,
            'repository': self.repository,
            'imported_in': self.imported_in,
            'import_count': self.import_count,
            'import_statements': self.import_statements,
            'subdependencies': self.subdependencies,
            'is_used': self.is_used,
            'is_direct': self.is_direct,
            'installed': self.installed,
            'install_path': self.install_path
        }


@dataclass
class ImportStatement:
    """Information about an import/require statement."""
    source_file: str
    line_number: int
    import_type: ImportType
    module_name: str
    imported_names: List[str] = field(default_factory=list)  # Named imports
    alias: Optional[str] = None  # Import alias
    is_type_only: bool = False  # TypeScript type-only import
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_file': self.source_file,
            'line_number': self.line_number,
            'import_type': self.import_type.value,
            'module_name': self.module_name,
            'imported_names': self.imported_names,
            'alias': self.alias,
            'is_type_only': self.is_type_only
        }


class DependencyParser:
    """
    External dependency parser for semantic analysis.
    
    Features:
    - Parse package.json files
    - Analyze node_modules structure
    - Track import/require statements
    - Detect unused dependencies
    - Build dependency trees
    """
    
    def __init__(self, project_root: str):
        """
        Initialize dependency parser.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.imports: List[ImportStatement] = []
        self.package_json_path = self.project_root / "package.json"
        self.node_modules_path = self.project_root / "node_modules"
        
        logging.info(f"DependencyParser initialized for: {project_root}")
    
    def parse_package_json(self) -> Dict[str, Any]:
        """
        Parse package.json file.
        
        Returns:
            Parsed package.json data
        """
        if not self.package_json_path.exists():
            logging.warning(f"package.json not found at: {self.package_json_path}")
            return {}
        
        try:
            with open(self.package_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logging.info(f"Parsed package.json: {data.get('name', 'unknown')}")
            
            # Extract dependencies
            self._extract_dependencies(data.get('dependencies', {}), DependencyType.PRODUCTION)
            self._extract_dependencies(data.get('devDependencies', {}), DependencyType.DEVELOPMENT)
            self._extract_dependencies(data.get('peerDependencies', {}), DependencyType.PEER)
            self._extract_dependencies(data.get('optionalDependencies', {}), DependencyType.OPTIONAL)
            
            return data
            
        except Exception as e:
            logging.error(f"Error parsing package.json: {e}")
            return {}
    
    def _extract_dependencies(self, deps: Dict[str, str], dep_type: DependencyType):
        """
        Extract dependencies from package.json section.
        
        Args:
            deps: Dependency dictionary from package.json
            dep_type: Type of dependencies
        """
        for name, version in deps.items():
            if name not in self.dependencies:
                self.dependencies[name] = DependencyInfo(
                    name=name,
                    version=version,
                    type=dep_type,
                    is_direct=True
                )
            else:
                # Update if not already set
                if self.dependencies[name].type == DependencyType.DEVELOPMENT and dep_type == DependencyType.PRODUCTION:
                    self.dependencies[name].type = dep_type
    
    def analyze_node_modules(self) -> Dict[str, DependencyInfo]:
        """
        Analyze node_modules directory structure.
        
        Returns:
            Dict of installed dependencies with metadata
        """
        if not self.node_modules_path.exists():
            logging.warning(f"node_modules not found at: {self.node_modules_path}")
            return {}
        
        installed = {}
        
        try:
            # Iterate through node_modules
            for item in self.node_modules_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Handle scoped packages (@org/package)
                    if item.name.startswith('@'):
                        for scoped_item in item.iterdir():
                            if scoped_item.is_dir():
                                package_name = f"{item.name}/{scoped_item.name}"
                                self._analyze_package(package_name, scoped_item, installed)
                    else:
                        self._analyze_package(item.name, item, installed)
            
            logging.info(f"Analyzed node_modules: {len(installed)} packages found")
            return installed
            
        except Exception as e:
            logging.error(f"Error analyzing node_modules: {e}")
            return {}
    
    def _analyze_package(self, package_name: str, package_path: Path, installed: Dict[str, DependencyInfo]):
        """
        Analyze a single package in node_modules.
        
        Args:
            package_name: Name of the package
            package_path: Path to the package directory
            installed: Dict to store installed packages
        """
        package_json = package_path / "package.json"
        
        if not package_json.exists():
            return
        
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                pkg_data = json.load(f)
            
            # Create or update dependency info
            if package_name in self.dependencies:
                dep_info = self.dependencies[package_name]
            else:
                # Transitive dependency
                dep_info = DependencyInfo(
                    name=package_name,
                    version=pkg_data.get('version', 'unknown'),
                    type=DependencyType.PRODUCTION,
                    is_direct=False
                )
                self.dependencies[package_name] = dep_info
            
            # Update metadata
            dep_info.installed = True
            dep_info.install_path = str(package_path)
            dep_info.description = pkg_data.get('description')
            dep_info.license = pkg_data.get('license')
            dep_info.homepage = pkg_data.get('homepage')
            
            # Extract repository
            repo = pkg_data.get('repository')
            if isinstance(repo, dict):
                dep_info.repository = repo.get('url')
            elif isinstance(repo, str):
                dep_info.repository = repo
            
            # Extract subdependencies
            deps = pkg_data.get('dependencies', {})
            dep_info.subdependencies = list(deps.keys())
            
            installed[package_name] = dep_info
            
        except Exception as e:
            logging.debug(f"Error analyzing package {package_name}: {e}")
    
    def extract_imports_from_ast(self, nodes: List[Dict[str, Any]], file_path: str):
        """
        Extract import/require statements from AST nodes.
        
        Args:
            nodes: List of AST nodes
            file_path: Path to the source file
        """
        for node in nodes:
            node_type = node.get('type', '')
            
            # ES6 imports
            if node_type == 'import_statement':
                self._extract_es6_import(node, file_path)
            
            # CommonJS require
            elif node_type == 'call_expression':
                callee = node.get('function', {})
                if isinstance(callee, dict) and callee.get('name') == 'require':
                    self._extract_require(node, file_path)
                # Dynamic import
                elif callee.get('name') == 'import':
                    self._extract_dynamic_import(node, file_path)
    
    def _extract_es6_import(self, node: Dict[str, Any], file_path: str):
        """Extract ES6 import statement."""
        source = node.get('source', {})
        module_name = source.get('value', '') if isinstance(source, dict) else str(source)
        
        if not module_name:
            return
        
        # Determine import type
        import_type = ImportType.ES6_IMPORT
        imported_names = []
        alias = None
        is_type_only = node.get('import_kind') == 'type'
        
        # Named imports
        if 'specifiers' in node:
            specifiers = node['specifiers']
            if isinstance(specifiers, list):
                for spec in specifiers:
                    if isinstance(spec, dict):
                        if spec.get('type') == 'import_specifier':
                            import_type = ImportType.ES6_IMPORT_NAMED
                            imported_names.append(spec.get('imported', {}).get('name', ''))
                        elif spec.get('type') == 'namespace_import':
                            import_type = ImportType.ES6_IMPORT_NAMESPACE
                            alias = spec.get('local', {}).get('name')
                        elif spec.get('type') == 'import_default_specifier':
                            import_type = ImportType.ES6_IMPORT_DEFAULT
                            alias = spec.get('local', {}).get('name')
        
        self._add_import(
            file_path=file_path,
            line_number=node.get('start_line', 0),
            import_type=import_type,
            module_name=module_name,
            imported_names=imported_names,
            alias=alias,
            is_type_only=is_type_only
        )
    
    def _extract_require(self, node: Dict[str, Any], file_path: str):
        """Extract CommonJS require statement."""
        args = node.get('arguments', [])
        if args and isinstance(args, list):
            first_arg = args[0]
            if isinstance(first_arg, dict):
                module_name = first_arg.get('value', '')
            else:
                module_name = str(first_arg)
            
            if module_name:
                self._add_import(
                    file_path=file_path,
                    line_number=node.get('start_line', 0),
                    import_type=ImportType.COMMONJS_REQUIRE,
                    module_name=module_name
                )
    
    def _extract_dynamic_import(self, node: Dict[str, Any], file_path: str):
        """Extract dynamic import() statement."""
        args = node.get('arguments', [])
        if args and isinstance(args, list):
            first_arg = args[0]
            module_name = first_arg.get('value', '') if isinstance(first_arg, dict) else str(first_arg)
            
            if module_name:
                self._add_import(
                    file_path=file_path,
                    line_number=node.get('start_line', 0),
                    import_type=ImportType.DYNAMIC_IMPORT,
                    module_name=module_name
                )
    
    def _add_import(
        self,
        file_path: str,
        line_number: int,
        import_type: ImportType,
        module_name: str,
        imported_names: Optional[List[str]] = None,
        alias: Optional[str] = None,
        is_type_only: bool = False
    ):
        """
        Add an import statement and update dependency tracking.
        
        Args:
            file_path: Source file path
            line_number: Line number of import
            import_type: Type of import
            module_name: Name of imported module
            imported_names: Named imports
            alias: Import alias
            is_type_only: Whether it's a type-only import
        """
        import_stmt = ImportStatement(
            source_file=file_path,
            line_number=line_number,
            import_type=import_type,
            module_name=module_name,
            imported_names=imported_names or [],
            alias=alias,
            is_type_only=is_type_only
        )
        
        self.imports.append(import_stmt)
        
        # Extract package name from module path
        package_name = self._extract_package_name(module_name)
        
        # Update dependency usage
        if package_name in self.dependencies:
            dep = self.dependencies[package_name]
            dep.is_used = True
            dep.import_count += 1
            if file_path not in dep.imported_in:
                dep.imported_in.append(file_path)
            dep.import_statements.append(import_stmt.to_dict())
    
    def _extract_package_name(self, module_path: str) -> str:
        """
        Extract package name from module path.
        
        Examples:
            'react' → 'react'
            'react/jsx-runtime' → 'react'
            '@mui/material' → '@mui/material'
            '@mui/material/Button' → '@mui/material'
            './utils' → '' (relative import)
            
        Args:
            module_path: Import module path
            
        Returns:
            Package name or empty string for relative imports
        """
        # Relative imports
        if module_path.startswith('.') or module_path.startswith('/'):
            return ''
        
        # Scoped packages
        if module_path.startswith('@'):
            parts = module_path.split('/')
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return module_path
        
        # Regular packages
        return module_path.split('/')[0]
    
    def detect_unused_dependencies(self) -> List[str]:
        """
        Detect dependencies that are declared but never imported.
        
        Returns:
            List of unused dependency names
        """
        unused = []
        
        for name, dep in self.dependencies.items():
            # Skip dev dependencies (might be used in build/test)
            if dep.type == DependencyType.DEVELOPMENT:
                continue
            
            # Skip peer dependencies (might be used by other deps)
            if dep.type == DependencyType.PEER:
                continue
            
            if not dep.is_used:
                unused.append(name)
        
        logging.info(f"Detected {len(unused)} unused dependencies")
        return unused
    
    def build_dependency_tree(self) -> Dict[str, List[str]]:
        """
        Build dependency tree showing relationships.
        
        Returns:
            Dict mapping package names to their dependencies
        """
        tree = {}
        
        for name, dep in self.dependencies.items():
            if dep.subdependencies:
                tree[name] = dep.subdependencies
        
        return tree
    
    def get_dependency_stats(self) -> Dict[str, Any]:
        """
        Get statistics about dependencies.
        
        Returns:
            Dict with dependency statistics
        """
        total = len(self.dependencies)
        direct = sum(1 for dep in self.dependencies.values() if dep.is_direct)
        transitive = total - direct
        
        by_type = {}
        for dep_type in DependencyType:
            count = sum(1 for dep in self.dependencies.values() if dep.type == dep_type)
            by_type[dep_type.value] = count
        
        used = sum(1 for dep in self.dependencies.values() if dep.is_used)
        unused = total - used
        installed = sum(1 for dep in self.dependencies.values() if dep.installed)
        
        return {
            'total_dependencies': total,
            'direct_dependencies': direct,
            'transitive_dependencies': transitive,
            'dependencies_by_type': by_type,
            'used_dependencies': used,
            'unused_dependencies': unused,
            'installed_dependencies': installed,
            'total_imports': len(self.imports)
        }
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export all dependency data.
        
        Returns:
            Dict with all dependency information
        """
        return {
            'dependencies': {name: dep.to_dict() for name, dep in self.dependencies.items()},
            'imports': [imp.to_dict() for imp in self.imports],
            'unused_dependencies': self.detect_unused_dependencies(),
            'dependency_tree': self.build_dependency_tree(),
            'statistics': self.get_dependency_stats()
        }
