"""
Configuration loader for FlowLens.

Loads user configuration from .flowlens.config.json or auto-detects from project structure.
Provides flexible, user-configurable project analysis settings.
"""

import json
import yaml
import logging
import dataclasses
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class FlowLensConfig:
    """User configuration for project analysis."""
    
    # Project identification
    project_type: Optional[str] = None
    project_name: Optional[str] = None
    
    # Directory exclusions (glob patterns supported)
    exclude: List[str] = field(default_factory=lambda: [
        '.git', 'node_modules', '__pycache__', '.venv', 'venv',
        'target', 'build', 'dist', '.next', '.nuxt', '.turbo',
        'bazel-*', 'vendor', '.bundle', 'bin', 'obj', '.cargo',
        '.nx', 'coverage', '.pytest_cache', '.mypy_cache', '.tox',
        'out', 'tmp', '.temp', '.cache', '.angular', '.swc'
    ])
    
    # Entry points
    entry_points: List[str] = field(default_factory=list)
    entry_point_patterns: List[str] = field(default_factory=lambda: [
        'src/main.*',
        'src/index.*',
        'app/layout.*',
        'app/page.*',
        'pages/_app.*',
        'pages/index.*',
        'main.py',
        'app.py',
        'manage.py'
    ])
    
    # API routes
    api_route_patterns: List[str] = field(default_factory=list)
    api_framework: Optional[str] = None
    
    # Component detection
    component_extensions: List[str] = field(default_factory=lambda: [
        '.tsx', '.jsx', '.vue', '.svelte', '.astro'
    ])
    component_naming: str = 'PascalCase'  # 'PascalCase', 'kebab-case', 'any'
    component_locations: List[str] = field(default_factory=lambda: [
        'components/**',
        'src/**',
        'app/**'
    ])
    
    # Import resolution
    import_aliases: Dict[str, str] = field(default_factory=dict)
    resolve_external: bool = False
    
    # C4 overrides
    c4_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    max_memory_mb: int = 2048
    max_files: int = 50000
    parallel_parsing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlowLensConfig':
        """Create from dictionary, filtering unknown keys."""
        # Get valid field names
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        # Filter out unknown keys to avoid TypeError
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class ConfigLoader:
    """Loads user configuration with smart defaults."""
    
    CONFIG_FILES = [
        '.flowlens.config.json',
        '.flowlens.config.yaml',
        '.flowlens.config.yml',
        '.devscope.yml'
    ]
    
    @classmethod
    def load(cls, project_path: Path) -> FlowLensConfig:
        """
        Load configuration from project directory.
        
        Priority:
        1. Explicit config file (.flowlens.config.json)
        2. Auto-detect from project structure
        3. Smart defaults
        
        Args:
            project_path: Path to project root
            
        Returns:
            FlowLensConfig with loaded or detected settings
        """
        if isinstance(project_path, str):
            project_path = Path(project_path)
        
        # Try to load explicit config
        for config_file in cls.CONFIG_FILES:
            config_path = project_path / config_file
            if config_path.exists():
                logging.info(f"Loading config from: {config_file}")
                return cls._load_from_file(config_path)
        
        # Auto-detect from project structure
        logging.info("No config file found, auto-detecting project structure")
        return cls._auto_detect(project_path)
    
    @classmethod
    def _load_from_file(cls, config_path: Path) -> FlowLensConfig:
        """Load config from JSON or YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f) or {}
                else:
                    data = json.load(f)
            
            logging.info(f"Loaded config: {data.get('project_type', 'custom')}")
            return FlowLensConfig.from_dict(data)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logging.error(f"Failed to parse config file {config_path}: {e}")
            return FlowLensConfig()
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")
            return FlowLensConfig()
    
    @classmethod
    def _auto_detect(cls, project_path: Path) -> FlowLensConfig:
        """Auto-detect configuration from project structure."""
        config = FlowLensConfig()
        
        # Detect Next.js
        if (project_path / 'next.config.js').exists() or (project_path / 'next.config.mjs').exists():
            logging.info("Detected Next.js project")
            config.project_type = 'nextjs'
            config.exclude.extend(['.next', '.swc'])
            
            # Check for App Router vs Pages Router
            if (project_path / 'app').exists():
                logging.info("Detected Next.js App Router")
                config.entry_points = ['app/layout.tsx', 'app/layout.ts', 'app/page.tsx', 'app/page.ts']
                config.api_route_patterns = ['app/**/route.ts', 'app/**/route.js']
            else:
                logging.info("Detected Next.js Pages Router")
                config.entry_points = ['pages/_app.tsx', 'pages/_app.js', 'pages/index.tsx', 'pages/index.js']
                config.api_route_patterns = ['pages/api/**/*.ts', 'pages/api/**/*.js']
        
        # Detect Nx monorepo
        elif (project_path / 'nx.json').exists():
            logging.info("Detected Nx monorepo")
            config.project_type = 'nx-monorepo'
            config.exclude.extend(['.nx', 'dist', 'tmp'])
            config.entry_point_patterns = [
                'apps/*/src/main.*',
                'apps/*/src/index.*',
                'libs/*/src/index.*'
            ]
        
        # Detect Turborepo
        elif (project_path / 'turbo.json').exists():
            logging.info("Detected Turborepo")
            config.project_type = 'turborepo'
            config.exclude.extend(['.turbo', 'out'])
            config.entry_point_patterns = [
                'apps/*/src/main.*',
                'apps/*/src/index.*',
                'packages/*/src/index.*'
            ]
        
        # Detect Vite
        elif any((project_path / f).exists() for f in ['vite.config.js', 'vite.config.ts', 'vite.config.mjs']):
            logging.info("Detected Vite project")
            config.project_type = 'vite'
            config.entry_points = ['src/main.tsx', 'src/main.ts', 'src/main.jsx', 'src/main.js', 'index.html']
        
        # Detect Angular
        elif (project_path / 'angular.json').exists():
            logging.info("Detected Angular project")
            config.project_type = 'angular'
            config.exclude.extend(['.angular', 'dist'])
            # Angular entry points are typically defined in angular.json
        
        # Detect Django
        elif (project_path / 'manage.py').exists():
            logging.info("Detected Django project")
            config.project_type = 'django'
            config.entry_points = ['manage.py']
            config.entry_point_patterns.extend(['*/settings.py', '*/urls.py'])
            config.api_route_patterns = ['*/urls.py', '**/views.py']
        
        # Detect Spring Boot (Maven)
        elif (project_path / 'pom.xml').exists():
            logging.info("Detected Maven/Spring Boot project")
            config.project_type = 'maven'
            config.exclude.extend(['target', '.mvn'])
            # Spring Boot entry points detected from @SpringBootApplication annotation
        
        # Detect Gradle
        elif (project_path / 'build.gradle').exists() or (project_path / 'build.gradle.kts').exists():
            logging.info("Detected Gradle project")
            config.project_type = 'gradle'
            config.exclude.extend(['.gradle', 'build'])
        
        # Detect generic Node.js project
        elif (project_path / 'package.json').exists():
            logging.info("Detected Node.js project")
            config.project_type = 'nodejs'
            # Use default entry point patterns
        
        # Detect Python project
        elif (project_path / 'requirements.txt').exists() or (project_path / 'setup.py').exists():
            logging.info("Detected Python project")
            config.project_type = 'python'
            config.entry_point_patterns.extend(['__main__.py', 'cli.py'])
        
        # Default fallback
        else:
            logging.info("Using default configuration")
            config.project_type = 'generic'
        
        return config


# Utility function for backward compatibility
def load_config(project_path: Path) -> FlowLensConfig:
    """Shorthand for ConfigLoader.load()"""
    return ConfigLoader.load(project_path)
