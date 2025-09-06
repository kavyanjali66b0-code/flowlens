"""
Spring Boot project scanner plugin.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class SpringBootPlugin(ScannerPlugin):
    """Plugin for detecting Spring Boot projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is a Spring Boot project."""
        file_names = [f.name for f in files]
        
        # Check for Maven with Spring Boot
        if 'pom.xml' in file_names:
            pom_path = self.project_path / 'pom.xml'
            if pom_path.exists():
                try:
                    with open(pom_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if 'spring-boot' in content:
                            return True
                except (OSError, UnicodeDecodeError):
                    pass
        
        # Check for Gradle with Spring Boot
        if 'build.gradle' in file_names:
            gradle_path = self.project_path / 'build.gradle'
            if gradle_path.exists():
                try:
                    with open(gradle_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if 'spring-boot' in content:
                            return True
                except (OSError, UnicodeDecodeError):
                    pass
        
        # Check for Java files with Spring Boot annotations
        for file_path in files:
            if file_path.suffix == '.java':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '@SpringBootApplication' in content:
                            return True
                except (OSError, UnicodeDecodeError):
                    pass
        
        return False
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.SPRING_BOOT
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Spring Boot configuration files."""
        config_files = []
        
        # Check for pom.xml
        pom_path = self.project_path / 'pom.xml'
        if pom_path.exists():
            try:
                with open(pom_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'pom.xml',
                    'path': 'pom.xml',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        # Check for build.gradle
        gradle_path = self.project_path / 'build.gradle'
        if gradle_path.exists():
            try:
                with open(gradle_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'build.gradle',
                    'path': 'build.gradle',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        # Check for application.properties/yml
        app_configs = ['application.properties', 'application.yml', 'application.yaml']
        for config_name in app_configs:
            config_path = self.project_path / config_name
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    config_files.append({
                        'type': 'application_config',
                        'path': config_name,
                        'content': content
                    })
                except (OSError, UnicodeDecodeError):
                    pass
        
        return config_files
