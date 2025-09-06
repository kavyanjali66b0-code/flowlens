"""
Plugin system for framework-specific entry point detection.
"""

# --- Imports ---
import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from .models import ProjectType, ConfigFile
from .plugins.base import BasePlugin 

# --- Plugin Base Class ---


# ------------------------- Concrete Plugins -------------------------

class MavenJavaPlugin(BasePlugin):
    name: str = "Maven (Java)"
    project_type: ProjectType = ProjectType.MAVEN_JAVA

    def find_entry_points(self) -> List[Dict]:
        for config in self.config_files:
            if config.type == 'pom.xml':
                pom_path = self.project_path / config.path
                try:
                    tree = ET.parse(pom_path)
                    root = tree.getroot()
                    for main_class in root.iter():
                        if 'mainClass' in main_class.tag:
                            self.entry_points.append({
                                "type": "main_class",
                                "value": main_class.text,
                                "source": "pom.xml"
                            })
                            logging.debug(f"Found Maven main class: {main_class.text}")
                except ET.ParseError as e:
                    logging.warning(f"Failed to parse pom.xml: {e}")
        return self.entry_points


class ReactVitePlugin(BasePlugin):
    """Plugin to find entry points in React + Vite projects."""
    name: str = "React (Vite)"
    project_type: ProjectType = ProjectType.REACT_VITE

    def find_entry_points(self) -> List[Dict]:
        index_path = self.project_path / 'index.html'
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    match = re.search(r'<script.*?src=["\'](.+?)["\']', content)
                    if match:
                        script_src = match.group(1).lstrip('/')
                        self.entry_points.append({
                            "type": "html_script",
                            "value": script_src,
                            "source": "index.html"
                        })
            except (UnicodeDecodeError, OSError) as e:
                logging.warning(f"Failed to read index.html: {e}")
        
        for ext in ['tsx', 'jsx', 'ts', 'js']:
            main_path = self.project_path / 'src' / f'main.{ext}'
            if main_path.exists():
                self.entry_points.append({
                    "type": "react_main",
                    "value": f"src/main.{ext}",
                    "source": "convention"
                })
                break
        
        return self.entry_points


class AngularPlugin(BasePlugin):
    name: str = "Angular"
    project_type: ProjectType = ProjectType.ANGULAR

    def find_entry_points(self) -> List[Dict]:
        for config in self.config_files:
            if config.type == 'angular.json' and config.content:
                try:
                    projects = config.content.get('projects', {})
                    for project in projects.values():
                        architect = project.get('architect', {})
                        build = architect.get('build', {})
                        options = build.get('options', {})
                        main = options.get('main')
                        if main:
                            self.entry_points.append({
                                "type": "angular_main",
                                "value": main,
                                "source": "angular.json"
                            })
                except (KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse angular.json: {e}")
        return self.entry_points


class DjangoPlugin(BasePlugin):
    name: str = "Django"
    project_type: ProjectType = ProjectType.DJANGO

    def find_entry_points(self) -> List[Dict]:
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file == 'urls.py':
                    relative_path = Path(root).relative_to(self.project_path) / file
                    self.entry_points.append({
                        "type": "django_urls",
                        "value": str(relative_path),
                        "source": "convention"
                    })
                elif file == 'settings.py':
                    relative_path = Path(root).relative_to(self.project_path) / file
                    self.entry_points.append({
                        "type": "django_settings",
                        "value": str(relative_path),
                        "source": "convention"
                    })
        return self.entry_points


class ExpressNodePlugin(BasePlugin):
    name: str = "Express (Node)"
    project_type: ProjectType = ProjectType.EXPRESS_NODE

    def find_entry_points(self) -> List[Dict]:
        for filename in ['index.js', 'server.js', 'app.js']:
            entry_path = self.project_path / filename
            if entry_path.exists():
                try:
                    with open(entry_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'app.listen' in content or 'express()' in content:
                            self.entry_points.append({
                                "type": "express_main",
                                "value": filename,
                                "source": "convention"
                            })
                            break
                except (UnicodeDecodeError, OSError) as e:
                    logging.warning(f"Failed to read {filename}: {e}")
        return self.entry_points


class SpringBootPlugin(BasePlugin):
    name: str = "Spring Boot"
    project_type: ProjectType = ProjectType.SPRING_BOOT

    def find_entry_points(self) -> List[Dict]:
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '@SpringBootApplication' in content and 'public static void main' in content:
                                relative_path = file_path.relative_to(self.project_path)
                                self.entry_points.append({
                                    "type": "spring_boot_main",
                                    "value": str(relative_path),
                                    "source": "annotation"
                                })
                    except (UnicodeDecodeError, OSError) as e:
                        logging.warning(f"Failed to read {file_path}: {e}")
        return self.entry_points


class AndroidPlugin(BasePlugin):
    name: str = "Android"
    project_type: ProjectType = ProjectType.ANDROID

    def find_entry_points(self) -> List[Dict]:
        manifest_path = self.project_path / 'app' / 'src' / 'main' / 'AndroidManifest.xml'
        if manifest_path.exists():
            try:
                tree = ET.parse(manifest_path)
                root = tree.getroot()
                for activity in root.iter('activity'):
                    for intent in activity.iter('intent-filter'):
                        for action in intent.iter('action'):
                            if 'MAIN' in action.get('android:name', ''):
                                activity_name = activity.get('android:name', '')
                                self.entry_points.append({
                                    "type": "android_main_activity",
                                    "value": activity_name,
                                    "source": "AndroidManifest.xml"
                                })
            except ET.ParseError as e:
                logging.warning(f"Failed to parse AndroidManifest.xml: {e}")
        return self.entry_points


# ------------------------- Discovery -------------------------

def get_all_plugins():
    """Returns a list of all available plugin classes."""
    return [cls for cls in BasePlugin.__subclasses__()]
