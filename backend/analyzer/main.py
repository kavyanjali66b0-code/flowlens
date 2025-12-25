"""
Main codebase analyzer orchestrator.
"""

import logging
import os
from typing import Dict, Any, Optional, Callable
import yaml

from .models import ProjectType
from .scanner import ProjectScanner
from .entry_points import EntryPointIdentifier
from .parser import LanguageParser
from .m4_semantic_analyzer import M4SemanticAnalyzer  # M4.0 Enhanced
from .progress_tracker import ProgressTracker, AnalysisPhase
from .api_response import ResponseBuilder, ParseError, Insight, InsightType, InsightSeverity
from .edge_validator import EdgeValidator


class WorkflowGraphGenerator:
    """Generates workflow graph from parsed nodes and edges."""
    
    def __init__(self, nodes, edges):
        """Initialize the graph generator.
        
        Args:
            nodes: List of parsed nodes
            edges: List of parsed edges
        """
        self.nodes = nodes
        self.edges = edges
        
    def generate(self) -> Dict[str, Any]:
        """Generate graph JSON for frontend.
        
        Returns:
            Dictionary containing nodes, edges, and metadata
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": list(set(node.type.value for node in self.nodes)),
                "edge_types": list(set(edge.type.value for edge in self.edges))
            }
        }


class CodebaseAnalyzer:
    """Main codebase analyzer orchestrator."""
    
    def __init__(self, enable_enhanced_analysis: bool = False):
        """Initialize the codebase analyzer.
        
        Args:
            enable_enhanced_analysis: Whether to use ML/NLP enhanced semantic analysis
        """
        self.project_path = None
        self.project_type = ProjectType.UNKNOWN
        self.scanner = None
        self.entry_identifier = None
        self.parser = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.enable_enhanced_analysis = enable_enhanced_analysis
        
    def analyze(
        self, 
        folder_path: str, 
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Main analysis pipeline.
        
        Args:
            folder_path: Path to the project directory to analyze
            session_id: Optional session ID for progress tracking
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing project info and workflow graph
        """
        logging.info(f"Starting codebase analysis for: {folder_path}")
        
        self.project_path = folder_path
        self.progress_callback = progress_callback
        
        # Initialize progress tracker
        if session_id:
            self.progress_tracker = ProgressTracker(session_id)
            self._report_progress(AnalysisPhase.INITIALIZING, 0, 1, "Initializing analysis")
        
        try:
            # Step 0: Load optional user config (.devscope.yml)
            user_config = self._load_user_config(folder_path)

            # Step 1: Scan project
            self._report_progress(AnalysisPhase.SCANNING, 0, 1, "Scanning project structure")
            self.scanner = ProjectScanner(folder_path)
            scan_results = self.scanner.scan()
            self._report_progress(AnalysisPhase.SCANNING, 1, 1, "Project scan complete")
            
            # Step 2: Determine project type
            self._report_progress(AnalysisPhase.DETECTING_TYPE, 0, 1, "Detecting project type")
            self._determine_project_type()
            self._report_progress(AnalysisPhase.DETECTING_TYPE, 1, 1, f"Detected: {self.project_type.value}")
            
            # Step 3: Identify entry points
            self._report_progress(AnalysisPhase.IDENTIFYING_ENTRIES, 0, 1, "Identifying entry points")
            self.entry_identifier = EntryPointIdentifier(
                folder_path, 
                self.scanner.config_files,
                self.project_type
            )
            entry_points = self.entry_identifier.identify()
            self._report_progress(AnalysisPhase.IDENTIFYING_ENTRIES, 1, 1, f"Found {len(entry_points or [])} entry points")
            
            # Step 4: Parse
            self._report_progress(AnalysisPhase.PARSING_FILES, 0, 1, "Starting file parsing")
            
            # Create symbol table for cross-file resolution
            from .symbol_table import SymbolTable
            symbol_table = SymbolTable()
            
            self.parser = LanguageParser(
                folder_path, 
                user_config=user_config,
                progress_callback=self._create_parser_progress_callback(),
                symbol_table=symbol_table
            )
            self.parser.parse_project(entry_points, self.project_type)
            self._report_progress(AnalysisPhase.PARSING_FILES, 1, 1, "File parsing complete")
            
            # Log symbol table statistics
            stats = symbol_table.get_stats()
            logging.info(f"Symbol table populated: {stats['total_symbols']} symbols, "
                        f"{stats['total_references']} references across {stats['files_with_symbols']} files")
            
            # Store symbol table for later use
            self.symbol_table = symbol_table
            
            # Step 5: Semantic analysis with M4.0 enhancements (edges, relationships)
            analysis_method = 'basic'  # Track which method was used
            analysis_warnings = []
            
            if self.enable_enhanced_analysis:
                self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Analyzing relationships with ML/NLP Enhanced Analysis (CodeBERT)")
                # Use enhanced semantic analyzer for ML/NLP capabilities with CodeBERT
                # Lazy import to avoid heavy deps unless enabled
                try:
                    from .enhanced_semantic_analyzer import EnhancedSemanticAnalyzer
                    
                    # Initialize enhanced semantic analyzer with CodeBERT
                    enhanced_semantic = EnhancedSemanticAnalyzer(
                        nodes=self.parser.nodes,
                        file_symbols=self.parser.file_symbols,
                        symbol_table=self.symbol_table,
                        enable_embeddings=True,
                        enable_clustering=True,
                        use_codebert=True,  # Use CodeBERT for better code understanding
                        batch_size=int(os.environ.get('FLOWLENS_CODEBERT_BATCH_SIZE', '32'))
                    )
                    
                    # Run enhanced semantic analysis
                    derived_edges = enhanced_semantic.analyze()
                    
                    # Check what actually ran
                    if enhanced_semantic.use_codebert and enhanced_semantic.codebert_embedder:
                        logging.info(f"✓ CodeBERT analysis successful - generated {len(derived_edges)} edges")
                        analysis_method = 'codebert'
                    else:
                        logging.warning(f"WARNING: CodeBERT unavailable, using sentence transformer - generated {len(derived_edges)} edges")
                        analysis_method = 'sentence_transformer'
                        analysis_warnings.append({
                            'type': 'ml_fallback',
                            'message': 'CodeBERT not available. Using SentenceTransformer for embeddings.',
                            'severity': 'info'
                        })
                    
                except (ImportError, OSError, RuntimeError) as e:
                    logging.warning(f"Enhanced analysis unavailable (ML dependencies may have DLL issues): {e}. Falling back to standard semantic analysis.")
                    analysis_warnings.append({
                        'type': 'ml_unavailable',
                        'message': 'ML dependencies unavailable. Using standard semantic analysis without embeddings.',
                        'severity': 'warning',
                        'details': str(e)
                    })
                    # Fall back to standard semantic analysis
                    self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Analyzing relationships with M4.0 (fallback)")
                    semantic = M4SemanticAnalyzer(
                        self.parser.nodes, 
                        getattr(self.parser, 'file_symbols', {}),
                        symbol_table=symbol_table
                    )
                    derived_edges = semantic.analyze()
                    analysis_method = 'basic'
                    self.enhanced_results = None
            else:
                self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Analyzing relationships with M4.0")
                semantic = M4SemanticAnalyzer(
                    self.parser.nodes, 
                    getattr(self.parser, 'file_symbols', {}),
                    symbol_table=symbol_table
                )
                derived_edges = semantic.analyze()
                self.enhanced_results = None

            # Merge parser edges and semantic edges, preserving different edge types
            # Allow multiple edge types between same nodes (e.g., A imports B AND A calls B)
            for e in derived_edges:
                # Check if this exact edge (source, target, type) already exists
                edge_exists = any(
                    existing_edge.source == e.source and 
                    existing_edge.target == e.target and 
                    existing_edge.type == e.type
                    for existing_edge in self.parser.edges
                )
                if not edge_exists:
                    self.parser.edges.append(e)
                    logging.debug(f"Added {e.type.value} edge: {e.source} -> {e.target}")
            
            # Final cleanup: Remove any self-referencing edges that might have slipped through
            initial_edge_count = len(self.parser.edges)
            self.parser.edges = [
                e for e in self.parser.edges 
                if e.source != e.target
            ]
            removed_self_refs = initial_edge_count - len(self.parser.edges)
            if removed_self_refs > 0:
                logging.warning(f"Removed {removed_self_refs} self-referencing edges during final cleanup")
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"Analysis complete: {len(derived_edges)} relationships found")

            # Step 6: Generate structured response
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 0, 1, "Generating response")
            
            project_name = self._derive_project_name(folder_path)
            language, framework = self._derive_language_and_framework(self.project_type)

            # Build structured response using ResponseBuilder
            builder = ResponseBuilder()
            builder.set_project_info(
                name=project_name,
                path=folder_path,
                project_type=self.project_type.value,
                language=language,
                framework=framework,
                config_files=scan_results.get("config_files", []),
                entry_points=entry_points or []
            )
            
            # Add nodes and edges
            builder.add_nodes(self.parser.nodes)
            builder.add_edges(self.parser.edges)
            
            # Add parse errors if any
            if hasattr(self.parser, 'parse_errors') and self.parser.parse_errors:
                for error in self.parser.parse_errors:
                    builder.add_error(error)
            
            # Add analysis insights
            insights = self._generate_insights(self.parser.nodes, self.parser.edges)
            
            # Optional: Validate edges (development/debugging)
            if os.environ.get('FLOWLENS_VALIDATE_EDGES', 'false').lower() == 'true':
                logging.info("Running edge validation...")
                validator = EdgeValidator(self.parser.nodes, self.parser.edges)
                validation_issues = validator.validate_all()
                
                if validation_issues:
                    # Log validation summary
                    summary = validator.get_summary()
                    logging.warning(f"Edge validation found {summary['total_issues']} issues: "
                                   f"{summary['errors']} errors, {summary['warnings']} warnings")
                    
                    # Convert validation issues to insights
                    for issue in validation_issues[:20]:  # Limit to 20 most important
                        if issue.severity.value in ['error', 'warning']:
                            insights.append(Insight(
                                type=InsightType.ARCHITECTURE_SMELL,
                                severity=InsightSeverity.WARNING,
                                title=f"Edge validation: {issue.issue_type}",
                                description=issue.message,
                                affected_nodes=[issue.edge.source, issue.edge.target],
                                metadata={'validation_issue': issue.to_dict()}
                            ))
            
            # Add all insights to response
            for insight in insights:
                builder.add_insight(insight)
            
            # Add analysis warnings if any
            for warning in analysis_warnings:
                builder.add_insight(Insight(
                    type=InsightType.ARCHITECTURE_SMELL,
                    severity=InsightSeverity.INFO if warning['severity'] == 'info' else InsightSeverity.WARNING,
                    title=warning['message'],
                    description=warning.get('details', ''),
                    metadata={'warning_type': warning['type']}
                ))
            
            # Add metadata
            builder.set_metadata('analysis_duration', 
                                self.progress_tracker.get_elapsed_seconds() if self.progress_tracker else 0)
            builder.set_metadata('enable_enhanced_analysis', self.enable_enhanced_analysis)
            builder.set_metadata('analysis_method', analysis_method)
            
            response = builder.build()
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 1, 1, "Response generation complete")
            
            # Mark as completed
            if self.progress_tracker:
                self.progress_tracker.mark_completed()
                if self.progress_callback:
                    self.progress_callback(self.progress_tracker.get_current_status())
            
            logging.info(f"Analysis complete. Generated {len(self.parser.nodes)} nodes "
                        f"and {len(self.parser.edges)} edges")
            
            return response.to_dict()
        
        except Exception as e:
            # Handle any errors during analysis
            logging.error(f"Analysis failed: {e}", exc_info=True)
            
            if self.progress_tracker:
                self.progress_tracker.mark_failed(str(e))
            
            # Return minimal error response
            builder = ResponseBuilder()
            
            # Try to set basic project info
            try:
                project_name = self._derive_project_name(folder_path)
                builder.set_project_info(
                    name=project_name,
                    path=folder_path,
                    project_type="unknown",
                    language="unknown",
                    framework="unknown"
                )
            except:
                builder.set_project_info(
                    name="Unknown",
                    path=folder_path,
                    project_type="unknown",
                    language="unknown",
                    framework="unknown"
                )
            
            # Add error
            builder.add_error(ParseError(
                file=folder_path,
                error_type="analysis_error",
                message=str(e),
                severity="error"
            ))
            
            response = builder.build()
            return response.to_dict()
    
    def _group_nodes_by_file(self):
        """Group nodes by file path."""
        file_groups = {}
        for node in self.parser.nodes:
            if node.file not in file_groups:
                file_groups[node.file] = []
            file_groups[node.file].append(node)
        return file_groups.items()
    
    def _get_file_code(self, file_path: str) -> Optional[str]:
        """Get file code content."""
        try:
            full_path = os.path.join(self.project_path, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logging.warning(f"Could not read file {file_path}: {e}")
        return None
    
    def _determine_project_type(self):
        """Determine the most specific project type from detected types."""
        if not self.scanner.detected_types:
            self.project_type = ProjectType.UNKNOWN
            return
            
        # Priority order for project types (most specific first)
        priority_order = [
            ProjectType.SPRING_BOOT,
            ProjectType.REACT_VITE,
            ProjectType.ANGULAR,
            ProjectType.DJANGO,
            ProjectType.EXPRESS_NODE,
            ProjectType.MAVEN_JAVA,
            ProjectType.GRADLE_JAVA,
            ProjectType.ANDROID,
            ProjectType.PYTHON_APP,
        ]
        
        for project_type in priority_order:
            if project_type in self.scanner.detected_types:
                self.project_type = project_type
                logging.info(f"Selected project type: {project_type.value}")
                return
        
        # Fallback to first detected type
        self.project_type = list(self.scanner.detected_types)[0]
        logging.info(f"Using fallback project type: {self.project_type.value}")

    def _derive_project_name(self, folder_path: str) -> str:
        try:
            return folder_path.rstrip("/\\").split(os.sep)[-1]
        except Exception:
            return "unknown_project"

    def _derive_language_and_framework(self, project_type: ProjectType) -> tuple[str, str]:
        mapping = {
            ProjectType.REACT_VITE: ("typescript", "react"),
            ProjectType.ANGULAR: ("typescript", "angular"),
            ProjectType.EXPRESS_NODE: ("javascript", "express"),
            ProjectType.DJANGO: ("python", "django"),
            ProjectType.PYTHON_APP: ("python", "python"),
            ProjectType.SPRING_BOOT: ("java", "spring_boot"),
            ProjectType.MAVEN_JAVA: ("java", "java"),
            ProjectType.GRADLE_JAVA: ("java", "java"),
            ProjectType.ANDROID: ("java", "android"),
        }
        return mapping.get(project_type, ("unknown", "unknown"))

    def _load_user_config(self, folder_path: str) -> Optional[Dict[str, Any]]:
        try:
            config_path = os.path.join(folder_path, '.devscope.yml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Failed to load user config: {e}")
        return {}
    
    def _report_progress(
        self, 
        phase: AnalysisPhase, 
        current: int, 
        total: int, 
        message: str = ""
    ):
        """Report progress update."""
        if self.progress_tracker:
            self.progress_tracker.update(phase, current, total, message)
            if self.progress_callback:
                self.progress_callback(self.progress_tracker.get_current_status())
    
    def _create_parser_progress_callback(self) -> Optional[Callable[[int, int], None]]:
        """Create a progress callback for the parser."""
        if not self.progress_tracker:
            return None
        
        def callback(current: int, total: int):
            """Parser progress callback."""
            self._report_progress(
                AnalysisPhase.PARSING_FILES,
                current,
                total,
                f"Parsing files: {current}/{total}"
            )
        
        return callback
    
    def _generate_insights(self, nodes, edges) -> list:
        """
        Generate analysis insights from nodes and edges.
        
        This method detects issues, patterns, and provides suggestions.
        """
        insights = []
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies(edges)
        if circular_deps:
            insights.append(Insight(
                type=InsightType.CIRCULAR_DEPENDENCY,
                severity=InsightSeverity.WARNING,
                title=f"Found {len(circular_deps)} circular dependencies",
                description="Circular dependencies can make code harder to maintain and test.",
                affected_nodes=[node for cycle in circular_deps for node in cycle],
                metadata={'cycles': circular_deps}
            ))
        
        # Detect large files
        file_sizes = {}
        for node in nodes:
            file_path = node.file
            if file_path not in file_sizes:
                file_sizes[file_path] = 0
            file_sizes[file_path] += 1
        
        large_files = [(f, count) for f, count in file_sizes.items() if count > 50]
        if large_files:
            for file_path, node_count in large_files[:5]:  # Top 5
                insights.append(Insight(
                    type=InsightType.LARGE_FILE,
                    severity=InsightSeverity.INFO,
                    title=f"Large file: {file_path}",
                    description=f"File contains {node_count} nodes. Consider splitting into smaller modules.",
                    affected_files=[file_path],
                    metadata={'node_count': node_count}
                ))
        
        # Detect unused components (nodes with no incoming edges)
        node_ids = {node.id for node in nodes}
        target_ids = {edge.target for edge in edges}
        unused_nodes = node_ids - target_ids
        
        # Filter out entry points (they're supposed to have no incoming edges)
        unused_non_entry = [
            node for node in nodes 
            if node.id in unused_nodes and not (node.metadata or {}).get('is_entry', False)
        ]
        
        if len(unused_non_entry) > 10:  # Only report if significant
            insights.append(Insight(
                type=InsightType.UNUSED_CODE,
                severity=InsightSeverity.INFO,
                title=f"Found {len(unused_non_entry)} potentially unused components",
                description="These components are not referenced by any other code.",
                affected_nodes=[node.id for node in unused_non_entry[:20]],  # Limit to 20
                metadata={'total_unused': len(unused_non_entry)}
            ))
        
        return insights
    
    def _detect_circular_dependencies(self, edges) -> list:
        """
        Detect circular dependencies in the graph.
        
        Returns list of cycles (each cycle is a list of node IDs).
        """
        # Build adjacency list
        from collections import defaultdict
        
        graph = defaultdict(list)
        for edge in edges:
            # Only consider dependency edges
            if edge.type.value in ['imports', 'depends_on', 'uses']:
                graph[edge.source].append(edge.target)
        
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Use get() to avoid mutating the dict via defaultdict during reads
            for neighbor in list(graph.get(node, [])):
                if neighbor not in visited:
                    if dfs(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle - check if neighbor is in path
                    if neighbor in path:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:]
                        if len(cycle) > 1 and cycle not in cycles:
                            cycles.append(cycle)
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Iterate over a stable snapshot of keys to avoid mutation during iteration
        for node in list(graph.keys()):
            if node not in visited:
                dfs(node, [])
        
        return cycles[:10]  # Return max 10 cycles
