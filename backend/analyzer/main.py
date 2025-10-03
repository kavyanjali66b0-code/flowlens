"""
Main codebase analyzer orchestrator.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, TYPE_CHECKING
import yaml

from .models import ProjectType, Node, Edge
from .scanner import ProjectScanner
from .entry_points import EntryPointIdentifier
from .parser import LanguageParser
from .m4_semantic_analyzer import M4SemanticAnalyzer  # M4.0 Enhanced
from .progress_tracker import ProgressTracker, AnalysisPhase

if TYPE_CHECKING:
    from .symbol_table import SymbolTable


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
    
    def __init__(self):
        """Initialize the codebase analyzer."""
        self.project_path = None
        self.project_type = ProjectType.UNKNOWN
        self.scanner = None
        self.entry_identifier = None
        self.parser = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
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
        
        # NEW: Initialize memory monitor
        from .memory_monitor import MemoryMonitor
        from .exceptions import MemoryLimitExceeded
        memory_monitor = MemoryMonitor(max_memory_mb=2048)  # 2GB limit
        
        try:
            # Step 0: Check memory before starting
            exceeded, msg = memory_monitor.check_memory_threshold()
            if exceeded:
                raise MemoryLimitExceeded(
                    current_mb=memory_monitor.get_current_usage().rss_mb,
                    limit_mb=2048
                )
            
            # Step 1: Load optional user config (.devscope.yml)
            user_config = self._load_user_config(folder_path)

            # Step 2: Scan project
            self._report_progress(AnalysisPhase.SCANNING, 0, 1, "Scanning project structure")
            self.scanner = ProjectScanner(folder_path)
            scan_results = self.scanner.scan()
            self._report_progress(AnalysisPhase.SCANNING, 1, 1, "Project scan complete")
            
            # Step 3: Determine project type
            self._report_progress(AnalysisPhase.DETECTING_TYPE, 0, 1, "Detecting project type")
            self._determine_project_type()
            self._report_progress(AnalysisPhase.DETECTING_TYPE, 1, 1, f"Detected: {self.project_type.value}")
            
            # Step 4: Identify entry points
            self._report_progress(AnalysisPhase.IDENTIFYING_ENTRIES, 0, 1, "Identifying entry points")
            self.entry_identifier = EntryPointIdentifier(
                folder_path, 
                self.scanner.config_files,
                self.project_type
            )
            entry_points = self.entry_identifier.identify()
            self._report_progress(AnalysisPhase.IDENTIFYING_ENTRIES, 1, 1, f"Found {len(entry_points or [])} entry points")
            
            # Step 5: Check memory before heavy parsing
            exceeded, msg = memory_monitor.check_memory_threshold()
            if exceeded:
                logging.warning(f"High memory usage before parsing: {msg}")
                memory_monitor.trigger_cleanup()
            
            # Step 6: Parse
            self._report_progress(AnalysisPhase.PARSING_FILES, 0, 1, "Starting file parsing")
            
            # Create symbol table for cross-file resolution
            from .symbol_table import SymbolTable
            symbol_table = SymbolTable()
            
            # Load user configuration for parallel parsing setting
            from .config_loader import ConfigLoader
            flow_config = ConfigLoader.load(folder_path)
            use_parallel = flow_config.parallel_parsing
            
            if use_parallel:
                logging.info("Using parallel parsing mode")
                try:
                    # Use parallel parser
                    nodes, edges, file_symbols = self._parse_with_parallel(
                        folder_path,
                        entry_points,
                        self.project_type,
                        user_config,
                        symbol_table
                    )
                    self.parser = None  # No parser object in parallel mode
                    
                    # Create a minimal parser-like object for compatibility
                    class ParserStub:
                        def __init__(self, nodes, edges, file_symbols):
                            self.nodes = nodes
                            self.edges = edges
                            self.file_symbols = file_symbols
                    
                    self.parser = ParserStub(nodes, edges, file_symbols)
                    
                except Exception as parallel_error:
                    logging.warning(f"Parallel parsing failed, falling back to sequential: {parallel_error}")
                    use_parallel = False
            
            if not use_parallel:
                logging.info("Using sequential parsing mode")
                # Use traditional sequential parser
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
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Analyzing relationships with M4.0")
            semantic = M4SemanticAnalyzer(
                self.parser.nodes, 
                getattr(self.parser, 'file_symbols', {}),
                symbol_table=symbol_table
            )
            derived_edges = semantic.analyze()

            # Merge parser edges and semantic edges (avoid duplicates in a simple way)
            # Merge parser edges and semantic edges (avoid duplicates in a simple way)
            existing = {(e.source, e.target, e.type.value) for e in self.parser.edges}
            for e in derived_edges:
                key = (e.source, e.target, e.type.value)
                if key not in existing:
                    self.parser.edges.append(e)
                    existing.add(key)
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"Analysis complete: {len(derived_edges)} relationships found")
            
            # Step 5.5: M3 Data Flow Tracking
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Tracking data flows")
            from .data_flow_tracker import DataFlowTracker
            data_flow_tracker = DataFlowTracker()
            
            # Debug: Check if nodes exist
            node_count = len(self.parser.nodes) if hasattr(self.parser, 'nodes') and self.parser.nodes else 0
            logging.info(f"DEBUG: Parser has {node_count} nodes before data flow tracking")
            if node_count == 0:
                logging.warning("WARNING: No nodes available for data flow tracking!")
                if hasattr(self.parser, 'nodes'):
                    logging.warning(f"  - self.parser.nodes exists but is: {type(self.parser.nodes)}")
                else:
                    logging.warning("  - self.parser.nodes does not exist")
            
            # Track data flows through the AST nodes
            self._track_data_flows(data_flow_tracker, self.parser.nodes, getattr(self.parser, 'file_symbols', {}))
            
            # Store flow statistics and export flows
            flow_stats = data_flow_tracker.get_statistics()
            logging.info(f"Data flow tracking complete: {flow_stats['total_flows']} flows across {flow_stats['total_nodes']} nodes")
            
            # Add flow data to node metadata
            for node in self.parser.nodes:
                node_flows = data_flow_tracker.get_flows_for_node(node.id)
                if node_flows['incoming'] or node_flows['outgoing']:
                    if 'metadata' not in node.__dict__:
                        node.metadata = {}
                    node.metadata['data_flows'] = {
                        'incoming_count': len(node_flows['incoming']),
                        'outgoing_count': len(node_flows['outgoing']),
                        'total_flows': len(node_flows['incoming']) + len(node_flows['outgoing'])
                    }
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"Data flow tracking complete: {flow_stats['total_flows']} flows tracked")
            
            # Step 5.6: M3 State Pattern Detection
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Detecting state patterns")
            from .state_detector import StateDetector
            state_detector = StateDetector()
            
            # Detect state management patterns
            state_patterns = state_detector.detect_patterns(self.parser.nodes, getattr(self.parser, 'file_symbols', {}))
            
            # Store state pattern statistics
            state_stats = state_detector.get_statistics()
            logging.info(f"State pattern detection complete: {state_stats['total_patterns']} patterns found "
                        f"({state_stats['react_hooks']} hooks, {state_stats['zustand_stores']} Zustand, "
                        f"{state_stats['redux_stores']} Redux)")
            
            # Add state pattern data to node metadata
            for pattern in state_patterns:
                # Find the node and add pattern info
                for node in self.parser.nodes:
                    if node.id == pattern.node_id:
                        if 'metadata' not in node.__dict__:
                            node.metadata = {}
                        if 'state_patterns' not in node.metadata:
                            node.metadata['state_patterns'] = []
                        node.metadata['state_patterns'].append(pattern.to_dict())
                        break
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"State pattern detection complete: {state_stats['total_patterns']} patterns found")
            
            # Step 5.7: M3 Call Graph Construction
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Building call graph")
            from .call_graph import CallGraph
            call_graph = CallGraph()
            
            # Build the call graph from parsed nodes
            self._build_call_graph(call_graph, self.parser.nodes, getattr(self.parser, 'file_symbols', {}))
            
            # Store call graph statistics
            call_stats = call_graph.get_statistics()
            logging.info(f"Call graph construction complete: {call_stats['total_nodes']} functions, "
                        f"{call_stats['total_calls']} calls ({call_stats['method_calls']} methods, "
                        f"{call_stats['async_calls']} async)")
            
            # Add call graph data to node metadata
            for node in self.parser.nodes:
                if node.id in call_graph.nodes:
                    call_node = call_graph.nodes[node.id]
                    if 'metadata' not in node.__dict__:
                        node.metadata = {}
                    node.metadata['call_graph'] = {
                        'callers': list(call_node.callers),
                        'callees': list(call_node.callees),
                        'call_count': call_node.call_count,
                        'is_entry_point': len(call_node.callers) == 0,
                        'is_leaf': len(call_node.callees) == 0
                    }
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"Call graph complete: {call_stats['total_calls']} calls tracked")
            
            # Step 5.8: M3 DI Pattern Detection
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Detecting DI patterns")
            from .di_detector import DIDetector
            di_detector = DIDetector()
            
            # Detect dependency injection patterns
            di_patterns = di_detector.detect_patterns(self.parser.nodes)
            
            # Store DI pattern statistics
            di_stats = di_detector.get_statistics()
            logging.info(f"DI pattern detection complete: {di_stats['total_patterns']} patterns found "
                        f"({di_stats['constructor_injection']} constructor, "
                        f"{di_stats['setter_injection']} setter, "
                        f"{di_stats['property_injection']} property)")
            
            # Add DI pattern data to node metadata
            for pattern_id, pattern in di_patterns.items():
                # Find the class node and add DI info
                for node in self.parser.nodes:
                    node_metadata = getattr(node, 'metadata', {}) or {}
                    node_class_name = node_metadata.get('name', '')
                    
                    if node_class_name == pattern.class_name:
                        if 'metadata' not in node.__dict__:
                            node.metadata = {}
                        if 'di_patterns' not in node.metadata:
                            node.metadata['di_patterns'] = []
                        node.metadata['di_patterns'].append(pattern.to_dict())
                        break
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"DI pattern detection complete: {di_stats['total_patterns']} patterns found")
            
            # Step 5.9: M3 Type Inference
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Inferring types")
            from .type_inference_engine import TypeInferenceEngine, infer_type_from_node
            type_engine = TypeInferenceEngine()
            
            # Infer types for all nodes
            type_info_map = {}
            for node in self.parser.nodes:
                node_metadata = getattr(node, 'metadata', {}) or {}
                
                # Try to infer type based on node data
                if node.type.value in ['variable_declaration', 'parameter', 'function']:
                    try:
                        type_info = infer_type_from_node(node.__dict__, type_engine)
                        if type_info and type_info.type_name:
                            type_info_map[node.id] = type_info
                            
                            # Add type info to node metadata
                            if 'metadata' not in node.__dict__:
                                node.metadata = {}
                            node.metadata['type_info'] = {
                                'type': type_info.type_name,
                                'category': type_info.category,
                                'confidence': type_info.confidence
                            }
                    except Exception as e:
                        logging.debug(f"Could not infer type for node {node.id}: {e}")
                        continue
            
            type_stats = type_engine.get_stats()
            logging.info(f"Type inference complete: {type_stats.get('total_types', 0)} types inferred")
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"Type inference complete: {type_stats.get('total_types', 0)} types inferred")
            
            # Step 5.10: M3 External Dependency Analysis
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Analyzing dependencies")
            from .dependency_parser import DependencyParser
            dep_parser = DependencyParser(folder_path)
            
            # Parse package.json and node_modules
            dep_parser.parse_package_json()
            dep_parser.analyze_node_modules()
            
            # Extract imports from parsed nodes (skip if nodes are not dict-like)
            # Note: Current parser creates Node objects, not raw AST dicts
            # Import extraction needs raw AST nodes, so we skip this for now
            try:
                # Convert Node objects to dict if needed
                if self.parser.nodes and hasattr(self.parser.nodes[0], '__dict__'):
                    # These are Node objects, not raw AST nodes
                    # Skip import extraction as it needs raw AST structure
                    logging.debug("Skipping AST import extraction (Node objects instead of raw AST)")
                else:
                    dep_parser.extract_imports_from_ast(self.parser.nodes, folder_path)
            except Exception as e:
                logging.debug(f"Could not extract imports from AST: {e}")
            
            # Get dependency statistics
            dep_stats = dep_parser.get_dependency_stats()
            unused_deps = dep_parser.detect_unused_dependencies()
            
            logging.info(f"Dependency analysis complete: {dep_stats['total_dependencies']} dependencies, "
                        f"{dep_stats.get('used_dependencies', 0)} used, {len(unused_deps)} unused")
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"Dependency analysis complete: {dep_stats['total_dependencies']} dependencies")
            
            # Step 5.11: M3 API Call Tracking
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 0, 1, "Tracking API calls")
            from .api_tracker import APITracker
            api_tracker = APITracker()
            
            # Track API calls from parsed nodes using new tree-sitter compatible method
            try:
                api_tracker.track_api_calls_from_files(self.parser.nodes, folder_path)
                logging.info(f"API tracking: {len(api_tracker.api_calls)} calls detected")
            except Exception as e:
                logging.warning(f"Could not track API calls: {e}")
            
            # Get API statistics
            api_stats = api_tracker.get_api_stats()
            services = api_tracker.services
            
            logging.info(f"API tracking complete: {api_stats.get('total_calls', 0)} calls, "
                        f"{api_stats['unique_services']} services, "
                        f"{api_stats['authenticated_calls']} authenticated")
            
            # Add API call data to project metadata
            api_summary = {
                'total_calls': api_stats['total_calls'],
                'third_party_calls': api_stats['third_party_calls'],
                'local_calls': api_stats['local_calls'],
                'unique_domains': api_stats.get('unique_domains', 0),
                'authenticated_calls': api_stats['authenticated_calls'],
                'endpoints': [call.to_dict() for call in api_tracker.api_calls],
                'services': {name: {
                    'call_count': service.call_count,
                    'category': service.category,
                    'domain': service.domain
                } for name, service in services.items()},
                'by_category': api_tracker.get_services_by_category(),
                'by_type': api_stats.get('by_type', {})
            }
            
            self._report_progress(AnalysisPhase.ANALYZING_RELATIONSHIPS, 1, 1, 
                                f"API tracking complete: {api_stats['total_calls']} calls tracked")

            # Step 6: Generate workflow graph
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 0, 1, "Generating graph")
            graph_generator = WorkflowGraphGenerator(
                self.parser.nodes,
                self.parser.edges
            )
            graph_data = graph_generator.generate()
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 1, 1, "Graph generation complete")
            
            # Step 6.1: M4 Edge Intelligence Enhancement
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 0, 1, "Enriching edges with intelligence")
            try:
                from .edge_intelligence import EdgeIntelligenceEnricher
                edge_enricher = EdgeIntelligenceEnricher(
                    nodes=self.parser.nodes,
                    edges=self.parser.edges,
                    data_flows={'statistics': flow_stats},
                    api_calls=api_summary
                )
                edge_enricher.enrich_all_edges()
                edge_enrichment_stats = edge_enricher.get_enrichment_stats()
                logging.info(f"Edge enrichment: {edge_enrichment_stats['enriched_edges']}/{edge_enrichment_stats['total_edges']} edges enriched, "
                           f"avg weight: {edge_enrichment_stats['avg_weight']:.2f}, "
                           f"avg criticality: {edge_enrichment_stats['avg_criticality']:.2f}")
            except Exception as e:
                logging.warning(f"Edge intelligence enrichment failed: {e}")
                edge_enrichment_stats = {}
            
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 1, 1, 
                                f"Edge enrichment complete: {edge_enrichment_stats.get('enriched_edges', 0)} edges")
            
            # Step 6.2: M4 Layout Hints Analysis
            layout_hints = {}
            try:
                from .layout_analyzer import LayoutHintsAnalyzer
                
                # Extract entry point IDs from entry_points list
                # entry_points is a list of dicts like [{"type": "react_app", "value": "src/main.tsx", ...}]
                entry_ids = []
                if entry_points:
                    for ep in entry_points:
                        # Try to find matching node by file path
                        ep_value = ep.get('value', '')
                        for node in self.parser.nodes:
                            if ep_value in node.file:
                                entry_ids.append(node.id)
                                break
                
                layout_analyzer = LayoutHintsAnalyzer(
                    nodes=self.parser.nodes,
                    edges=self.parser.edges,
                    entry_points=entry_ids,
                    project_root=folder_path
                )
                layout_hints = layout_analyzer.analyze()
                layout_recommendations = layout_analyzer.get_layout_recommendations()
                layout_hints['recommendations'] = layout_recommendations
                logging.info(f"Layout hints: {layout_hints['statistics']['total_clusters']} clusters, "
                           f"{layout_hints['statistics']['total_islands']} islands, "
                           f"{layout_hints['statistics']['total_layers']} layers")
            except Exception as e:
                logging.warning(f"Layout hints analysis failed: {e}")
                layout_hints = {}
            
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 1, 1, 
                                f"Layout hints complete: {layout_hints.get('statistics', {}).get('total_clusters', 0)} clusters")
            
            # Step 6.3: M4 Module Boundaries with Metrics
            module_boundaries = {}
            try:
                from .module_boundaries import ModuleBoundariesAnalyzer
                module_analyzer = ModuleBoundariesAnalyzer(
                    nodes=self.parser.nodes,
                    edges=self.parser.edges,
                    project_root=folder_path
                )
                module_boundaries = module_analyzer.analyze()
                logging.info(f"Module boundaries: {module_boundaries['statistics']['total_modules']} modules, "
                           f"avg cohesion: {module_boundaries['statistics']['avg_cohesion']:.2f}, "
                           f"avg coupling: {module_boundaries['statistics']['avg_coupling']:.2f}, "
                           f"{len(module_boundaries['hotspots'])} hotspots")
            except Exception as e:
                logging.warning(f"Module boundaries analysis failed: {e}")
                module_boundaries = {}
            
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 1, 1, 
                                f"Module boundaries complete: {module_boundaries.get('statistics', {}).get('total_modules', 0)} modules")
            
            # Step 6.4: M4 Runtime Instrumentation Metadata
            instrumentation_metadata = {}
            try:
                from .runtime_instrumentation import RuntimeInstrumentationAnalyzer
                
                # Extract entry point IDs (same as layout hints)
                entry_ids = []
                if entry_points:
                    for ep in entry_points:
                        ep_value = ep.get('value', '')
                        for node in self.parser.nodes:
                            if ep_value in node.file:
                                entry_ids.append(node.id)
                                break
                
                instrumentation_analyzer = RuntimeInstrumentationAnalyzer(
                    nodes=self.parser.nodes,
                    edges=self.parser.edges,
                    entry_points=entry_ids,
                    data_flows={'statistics': flow_stats}
                )
                instrumentation_metadata = instrumentation_analyzer.analyze()
                logging.info(f"Runtime instrumentation: {instrumentation_metadata['statistics']['total_hookable_points']} hookable points, "
                           f"{instrumentation_metadata['statistics']['total_state_variables']} state vars, "
                           f"{instrumentation_metadata['statistics']['total_execution_paths']} paths")
            except Exception as e:
                logging.warning(f"Runtime instrumentation analysis failed: {e}")
                instrumentation_metadata = {}
            
            self._report_progress(AnalysisPhase.GENERATING_GRAPH, 1, 1, 
                                f"Instrumentation complete: {instrumentation_metadata.get('statistics', {}).get('total_hookable_points', 0)} hooks")
            
            project_name = self._derive_project_name(folder_path)
            language, framework = self._derive_language_and_framework(self.project_type)

            result = {
                "project_info": {
                    "name": project_name,
                    "path": folder_path,
                    "type": self.project_type.value,
                    "language": language,
                    "framework": framework,
                    "config_files": scan_results.get("config_files", []),
                    "entry_points": entry_points or []
                },
                "graph": graph_data or {"nodes": [], "edges": [], "metadata": {"total_nodes": 0, "total_edges": 0, "node_types": [], "edge_types": []}},
                "m3_analysis": {
                    "data_flows": {
                        "statistics": flow_stats,
                        "total_flows": flow_stats['total_flows']
                    },
                    "state_patterns": {
                        "statistics": state_stats,
                        "patterns": [p.to_dict() for p in state_patterns]
                    },
                    "call_graph": {
                        "statistics": call_stats,
                        "entry_points": [node_id for node_id, node in call_graph.nodes.items() 
                                        if len(node.callers) == 0]
                    },
                    "dependency_injection": {
                        "statistics": di_stats,
                        "patterns": {k: v.to_dict() for k, v in di_patterns.items()}
                    },
                    "type_inference": {
                        "statistics": type_stats,
                        "inferred_types": {node_id: {
                            'type': ti.type_name,
                            'category': ti.category,
                            'confidence': ti.confidence
                        } for node_id, ti in type_info_map.items()}
                    },
                    "dependencies": {
                        "statistics": dep_stats,
                        "unused": unused_deps,
                        "by_type": dep_stats.get('by_type', {})
                    },
                    "api_calls": api_summary,
                    "edge_intelligence": {
                        "statistics": edge_enrichment_stats,
                        "enriched_count": edge_enrichment_stats.get('enriched_edges', 0),
                        "avg_weight": edge_enrichment_stats.get('avg_weight', 0.0),
                        "avg_criticality": edge_enrichment_stats.get('avg_criticality', 0.0)
                    },
                    "layout_hints": {
                        "statistics": layout_hints.get('statistics', {}),
                        "directory_clusters": layout_hints.get('directory_clusters', []),
                        "dependency_islands": layout_hints.get('dependency_islands', []),
                        "architectural_layers": layout_hints.get('architectural_layers', []),
                        "entry_hierarchy": layout_hints.get('entry_hierarchy', []),
                        "recommendations": layout_hints.get('recommendations', {})
                    },
                    "module_boundaries": {
                        "statistics": module_boundaries.get('statistics', {}),
                        "modules": module_boundaries.get('modules', []),
                        "coupling_matrix": module_boundaries.get('coupling_matrix', {}),
                        "hotspots": module_boundaries.get('hotspots', [])
                    },
                    "runtime_instrumentation": {
                        "statistics": instrumentation_metadata.get('statistics', {}),
                        "hookable_points": instrumentation_metadata.get('hookable_points', []),
                        "observable_state": instrumentation_metadata.get('observable_state', []),
                        "execution_paths": instrumentation_metadata.get('execution_paths', [])
                    }
                }
            }
            
            # NEW: Save incremental analysis snapshot
            try:
                from .incremental_analyzer import IncrementalAnalyzer
                incremental = IncrementalAnalyzer(Path(folder_path))
                
                # Collect analyzed files
                analyzed_files = []
                if self.parser:
                    # Extract file paths from parsed files
                    for relative_path in getattr(self.parser, 'parsed_files', set()):
                        analyzed_files.append(Path(folder_path) / relative_path)
                
                # Save snapshot
                incremental.save_snapshot(
                    files=analyzed_files,
                    total_nodes=graph_data['metadata']['total_nodes'],
                    total_edges=graph_data['metadata']['total_edges'],
                    project_type=self.project_type.value
                )
            except Exception as snapshot_error:
                logging.warning(f"Failed to save incremental snapshot: {snapshot_error}")
            
            # Mark as completed
            if self.progress_tracker:
                self.progress_tracker.mark_completed()
                if self.progress_callback:
                    self.progress_callback(self.progress_tracker.get_current_status())
            
            logging.info(f"Analysis complete. Generated {graph_data['metadata']['total_nodes']} nodes "
                        f"and {graph_data['metadata']['total_edges']} edges")
            
            return result
            
        except Exception as e:
            # Mark as failed
            if self.progress_tracker:
                self.progress_tracker.mark_failed(str(e))
                if self.progress_callback:
                    self.progress_callback(self.progress_tracker.get_current_status())
            
            logging.error(f"Analysis failed for {folder_path}: {e}")
            raise
    
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
    
    def _parse_with_parallel(
        self,
        folder_path: str,
        entry_points: List[Dict],
        project_type: ProjectType,
        user_config: Dict[str, Any],
        symbol_table: 'SymbolTable'
    ) -> tuple[List[Node], List[Edge], Dict[str, Any]]:
        """Parse files using parallel processing.
        
        Args:
            folder_path: Root path of the project
            entry_points: List of entry point dictionaries
            project_type: Detected project type
            user_config: User configuration
            symbol_table: Symbol table for cross-file resolution
            
        Returns:
            Tuple of (nodes, edges, file_symbols)
        """
        from pathlib import Path
        from .parallel_parser import ParallelParser, create_parse_task
        from .models import Node, Edge, EdgeType
        
        project_path = Path(folder_path)
        
        # Determine which files to parse
        extensions_to_parse = {'.js', '.jsx', '.ts', '.tsx', '.py', '.java'}
        files_to_parse = []
        entry_file_paths = set()
        
        # Collect entry point files
        for entry in entry_points:
            entry_path = project_path / entry['value']
            if entry_path.exists() and entry_path.is_file():
                entry_file_paths.add(entry_path)
        
        # Scan for additional files (same logic as parser.py)
        for location in ['src', 'app', 'lib', 'components', 'pages', 'api', 'server']:
            location_path = project_path / location
            if location_path.exists():
                for root, dirs, files in os.walk(location_path):
                    # Skip common directories
                    dirs[:] = [d for d in dirs if d not in {
                        'node_modules', '__pycache__', '.git', '__pycache__'
                    }]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix.lower() in extensions_to_parse:
                            files_to_parse.append(file_path)
        
        # Create parse tasks
        parse_tasks = []
        for file_path in files_to_parse:
            is_entry = file_path in entry_file_paths
            task = create_parse_task(file_path, project_path, is_entry)
            if task:
                parse_tasks.append(task)
        
        logging.info(f"Parallel parser: {len(parse_tasks)} files to parse ({len(entry_file_paths)} entry points)")
        
        # Plugin selector function
        from .plugins.languages import JavaScriptPlugin, JavaPlugin, PythonPlugin
        
        def select_plugin(file_path: Path) -> Optional[str]:
            ext = file_path.suffix.lower()
            if ext in ['.js', '.jsx', '.ts', '.tsx']:
                return 'JavaScriptPlugin'
            elif ext == '.py':
                return 'PythonPlugin'
            elif ext == '.java':
                return 'JavaPlugin'
            return None
        
        # Create parallel parser with optimal settings
        worker_count = ParallelParser.get_optimal_worker_count()
        chunk_size = ParallelParser.estimate_chunk_size(len(parse_tasks), worker_count)
        
        parallel_parser = ParallelParser(
            project_path,
            worker_count=worker_count,
            chunk_size=chunk_size,
            progress_callback=self._create_parser_progress_callback()
        )
        
        # Parse files in parallel
        results = parallel_parser.parse_files(parse_tasks, select_plugin)
        
        # Aggregate results
        all_nodes = []
        all_edges = []
        file_symbols = {}
        
        for relative_path, result in results.items():
            if result.success:
                all_nodes.extend(result.nodes)
                file_symbols[relative_path] = result.symbols
        
        logging.info(f"Parallel parsing complete: {len(all_nodes)} nodes from {len(results)} files")
        
        return all_nodes, all_edges, file_symbols

    def _load_user_config(self, folder_path: str) -> Optional[Dict[str, Any]]:
        try:
            config_path = os.path.join(folder_path, '.devscope.yml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Failed to load user config: {e}")
        return {}
    
    def _track_data_flows(self, tracker, nodes: List[Node], file_symbols: Dict[str, Dict]):
        """
        Track data flows through the parsed nodes.
        
        This method analyzes nodes and tracks:
        - Variable assignments (from metadata)
        - React useState hooks (from detected hooks)
        - Function calls with parameters
        - Component prop flows
        
        Args:
            tracker: DataFlowTracker instance
            nodes: List of parsed nodes (our Node objects, not raw AST)
            file_symbols: Symbol information per file
        """
        logging.info(f"Starting data flow tracking for {len(nodes)} nodes")
        
        # Create a mapping of node IDs for quick lookup
        node_map = {node.id: node for node in nodes}
        
        # Track flows for each node
        flow_count = 0
        for node in nodes:
            try:
                # Get node metadata
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                # Track React component flows (component nodes)
                if node_type == 'component':
                    self._track_component_flow(tracker, node, node_map, metadata)
                    flow_count += 1
                    
                    # Check for React hooks in components
                    hooks = metadata.get('hooks', [])
                    if hooks and any('useState' in str(h) for h in hooks):
                        self._track_react_state_flow(tracker, node, node_map, metadata)
                        flow_count += 1
                
                # Track function flows (function nodes)
                if node_type == 'function':
                    # Check for React hooks in functions too (for custom hooks)
                    hooks = metadata.get('hooks', [])
                    if hooks and any('useState' in str(h) for h in hooks):
                        self._track_react_state_flow(tracker, node, node_map, metadata)
                        flow_count += 1
                    
                    # Track function parameter flows
                    if metadata.get('parameters'):
                        self._track_function_flow(tracker, node, node_map, metadata)
                        flow_count += 1
                
                # Track variable assignments from metadata
                if metadata.get('variables'):
                    self._track_variables_flow(tracker, node, node_map, metadata)
                    flow_count += 1
                
            except Exception as e:
                logging.warning(f"Failed to track data flow for node {node.id}: {e}")
        
        logging.info(f"Data flow tracking complete: {flow_count} nodes processed")
        
        logging.info(f"Data flow tracking completed: {len(tracker.flows)} flows tracked")
    
    def _track_component_flow(self, tracker, node, node_map, metadata):
        """Track React component prop flows"""
        props = metadata.get('props', [])
        if props:
            for prop in props:
                prop_name = prop if isinstance(prop, str) else prop.get('name', 'prop')
                # Create flow for prop into component
                tracker.track_assignment(
                    target_id=f"{node.id}_prop_{prop_name}",
                    source_id=f"{node.id}_parent",
                    variable_name=prop_name,
                    line_number=getattr(node, 'line_number', None),
                    is_const=True
                )
    
    def _track_variables_flow(self, tracker, node, node_map, metadata):
        """Track variable declarations from metadata"""
        variables = metadata.get('variables', [])
        for var in variables:
            if isinstance(var, dict):
                var_name = var.get('name', 'var')
                is_const = var.get('kind') == 'const'
                tracker.track_assignment(
                    target_id=f"{node.id}_var_{var_name}",
                    source_id=f"{node.id}_init",
                    variable_name=var_name,
                    line_number=var.get('line', getattr(node, 'line_number', None)),
                    is_const=is_const
                )
    
    def _track_assignment_flow(self, tracker, node, node_map, metadata):
        """Track variable assignment flow"""
        var_name = node.name or metadata.get('name', 'unknown')
        is_const = metadata.get('is_const', False)
        
        # Create a simple flow edge
        tracker.track_assignment(
            target_id=node.id,
            source_id=metadata.get('initializer_id', f"{node.id}_init"),
            variable_name=var_name,
            line_number=getattr(node, 'line_number', None),
            is_const=is_const
        )
    
    def _track_react_state_flow(self, tracker, node, node_map, metadata):
        """Track React useState hook flow"""
        hooks = metadata.get('hooks', [])
        for hook in hooks:
            hook_str = str(hook)
            if 'useState' in hook_str:
                # Try to extract state variable name from hook string or metadata
                state_name = 'state'
                setter_name = 'setState'
                
                # Parse hook string like "useState" or "const [count, setCount] = useState(0)"
                if isinstance(hook, dict):
                    state_name = hook.get('state_var', 'state')
                    setter_name = hook.get('setter_var', 'setState')
                
                tracker.track_react_state_init(
                    hook_call_id=f"{node.id}_hook_{state_name}",
                    state_var_id=f"{node.id}_state_{state_name}",
                    setter_id=f"{node.id}_setter_{state_name}",
                    state_name=state_name,
                    setter_name=setter_name,
                    line_number=getattr(node, 'line_number', None)
                )
    
    def _track_function_flow(self, tracker, node, node_map, metadata):
        """Track function call and parameter flow"""
        params = metadata.get('parameters', [])
        if params:
            # Track parameter flows
            for i, param in enumerate(params):
                param_name = param if isinstance(param, str) else param.get('name', f'param{i}')
                tracker.create_node(
                    node_id=f"{node.id}_param_{i}",
                    node_type='parameter',
                    name=param_name,
                    scope=node.id
                )
    
    def _track_property_flow(self, tracker, node, node_map, metadata):
        """Track property access flow"""
        obj_name = metadata.get('object', 'object')
        prop_name = metadata.get('property', 'property')
        
        tracker.track_property_access(
            object_id=f"{node.id}_obj",
            property_id=node.id,
            property_name=prop_name,
            line_number=getattr(node, 'line_number', None)
        )
    
    def _build_call_graph(self, call_graph, nodes: List[Node], file_symbols: Dict[str, Dict]):
        """
        Build call graph from AST nodes.
        
        This method analyzes nodes and constructs a call graph showing:
        - Function definitions
        - Function calls (direct, method, static, constructor)
        - Caller-callee relationships
        - Call sites
        
        Args:
            call_graph: CallGraph instance
            nodes: List of parsed nodes
            file_symbols: Symbol information per file
        """
        logging.info("Starting call graph construction")
        
        # Create a mapping of node IDs for quick lookup
        node_map = {node.id: node for node in nodes}
        
        # Phase 1: Add all functions to the call graph
        for node in nodes:
            try:
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                # Add functions, methods, and classes
                if node_type in ['function', 'arrow_function', 'function_declaration', 
                                'method_definition', 'class_declaration']:
                    self._add_function_to_call_graph(call_graph, node, metadata)
                    
            except Exception as e:
                logging.warning(f"Failed to add function {node.id} to call graph: {e}")
        
        # Phase 2: Track function calls and build relationships
        for node in nodes:
            try:
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                
                # Track function calls
                if node_type in ['call_expression', 'new_expression']:
                    self._track_function_call(call_graph, node, node_map, metadata)
                
                # Track method invocations
                elif node_type in ['member_expression', 'member_access']:
                    if metadata.get('is_call', False):
                        self._track_method_invocation(call_graph, node, node_map, metadata)
                
                # Track constructor calls
                elif node_type == 'new_expression':
                    self._track_constructor_call(call_graph, node, node_map, metadata)
                
            except Exception as e:
                logging.warning(f"Failed to track call for node {node.id}: {e}")
        
        # Phase 3: Track higher-order functions (callbacks, promises, async)
        for node in nodes:
            try:
                metadata = getattr(node, 'metadata', {}) or {}
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                code_snippet = metadata.get('code_snippet', '')
                
                # Track callbacks (functions passed as arguments)
                if self._is_callback(node, metadata, code_snippet):
                    self._track_callback(call_graph, node, node_map, metadata)
                
                # Track promise chains (.then, .catch, .finally)
                if self._is_promise_chain(node, metadata, code_snippet):
                    self._track_promise_chain(call_graph, node, node_map, metadata)
                
                # Track async/await patterns
                if self._is_async_await(node, metadata, code_snippet):
                    self._track_async_await(call_graph, node, node_map, metadata)
                
                # Track event handlers (addEventListener, onClick, etc.)
                if self._is_event_handler(node, metadata, code_snippet):
                    self._track_event_handler(call_graph, node, node_map, metadata)
                
            except Exception as e:
                logging.warning(f"Failed to track higher-order function for node {node.id}: {e}")
        
        logging.info(f"Call graph construction completed: {len(call_graph.nodes)} functions, {len(call_graph.call_sites)} calls")
    
    def _add_function_to_call_graph(self, call_graph, node, metadata):
        """Add a function to the call graph"""
        from .call_graph import CallType
        
        node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
        
        # Determine function type
        func_type = 'function'
        if node_type == 'method_definition':
            func_type = 'method'
        elif node_type == 'class_declaration':
            func_type = 'constructor'
        
        # Check if async or generator
        is_async = metadata.get('is_async', False) or metadata.get('async', False)
        is_generator = metadata.get('is_generator', False) or metadata.get('generator', False)
        
        # Get parameters
        parameters = metadata.get('parameters', [])
        if isinstance(parameters, list):
            param_names = [p if isinstance(p, str) else p.get('name', '') for p in parameters]
        else:
            param_names = []
        
        call_graph.add_function(
            node_id=node.id,
            name=node.name or metadata.get('name', 'anonymous'),
            node_type=func_type,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            is_async=is_async,
            is_generator=is_generator,
            parameters=param_names,
            metadata=metadata
        )
    
    def _track_function_call(self, call_graph, node, node_map, metadata):
        """Track a function call"""
        from .call_graph import CallType
        
        # Get caller (parent function)
        caller_id = metadata.get('parent_function') or metadata.get('scope', 'global')
        
        # Get callee (function being called)
        callee_name = metadata.get('callee') or metadata.get('function_name', 'unknown')
        
        # Try to find callee in call graph
        callee_nodes = call_graph.get_functions_by_name(callee_name)
        callee_id = callee_nodes[0].node_id if callee_nodes else f"external_{callee_name}"
        
        # Get arguments
        arguments = metadata.get('arguments', [])
        arg_ids = [f"{node.id}_arg_{i}" for i in range(len(arguments))]
        
        # Check if async
        is_async = metadata.get('is_async', False) or 'await' in metadata.get('code_snippet', '')
        
        # Add the call
        call_graph.add_call(
            call_id=f"{node.id}_call",
            caller_id=caller_id,
            callee_id=callee_id,
            call_type=CallType.ASYNC_CALL if is_async else CallType.FUNCTION_CALL,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            arguments=arg_ids,
            is_async=is_async,
            confidence=0.8
        )
    
    def _track_method_invocation(self, call_graph, node, node_map, metadata):
        """Track a method invocation (obj.method() or Class.staticMethod())"""
        from .call_graph import CallType
        
        # Get caller
        caller_id = metadata.get('parent_function') or metadata.get('scope', 'global')
        
        # Get object and method
        object_name = metadata.get('object', 'unknown')
        method_name = metadata.get('property', metadata.get('method', 'unknown'))
        
        # Determine if it's a static method (starts with uppercase)
        is_static = object_name and object_name[0].isupper()
        
        # Create callee ID
        callee_id = f"{object_name}.{method_name}"
        
        # Get arguments
        arguments = metadata.get('arguments', [])
        arg_ids = [f"{node.id}_arg_{i}" for i in range(len(arguments))]
        
        # Add the call
        call_graph.add_call(
            call_id=f"{node.id}_method_call",
            caller_id=caller_id,
            callee_id=callee_id,
            call_type=CallType.STATIC_CALL if is_static else CallType.METHOD_CALL,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            arguments=arg_ids,
            confidence=0.9,
            metadata={'object': object_name, 'method': method_name}
        )
    
    def _track_constructor_call(self, call_graph, node, node_map, metadata):
        """Track a constructor call (new ClassName())"""
        from .call_graph import CallType
        
        # Get caller
        caller_id = metadata.get('parent_function') or metadata.get('scope', 'global')
        
        # Get class name
        class_name = metadata.get('class_name') or metadata.get('callee', 'Unknown')
        callee_id = f"new_{class_name}"
        
        # Get arguments
        arguments = metadata.get('arguments', [])
        arg_ids = [f"{node.id}_arg_{i}" for i in range(len(arguments))]
        
        # Add the call
        call_graph.add_call(
            call_id=f"{node.id}_constructor",
            caller_id=caller_id,
            callee_id=callee_id,
            call_type=CallType.CONSTRUCTOR_CALL,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            arguments=arg_ids,
            confidence=1.0,
            metadata={'class_name': class_name}
        )
    
    # Higher-Order Function Analysis Methods (M3.2.3)
    
    def _is_callback(self, node, metadata, code_snippet):
        """Check if node is a callback function"""
        # Look for function expressions passed as arguments
        parent = metadata.get('parent_type', '')
        return (parent == 'call_expression' and 
                metadata.get('is_argument', False) and
                metadata.get('node_kind') in ['arrow_function', 'function_expression'])
    
    def _track_callback(self, call_graph, node, node_map, metadata):
        """Track callback function"""
        from .call_graph import CallType
        
        # Get the function receiving the callback
        parent_function = metadata.get('parent_function', 'unknown')
        callback_id = node.id
        
        # Add the callback relationship
        call_graph.add_call(
            call_id=f"{node.id}_callback",
            caller_id=parent_function,
            callee_id=callback_id,
            call_type=CallType.CALLBACK,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            confidence=0.9,
            metadata={'callback_type': 'function_argument'}
        )
    
    def _is_promise_chain(self, node, metadata, code_snippet):
        """Check if node is part of a promise chain"""
        return ('.then(' in code_snippet or 
                '.catch(' in code_snippet or 
                '.finally(' in code_snippet or
                'Promise.' in code_snippet)
    
    def _track_promise_chain(self, call_graph, node, node_map, metadata):
        """Track promise chain (.then, .catch, .finally)"""
        from .call_graph import CallType
        
        code_snippet = metadata.get('code_snippet', '')
        caller_id = metadata.get('parent_function', 'global')
        
        # Determine promise method
        if '.then(' in code_snippet:
            promise_method = 'then'
            call_type = CallType.PROMISE_THEN
        elif '.catch(' in code_snippet:
            promise_method = 'catch'
            call_type = CallType.PROMISE_CATCH
        elif '.finally(' in code_snippet:
            promise_method = 'finally'
            call_type = CallType.PROMISE_THEN  # Use THEN for finally
        else:
            promise_method = 'promise'
            call_type = CallType.ASYNC_CALL
        
        # Track the promise chain
        call_graph.add_call(
            call_id=f"{node.id}_promise_{promise_method}",
            caller_id=caller_id,
            callee_id=f"{node.id}_promise_handler",
            call_type=call_type,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            is_async=True,
            confidence=0.95,
            metadata={'promise_method': promise_method, 'is_promise_chain': True}
        )
    
    def _is_async_await(self, node, metadata, code_snippet):
        """Check if node uses async/await"""
        return ('await ' in code_snippet or 
                metadata.get('is_async', False) or
                metadata.get('has_await', False))
    
    def _track_async_await(self, call_graph, node, node_map, metadata):
        """Track async/await patterns"""
        from .call_graph import CallType
        
        code_snippet = metadata.get('code_snippet', '')
        
        # If this is an async function definition, it's already tracked
        # We're interested in await expressions
        if 'await ' in code_snippet:
            caller_id = metadata.get('parent_function', 'global')
            
            # Extract awaited function (simplified)
            awaited_func = code_snippet.split('await ')[1].split('(')[0].strip() if 'await ' in code_snippet else 'unknown'
            
            call_graph.add_call(
                call_id=f"{node.id}_await",
                caller_id=caller_id,
                callee_id=f"async_{awaited_func}",
                call_type=CallType.ASYNC_CALL,
                file_path=node.file if hasattr(node, 'file') else None,
                line_number=metadata.get('line_number'),
                is_async=True,
                confidence=0.85,
                metadata={'is_await': True, 'awaited_function': awaited_func}
            )
    
    def _is_event_handler(self, node, metadata, code_snippet):
        """Check if node is an event handler"""
        return ('addEventListener' in code_snippet or
                'onClick' in code_snippet or
                'onChange' in code_snippet or
                'onSubmit' in code_snippet or
                metadata.get('is_event_handler', False))
    
    def _track_event_handler(self, call_graph, node, node_map, metadata):
        """Track event handler"""
        from .call_graph import CallType
        
        code_snippet = metadata.get('code_snippet', '')
        
        # Determine event type
        if 'onClick' in code_snippet:
            event_type = 'click'
        elif 'onChange' in code_snippet:
            event_type = 'change'
        elif 'onSubmit' in code_snippet:
            event_type = 'submit'
        elif 'addEventListener' in code_snippet:
            # Try to extract event name
            if "'" in code_snippet:
                event_type = code_snippet.split("'")[1]
            elif '"' in code_snippet:
                event_type = code_snippet.split('"')[1]
            else:
                event_type = 'event'
        else:
            event_type = 'unknown'
        
        caller_id = metadata.get('parent_function', 'global')
        
        call_graph.add_call(
            call_id=f"{node.id}_event_{event_type}",
            caller_id=caller_id,
            callee_id=f"{node.id}_handler",
            call_type=CallType.EVENT_HANDLER,
            file_path=node.file if hasattr(node, 'file') else None,
            line_number=metadata.get('line_number'),
            confidence=0.9,
            metadata={'event_type': event_type, 'is_event_handler': True}
        )
    
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
