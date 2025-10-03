"""
External API Tracker for M3.5.2

This module provides comprehensive external API call tracking including:
- HTTP request detection (fetch, axios, XHR, node-fetch)
- WebSocket connection tracking
- GraphQL query/mutation detection
- Third-party service identification
- API endpoint extraction
- Request method tracking (GET, POST, PUT, DELETE, etc.)
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse


class APICallType(Enum):
    """Type of API call."""
    HTTP_FETCH = "http_fetch"  # fetch() API
    HTTP_AXIOS = "http_axios"  # axios library
    HTTP_XHR = "http_xhr"  # XMLHttpRequest
    HTTP_NODE_FETCH = "http_node_fetch"  # node-fetch
    HTTP_REQUEST = "http_request"  # request library
    HTTP_GOT = "http_got"  # got library
    WEBSOCKET = "websocket"  # WebSocket connection
    GRAPHQL = "graphql"  # GraphQL query/mutation
    REST = "rest"  # REST API call
    GRPC = "grpc"  # gRPC call


class HTTPMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"
    CONNECT = "CONNECT"


@dataclass
class APIEndpoint:
    """Information about an API endpoint."""
    url: str
    method: Optional[HTTPMethod] = None
    call_type: APICallType = APICallType.HTTP_FETCH
    
    # Source information
    source_file: str = ""
    line_number: int = 0
    
    # Request details
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body_type: Optional[str] = None  # json, form-data, text, etc.
    
    # Service identification
    service_name: Optional[str] = None  # e.g., "GitHub API", "Stripe"
    is_third_party: bool = True
    domain: Optional[str] = None
    
    # Metadata
    is_authenticated: bool = False
    auth_type: Optional[str] = None  # bearer, basic, api-key, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'url': self.url,
            'method': self.method.value if self.method else None,
            'call_type': self.call_type.value,
            'source_file': self.source_file,
            'line_number': self.line_number,
            'headers': self.headers,
            'query_params': self.query_params,
            'body_type': self.body_type,
            'service_name': self.service_name,
            'is_third_party': self.is_third_party,
            'domain': self.domain,
            'is_authenticated': self.is_authenticated,
            'auth_type': self.auth_type
        }


@dataclass
class ThirdPartyService:
    """Information about a third-party service."""
    name: str
    domain: str
    endpoints: List[APIEndpoint] = field(default_factory=list)
    call_count: int = 0
    
    # Service metadata
    category: Optional[str] = None  # payment, analytics, auth, storage, etc.
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'domain': self.domain,
            'endpoints': [e.to_dict() for e in self.endpoints],
            'call_count': self.call_count,
            'category': self.category,
            'description': self.description
        }


class APITracker:
    """
    External API tracker for semantic analysis.
    
    Features:
    - Detect HTTP calls (fetch, axios, XHR, etc.)
    - Track WebSocket connections
    - Identify GraphQL operations
    - Extract API endpoints
    - Identify third-party services
    - Track authentication methods
    """
    
    def __init__(self):
        """Initialize API tracker."""
        self.api_calls: List[APIEndpoint] = []
        self.services: Dict[str, ThirdPartyService] = {}
        
        # Known third-party service patterns
        self.service_patterns = self._init_service_patterns()
        
        logging.info("APITracker initialized")
    
    def _init_service_patterns(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize known third-party service patterns.
        
        Returns:
            Dict mapping domain patterns to service info
        """
        return {
            # Payment Services
            'stripe.com': {'name': 'Stripe', 'category': 'payment'},
            'paypal.com': {'name': 'PayPal', 'category': 'payment'},
            'square.com': {'name': 'Square', 'category': 'payment'},
            
            # Cloud Services
            'amazonaws.com': {'name': 'AWS', 'category': 'cloud'},
            'azure.com': {'name': 'Azure', 'category': 'cloud'},
            'googleapis.com': {'name': 'Google Cloud', 'category': 'cloud'},
            
            # Analytics
            'google-analytics.com': {'name': 'Google Analytics', 'category': 'analytics'},
            'mixpanel.com': {'name': 'Mixpanel', 'category': 'analytics'},
            'segment.com': {'name': 'Segment', 'category': 'analytics'},
            
            # Authentication
            'auth0.com': {'name': 'Auth0', 'category': 'auth'},
            'okta.com': {'name': 'Okta', 'category': 'auth'},
            'firebase.com': {'name': 'Firebase Auth', 'category': 'auth'},
            
            # Communication
            'twilio.com': {'name': 'Twilio', 'category': 'communication'},
            'sendgrid.com': {'name': 'SendGrid', 'category': 'communication'},
            'mailgun.com': {'name': 'Mailgun', 'category': 'communication'},
            
            # Storage
            's3.amazonaws.com': {'name': 'AWS S3', 'category': 'storage'},
            'cloudinary.com': {'name': 'Cloudinary', 'category': 'storage'},
            
            # Social/APIs
            'api.github.com': {'name': 'GitHub API', 'category': 'development'},
            'api.twitter.com': {'name': 'Twitter API', 'category': 'social'},
            'graph.facebook.com': {'name': 'Facebook Graph API', 'category': 'social'},
            'slack.com/api': {'name': 'Slack API', 'category': 'communication'},
            
            # Database
            'mongodb.com': {'name': 'MongoDB Atlas', 'category': 'database'},
            'supabase.co': {'name': 'Supabase', 'category': 'database'},
            
            # Monitoring
            'sentry.io': {'name': 'Sentry', 'category': 'monitoring'},
            'datadog.com': {'name': 'Datadog', 'category': 'monitoring'}
        }
    
    def track_api_calls_from_ast(self, nodes: List[Dict[str, Any]], file_path: str):
        """
        Extract API calls from AST nodes.
        
        Args:
            nodes: List of AST nodes
            file_path: Path to the source file
        """
        for node in nodes:
            node_type = node.get('type', '')
            
            # fetch() calls
            if node_type == 'call_expression':
                self._check_fetch_call(node, file_path)
                self._check_axios_call(node, file_path)
                self._check_xhr_call(node, file_path)
                self._check_websocket_call(node, file_path)
            
            # Method calls (e.g., axios.get, axios.post)
            elif node_type == 'member_expression':
                self._check_method_call(node, file_path)
    
    def _check_fetch_call(self, node: Dict[str, Any], file_path: str):
        """Check for fetch() API calls."""
        callee = node.get('function', {})
        if not isinstance(callee, dict):
            return
        
        func_name = callee.get('name', '')
        if func_name not in ['fetch', 'node-fetch']:
            return
        
        # Extract URL
        args = node.get('arguments', [])
        if not args:
            return
        
        first_arg = args[0]
        url = self._extract_url_from_arg(first_arg)
        
        if not url:
            return
        
        # Extract options (second argument)
        method = HTTPMethod.GET  # default
        headers = {}
        body_type = None
        
        if len(args) > 1:
            options = args[1]
            if isinstance(options, dict):
                # Extract method
                method_value = options.get('method', 'GET')
                if isinstance(method_value, str):
                    try:
                        method = HTTPMethod[method_value.upper()]
                    except KeyError:
                        pass
                
                # Extract headers
                headers_obj = options.get('headers', {})
                if isinstance(headers_obj, dict):
                    headers = headers_obj
                
                # Detect body type
                if 'body' in options:
                    body_type = self._detect_body_type(options.get('body'))
        
        self._add_api_call(
            url=url,
            method=method,
            call_type=APICallType.HTTP_FETCH,
            file_path=file_path,
            line_number=node.get('start_line', 0),
            headers=headers,
            body_type=body_type
        )
    
    def _check_axios_call(self, node: Dict[str, Any], file_path: str):
        """Check for axios library calls."""
        callee = node.get('function', {})
        if not isinstance(callee, dict):
            return
        
        # axios(config) or axios(url, config)
        if callee.get('name') == 'axios':
            args = node.get('arguments', [])
            if not args:
                return
            
            # axios(url, config) or axios(config)
            first_arg = args[0]
            if isinstance(first_arg, dict) and 'url' in first_arg:
                # axios(config)
                self._extract_axios_config(first_arg, file_path, node.get('start_line', 0))
            else:
                # axios(url, config)
                url = self._extract_url_from_arg(first_arg)
                if url:
                    config = args[1] if len(args) > 1 else {}
                    self._extract_axios_config(
                        {**config, 'url': url} if isinstance(config, dict) else {'url': url},
                        file_path,
                        node.get('start_line', 0)
                    )
        
        # axios.method() calls handled in _check_method_call
    
    def _check_method_call(self, node: Dict[str, Any], file_path: str):
        """Check for method-based API calls (e.g., axios.get, axios.post)."""
        obj = node.get('object', {})
        property = node.get('property', {})
        
        if not isinstance(obj, dict) or not isinstance(property, dict):
            return
        
        obj_name = obj.get('name', '')
        method_name = property.get('name', '')
        
        # axios.get(), axios.post(), etc.
        if obj_name == 'axios' and method_name in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
            # This is handled by parent call_expression
            pass
    
    def _check_xhr_call(self, node: Dict[str, Any], file_path: str):
        """Check for XMLHttpRequest calls."""
        callee = node.get('function', {})
        if not isinstance(callee, dict):
            return
        
        # new XMLHttpRequest()
        if callee.get('name') == 'XMLHttpRequest':
            # Track XHR creation, actual calls are in open/send methods
            pass
    
    def _check_websocket_call(self, node: Dict[str, Any], file_path: str):
        """Check for WebSocket connections."""
        callee = node.get('function', {})
        if not isinstance(callee, dict):
            return
        
        # new WebSocket(url)
        if callee.get('name') == 'WebSocket':
            args = node.get('arguments', [])
            if args:
                url = self._extract_url_from_arg(args[0])
                if url:
                    self._add_api_call(
                        url=url,
                        method=None,
                        call_type=APICallType.WEBSOCKET,
                        file_path=file_path,
                        line_number=node.get('start_line', 0)
                    )
    
    def _extract_axios_config(self, config: Dict[str, Any], file_path: str, line_number: int):
        """Extract API call from axios config object."""
        url = config.get('url', '')
        if not url:
            return
        
        # Extract method
        method = HTTPMethod.GET
        method_str = config.get('method', 'get').upper()
        try:
            method = HTTPMethod[method_str]
        except KeyError:
            pass
        
        # Extract headers
        headers = config.get('headers', {})
        
        # Extract body type
        body_type = None
        if 'data' in config:
            body_type = 'json'
        elif 'params' in config:
            body_type = 'query'
        
        self._add_api_call(
            url=url,
            method=method,
            call_type=APICallType.HTTP_AXIOS,
            file_path=file_path,
            line_number=line_number,
            headers=headers if isinstance(headers, dict) else {},
            body_type=body_type
        )
    
    def _extract_url_from_arg(self, arg: Any) -> Optional[str]:
        """Extract URL from function argument."""
        if isinstance(arg, dict):
            # Literal string
            if arg.get('type') in ['string', 'string_literal']:
                return arg.get('value', '')
            # Template string
            elif arg.get('type') == 'template_string':
                # Try to extract URL pattern
                parts = arg.get('parts', [])
                if parts and isinstance(parts[0], str):
                    return parts[0]
        elif isinstance(arg, str):
            return arg
        
        return None
    
    def _detect_body_type(self, body: Any) -> Optional[str]:
        """Detect the type of request body."""
        if isinstance(body, dict):
            body_type = body.get('type', '')
            if 'json' in body_type.lower():
                return 'json'
            elif 'form' in body_type.lower():
                return 'form-data'
        return 'unknown'
    
    def _add_api_call(
        self,
        url: str,
        method: Optional[HTTPMethod],
        call_type: APICallType,
        file_path: str,
        line_number: int,
        headers: Optional[Dict[str, str]] = None,
        body_type: Optional[str] = None
    ):
        """
        Add an API call to tracking.
        
        Args:
            url: API endpoint URL
            method: HTTP method
            call_type: Type of API call
            file_path: Source file path
            line_number: Line number
            headers: Request headers
            body_type: Type of request body
        """
        # Parse URL
        parsed_url = self._parse_url(url)
        domain = parsed_url.get('domain')
        
        # Identify service
        service_name = self._identify_service(domain)
        
        # Check for authentication
        is_authenticated, auth_type = self._check_authentication(headers or {})
        
        # Create endpoint
        endpoint = APIEndpoint(
            url=url,
            method=method,
            call_type=call_type,
            source_file=file_path,
            line_number=line_number,
            headers=headers or {},
            body_type=body_type,
            service_name=service_name,
            is_third_party=domain is not None and not self._is_local_url(url),
            domain=domain,
            is_authenticated=is_authenticated,
            auth_type=auth_type
        )
        
        self.api_calls.append(endpoint)
        
        # Track service
        if service_name and domain:
            if service_name not in self.services:
                # Find category by matching domain pattern
                category = None
                domain_lower = domain.lower()
                for pattern, info in self.service_patterns.items():
                    pattern_lower = pattern.lower()
                    if pattern_lower in domain_lower or domain_lower in pattern_lower:
                        category = info.get('category')
                        break
                
                self.services[service_name] = ThirdPartyService(
                    name=service_name,
                    domain=domain,
                    category=category
                )
            
            self.services[service_name].endpoints.append(endpoint)
            self.services[service_name].call_count += 1
    
    def _parse_url(self, url: str) -> Dict[str, Optional[str]]:
        """
        Parse URL and extract components.
        
        Args:
            url: URL to parse
            
        Returns:
            Dict with domain, path, protocol
        """
        # Handle template strings and variables
        if '${' in url or url.startswith('/'):
            return {'domain': None, 'path': url, 'protocol': None}
        
        try:
            parsed = urlparse(url)
            return {
                'domain': parsed.netloc or None,
                'path': parsed.path or None,
                'protocol': parsed.scheme or None
            }
        except Exception:
            return {'domain': None, 'path': url, 'protocol': None}
    
    def _identify_service(self, domain: Optional[str]) -> Optional[str]:
        """
        Identify third-party service from domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Service name or None
        """
        if not domain:
            return None
        
        domain_lower = domain.lower()
        
        # Check exact matches first
        if domain_lower in self.service_patterns:
            return self.service_patterns[domain_lower]['name']
        
        # Check partial matches (domain contains pattern or pattern contains domain)
        for pattern, info in self.service_patterns.items():
            pattern_lower = pattern.lower()
            # Check if pattern is in domain OR domain is in pattern
            if pattern_lower in domain_lower or domain_lower in pattern_lower:
                return info['name']
        
        return None
    
    def _is_local_url(self, url: str) -> bool:
        """Check if URL is local."""
        local_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
        return any(pattern in url.lower() for pattern in local_patterns) or url.startswith('/')
    
    def _check_authentication(self, headers: Dict[str, str]) -> tuple[bool, Optional[str]]:
        """
        Check if request has authentication.
        
        Args:
            headers: Request headers
            
        Returns:
            Tuple of (is_authenticated, auth_type)
        """
        # Check Authorization header
        for key, value in headers.items():
            if key.lower() == 'authorization':
                if 'bearer' in value.lower():
                    return (True, 'bearer')
                elif 'basic' in value.lower():
                    return (True, 'basic')
                return (True, 'custom')
            
            # API key in header
            if 'api' in key.lower() and 'key' in key.lower():
                return (True, 'api-key')
        
        return (False, None)
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API calls.
        
        Returns:
            Dict with API call statistics
        """
        total_calls = len(self.api_calls)
        
        # Count by type
        by_type = {}
        for call_type in APICallType:
            count = sum(1 for call in self.api_calls if call.call_type == call_type)
            by_type[call_type.value] = count
        
        # Count by method
        by_method = {}
        for method in HTTPMethod:
            count = sum(1 for call in self.api_calls if call.method == method)
            by_method[method.value] = count
        
        # Third-party vs local
        third_party = sum(1 for call in self.api_calls if call.is_third_party)
        local = total_calls - third_party
        
        # Authenticated calls
        authenticated = sum(1 for call in self.api_calls if call.is_authenticated)
        
        return {
            'total_calls': total_calls,
            'by_type': by_type,
            'by_method': by_method,
            'third_party_calls': third_party,
            'local_calls': local,
            'authenticated_calls': authenticated,
            'unique_services': len(self.services),
            'unique_domains': len(set(call.domain for call in self.api_calls if call.domain))
        }
    
    def get_services_by_category(self) -> Dict[str, List[str]]:
        """
        Get services grouped by category.
        
        Returns:
            Dict mapping categories to service names
        """
        by_category = {}
        
        for service in self.services.values():
            if service.category:
                if service.category not in by_category:
                    by_category[service.category] = []
                by_category[service.category].append(service.name)
        
        return by_category
    
    def track_api_calls_from_files(self, file_nodes: List[Any], project_path: str):
        """
        Extract API calls from parsed file nodes (works with tree-sitter).
        
        Args:
            file_nodes: List of Node objects from parser
            project_path: Root path of the project
        """
        from pathlib import Path
        
        for node in file_nodes:
            # Skip non-module/component/function nodes
            node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
            if node_type not in ['module', 'component', 'function']:
                continue
            
            # Get the source file
            file_path = getattr(node, 'file', '')
            if not file_path:
                continue
            
            full_path = Path(project_path) / file_path
            if not full_path.exists():
                continue
            
            # Read file content
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logging.debug(f"Could not read {file_path}: {e}")
                continue
            
            # Extract API calls from source code
            self._extract_api_calls_from_source(content, file_path)
    
    def _extract_api_calls_from_source(self, content: str, file_path: str):
        """
        Extract API calls from source code using regex patterns.
        
        Args:
            content: Source code content
            file_path: Path to the source file
        """
        # Pattern 1: fetch() calls
        fetch_pattern = r'fetch\s*\(\s*[\'"`]([^\'"`]+)[\'"`]'
        for match in re.finditer(fetch_pattern, content):
            url = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            self._add_api_endpoint(url, HTTPMethod.GET, APICallType.HTTP_FETCH, file_path, line_number)
        
        # Pattern 2: axios calls - axios.get/post/etc
        axios_methods = ['get', 'post', 'put', 'delete', 'patch']
        for method_name in axios_methods:
            pattern = rf'axios\.{method_name}\s*\(\s*[\'"`]([^\'"`]+)[\'"`]'
            for match in re.finditer(pattern, content):
                url = match.group(1)
                line_number = content[:match.start()].count('\n') + 1
                method = HTTPMethod[method_name.upper()]
                self._add_api_endpoint(url, method, APICallType.HTTP_AXIOS, file_path, line_number)
        
        # Pattern 3: axios() with config
        axios_config_pattern = r'axios\s*\(\s*\{[^}]*url\s*:\s*[\'"`]([^\'"`]+)[\'"`][^}]*method\s*:\s*[\'"`]([^\'"`]+)[\'"`]'
        for match in re.finditer(axios_config_pattern, content):
            url = match.group(1)
            method_str = match.group(2).upper()
            line_number = content[:match.start()].count('\n') + 1
            try:
                method = HTTPMethod[method_str]
            except KeyError:
                method = HTTPMethod.GET
            self._add_api_endpoint(url, method, APICallType.HTTP_AXIOS, file_path, line_number)
        
        # Pattern 4: WebSocket connections
        ws_pattern = r'new\s+WebSocket\s*\(\s*[\'"`]([^\'"`]+)[\'"`]'
        for match in re.finditer(ws_pattern, content):
            url = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            self._add_api_endpoint(url, None, APICallType.WEBSOCKET, file_path, line_number)
        
        # Pattern 5: GraphQL queries/mutations
        graphql_pattern = r'(query|mutation)\s+\w+\s*\{'
        for match in re.finditer(graphql_pattern, content):
            # Extract operation type
            operation = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            # GraphQL endpoints are usually in the context, try to find them
            # For now, just mark as GraphQL
            self._add_api_endpoint(f'graphql_{operation}', None, APICallType.GRAPHQL, file_path, line_number)
    
    def _add_api_endpoint(self, url: str, method: Optional[HTTPMethod], 
                         call_type: APICallType, file_path: str, line_number: int):
        """
        Add an API endpoint to tracking.
        
        Args:
            url: API endpoint URL
            method: HTTP method
            call_type: Type of API call
            file_path: Source file path
            line_number: Line number in source
        """
        # Parse URL to extract domain
        domain = None
        service_name = None
        is_third_party = True
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Check if it's a known service
            for pattern, service_info in self.service_patterns.items():
                if pattern in domain:
                    service_name = service_info['name']
                    break
            
            # Check if it's a relative URL (internal API)
            if not domain or url.startswith('/'):
                is_third_party = False
                service_name = "Internal API"
        except Exception as e:
            logging.debug(f"Could not parse URL {url}: {e}")
        
        # Create API endpoint
        endpoint = APIEndpoint(
            url=url,
            method=method,
            call_type=call_type,
            source_file=file_path,
            line_number=line_number,
            domain=domain,
            service_name=service_name,
            is_third_party=is_third_party
        )
        
        self.api_calls.append(endpoint)
        
        # Track service
        if service_name and is_third_party:
            if service_name not in self.services:
                category = None
                for pattern, service_info in self.service_patterns.items():
                    if pattern in (domain or ''):
                        category = service_info.get('category')
                        break
                
                self.services[service_name] = ThirdPartyService(
                    name=service_name,
                    domain=domain or '',
                    category=category
                )
            
            self.services[service_name].endpoints.append(endpoint)
            self.services[service_name].call_count += 1
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export all API tracking data.
        
        Returns:
            Dict with all API call information
        """
        return {
            'api_calls': [call.to_dict() for call in self.api_calls],
            'services': {name: service.to_dict() for name, service in self.services.items()},
            'statistics': self.get_api_stats(),
            'services_by_category': self.get_services_by_category()
        }
