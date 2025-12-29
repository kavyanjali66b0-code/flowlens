import { ProjectMeta, File, GraphNode, GraphEdge } from '../store/useFlowLensStore';

const API_URL = "http://localhost:5000";

export interface ProjectData {
  projectMeta: ProjectMeta;
  files: File[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
}

/**
 * Calculate node importance based on connectivity
 */
function calculateNodeImportance(nodes: any[], edges: any[]): Map<string, number> {
  const importance = new Map<string, number>();

  // Initialize all nodes with base importance
  nodes.forEach(node => {
    importance.set(node.id, 1);
  });

  // Calculate in-degree and out-degree
  const inDegree = new Map<string, number>();
  const outDegree = new Map<string, number>();

  edges.forEach(edge => {
    // Count incoming connections
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    // Count outgoing connections  
    outDegree.set(edge.source, (outDegree.get(edge.source) || 0) + 1);
  });

  // Calculate importance score (nodes that are highly connected)
  nodes.forEach(node => {
    const incoming = inDegree.get(node.id) || 0;
    const outgoing = outDegree.get(node.id) || 0;

    // Nodes with many incoming connections are important (many things depend on them)
    // Nodes with many outgoing connections are also important (they coordinate many things)
    const score = Math.log(incoming + 1) * 2 + Math.log(outgoing + 1);
    importance.set(node.id, score);
  });

  return importance;
}

/**
 * Create layout based on dependency hierarchy
 */
function createDependencyLayout(nodes: any[], edges: any[]): { position: { x: number; y: number } }[] {
  const importance = calculateNodeImportance(nodes, edges);

  // Sort nodes by importance
  const sortedNodes = [...nodes].sort((a, b) => {
    const importanceA = importance.get(a.id) || 0;
    const importanceB = importance.get(b.id) || 0;
    return importanceB - importanceA;
  });

  // Create layers based on importance
  const positions: { position: { x: number; y: number } }[] = [];
  const nodePositionMap = new Map();

  sortedNodes.forEach((node, index) => {
    const importance_score = importance.get(node.id) || 0;

    // Create layers: high importance at top, lower importance below
    let layer = 0;
    if (importance_score > 3) layer = 0;      // Most important
    else if (importance_score > 2) layer = 1; // Important  
    else if (importance_score > 1) layer = 2; // Moderate
    else layer = 3;                          // Least important

    // Count nodes in this layer so far
    const nodesInLayer = sortedNodes.slice(0, index).filter(n => {
      const score = importance.get(n.id) || 0;
      if (importance_score > 3) return score > 3;
      else if (importance_score > 2) return score > 2 && score <= 3;
      else if (importance_score > 1) return score > 1 && score <= 2;
      else return score <= 1;
    }).length;

    const nodesPerRow = Math.min(6, Math.ceil(Math.sqrt(sortedNodes.length / 4)));
    const x = 150 + (nodesInLayer % nodesPerRow) * 200;
    const y = 100 + layer * 150 + Math.floor(nodesInLayer / nodesPerRow) * 100;

    const position = { x, y };
    positions[nodes.indexOf(node)] = { position };
    nodePositionMap.set(node.id, position);
  });

  return positions;
}

/**
 * Build hierarchical relationships between nodes
 */
function buildNodeRelationships(nodes: any[]): Map<string, { fileId?: string, classId?: string }> {
  const relationships = new Map<string, { fileId?: string, classId?: string }>();

  // Create lookup maps
  const fileNodes = nodes.filter(n => n.type === 'module' || n.type === 'file');
  const classNodes = nodes.filter(n => n.type === 'class' || n.type === 'component');
  const methodNodes = nodes.filter(n => n.type === 'function');

  // For each node, find its parent relationships based on file paths
  nodes.forEach(node => {
    if (!node.file) return;

    const nodePath = node.file.replace(/\\/g, '/');

    // Find parent file
    const parentFile = fileNodes.find(f => {
      const filePath = f.file?.replace(/\\/g, '/');
      return filePath === nodePath;
    });

    if (parentFile) {
      relationships.set(node.id, {
        ...relationships.get(node.id),
        fileId: parentFile.id
      });
    }

    // For methods, find parent class
    if (node.type === 'function') {
      // Look for a class in the same file
      const parentClass = classNodes.find(c => {
        const classPath = c.file?.replace(/\\/g, '/');
        return classPath === nodePath;
      });

      if (parentClass) {
        relationships.set(node.id, {
          ...relationships.get(node.id),
          classId: parentClass.id
        });
      }
    }
  });

  return relationships;
}

/**
 * Get visual style for edge based on type
 */
function getEdgeStyle(edgeType: string) {
  const styleMap: Record<string, { stroke: string; strokeWidth: number; strokeDasharray?: string; opacity: number }> = {
    'imports': {
      stroke: '#3b82f6',  // blue
      strokeWidth: 2,
      opacity: 0.8
    },
    'calls': {
      stroke: '#f97316',  // orange
      strokeWidth: 1.5,
      strokeDasharray: '5,5',
      opacity: 0.7
    },
    'renders': {
      stroke: '#10b981',  // green
      strokeWidth: 1.5,
      strokeDasharray: '2,2',
      opacity: 0.7
    },
    'instantiates': {
      stroke: '#8b5cf6',  // purple
      strokeWidth: 2,
      opacity: 0.8
    },
    'async_calls': {
      stroke: '#ef4444',  // red
      strokeWidth: 1.5,
      strokeDasharray: '8,4',
      opacity: 0.7
    }
  };

  return styleMap[edgeType] || {
    stroke: '#6b7280',  // gray fallback
    strokeWidth: 1,
    opacity: 0.5
  };
}

/**
 * Normalize backend response into ProjectData
 */
export async function normalizeResponse(raw: any): Promise<ProjectData> {
  const nodes = raw.graph?.nodes || [];
  const edges = raw.graph?.edges || [];

  console.log('Backend analysis:', {
    totalNodes: nodes.length,
    totalEdges: edges.length,
    nodeTypes: [...new Set(nodes.map((n: any) => n.type))],
    edgeTypes: [...new Set(edges.map((e: any) => e.type))]
  });

  // Map backend node types to frontend types
  const typeMapping: Record<string, 'file' | 'class' | 'method'> = {
    'module': 'file',
    'file': 'file',
    'class': 'class',
    'component': 'class',
    'function': 'method'
  };

  // Build relationships between nodes
  const relationships = buildNodeRelationships(nodes);

  // ID sanitization function (must match across all transforms)
  const sanitizeId = (id: string): string => {
    return id.replace(/\\/g, '/').replace(/[^a-zA-Z0-9/_-]/g, '_');
  };

  // === DEDUPLICATE BACKEND NODES (Backend path normalization bug) ===
  // Backend sends duplicates with same ID but different path formats (\ vs /)
  const seenIds = new Set<string>();
  const deduplicatedNodes = nodes.filter(node => {
    if (seenIds.has(node.id)) {
      console.warn(`Duplicate node from backend: ${node.id}`);
      return false; // Skip duplicate
    }
    seenIds.add(node.id);
    return true;
  });

  console.log(`Deduplicated: ${nodes.length} → ${deduplicatedNodes.length} nodes`);

  // === SANITIZE NODE IDS FIRST (before layout) ===
  const sanitizedNodes = deduplicatedNodes.map(node => ({
    ...node,
    id: sanitizeId(node.id),
  }));

  // === FOLDER-BASED GRID LAYOUT (Mockup Style) ===
  const { applyFolderLayout } = await import('./folderLayout');
  const { nodes: layoutedNodes, folderNodes } = applyFolderLayout(sanitizedNodes);

  // Update positions from layout (IDs are already sanitized)
  const layoutPositions = new Map<string, { x: number; y: number }>();
  layoutedNodes.forEach(node => {
    if (node.position) {
      layoutPositions.set(node.id, node.position); // IDs already sanitized!
    }
  });

  // Create unique nodes with proper relationships and layout positions
  const uniqueNodes = new Map();
  const nodeIdMapping = new Map();

  // CRITICAL: Use sanitizedNodes (already processed) instead of nodes to prevent duplicates
  sanitizedNodes.forEach((node: any, i: number) => {
    const originalId = node.id || `node-${i}`; // This is already sanitized!
    let sanitizedId = originalId; // No need to resanitize

    // Check for duplicates (should not happen now, but keep as safety)
    if (uniqueNodes.has(sanitizedId)) {
      let counter = 1;
      while (uniqueNodes.has(`${sanitizedId}-${counter}`)) {
        counter++;
      }
      sanitizedId = `${sanitizedId}-${counter}`;
    }

    nodeIdMapping.set(originalId, sanitizedId);

    const mappedType = typeMapping[node.type] || 'file';
    const nodeRelations = relationships.get(originalId) || {};

    // Extract meaningful label from the node
    let label = node.name;
    if (!label && node.file) {
      const parts = node.file.split(/[\\\/]/);
      label = parts[parts.length - 1].replace(/\.(tsx?|jsx?|css|ts|js)$/, '');
    }
    if (!label) {
      label = originalId.split('_')[0];
    }

    // Build proper relationships with sanitized IDs
    const fileId = nodeRelations.fileId ? sanitizeId(nodeRelations.fileId) :
      (mappedType === 'file' ? sanitizedId : sanitizeId(node.file || ''));
    const classId = nodeRelations.classId ? sanitizeId(nodeRelations.classId) :
      (mappedType === 'class' ? sanitizedId : undefined);
    const methodId = mappedType === 'method' ? sanitizedId : undefined;

    uniqueNodes.set(sanitizedId, {
      id: sanitizedId,
      type: mappedType,
      position: layoutPositions.get(sanitizedId) || { x: 100 + (i % 8) * 150, y: 100 + Math.floor(i / 8) * 120 },
      data: {
        label: label,
        fileId: fileId,
        classId: classId,
        methodId: methodId,
        description: `${node.type}${node.file ? ` • ${node.file.split(/[\\\/]/).pop()}` : ''}`,
        originalId: originalId,
        nodeType: node.type, // Keep original backend type for filtering
      },
    });
  });

  // Process edges with sanitized IDs
  const processedEdges = new Map();
  const validEdges: any[] = [];

  edges.forEach((edge: any, i: number) => {
    const sourceId = nodeIdMapping.get(edge.source);
    const targetId = nodeIdMapping.get(edge.target);

    if (!sourceId || !targetId) {
      return;
    }

    if (!uniqueNodes.has(sourceId) || !uniqueNodes.has(targetId)) {
      return;
    }

    const edgeKey = `${sourceId}-${targetId}`;
    if (processedEdges.has(edgeKey)) {
      return;
    }

    const edgeData = {
      id: `edge-${sourceId}-${targetId}`,
      source: sourceId,
      target: targetId,
      type: 'default',
      animated: edge.type === 'async_calls',
      label: edge.type || '',
      edgeType: edge.type,
      style: getEdgeStyle(edge.type || 'unknown'),
    };

    processedEdges.set(edgeKey, edgeData);
    validEdges.push(edgeData);
  });

  let finalNodes = Array.from(uniqueNodes.values());

  // === ADD FOLDER BACKGROUND NODES (Mockup Style) ===
  // Folder nodes are non-interactive backgrounds, NOT parent containers
  folderNodes.forEach(folderNode => {
    finalNodes.push({
      id: folderNode.id,
      type: 'folder',
      position: folderNode.position,
      data: folderNode.data,
      style: {
        width: folderNode.style?.width || 250,
        height: folderNode.style?.height || 150,
        zIndex: -1,
      },
      // These props tell React Flow the node dimensions
      width: folderNode.style?.width || 250,
      height: folderNode.style?.height || 150,
      selectable: false,
      draggable: false,
    });
  });

  console.log('Folder layout applied:', {
    totalNodes: finalNodes.length,
    fileNodes: finalNodes.filter(n => n.type === 'file').length,
    folderNodes: folderNodes.length,
    sampleFolder: folderNodes[0],
  });

  // Count nodes correctly
  const fileCount = finalNodes.filter(n => n.type === 'file').length;
  const classCount = finalNodes.filter(n => n.type === 'class').length;
  const methodCount = finalNodes.filter(n => n.type === 'method').length;

  console.log('Processed visualization data:', {
    nodes: finalNodes.length,
    edges: validEdges.length,
    typeBreakdown: {
      files: fileCount,
      classes: classCount,
      methods: methodCount,
    },
    edgeTypes: [...new Set(edges.map((e: any) => e.type))]
  });

  return {
    projectMeta: {
      name: raw.project_info?.name || 'Unknown Project',
      fileCount: fileCount,
      classCount: classCount,
      methodCount: methodCount,
      framework: raw.project_info?.framework,
      language: raw.project_info?.language,
    },
    files: [],
    graphNodes: finalNodes,
    graphEdges: validEdges
  };
}

export async function loadProjectData(): Promise<ProjectData> {
  try {
    const response = await fetch(`${API_URL}/parse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ demo: true })
    });

    if (!response.ok) {
      throw new Error(`Failed to load project data: ${response.statusText}`);
    }

    const raw = await response.json();
    const normalized = normalizeResponse(raw);
    return normalized;
  } catch (error) {
    console.error('Error loading project data:', error);
    throw error;
  }
}

export async function fetchAnalyzeRepo(repoUrl: string): Promise<ProjectData> {
  try {
    const response = await fetch(`${API_URL}/parse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoUrl }),
    });

    if (!response.ok) {
      throw new Error(`Failed to analyze repo: ${response.statusText}`);
    }

    const raw = await response.json();
    const normalized = normalizeResponse(raw);
    return normalized;
  } catch (error) {
    console.error('Error analyzing repo:', error);
    throw error;
  }
}

export function filterNodesByQuery(nodes: GraphNode[], query: string): GraphNode[] {
  if (!query.trim()) return nodes;

  const lowerQuery = query.toLowerCase();
  return nodes.filter(node =>
    node.data.label.toLowerCase().includes(lowerQuery) ||
    node.data.description?.toLowerCase().includes(lowerQuery)
  );
}

export function filterNodesByType(nodes: GraphNode[], type: 'files' | 'classes' | 'methods'): GraphNode[] {
  const typeMap = {
    'files': 'file',
    'classes': 'class',
    'methods': 'method'
  };

  return nodes.filter(node => node.type === typeMap[type]);
}

/**
 * Intelligent view filtering with proper context awareness
 */
export function getNodesForView(
  allNodes: GraphNode[],
  viewLevel: 'repo' | 'file' | 'class' | 'method',
  selectedFileId?: string | null,
  selectedClassId?: string | null
): GraphNode[] {
  console.log('getNodesForView called:', { viewLevel, selectedFileId, selectedClassId, totalNodes: allNodes.length });

  let filteredNodes: GraphNode[] = [];

  switch (viewLevel) {
    case 'repo':
      // Show all files AND folder background nodes at repo level
      filteredNodes = allNodes.filter(node => node.type === 'file' || node.type === 'folder');
      console.log(`Repo view: showing ${filteredNodes.filter(n => n.type === 'file').length} files + ${filteredNodes.filter(n => n.type === 'folder').length} folders`);
      break;

    case 'file':
      if (selectedFileId) {
        // Show classes in the selected file
        filteredNodes = allNodes.filter(node =>
          node.type === 'class' && node.data.fileId === selectedFileId
        );
      } else {
        // Show all classes if no file is selected
        filteredNodes = allNodes.filter(node => node.type === 'class');
      }
      break;

    case 'class':
      if (selectedClassId) {
        // Show methods in the selected class
        filteredNodes = allNodes.filter(node =>
          node.type === 'method' && node.data.classId === selectedClassId
        );
      } else {
        // Show all classes if no class is selected
        filteredNodes = allNodes.filter(node => node.type === 'class');
      }
      break;

    case 'method':
      if (selectedClassId) {
        // Show methods in the selected class
        filteredNodes = allNodes.filter(node =>
          node.type === 'method' && node.data.classId === selectedClassId
        );
      } else if (selectedFileId) {
        // Show methods in the selected file
        filteredNodes = allNodes.filter(node =>
          node.type === 'method' && node.data.fileId === selectedFileId
        );
      } else {
        // Show all methods
        filteredNodes = allNodes.filter(node => node.type === 'method');
      }
      break;

    default:
      filteredNodes = allNodes;
  }

  console.log('Filtered nodes:', filteredNodes.length, filteredNodes.map(n => ({ id: n.id, type: n.type, label: n.data.label })));
  return filteredNodes;
}

export function getEdgesForView(
  allEdges: GraphEdge[],
  visibleNodeIds: Set<string>,
  viewLevel?: 'repo' | 'file' | 'class' | 'method'
): GraphEdge[] {
  // At repo level, show ALL edges for complete architecture view
  if (viewLevel === 'repo') {
    console.log(`Repo view: showing all ${allEdges.length} edges`);
    return allEdges;
  }

  // At other levels, filter to visible nodes
  const filteredEdges = allEdges.filter(edge =>
    visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
  );

  console.log('Filtered edges:', filteredEdges.length, 'from', allEdges.length, 'total edges');
  return filteredEdges;
}