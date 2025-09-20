import { ProjectMeta, File, GraphNode, GraphEdge } from '../store/useFlowLensStore';

const API_URL = import.meta.env.VITE_API_URL;

export interface ProjectData {
  projectMeta: ProjectMeta;
  files: File[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
}

// export async function loadMockData(): Promise<ProjectData> {
//   try {
//     const response = await fetch('/mocks/hier-sample.json');
//     if (!response.ok) {
//       throw new Error(`Failed to load mock data: ${response.statusText}`);
//     }
//     return response.json();
//   } catch (error) {
//     console.error('Error loading mock data:', error);
//     throw error;
//   }
// }

/**
 * Load demo project data from backend
 */
export async function loadProjectData(): Promise<ProjectData> {
  try {
    const response = await fetch(`${API_URL}/parse`, {   // ✅ changed from /analyze
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ demo: true }) // or whatever your backend expects
    });
    if (!response.ok) {
      throw new Error(`Failed to load project data: ${response.statusText}`);
    }
    return response.json();
  } catch (error) {
    console.error('Error loading project data:', error);
    throw error;
  }
}

/**
 * Analyze a GitHub repo using backend
 */
export async function fetchAnalyzeRepo(repoUrl: string): Promise<ProjectData> {
  try {
    const response = await fetch(`${API_URL}/parse`, {   // ✅ changed from /analyze
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoUrl }),
    });

    if (!response.ok) {
      throw new Error(`Failed to analyze repo: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    console.error('Error analyzing repo:', error);
    throw error;
  }
}

/**
 * Filter nodes by search query (case-insensitive)
 */
export function filterNodesByQuery(nodes: GraphNode[], query: string): GraphNode[] {
  if (!query.trim()) return nodes;
  
  const lowerQuery = query.toLowerCase();
  return nodes.filter(node => 
    node.data.label.toLowerCase().includes(lowerQuery) ||
    node.data.description?.toLowerCase().includes(lowerQuery)
  );
}

/**
 * Filter nodes by type
 */
export function filterNodesByType(nodes: GraphNode[], type: 'files' | 'classes' | 'methods'): GraphNode[] {
  const typeMap = {
    'files': 'file',
    'classes': 'class',
    'methods': 'method'
  };
  
  return nodes.filter(node => node.type === typeMap[type]);
}

/**
 * Get nodes for specific view level and selection context
 */
export function getNodesForView(
  allNodes: GraphNode[], 
  viewLevel: 'repo' | 'file' | 'class' | 'method',
  selectedFileId?: string | null,
  selectedClassId?: string | null
): GraphNode[] {
  switch (viewLevel) {
    case 'repo':
      return allNodes.filter(node => node.type === 'file');
    
    case 'file':
      if (!selectedFileId) return [];
      return allNodes.filter(node => 
        node.type === 'class' && node.data.fileId === selectedFileId
      );
    
    case 'class':
      if (!selectedClassId) return [];
      return allNodes.filter(node => 
        node.type === 'method' && node.data.classId === selectedClassId
      );
    
    case 'method':
      return allNodes.filter(node => node.type === 'method');
    
    default:
      return allNodes;
  }
}

/**
 * Get edges for specific view level
 */
export function getEdgesForView(
  allEdges: GraphEdge[],
  visibleNodeIds: Set<string>
): GraphEdge[] {
  return allEdges.filter(edge => 
    visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
  );
}
