import { create } from 'zustand';

export interface ProjectMeta {
  name: string;
  fileCount: number;
  classCount: number;
  methodCount: number;
  framework?: string;
  language?: string;
}

export interface Method {
  id: string;
  name: string;
  startLine: number;
  endLine: number;
  code: string;
  description: string;
}

export interface Class {
  id: string;
  name: string;
  startLine: number;
  endLine: number;
  methods: Method[];
}

export interface File {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  content?: string;
  classes: Class[];
}

export interface GraphNodeData {
  label: string;
  fileId?: string;
  classId?: string;
  methodId?: string;
  description?: string;
  dimmed?: boolean;
  originalId?: string;
  nodeType?: string;
  type?: 'file' | 'class' | 'method';
}

export interface GraphNode {
  id: string;
  type: 'file' | 'class' | 'method' | 'folder';
  position: { x: number; y: number };
  data: GraphNodeData;
  parentNode?: string;
  extent?: 'parent';
  width?: number;
  height?: number;
  style?: any;
  selectable?: boolean;
  draggable?: boolean;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
  animated?: boolean;
  label?: string;
  style?: {
    strokeWidth?: number;
    stroke?: string;
    opacity?: number;
  };
}

export type ViewLevel = 'repo' | 'file' | 'class' | 'method';
export type FilterType = 'files' | 'classes' | 'methods';

interface FlowLensState {
  // Data
  projectMeta: ProjectMeta | null;
  files: File[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];

  // UI State
  darkMode: boolean;
  searchQuery: string;
  filterType: FilterType;
  viewLevel: ViewLevel;

  // Selection (for navigation)
  selectedFileId: string | null;
  selectedClassId: string | null;
  selectedMethodId: string | null;

  // Inspection (for code viewing - separate from navigation)
  inspectedNodeId: string | null;

  // Edge filtering
  visibleEdgeTypes: Set<string>;

  // Loading
  isLoading: boolean;
  error: string | null;

  // Actions
  setProjectData: (data: {
    projectMeta: ProjectMeta;
    files: File[];
    graphNodes: GraphNode[];
    graphEdges: GraphEdge[];
  }) => void;
  setDarkMode: (darkMode: boolean) => void;
  setSearchQuery: (query: string) => void;
  setFilterType: (type: FilterType) => void;
  setViewLevel: (level: ViewLevel) => void;

  // Navigation actions (drill down into components)
  setSelectedFile: (fileId: string | null) => void;
  setSelectedClass: (classId: string | null) => void;
  setSelectedMethod: (methodId: string | null) => void;
  clearSelection: () => void;

  // Inspection actions (view code without navigation)
  setInspectedNode: (nodeId: string | null) => void;

  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Edge filtering
  setVisibleEdgeTypes: (types: Set<string>) => void;
  toggleEdgeType: (type: string) => void;

  // Helper methods
  getSelectedNode: () => GraphNode | null;
  getInspectedNode: () => GraphNode | null;
  getNodeById: (id: string) => GraphNode | null;
  getRelatedNodes: (nodeId: string) => GraphNode[];
}

export const useFlowLensStore = create<FlowLensState>((set, get) => ({
  // Initial state
  projectMeta: null,
  files: [],
  graphNodes: [],
  graphEdges: [],

  darkMode: typeof window !== 'undefined' ? localStorage.getItem('flowlens-dark-mode') === 'true' : false,
  searchQuery: '',
  filterType: 'files',
  viewLevel: 'repo',

  selectedFileId: null,
  selectedClassId: null,
  selectedMethodId: null,
  inspectedNodeId: null,

  isLoading: false,
  error: null,

  visibleEdgeTypes: new Set(['imports', 'calls', 'renders', 'instantiates', 'async_calls']),

  // Actions
  setProjectData: (data) => {
    console.log('Setting project data:', data);
    set({
      ...data,
      viewLevel: 'repo',
      selectedFileId: null,
      selectedClassId: null,
      selectedMethodId: null,
      inspectedNodeId: null,
      error: null
    });
  },

  setDarkMode: (darkMode) => {
    set({ darkMode });
    if (typeof window !== 'undefined') {
      localStorage.setItem('flowlens-dark-mode', darkMode.toString());
      document.documentElement.classList.toggle('dark', darkMode);
    }
  },

  setSearchQuery: (searchQuery) => set({ searchQuery }),

  setFilterType: (filterType) => set({ filterType }),

  setViewLevel: (viewLevel) => {
    console.log('Setting view level to:', viewLevel);
    set({ viewLevel });
  },

  // Navigation actions (these change the view)
  setSelectedFile: (selectedFileId) => {
    console.log('Setting selected file:', selectedFileId);
    set({
      selectedFileId,
      selectedClassId: null,
      selectedMethodId: null,
      viewLevel: selectedFileId ? 'file' : 'repo',
      error: null
    });
  },

  setSelectedClass: (selectedClassId) => {
    console.log('Setting selected class:', selectedClassId);
    const state = get();
    set({
      selectedClassId,
      selectedMethodId: null,
      viewLevel: selectedClassId ? 'class' : state.selectedFileId ? 'file' : 'repo',
      error: null
    });
  },

  setSelectedMethod: (selectedMethodId) => {
    console.log('Setting selected method:', selectedMethodId);
    const state = get();
    set({
      selectedMethodId,
      viewLevel: selectedMethodId ? 'method' :
        state.selectedClassId ? 'class' :
          state.selectedFileId ? 'file' : 'repo',
      error: null
    });
  },

  clearSelection: () => set({
    selectedFileId: null,
    selectedClassId: null,
    selectedMethodId: null,
    viewLevel: 'repo',
    searchQuery: '',
    error: null
  }),

  // Inspection action (for code viewing)
  setInspectedNode: (inspectedNodeId) => {
    console.log('Setting inspected node:', inspectedNodeId);
    set({ inspectedNodeId });
  },

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error }),

  // Edge filtering
  setVisibleEdgeTypes: (visibleEdgeTypes) => set({ visibleEdgeTypes }),

  toggleEdgeType: (type: string) => {
    const state = get();
    const newTypes = new Set(state.visibleEdgeTypes);
    if (newTypes.has(type)) {
      newTypes.delete(type);
    } else {
      newTypes.add(type);
    }
    set({ visibleEdgeTypes: newTypes });
  },

  // Helper methods
  getSelectedNode: () => {
    const state = get();
    const selectedId = state.selectedMethodId || state.selectedClassId || state.selectedFileId;
    if (!selectedId) return null;

    return state.graphNodes.find(node => node.id === selectedId) || null;
  },

  getInspectedNode: () => {
    const state = get();
    if (!state.inspectedNodeId) return null;

    return state.graphNodes.find(node => node.id === state.inspectedNodeId) || null;
  },

  getNodeById: (id: string) => {
    const state = get();
    return state.graphNodes.find(node => node.id === id) || null;
  },

  getRelatedNodes: (nodeId: string) => {
    const state = get();
    const node = state.graphNodes.find(n => n.id === nodeId);
    if (!node) return [];

    const related: GraphNode[] = [];

    if (node.type === 'file') {
      related.push(...state.graphNodes.filter(n => n.data.fileId === nodeId));
    } else if (node.type === 'class') {
      related.push(...state.graphNodes.filter(n => n.data.classId === nodeId));
      if (node.data.fileId) {
        const parentFile = state.graphNodes.find(n => n.id === node.data.fileId);
        if (parentFile) related.push(parentFile);
      }
    } else if (node.type === 'method') {
      if (node.data.classId) {
        const parentClass = state.graphNodes.find(n => n.id === node.data.classId);
        if (parentClass) related.push(parentClass);
      }
      if (node.data.fileId) {
        const parentFile = state.graphNodes.find(n => n.id === node.data.fileId);
        if (parentFile) related.push(parentFile);
      }
    }

    return related;
  },
}));