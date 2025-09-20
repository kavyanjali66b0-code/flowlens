import { create } from 'zustand';

export interface ProjectMeta {
  name: string;
  fileCount: number;
  classCount: number;
  methodCount: number;
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

export interface GraphNode {
  id: string;
  type: 'file' | 'class' | 'method';
  position: { x: number; y: number };
  data: {
    label: string;
    fileId?: string;
    classId?: string;
    methodId?: string;
    description?: string;
    dimmed?: boolean;
  };
  parentNode?: string;
  extent?: 'parent';
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
  animated?: boolean;
  label?: string;
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
  
  // Selection
  selectedFileId: string | null;
  selectedClassId: string | null;
  selectedMethodId: string | null;
  
  // Loading
  isLoading: boolean;
  
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
  setSelectedFile: (fileId: string | null) => void;
  setSelectedClass: (classId: string | null) => void;
  setSelectedMethod: (methodId: string | null) => void;
  clearSelection: () => void;
  setLoading: (loading: boolean) => void;
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
  
  isLoading: false,
  
  // Actions
  setProjectData: (data) => set(data),
  
  setDarkMode: (darkMode) => {
    set({ darkMode });
    if (typeof window !== 'undefined') {
      localStorage.setItem('flowlens-dark-mode', darkMode.toString());
      document.documentElement.classList.toggle('dark', darkMode);
    }
  },
  
  setSearchQuery: (searchQuery) => set({ searchQuery }),
  
  setFilterType: (filterType) => set({ filterType }),
  
  setViewLevel: (viewLevel) => set({ viewLevel }),
  
  setSelectedFile: (selectedFileId) => {
    set({ 
      selectedFileId,
      selectedClassId: null,
      selectedMethodId: null,
      viewLevel: selectedFileId ? 'file' : 'repo'
    });
  },
  
  setSelectedClass: (selectedClassId) => {
    set({ 
      selectedClassId,
      selectedMethodId: null,
      viewLevel: selectedClassId ? 'class' : get().selectedFileId ? 'file' : 'repo'
    });
  },
  
  setSelectedMethod: (selectedMethodId) => {
    set({ 
      selectedMethodId,
      viewLevel: selectedMethodId ? 'method' : get().selectedClassId ? 'class' : get().selectedFileId ? 'file' : 'repo'
    });
  },
  
  clearSelection: () => set({
    selectedFileId: null,
    selectedClassId: null,
    selectedMethodId: null,
    viewLevel: 'repo'
  }),
  
  setLoading: (isLoading) => set({ isLoading }),
}));