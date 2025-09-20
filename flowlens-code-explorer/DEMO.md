# FlowLens - Code Structure Visualization

FlowLens is a production-quality React + TypeScript web application for visualizing and exploring code structures through interactive diagrams. It provides a professional drill-down workflow from repositories to individual methods.

## 🚀 Features

- **Interactive Code Visualization**: React Flow-powered graphs showing relationships between files, classes, and methods
- **Multi-Level Navigation**: Drill down from repository → files → classes → methods with smooth transitions
- **Smart Search & Filtering**: Real-time search with live dimming of non-matching nodes
- **Monaco Editor Integration**: Syntax-highlighted code viewing with method highlighting
- **Dark/Light Theme**: Persistent theme switching with localStorage
- **Export Functionality**: Export diagrams as PNG images
- **Responsive Design**: Three-panel layout that adapts to different screen sizes
- **Professional UI**: Clean, developer-focused design with subtle animations

## 🛠 Tech Stack

- **Frontend**: Vite + React 18 + TypeScript
- **Styling**: TailwindCSS with custom design system
- **State Management**: Zustand (lightweight store)
- **Visualization**: React Flow (@xyflow/react)
- **Code Editor**: Monaco Editor
- **Icons**: Lucide React
- **Notifications**: React Toastify
- **Export**: html-to-image

## 📦 Installation & Setup

### Prerequisites
- Node.js 18+ and npm
- Modern web browser

### Local Development

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd flowlens
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Open in browser**:
   Navigate to `http://localhost:8080`

### Build for Production

```bash
npm run build
npm run preview  # Preview production build
```

## 🎯 Usage Guide

### Getting Started

1. **Load a Project**: 
   - Use "Upload Project" for local files (coming soon)
   - Or "GitHub Repository" for analyzing repos
   - Click "Analyze Project" to load demo data

2. **Navigate the Interface**:
   - **Left Panel**: Project overview, search, and statistics
   - **Center Panel**: Interactive graph visualization
   - **Right Panel**: Code viewer with syntax highlighting
   - **Bottom Toolbar**: Zoom, fit view, and export controls

### Navigation Workflow

1. **Repository Level**: View files and their dependencies
2. **File Level**: Click a file to see its classes
3. **Class Level**: Click a class to see its methods  
4. **Method Level**: Click a method to view its code

### Key Features

- **Search**: Press `/` to focus search, filters nodes in real-time
- **Statistics Cards**: Click to filter graph by type (files/classes/methods)
- **Dark Mode**: Toggle in header, preference saved automatically
- **Export**: Use toolbar to export current view as PNG
- **Breadcrumbs**: Header shows current navigation path

## 🏗 Architecture

### Component Structure
```
src/
├── components/
│   ├── Header.tsx           # App header with branding and theme toggle
│   ├── UploadBar.tsx        # Project upload interface
│   ├── Overview.tsx         # Left sidebar with stats and search
│   ├── GraphView.tsx        # Main React Flow graph visualization
│   ├── CodePanel.tsx        # Right sidebar code viewer
│   └── BottomToolbar.tsx    # Graph controls toolbar
├── store/
│   └── useFlowLensStore.ts  # Zustand state management
├── services/
│   └── dataService.ts       # Data loading and filtering utilities
├── utils/
│   └── exportUtils.ts       # Export and event utilities
└── App.tsx                  # Main application component
```

### State Management

Uses Zustand for lightweight, TypeScript-first state management:

- **Project Data**: Files, classes, methods, graph nodes/edges
- **UI State**: Dark mode, search query, filter type, view level
- **Selection State**: Currently selected file/class/method
- **Loading State**: Analysis progress indication

### Design System

Custom design system built on TailwindCSS:
- **Colors**: Developer-focused neutral palette with indigo accents
- **Components**: Semantic design tokens for consistent styling
- **Animations**: Smooth transitions with custom timing functions
- **Responsive**: Mobile-first approach with collapsible panels

## 📊 Mock Data

The application currently uses static mock data (`/public/mocks/hier-sample.json`) representing a sample JavaScript project with:

- 4 files (userService.js, logger.js, apiClient.js, dashboard.js)
- 6 classes across files
- 18 methods with realistic code samples
- Dependency relationships between files

### Future Backend Integration

The architecture is designed for easy backend integration:

- `fetchAnalyzeRepo()` function ready for API calls
- Extensible data models for real repository analysis
- Separation of concerns between UI and data layers

## 🎨 Customization

### Themes
Modify `src/index.css` to customize the design system:
- Color palette variables
- Component styles
- Animation timings

### Graph Styling
React Flow styling in `src/components/GraphView.tsx`:
- Node appearances and interactions
- Edge styles and animations
- Background and controls

### Code Editor
Monaco Editor configuration in `src/components/CodePanel.tsx`:
- Syntax highlighting themes
- Editor options and features
- Language support

## 🧪 Development

### Code Quality
- TypeScript for type safety
- ESLint configuration included
- Component-based architecture
- Separation of concerns

### Performance
- Memoized computations for graph filtering
- Efficient state updates with Zustand
- Lazy loading ready for large codebases
- Optimized React Flow rendering

### Accessibility
- ARIA labels on interactive elements
- Keyboard navigation support
- Screen reader friendly
- High contrast theme support

## 🔄 Future Enhancements

1. **Backend Integration**: Real GitHub API analysis
2. **File Upload**: Support for zip archives and folders
3. **AI Features**: Code summarization and insights
4. **Advanced Filtering**: Complex queries and saved views
5. **Collaboration**: Shared diagrams and annotations
6. **Multiple Languages**: Support for Python, Java, C#, etc.

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is built with Lovable and follows their licensing terms.

---

**FlowLens** - Making code structure beautiful and understandable through visualization.