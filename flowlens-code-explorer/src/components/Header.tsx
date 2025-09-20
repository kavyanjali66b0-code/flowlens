import React from 'react';
import { Moon, Sun, Code2 } from 'lucide-react';
import { useFlowLensStore } from '../store/useFlowLensStore';

const Header: React.FC = () => {
  const { darkMode, setDarkMode, projectMeta, selectedFileId, selectedClassId, selectedMethodId } = useFlowLensStore();

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const getBreadcrumbs = () => {
    const breadcrumbs = [{ label: 'FlowLens', path: null }];
    
    if (projectMeta) {
      breadcrumbs.push({ label: projectMeta.name, path: 'repo' });
    }
    
    if (selectedFileId) {
      // Find file name from selectedFileId
      const fileName = selectedFileId.split('/').pop() || 'file';
      breadcrumbs.push({ label: fileName, path: 'file' });
    }
    
    if (selectedClassId) {
      breadcrumbs.push({ label: selectedClassId.replace('class-', ''), path: 'class' });
    }
    
    if (selectedMethodId) {
      breadcrumbs.push({ label: selectedMethodId.replace('method-', ''), path: 'method' });
    }
    
    return breadcrumbs;
  };

  const breadcrumbs = getBreadcrumbs();

  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6 shadow-sm">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <Code2 className="h-8 w-8 text-primary" />
          <h1 className="text-2xl font-bold text-foreground">FlowLens</h1>
        </div>
        
        {breadcrumbs.length > 1 && (
          <nav className="flex items-center space-x-2 text-sm text-muted-foreground">
            {breadcrumbs.map((crumb, index) => (
              <React.Fragment key={index}>
                {index > 0 && <span className="text-border">→</span>}
                <span className={index === breadcrumbs.length - 1 ? 'text-foreground font-medium' : 'hover:text-foreground cursor-pointer'}>
                  {crumb.label}
                </span>
              </React.Fragment>
            ))}
          </nav>
        )}
      </div>

      <button
        onClick={toggleDarkMode}
        className="flowlens-button-ghost h-10 w-10 p-0"
        aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {darkMode ? (
          <Sun className="h-5 w-5" />
        ) : (
          <Moon className="h-5 w-5" />
        )}
      </button>
    </header>
  );
};

export default Header;