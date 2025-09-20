import React, { useEffect } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import { useFlowLensStore } from './store/useFlowLensStore';
import Header from './components/Header';
import UploadBar from './components/UploadBar';
import Overview from './components/Overview';
import GraphView from './components/GraphView';
import CodePanel from './components/CodePanel';
import BottomToolbar from './components/BottomToolbar';

const App: React.FC = () => {
  const { darkMode, projectMeta } = useFlowLensStore();

  // Apply dark mode class to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  return (
    <div className="h-screen flex flex-col bg-background">
      <Header />
      
      {!projectMeta ? (
        <div className="flex-1">
          <UploadBar />
        </div>
      ) : (
        <div className="flex-1 flex">
          <Overview />
          
          <ReactFlowProvider>
            <GraphView />
            <BottomToolbar />
          </ReactFlowProvider>
          
          <CodePanel />
        </div>
      )}

      <ToastContainer
        position="bottom-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme={darkMode ? 'dark' : 'light'}
        toastClassName="font-mono text-sm"
      />
    </div>
  );
};

export default App;