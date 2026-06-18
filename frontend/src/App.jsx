import { useState } from "react";
import UploadForm from "./components/UploadForm";
import ResultDisplay from "./components/ResultDisplay";
import "./App.css";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const handlePredict = async (file) => {
    setSelectedFile(file);
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const API_BASE = import.meta.env.VITE_API_BASE || "";
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "预测失败");
      if (!data.success) throw new Error(data.error || "预测失败");
      setResult(data);
    } catch (err) {
      setError(err.message || "网络错误，请检查后端服务是否启动");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setSelectedFile(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-icon">🏥</div>
          <div>
            <h1>肺炎X光辅助诊断系统</h1>
            <p className="header-subtitle">ResNet50 + CBAM 注意力机制 · 新冠肺炎智能筛查</p>
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <div className="content-layout">
            <div className="upload-section">
              <UploadForm
                onPredict={handlePredict}
                loading={loading}
                selectedFile={selectedFile}
                onReset={handleReset}
                result={result}
              />
            </div>
            <div className="result-section">
              <ResultDisplay result={result} loading={loading} error={error} />
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p><span>⚠️</span> 本系统仅供辅助诊断参考，不可替代医生临床诊断。</p>
      </footer>
    </div>
  );
}

export default App;
