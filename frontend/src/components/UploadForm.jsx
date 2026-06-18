import { useState, useRef, useCallback } from "react";

export default function UploadForm({ onPredict, loading, onReset }) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileError, setFileError] = useState(null);
  const fileInputRef = useRef(null);
  const pendingFileRef = useRef(null);

  const validateFile = (file) => {
    if (!file) return "请选择文件";
    const ext = file.name.split(".").pop().toLowerCase();
    const valid = ["png", "jpg", "jpeg", "bmp", "tiff", "tif"];
    if (!valid.includes(ext)) return `不支持 .${ext} 格式`;
    if (file.size > 16 * 1024 * 1024) return `文件过大 (${(file.size / 1024 / 1024).toFixed(1)}MB)，最大16MB`;
    return null;
  };

  const handleFile = useCallback((file) => {
    setFileError(null);
    const err = validateFile(file);
    if (err) { setFileError(err); return; }
    pendingFileRef.current = file;
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleClick = () => fileInputRef.current?.click();
  const handleInputChange = (e) => { const f = e.target.files[0]; if (f) handleFile(f); };
  const handleReset = () => { setPreview(null); setFileError(null); pendingFileRef.current = null; if (fileInputRef.current) fileInputRef.current.value = ""; onReset(); };

  const handlePredict = () => {
    if (pendingFileRef.current) {
      onPredict(pendingFileRef.current);
    }
  };

  return (
    <div className="upload-form">
      <h2 className="section-title">上传X光影像</h2>
      <p className="section-desc">请上传胸部X光正位片（PNG/JPG格式）</p>

      {!preview && (
        <div
          className={`drop-zone ${dragOver ? "drag-over" : ""} ${loading ? "disabled" : ""}`}
          onDrop={handleDrop} onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={(e) => { e.preventDefault(); setDragOver(false); }}
          onClick={handleClick}
        >
          <div className="drop-zone-content">
            <div className="drop-icon">
              <svg viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
              </svg>
            </div>
            <p className="drop-text"><strong>点击选择</strong> 或将图像拖拽到此区域</p>
            <p className="drop-hint">支持 PNG/JPG/BMP/TIFF，最大16MB</p>
          </div>
        </div>
      )}

      {preview && (
        <div className="preview-section">
          <div className="preview-container">
            <img src={preview} alt="X光预览" className="preview-image" />
          </div>
          <div className="preview-actions">
            <button className="btn btn-outline" onClick={handleReset} disabled={loading}>重新选择</button>
            <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
              {loading ? "诊断中..." : "🔍 开始诊断"}
            </button>
          </div>
        </div>
      )}

      {fileError && <div className="error-message">⚠️ {fileError}</div>}

      <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg,.bmp,.tiff,.tif"
        onChange={handleInputChange} style={{ display: "none" }} />
    </div>
  );
}
