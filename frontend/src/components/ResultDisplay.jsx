const CLASS_ORDER = ["新冠肺炎", "正常", "肺部阴影", "病毒性肺炎"];

const CLASS_COLORS = {
  "新冠肺炎": "#dc3545",
  "正常": "#28a745",
  "肺部阴影": "#fd7e14",
  "病毒性肺炎": "#ffc107",
};

const CLASS_HINTS = {
  "新冠肺炎": "COVID-19 病毒感染，具有高传染性，需立即隔离",
  "正常": "双肺纹理清晰，未见明显实质性病变",
  "肺部阴影": "肺部磨玻璃影/实变，可能由多种病因引起",
  "病毒性肺炎": "病毒性感染引起的肺部炎症表现",
};

function ProgressBar({ label, value, color, isPrediction, hint }) {
  return (
    <div className={`prob-bar ${isPrediction ? "is-prediction" : ""}`}>
      <div className="prob-bar-label">
        <span className="prob-bar-name">{label}</span>
        <span className="prob-bar-value">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="prob-bar-track">
        <div className="prob-bar-fill" style={{ width: `${value * 100}%`, backgroundColor: color }} />
      </div>
      {hint && <span className="prob-bar-hint">{hint}</span>}
    </div>
  );
}

export default function ResultDisplay({ result, loading, error }) {
  if (loading) {
    return (
      <div className="loading-state">
        <div className="spinner" />
        <h3>正在分析影像...</h3>
        <p className="loading-hint">模型正在进行特征提取与分类，请稍候</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <div className="error-icon">⚠️</div>
        <h3>诊断失败</h3>
        <p className="error-text">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🫁</div>
        <h3>等待检测</h3>
        <p>请上传胸部X光影像开始智能筛查</p>
        <div className="steps-guide">
          <div className="step-item"><span className="step-num">1</span> 上传胸部 X 光影像</div>
          <div className="step-item"><span className="step-num">2</span> 点击「开始诊断」</div>
          <div className="step-item"><span className="step-num">3</span> 查看智能分析结果</div>
        </div>
      </div>
    );
  }

  const severityMap = {
    "新冠肺炎": "severity-high", "正常": "severity-normal",
    "肺部阴影": "severity-medium", "病毒性肺炎": "severity-low",
  };

  const sortedProbs = CLASS_ORDER.map(name => ({
    name, value: result.probabilities[name] || 0,
    isPrediction: name === result.prediction,
    hint: CLASS_HINTS[name],
  }));

  return (
    <div className="result-display">
      <h2 className="section-title">诊断结果</h2>

      <div className={`result-card ${severityMap[result.prediction] || ""}`}>
        <div className="result-badge" style={{ backgroundColor: result.class_color }}>
          {result.prediction}
        </div>
        <div className="result-confidence">
          <span className="confidence-label">置信度</span>
          <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {result.advice && (
        <div className="advice-box" style={{ borderLeftColor: result.class_color, backgroundColor: `${result.class_color}15` }}>
          <span className="advice-icon">{result.prediction === "正常" ? "✅" : "⚠️"}</span>
          <span className="advice-text">{result.advice}</span>
        </div>
      )}

      <div className="probabilities-section">
        <h3 className="subsection-title">各类别概率分布</h3>
        <div className="prob-list">
          {sortedProbs.map(item => (
            <ProgressBar key={item.name} {...item} color={CLASS_COLORS[item.name] || "#6c757d"} />
          ))}
        </div>
      </div>

      <div className="disclaimer">ℹ️ 本结果仅为AI辅助诊断参考，请以临床医生诊断为准</div>
    </div>
  );
}
