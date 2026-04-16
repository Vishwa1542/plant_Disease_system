import React, { useState, useRef, useCallback } from "react";
import "./App.css";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// SEVERITY BADGE COLOR MAPPING
// ─────────────────────────────────────────────────────────────────────────────
const SEVERITY_COLORS = {
  None: { bg: "#d1fae5", text: "#065f46", label: "✓ Healthy" },
  Moderate: { bg: "#fef3c7", text: "#92400e", label: "⚠ Moderate" },
  High: { bg: "#fee2e2", text: "#991b1b", label: "🔴 High Risk" },
  Critical: { bg: "#fce7f3", text: "#9d174d", label: "☠ Critical" },
  Unknown: { bg: "#f3f4f6", text: "#374151", label: "? Unknown" },
};

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY: Call predict API
// ─────────────────────────────────────────────────────────────────────────────
async function callPredictAPI(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Prediction failed. Please try again.");
  }

  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT: Spinner
// ─────────────────────────────────────────────────────────────────────────────
function Spinner() {
  return (
    <div className="spinner-wrapper">
      <div className="spinner" />
      <p className="spinner-text">Analyzing your plant...</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT: Confidence Bar
// ─────────────────────────────────────────────────────────────────────────────
function ConfidenceBar({ confidence }) {
  const color =
    confidence >= 85 ? "#22c55e" : confidence >= 70 ? "#f59e0b" : "#ef4444";
  return (
    <div className="confidence-bar-wrapper">
      <div className="confidence-label">
        <span>Confidence</span>
        <strong style={{ color }}>{confidence}%</strong>
      </div>
      <div className="confidence-track">
        <div
          className="confidence-fill"
          style={{ width: `${confidence}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT: Result Card
// ─────────────────────────────────────────────────────────────────────────────
function ResultCard({ result, imagePreview }) {
  const { disease, confidence, solution, top5 } = result;
  const severity = SEVERITY_COLORS[solution.severity] || SEVERITY_COLORS.Unknown;
  const isHealthy = solution.severity === "None";

  return (
    <div className={`result-card ${isHealthy ? "result-healthy" : "result-disease"}`}>
      {/* Header */}
      <div className="result-header">
        <div className="result-image-thumb">
          <img src={imagePreview} alt="Analyzed plant" />
        </div>
        <div className="result-title-block">
          <span
            className="severity-badge"
            style={{ background: severity.bg, color: severity.text }}
          >
            {severity.label}
          </span>
          <h2 className="disease-name">{disease}</h2>
          {solution.plant && (
            <p className="plant-name">🌿 {solution.plant}</p>
          )}
        </div>
      </div>

      {/* Confidence */}
      <ConfidenceBar confidence={confidence} />

      {/* Symptoms */}
      <div className="info-section">
        <h3>🔍 Symptoms</h3>
        <p>{solution.symptoms}</p>
      </div>

      {/* Pesticide */}
      {!isHealthy && (
        <div className="info-section pesticide-section">
          <h3>💊 Recommended Treatment</h3>
          <p className="pesticide-text">{solution.pesticide}</p>
        </div>
      )}

      {/* Precautions */}
      <div className="info-section">
        <h3>{isHealthy ? "✅ Maintenance Tips" : "🛡 Precautions"}</h3>
        <ul className="precautions-list">
          {solution.precautions?.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      </div>

      {/* Top 5 predictions */}
      {top5 && Object.keys(top5).length > 0 && (
        <details className="top5-section">
          <summary>📊 Top 5 Predictions</summary>
          <div className="top5-grid">
            {Object.entries(top5).map(([name, prob]) => (
              <div key={name} className="top5-item">
                <span className="top5-name">{name.replace(/_/g, " ")}</span>
                <span className="top5-prob">{prob}%</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT: Low Confidence Warning
// ─────────────────────────────────────────────────────────────────────────────
function LowConfidenceCard({ confidence, message }) {
  return (
    <div className="low-conf-card">
      <div className="low-conf-icon">🔍</div>
      <h3>Low Confidence Result</h3>
      <p>{message}</p>
      <p className="low-conf-score">Score: {confidence}%</p>
      <ul className="low-conf-tips">
        <li>Use natural lighting or good indoor light</li>
        <li>Focus clearly on the affected leaf area</li>
        <li>Avoid blurry or zoomed-out images</li>
        <li>Try uploading a different angle</li>
      </ul>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE: Home
// ─────────────────────────────────────────────────────────────────────────────
function HomePage({ onNavigate }) {
  return (
    <div className="home-page">
      {/* Hero */}
      <div className="hero">
        <div className="hero-badge">🌱 AI-Powered Diagnosis</div>
        <h1 className="hero-title">
          Know Your Plant's
          <em> Health Instantly</em>
        </h1>
        <p className="hero-subtitle">
          Upload a photo or use your camera to detect plant diseases, get
          confidence scores, and receive pesticide recommendations — in seconds.
        </p>
        <button className="cta-button" onClick={() => onNavigate("predict")}>
          Scan a Plant →
        </button>
      </div>

      {/* Features */}
      <div className="features-grid">
        {[
          {
            icon: "🧠",
            title: "Deep Learning CNN",
            desc: "Trained on 87,000+ images across 26 disease classes with 95%+ accuracy.",
          },
          {
            icon: "📸",
            title: "Camera & Upload",
            desc: "Use your phone camera or upload an existing photo from your gallery.",
          },
          {
            icon: "💊",
            title: "Treatment Advice",
            desc: "Get specific pesticide names, dosage, and precautions for each disease.",
          },
          {
            icon: "⚡",
            title: "Instant Results",
            desc: "Disease prediction and recommendations delivered in under 2 seconds.",
          },
        ].map((f) => (
          <div className="feature-card" key={f.title}>
            <span className="feature-icon">{f.icon}</span>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>

      {/* Supported Plants */}
      <div className="plants-section">
        <h2>Supported Plants</h2>
        <div className="plants-tags">
          {[
            "Apple 🍎",
            "Blueberry 🫐",
            "Cherry 🍒",
            "Corn 🌽",
            "Grape 🍇",
            "Orange 🍊",
            "Peach 🍑",
            "Bell Pepper 🫑",
            "Potato 🥔",
            "Raspberry 🫐",
            "Soybean 🌿",
            "Squash 🎃",
            "Strawberry 🍓",
            "Tomato 🍅",
          ].map((p) => (
            <span className="plant-tag" key={p}>
              {p}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE: Predict
// ─────────────────────────────────────────────────────────────────────────────
function PredictPage() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // ── Handle file upload ──────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    loadImage(file);
  };

  const loadImage = (file) => {
    setImageFile(file);
    setResult(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target.result);
    reader.readAsDataURL(file);
  };

  // ── Drag & Drop ─────────────────────────────────────────────────
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      loadImage(file);
    }
  }, []);

  const handleDragOver = (e) => e.preventDefault();

  // ── Camera ───────────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }, // Use back camera on mobile
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setCameraActive(true);
    } catch (err) {
      setError("Camera access denied. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  const capturePhoto = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      loadImage(file);
      stopCamera();
    }, "image/jpeg");
  };

  // ── Predict ─────────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!imageFile) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const data = await callPredictAPI(imageFile);
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    stopCamera();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="predict-page">
      <div className="predict-container">
        {/* Left panel — image input */}
        <div className="input-panel">
          <h2 className="panel-title">Upload or Capture</h2>

          {/* Drop zone */}
          {!cameraActive && (
            <div
              className={`drop-zone ${imagePreview ? "drop-zone--filled" : ""}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => !imagePreview && fileInputRef.current.click()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="preview-img" />
              ) : (
                <div className="drop-placeholder">
                  <span className="drop-icon">🖼</span>
                  <p>Drag & drop an image here</p>
                  <p className="drop-sub">or click to browse</p>
                </div>
              )}
            </div>
          )}

          {/* Camera preview */}
          {cameraActive && (
            <div className="camera-wrapper">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="camera-feed"
              />
            </div>
          )}

          {/* Hidden canvas for capture */}
          <canvas ref={canvasRef} style={{ display: "none" }} />

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />

          {/* Action buttons */}
          <div className="action-buttons">
            {!cameraActive ? (
              <>
                <button
                  className="btn btn-outline"
                  onClick={() => fileInputRef.current.click()}
                >
                  📁 Upload Image
                </button>
                <button className="btn btn-outline" onClick={startCamera}>
                  📷 Open Camera
                </button>
              </>
            ) : (
              <>
                <button className="btn btn-primary" onClick={capturePhoto}>
                  📸 Capture
                </button>
                <button className="btn btn-ghost" onClick={stopCamera}>
                  ✕ Cancel
                </button>
              </>
            )}
          </div>

          {/* Predict / Reset */}
          {imagePreview && !cameraActive && (
            <div className="predict-actions">
              <button
                className="btn btn-predict"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "🔬 Analyze Plant"}
              </button>
              <button className="btn btn-ghost" onClick={handleReset}>
                Reset
              </button>
            </div>
          )}
        </div>

        {/* Right panel — results */}
        <div className="result-panel">
          <h2 className="panel-title">Diagnosis Result</h2>

          {!result && !loading && !error && (
            <div className="result-empty">
              <span className="empty-icon">🌿</span>
              <p>Results will appear here after analysis.</p>
              <p className="empty-sub">
                Upload a clear photo of an affected plant leaf for best results.
              </p>
            </div>
          )}

          {loading && <Spinner />}

          {error && (
            <div className="error-card">
              <span>❌</span>
              <p>{error}</p>
            </div>
          )}

          {result && !loading && (
            <>
              {result.low_confidence ? (
                <LowConfidenceCard
                  confidence={result.confidence}
                  message={result.message}
                />
              ) : (
                <ResultCard result={result} imagePreview={imagePreview} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP — Navigation
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("home");

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <button className="nav-logo" onClick={() => setPage("home")}>
          🌱 PlantGuard <span>AI</span>
        </button>
        <div className="nav-links">
          <button
            className={`nav-link ${page === "home" ? "active" : ""}`}
            onClick={() => setPage("home")}
          >
            Home
          </button>
          <button
            className={`nav-link ${page === "predict" ? "active" : ""}`}
            onClick={() => setPage("predict")}
          >
            Diagnose
          </button>
        </div>
      </nav>

      {/* Page content */}
      <main className="main-content">
        {page === "home" ? (
          <HomePage onNavigate={setPage} />
        ) : (
          <PredictPage />
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>🌱 PlantGuard AI · CNN-powered disease detection · 26 plant classes</p>
      </footer>
    </div>
  );
}
