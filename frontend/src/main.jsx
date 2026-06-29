import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const contentTypes = [
  { id: "image", label: "Image", endpoint: "/api/v1/moderate/image", accept: "image/jpeg,image/png,image/webp,image/gif", max: "10 MB" },
  { id: "video", label: "Video", endpoint: "/api/v1/moderate/video", accept: "video/mp4,video/quicktime,video/webm,video/x-matroska", max: "250 MB" },
  { id: "text", label: "Text", endpoint: "/api/v1/moderate/text" },
  { id: "pdf", label: "PDF", endpoint: "/api/v1/moderate/pdf", accept: "application/pdf,.pdf", max: "25 MB" },
  { id: "docx", label: "DOCX", endpoint: "/api/v1/moderate/docx", accept: "application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx", max: "25 MB" },
];

const baseCategoryOrder = [
  ["adult_content", "Adult Content"],
  ["nudity", "Nudity"],
  ["suggestive_content", "Suggestive Content"],
  ["violence", "Violence"],
  ["graphic_violence", "Graphic Violence"],
  ["weapons", "Weapons"],
  ["drugs", "Drugs"],
  ["blood", "Blood"],
  ["medical_content", "Medical Content"],
  ["political_propaganda", "Political Content"],
  ["religious_extremism", "Religious Extremism"],
  ["hate_speech", "Hate Speech"],
  ["hate_symbol", "Hate Symbols"],
  ["toxic_text", "Toxic Language"],
  ["spam", "Spam"],
  ["scam", "Scam"],
  ["phishing", "Phishing"],
  ["malware_links", "Malware Links"],
  ["misinformation", "Misinformation"],
  ["self_harm", "Self Harm"],
  ["child_safety_risk", "Child Safety Risk"],
  ["pii_detection", "Personal Information"],
  ["qr_code", "QR Codes"],
  ["barcode", "Barcodes"],
  ["watermark", "Watermarks"],
  ["copyright_notice", "Copyright Notices"],
  ["sensitive_document", "Sensitive Documents"],
  ["financial_information", "Financial Information"],
  ["identity_document", "Identity Documents"],
  ["document", "Document"],
];

const stageMap = {
  image: ["Validation", "OCR", "Vision Models", "Classification", "Rule Engine"],
  video: ["Validation", "Frame Extraction", "Vision Models", "Transcript", "Rule Engine"],
  text: ["Validation", "Text Classifier", "PII Detection", "Rule Engine"],
  pdf: ["Validation", "Metadata", "Text Extraction", "OCR", "Rule Engine"],
  docx: ["Validation", "Text Extraction", "Embedded Images", "PII Detection", "Rule Engine"],
};

const theme = {
  bg: "#f5f7f8",
  panel: "rgba(255,255,255,0.96)",
  ink: "#16201d",
  muted: "#60716b",
  line: "#d9e1de",
  accent: "#0f766e",
  accentStrong: "#115e59",
  warn: "#b45309",
  danger: "#b91c1c",
  soft: "#edf3f1",
};

const styles = {
  page: {
    minHeight: "100vh",
    margin: 0,
    background: `radial-gradient(circle at 20% 0%, rgba(15,118,110,.12), transparent 30%), ${theme.bg}`,
    color: theme.ink,
    fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  shell: { width: "min(1220px, calc(100% - 32px))", margin: "0 auto", padding: "34px 0" },
  hero: { display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 24, padding: "10px 0 26px", flexWrap: "wrap" },
  eyebrow: { color: theme.muted, fontSize: 13, fontWeight: 800, textTransform: "uppercase" },
  title: { margin: "6px 0 8px", fontSize: "clamp(2.2rem, 5vw, 4.8rem)", lineHeight: 0.95, letterSpacing: 0 },
  subhead: { margin: 0, color: theme.muted, fontSize: 18, maxWidth: 760 },
  status: { border: `1px solid ${theme.line}`, borderRadius: 999, padding: "10px 14px", background: "rgba(255,255,255,.76)", color: theme.accentStrong, fontWeight: 800 },
  workspace: { display: "grid", gridTemplateColumns: "minmax(300px, 430px) 1fr", gap: 20, alignItems: "start" },
  panel: { background: theme.panel, border: `1px solid ${theme.line}`, borderRadius: 8, boxShadow: "0 18px 60px rgba(18,32,29,.12)" },
  inputPanel: { padding: 18, position: "sticky", top: 18 },
  selector: { display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 6, padding: 4, border: `1px solid ${theme.line}`, borderRadius: 8, background: theme.soft, marginBottom: 14 },
  tab: (active) => ({ border: 0, borderRadius: 6, padding: "10px 7px", background: active ? "#fff" : "transparent", color: active ? theme.ink : theme.muted, fontWeight: 900, cursor: "pointer", minHeight: 40 }),
  sourceTabs: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 16 },
  sourceTab: (active) => ({ border: `1px solid ${active ? theme.accent : theme.line}`, borderRadius: 8, padding: 10, background: active ? "#e7f3f1" : "#fff", color: active ? theme.accentStrong : theme.muted, fontWeight: 900, cursor: "pointer" }),
  dropzone: { display: "grid", placeItems: "center", minHeight: 170, border: "1.5px dashed #a7b8b2", borderRadius: 8, background: "#f8fbfa", padding: 22, textAlign: "center", cursor: "pointer" },
  field: { display: "grid", gap: 8, marginTop: 14 },
  label: { color: theme.muted, fontWeight: 850, fontSize: 14 },
  input: { width: "100%", border: `1px solid ${theme.line}`, borderRadius: 8, padding: 12, background: "#fbfdfc", color: theme.ink, boxSizing: "border-box", font: "inherit" },
  primary: { width: "100%", marginTop: 16, padding: "13px 16px", color: "#fff", background: theme.accent, border: 0, borderRadius: 8, fontWeight: 950, cursor: "pointer" },
  secondary: { padding: "10px 13px", color: theme.accentStrong, background: "#e7f3f1", border: 0, borderRadius: 8, fontWeight: 900, cursor: "pointer" },
  preview: { width: "100%", aspectRatio: "4 / 3", borderRadius: 8, border: `1px solid ${theme.line}`, objectFit: "cover", background: "#101816", marginTop: 14 },
  fileCard: { border: `1px solid ${theme.line}`, borderRadius: 8, padding: 14, marginTop: 14, background: "#fbfdfc" },
  progress: { marginTop: 16, display: "grid", gap: 8 },
  step: (done) => ({ display: "flex", alignItems: "center", gap: 8, color: done ? theme.accentStrong : theme.muted, fontWeight: 850 }),
  resultPanel: { padding: 20, minHeight: 590 },
  split: { display: "grid", gridTemplateColumns: "minmax(220px, 330px) 1fr", gap: 20, alignItems: "start" },
  resultImage: { width: "100%", aspectRatio: "1 / 1", objectFit: "cover", borderRadius: 8, border: `1px solid ${theme.line}`, background: "#101816" },
  empty: { minHeight: 520, display: "grid", placeContent: "center", textAlign: "center", color: theme.muted },
  sectionLabel: { display: "block", color: theme.muted, fontSize: 12, fontWeight: 900, textTransform: "uppercase" },
  score: { display: "block", fontSize: 54, lineHeight: 1, marginTop: 6 },
  scoreBar: { height: 14, overflow: "hidden", borderRadius: 999, background: "#e3ebe8", margin: "12px 0 18px" },
  scoreFill: (value) => ({ display: "block", width: `${clamp(value)}%`, height: "100%", borderRadius: "inherit", background: "linear-gradient(90deg,#0f766e,#eab308,#dc2626)" }),
  risk: (risk) => ({ display: "inline-flex", borderRadius: 999, padding: "10px 13px", background: risk.includes("CRITICAL") ? "#fee2e2" : risk.includes("HIGH") || risk.includes("MEDIUM") ? "#fff4de" : "#e7f3f1", color: risk.includes("CRITICAL") ? theme.danger : risk.includes("HIGH") || risk.includes("MEDIUM") ? theme.warn : theme.accentStrong, fontWeight: 950 }),
  categoryList: { display: "grid", gap: 8, marginTop: 10 },
  categoryRow: { display: "grid", gridTemplateColumns: "170px 1fr 58px", gap: 10, alignItems: "center", fontWeight: 850 },
  miniTrack: { height: 8, background: "#e3ebe8", borderRadius: 999, overflow: "hidden" },
  miniFill: (value) => ({ display: "block", width: `${clamp(value)}%`, height: "100%", background: theme.accent }),
  chips: { display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 },
  chip: { borderRadius: 999, padding: "7px 10px", background: theme.soft, color: theme.accentStrong, fontWeight: 850 },
  ocr: { minHeight: 96, margin: "10px 0 0", whiteSpace: "pre-wrap", border: `1px solid ${theme.line}`, borderRadius: 8, padding: 12, background: "#101816", color: "#e9f4f1", overflow: "auto" },
  notice: { marginTop: 16, border: "1px solid #f1c37d", borderRadius: 8, padding: 12, background: "#fff8eb", color: theme.warn, fontWeight: 850 },
  actions: { display: "flex", gap: 10, flexWrap: "wrap", marginTop: 18 },
  metaGrid: { display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8, marginTop: 10 },
  metaCell: { border: `1px solid ${theme.line}`, borderRadius: 8, padding: 10, background: "#fbfdfc" },
};

function clamp(value) {
  return Math.max(0, Math.min(100, Number(value) || 0));
}

function App() {
  const [contentType, setContentType] = useState("image");
  const [sourceMode, setSourceMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [remoteUrl, setRemoteUrl] = useState("");
  const [text, setText] = useState("");
  const [caption, setCaption] = useState("");
  const [preview, setPreview] = useState("");
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [activeStage, setActiveStage] = useState(-1);
  const [health, setHealth] = useState("Checking service");
  const fileRef = useRef(null);

  const current = contentTypes.find((item) => item.id === contentType) || contentTypes[0];
  const supportsUrl = contentType === "image" || contentType === "pdf" || contentType === "docx";
  const supportsUpload = contentType !== "text";
  const stages = stageMap[contentType] || stageMap.image;

  useEffect(() => {
    fetch("/api/v1/health")
      .then((response) => response.json())
      .then((payload) => setHealth(`${payload.status.toUpperCase()} / ${payload.mode}`))
      .catch(() => setHealth("Service unavailable"));
  }, []);

  useEffect(() => {
    setFile(null);
    setRemoteUrl("");
    setText("");
    setPreview("");
    setReport(null);
    setError("");
    setSourceMode(contentType === "text" ? "text" : "upload");
    setActiveStage(-1);
  }, [contentType]);

  useEffect(() => {
    if (!file || contentType !== "image") return undefined;
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file, contentType]);

  useEffect(() => {
    if (sourceMode === "url" && contentType === "image") setPreview(remoteUrl.trim());
  }, [remoteUrl, sourceMode, contentType]);

  useEffect(() => {
    if (!busy) return undefined;
    setActiveStage(0);
    const interval = window.setInterval(() => {
      setActiveStage((stage) => Math.min(stage + 1, stages.length - 1));
    }, 650);
    return () => window.clearInterval(interval);
  }, [busy, stages.length]);

  const categoryRows = useMemo(() => {
    const categories = report?.categories || {};
    const labels = report?.category_labels || {};
    const keys = new Set(baseCategoryOrder.map(([key]) => key));
    const rows = baseCategoryOrder.map(([key, label]) => [key, labels[key] || label, Number(categories[key] || 0)]);
    Object.keys(categories).forEach((key) => {
      if (!keys.has(key)) rows.push([key, labels[key] || prettify(key), Number(categories[key] || 0)]);
    });
    return rows;
  }, [report]);

  async function analyze(event) {
    event.preventDefault();
    setError("");
    setReport(null);

    let request;
    if (contentType === "text") {
      if (!text.trim()) {
        setError("Enter text to analyze.");
        return;
      }
      request = {
        url: current.endpoint,
        options: {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.trim() }),
        },
      };
    } else {
      const data = new FormData();
      if (caption.trim() && (contentType === "image" || contentType === "video")) data.append("caption", caption.trim());
      if (sourceMode === "upload") {
        if (!file) {
          setError(`Choose a ${current.label} file first.`);
          return;
        }
        data.append("file", file);
      } else {
        if (!remoteUrl.trim()) {
          setError(`Enter an HTTPS ${current.label} URL.`);
          return;
        }
        data.append(contentType === "image" ? "image_url" : "document_url", remoteUrl.trim());
      }
      request = { url: current.endpoint, options: { method: "POST", body: data } };
    }

    setBusy(true);
    try {
      const response = await fetch(request.url, request.options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Analysis failed.");
      setReport(payload);
      setActiveStage(stages.length - 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function downloadJson() {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `aegis-${contentType}-moderation-report.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main style={styles.shell}>
      <section style={styles.hero}>
        <div>
          <div style={styles.eyebrow}>AI-Powered Multimodal Moderation Platform</div>
          <h1 style={styles.title}>Aegis Moderation</h1>
          <p style={styles.subhead}>Analyze images, videos, text, PDFs, and DOCX files locally with detailed safety reports.</p>
        </div>
        <div style={styles.status}>{health}</div>
      </section>

      <section style={styles.workspace}>
        <form onSubmit={analyze} style={{ ...styles.panel, ...styles.inputPanel }}>
          <div style={styles.selector}>
            {contentTypes.map((item) => (
              <button key={item.id} type="button" style={styles.tab(contentType === item.id)} onClick={() => setContentType(item.id)}>
                {item.label}
              </button>
            ))}
          </div>

          {supportsUrl && (
            <div style={styles.sourceTabs}>
              <button type="button" style={styles.sourceTab(sourceMode === "upload")} onClick={() => setSourceMode("upload")}>Upload</button>
              <button type="button" style={styles.sourceTab(sourceMode === "url")} onClick={() => setSourceMode("url")}>URL</button>
            </div>
          )}

          {supportsUpload && sourceMode === "upload" && (
            <div style={styles.dropzone} onClick={() => fileRef.current?.click()}>
              <input ref={fileRef} type="file" accept={current.accept} hidden onChange={(event) => setFile(event.target.files?.[0] || null)} />
              <strong>{file ? file.name : `Upload ${current.label}`}</strong>
              <span style={{ color: theme.muted, marginTop: 6 }}>{current.label} file up to {current.max}</span>
            </div>
          )}

          {supportsUrl && sourceMode === "url" && (
            <label style={styles.field}>
              <span style={styles.label}>Paste {current.label} URL</span>
              <input style={styles.input} value={remoteUrl} onChange={(event) => setRemoteUrl(event.target.value)} placeholder={`https://example.com/file.${contentType}`} />
            </label>
          )}

          {contentType === "text" && (
            <label style={styles.field}>
              <span style={styles.label}>Text Content</span>
              <textarea style={{ ...styles.input, resize: "vertical", minHeight: 190 }} value={text} onChange={(event) => setText(event.target.value)} placeholder="Paste text to moderate..." />
            </label>
          )}

          {preview && contentType === "image" ? <img style={styles.preview} src={preview} alt="Selected preview" /> : null}
          {file && contentType !== "image" ? <div style={styles.fileCard}><strong>{file.name}</strong><br /><span style={{ color: theme.muted }}>{Math.round(file.size / 1024).toLocaleString()} KB selected</span></div> : null}

          {(contentType === "image" || contentType === "video") && (
            <label style={styles.field}>
              <span style={styles.label}>Optional context</span>
              <textarea style={{ ...styles.input, resize: "vertical" }} rows="3" value={caption} onChange={(event) => setCaption(event.target.value)} placeholder="Add context when it helps the analysis" />
            </label>
          )}

          <button style={{ ...styles.primary, opacity: busy ? 0.7 : 1 }} type="submit" disabled={busy}>
            {busy ? `Analyzing ${current.label.toLowerCase()}...` : `Analyze ${current.label}`}
          </button>
          {error ? <p style={{ color: theme.danger, fontWeight: 850 }}>{error}</p> : null}

          {(busy || activeStage >= 0) && (
            <div style={styles.progress}>
              <strong>Analyzing {current.label.toLowerCase()}...</strong>
              {stages.map((stage, index) => <div key={stage} style={styles.step(index <= activeStage)}>OK {stage}</div>)}
            </div>
          )}
        </form>

        <section style={{ ...styles.panel, ...styles.resultPanel }} aria-live="polite">
          {!report ? (
            <div style={styles.empty}>
              <h2 style={{ color: theme.ink, margin: "0 0 8px", fontSize: 32 }}>Moderation Report</h2>
              <p>Your preview and report will appear here after analysis.</p>
            </div>
          ) : (
            <div style={styles.split}>
              <ContentPreview contentType={contentType} preview={preview} file={file} text={text} report={report} />
              <div>
                <span style={styles.sectionLabel}>Overall Safety Score</span>
                <strong style={styles.score}>{Number(report.overall_score).toFixed(1)}%</strong>
                <div style={styles.scoreBar}><span style={styles.scoreFill(report.overall_score)} /></div>
                <span style={styles.sectionLabel}>Risk Level</span>
                <span style={styles.risk(report.risk_level || "")}>{report.risk_level}</span>

                {report.error ? <div style={styles.notice}>Pipeline notice: {report.error}</div> : null}
                <DocumentDetails report={report} />

                <h3>Detected Categories</h3>
                <div style={styles.categoryList}>
                  {categoryRows.map(([key, label, value]) => (
                    <div key={key} style={styles.categoryRow}>
                      <span>{label}</span>
                      <span style={styles.miniTrack}><span style={styles.miniFill(value)} /></span>
                      <span>{value.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>

                <h3>Detected Objects</h3>
                <div style={styles.chips}>{(report.objects?.length ? report.objects : ["None detected"]).map((item) => <span key={item} style={styles.chip}>{item}</span>)}</div>

                <h3>{report.content_type === "text" ? "Text Preview" : "OCR / Extracted Text"}</h3>
                <pre style={styles.ocr}>{report.ocr_text || report.extracted_text_preview || report.video?.transcript || "No text detected."}</pre>

                <h3>Recommendation</h3>
                <strong>{report.decision === "Reject" ? "Reject" : report.decision === "Review Required" ? "Review" : "Accept"} - {report.recommendation || report.decision}</strong>

                <div style={styles.actions}>
                  <button type="button" style={styles.secondary} onClick={downloadJson}>Download JSON</button>
                  <button type="button" style={styles.secondary} onClick={() => window.print()}>Download PDF</button>
                </div>
              </div>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function ContentPreview({ contentType, preview, file, text, report }) {
  if (contentType === "image" && preview) {
    return <img style={styles.resultImage} src={preview} alt="Analyzed content" />;
  }
  if (contentType === "text") {
    return <div style={styles.fileCard}><strong>Text Submission</strong><pre style={{ whiteSpace: "pre-wrap", color: theme.muted }}>{text.slice(0, 800)}</pre></div>;
  }
  const info = report.document?.file_info || {};
  return (
    <div style={styles.fileCard}>
      <span style={styles.sectionLabel}>Selected Content</span>
      <h3 style={{ marginTop: 8 }}>{file?.name || info.filename || prettify(contentType)}</h3>
      <p style={{ color: theme.muted }}>{prettify(contentType)} moderation report</p>
    </div>
  );
}

function DocumentDetails({ report }) {
  const document = report.document;
  if (!document) return null;
  const info = document.file_info || {};
  const cells = [
    ["File Type", info.file_type || report.content_type?.toUpperCase()],
    ["File Size", info.file_size_bytes ? `${Math.round(info.file_size_bytes / 1024).toLocaleString()} KB` : "Unknown"],
    ["Pages", document.page_count ?? "N/A"],
    ["Processing Time", `${document.processing_time_seconds ?? 0}s`],
    ["Embedded Images", document.embedded_images ?? 0],
    ["Tables", document.table_count ?? 0],
  ];
  return (
    <>
      <h3>File Information</h3>
      <div style={styles.metaGrid}>
        {cells.map(([label, value]) => (
          <div key={label} style={styles.metaCell}>
            <span style={styles.sectionLabel}>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      {document.links?.length ? (
        <>
          <h3>Detected Links</h3>
          <div style={styles.chips}>{document.links.slice(0, 8).map((link) => <span key={link} style={styles.chip}>{link}</span>)}</div>
        </>
      ) : null}
    </>
  );
}

function prettify(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <div style={styles.page}>
      <App />
    </div>
  </React.StrictMode>,
);
