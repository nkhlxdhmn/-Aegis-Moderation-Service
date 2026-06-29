const tabs = document.querySelectorAll(".mode-tab");
const uploadPane = document.querySelector("#uploadPane");
const urlPane = document.querySelector("#urlPane");
const form = document.querySelector("#analyzeForm");
const fileInput = document.querySelector("#imageFile");
const fileHint = document.querySelector("#fileHint");
const errorBox = document.querySelector("#formError");
const button = document.querySelector("#analyzeButton");
const emptyState = document.querySelector("#emptyState");
const results = document.querySelector("#results");
const statusPill = document.querySelector("#serviceStatus");
let mode = "upload";
let latestReport = null;
const palette = ["#0f766e", "#2563eb", "#eab308", "#dc2626", "#7c3aed", "#be123c"];

function setMode(nextMode) {
  mode = nextMode;
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  uploadPane.classList.toggle("hidden", mode !== "upload");
  urlPane.classList.toggle("hidden", mode !== "url");
  errorBox.textContent = "";
}

tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
fileInput.addEventListener("change", () => {
  fileHint.textContent = fileInput.files.length ? fileInput.files[0].name : "JPEG, PNG, WEBP, or GIF up to 10 MB";
});

async function checkHealth() {
  try {
    const response = await fetch("/api/v1/health");
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    statusPill.textContent = `${health.status.toUpperCase()} / ${health.mode}`;
    statusPill.classList.add("ok");
  } catch {
    statusPill.textContent = "Service unavailable";
  }
}

function riskClass(risk) {
  const normalized = String(risk).toLowerCase();
  if (normalized.includes("critical")) return "critical";
  if (normalized.includes("high")) return "high";
  if (normalized.includes("medium")) return "medium";
  return "";
}

function renderReport(report) {
  latestReport = report;
  emptyState.classList.add("hidden");
  results.classList.remove("hidden");
  document.querySelector("#overallScore").textContent = `${Number(report.overall_score).toFixed(1)}%`;
  document.querySelector("#scoreBar").style.width = `${report.overall_score}%`;
  const badge = document.querySelector("#riskLevel");
  badge.textContent = report.risk_level;
  badge.className = `risk-badge ${riskClass(report.risk_level)}`;
  document.querySelector("#decision").textContent = report.decision;
  document.querySelector("#recommendation").textContent = report.recommendation;

  const labels = report.category_labels || {};
  const entries = Object.entries(report.categories || {}).sort((a, b) => b[1] - a[1]);
  document.querySelector("#categoryList").innerHTML = entries.map(([key, value]) => `
    <div class="category-item">
      <div class="category-top"><span>${labels[key] || key}</span><span>${Number(value).toFixed(1)}%</span></div>
      <div class="mini-bar"><span style="width:${Math.max(0, Math.min(100, value))}%"></span></div>
    </div>`).join("");

  const top = entries.slice(0, 6);
  document.querySelector("#topSignals").innerHTML = top.map(([key, value]) => `
    <div class="top-signal"><span>${labels[key] || key}</span><strong>${Number(value).toFixed(1)}%</strong></div>`).join("");

  let start = 0;
  const total = Math.max(top.reduce((sum, [, value]) => sum + Number(value), 0), 1);
  const slices = top.map(([, value], index) => {
    const size = (Number(value) / total) * 100;
    const segment = `${palette[index % palette.length]} ${start}% ${start + size}%`;
    start += size;
    return segment;
  });
  document.querySelector("#pieChart").style.background = `conic-gradient(${slices.join(",")})`;

  const objects = report.objects && report.objects.length ? report.objects : ["None detected"];
  document.querySelector("#objects").innerHTML = objects.map((item) => `<span class="chip">${item}</span>`).join("");
  document.querySelector("#ocrText").textContent = report.ocr_text || "No OCR text detected.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";
  const data = new FormData();
  const caption = document.querySelector("#caption").value.trim();
  if (caption) data.append("caption", caption);

  if (mode === "upload") {
    if (!fileInput.files.length) {
      errorBox.textContent = "Choose an image first.";
      return;
    }
    data.append("file", fileInput.files[0]);
  } else {
    const url = document.querySelector("#imageUrl").value.trim();
    if (!url) {
      errorBox.textContent = "Enter an HTTPS image URL.";
      return;
    }
    data.append("image_url", url);
  }

  button.disabled = true;
  button.textContent = "Analyzing...";
  try {
    const response = await fetch("/api/v1/analyze", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed.");
    renderReport(payload);
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Analyze";
  }
});

document.querySelector("#downloadJson").addEventListener("click", () => {
  if (!latestReport) return;
  const blob = new Blob([JSON.stringify(latestReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "aegis-moderation-report.json";
  link.click();
  URL.revokeObjectURL(url);
});

document.querySelector("#downloadPdf").addEventListener("click", () => window.print());
checkHealth();
