const RULE_DB_CANDIDATES = [
  "./output.json",
  "./github_pages_code_checking/output.json",
  "/github_pages_code_checking/output.json",
];

const languageSelect = document.getElementById("languageSelect");
const codeInput = document.getElementById("codeInput");
const scanButton = document.getElementById("scanButton");
const resultsList = document.getElementById("resultsList");
const summaryText = document.getElementById("summaryText");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");

const semgrepToCodeMirrorMode = {
  bash: "shell",
  c: "text/x-csrc",
  csharp: "text/x-csharp",
  erlang: "erlang",
  fortran: "fortran",
  go: "go",
  html: "xml",
  java: "text/x-java",
  javascript: "javascript",
  js: "javascript",
  kotlin: "text/x-kotlin",
  lua: "lua",
  perl: "perl",
  php: "application/x-httpd-php",
  powershell: "powershell",
  python: "python",
  r: "r",
  ruby: "ruby",
  rust: "rust",
  scala: "text/x-scala",
  swift: "swift",
  ts: "text/typescript",
  typescript: "text/typescript",
  yaml: "yaml",
};

let allRules = [];
let editor = null;

function setLoading(isLoading, text = "Scanning with OpenGrep...") {
  loadingText.textContent = text;
  loadingOverlay.classList.toggle("hidden", !isLoading);
  scanButton.disabled = isLoading;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function initializeEditor() {
  if (!window.CodeMirror) {
    return;
  }

  editor = window.CodeMirror.fromTextArea(codeInput, {
    mode: "python",
    theme: "material-darker",
    lineNumbers: true,
    indentUnit: 2,
    tabSize: 2,
    indentWithTabs: false,
    lineWrapping: false,
  });
}

function getEditorCode() {
  return editor ? editor.getValue() : codeInput.value;
}

function setEditorMode() {
  if (!editor) {
    return;
  }
  const selected = languageSelect.value;
  editor.setOption("mode", semgrepToCodeMirrorMode[selected] || null);
}

function getSeverityClass(severity) {
  const value = String(severity || "INFO").toLowerCase();
  if (value.includes("error")) {
    return "error";
  }
  if (value.includes("warn")) {
    return "warning";
  }
  return "info";
}

function populateLanguageOptions(rules) {
  const langs = new Set();
  for (const rule of rules) {
    if (!Array.isArray(rule.languages)) {
      continue;
    }
    for (const lang of rule.languages) {
      const value = String(lang || "").trim().toLowerCase();
      if (value) {
        langs.add(value);
      }
    }
  }

  const sorted = [...langs].sort((a, b) => a.localeCompare(b));
  languageSelect.innerHTML = "";

  for (const lang of sorted) {
    const opt = document.createElement("option");
    opt.value = lang;
    opt.textContent = lang;
    languageSelect.append(opt);
  }

  if (sorted.includes("python")) {
    languageSelect.value = "python";
  }

  setEditorMode();
}

function renderResults(findings) {
  resultsList.innerHTML = "";

  if (!findings.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No findings from OpenGrep for this input.";
    resultsList.append(empty);
    return;
  }

  for (const item of findings) {
    const card = document.createElement("article");
    card.className = `result-card ${getSeverityClass(item.severity)}`;

    const where = item.line ? `${item.sourceFile}:${item.line}` : item.sourceFile;

    card.innerHTML = `
      <h3>${escapeHtml(item.id)}</h3>
      <div class="meta">${escapeHtml(item.severity)} | ${escapeHtml(where)}</div>
      <div class="meta">${escapeHtml(item.message || "No message")}</div>
      <div class="snippet">${escapeHtml(item.snippet || "")}</div>
    `;

    resultsList.append(card);
  }
}

async function fetchRuleDatabase() {
  const errors = [];

  for (const path of RULE_DB_CANDIDATES) {
    try {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (!Array.isArray(data.rules)) {
        throw new Error("Missing rules array");
      }
      return { data, path };
    } catch (error) {
      errors.push(`${path} -> ${error.message}`);
    }
  }

  throw new Error(errors.join(" | "));
}

async function runOpenGrepScan(code, language) {
  const response = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, language }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }

  return payload;
}

async function loadRules() {
  setLoading(true, "Loading rule index...");
  try {
    const { data, path } = await fetchRuleDatabase();
    allRules = data.rules;
    populateLanguageOptions(allRules);
    summaryText.textContent = `Loaded ${allRules.length} rules from ${path}.`;
  } catch (error) {
    summaryText.textContent = "Could not load rule index.";
    resultsList.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
  } finally {
    setLoading(false);
  }
}

async function handleScan() {
  const code = getEditorCode();
  const language = languageSelect.value;

  if (!code.trim()) {
    summaryText.textContent = "Please paste code first.";
    resultsList.innerHTML = '<p class="empty">No code to scan yet.</p>';
    return;
  }

  setLoading(true, "Running OpenGrep engine...");
  try {
    const result = await runOpenGrepScan(code, language);
    renderResults(result.findings || []);

    summaryText.textContent = [
      `Engine: ${result.engine || "opengrep"}`,
      `Configs: ${(result.configs || []).length}`,
      `Findings: ${result.findingCount || 0}`,
      `Errors: ${(result.errors || []).length}`,
    ].join(" | ");
  } catch (error) {
    summaryText.textContent = "OpenGrep scan failed.";
    resultsList.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
  } finally {
    setLoading(false);
  }
}

initializeEditor();
languageSelect.addEventListener("change", setEditorMode);
scanButton.addEventListener("click", handleScan);
loadRules();
