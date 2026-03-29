#!/usr/bin/env node

const http = require("http");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const os = require("os");
const { spawn } = require("child_process");

const HOST = process.env.HOST || "0.0.0.0";
const PORT = Number(process.env.PORT || 8000);
const OPENGREP_BIN = process.env.OPENGREP_BIN || "opengrep";
const ROOT = __dirname;
const REPO_ROOT = path.resolve(ROOT, "..");
const RULES_ROOT = path.join(REPO_ROOT, "opengrep-rules");
const OUTPUT_JSON_PATH = path.join(ROOT, "output.json");
const MAX_CONFIG_FILES = 20;
const OPENGREP_PROCESS_TIMEOUT_MS = 20000;
const MAX_REQUEST_BYTES = Number(process.env.MAX_REQUEST_BYTES || 200000);
const COMMON_ANCHORS = new Set([
  "import",
  "return",
  "class",
  "function",
  "def",
  "true",
  "false",
  "none",
  "http",
  "https",
  "string",
  "object",
  "select",
  "insert",
  "update",
  "delete",
]);

const LANG_TO_EXT = {
  apex: ".apex",
  bash: ".sh",
  c: ".c",
  clojure: ".clj",
  cpp: ".cpp",
  csharp: ".cs",
  dart: ".dart",
  dockerfile: "Dockerfile",
  elixir: ".ex",
  erlang: ".erl",
  generic: ".txt",
  go: ".go",
  html: ".html",
  java: ".java",
  javascript: ".js",
  js: ".js",
  json: ".json",
  kotlin: ".kt",
  lua: ".lua",
  php: ".php",
  powershell: ".ps1",
  python: ".py",
  r: ".r",
  ruby: ".rb",
  rust: ".rs",
  scala: ".scala",
  swift: ".swift",
  terraform: ".tf",
  ts: ".ts",
  typescript: ".ts",
  yaml: ".yaml",
};

const LANG_DIR_ALIASES = {
  js: ["javascript"],
  ts: ["typescript"],
  csharp: ["csharp"],
};

let cachedRules = [];
try {
  const parsed = JSON.parse(fs.readFileSync(OUTPUT_JSON_PATH, "utf-8"));
  cachedRules = Array.isArray(parsed?.rules) ? parsed.rules : [];
} catch {
  cachedRules = [];
}

function sendJson(res, statusCode, obj) {
  const payload = JSON.stringify(obj);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
    "Cache-Control": "no-store",
  });
  res.end(payload);
}

function safeJoin(root, requestPath) {
  const normalized = path.normalize(requestPath).replace(/^([.][.][/\\])+/, "");
  const fullPath = path.join(root, normalized);
  if (!fullPath.startsWith(root)) {
    return null;
  }
  return fullPath;
}

function contentTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".js") return "application/javascript; charset=utf-8";
  if (ext === ".json") return "application/json; charset=utf-8";
  return "text/plain; charset=utf-8";
}

async function parseBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let totalBytes = 0;
    let aborted = false;

    req.on("data", (chunk) => {
      if (aborted) {
        return;
      }

      totalBytes += chunk.length;
      if (totalBytes > MAX_REQUEST_BYTES) {
        aborted = true;
        reject(new Error(`Request body too large (max ${MAX_REQUEST_BYTES} bytes)`));
        req.destroy();
        return;
      }

      chunks.push(chunk);
    });

    req.on("end", () => {
      if (aborted) {
        return;
      }

      try {
        const text = Buffer.concat(chunks).toString("utf-8");
        resolve(text ? JSON.parse(text) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

async function exists(p) {
  try {
    await fsp.access(p);
    return true;
  } catch {
    return false;
  }
}

async function resolveConfigPaths(language) {
  const code = arguments[1] || "";
  const lang = String(language || "").trim().toLowerCase();
  return resolveFallbackConfigPaths(lang, String(code || ""));
}

async function resolveFallbackConfigPaths(language, code) {
  const codeLower = String(code || "").toLowerCase();
  const configs = [];

  const languageRoots = [language, ...(LANG_DIR_ALIASES[language] || [])].filter(Boolean);

  for (const langRoot of languageRoots) {
    const languagePath = path.join(RULES_ROOT, langRoot);
    if (await exists(languagePath)) {
      configs.push(languagePath);
    }
  }

  // Framework-specific boost for better precision and speed on common Python snippets.
  if (language === "python") {
    const frameworkHints = [
      ["pyramid", "python/pyramid"],
      ["flask", "python/flask"],
      ["django", "python/django"],
      ["jwt", "python/jwt"],
      ["sqlalchemy", "python/sqlalchemy"],
    ];

    for (const [token, relPath] of frameworkHints) {
      if (!codeLower.includes(token)) {
        continue;
      }
      const p = path.join(RULES_ROOT, relPath);
      if (await exists(p)) {
        configs.unshift(p);
      }
    }
  }

  return [...new Set(configs)];
}

function regexAnchor(rawRegex) {
  const cleaned = String(rawRegex || "")
    .replace(/\\./g, " ")
    .replace(/\[[^\]]*\]/g, " ")
    .replace(/[(){}?+*|^$]/g, " ");
  const tokens = cleaned.match(/[A-Za-z_][A-Za-z0-9_.:\/-]{3,}/g) || [];
  if (!tokens.length) {
    return "";
  }

  const filtered = tokens
    .map((t) => t.toLowerCase())
    .filter((t) => t.length >= 5 && !COMMON_ANCHORS.has(t));
  if (!filtered.length) {
    return "";
  }

  filtered.sort((a, b) => b.length - a.length);
  return filtered[0];
}

async function resolvePrefilteredRuleFiles(language, code) {
  if (!cachedRules.length) {
    return [];
  }

  const codeLower = code.toLowerCase();
  const candidates = [];

  for (const rule of cachedRules) {
    const langs = Array.isArray(rule.languages)
      ? rule.languages.map((x) => String(x || "").toLowerCase())
      : [];
    if (!(langs.includes(language) || langs.includes("generic"))) {
      continue;
    }

    const source = String(rule.source_file || "");
    if (!source) {
      continue;
    }

    const regexes = Array.isArray(rule.match_regexes) ? rule.match_regexes : [];
    let likely = false;

    for (const rx of regexes) {
      const anchor = regexAnchor(rx);
      if (!anchor) {
        continue;
      }
      if (codeLower.includes(anchor)) {
        likely = true;
        break;
      }
    }

    if (likely) {
      candidates.push(source);
    }
  }

  const unique = [...new Set(candidates)].slice(0, MAX_CONFIG_FILES);
  const resolved = [];
  for (const rel of unique) {
    const full = path.join(RULES_ROOT, rel);
    if (await exists(full)) {
      resolved.push(full);
    }
  }

  return resolved;
}

function extractJsonFromCli(stdoutText) {
  const lines = stdoutText.split(/\r?\n/);
  const start = lines.findIndex((line) => line.trim().startsWith("{"));
  if (start === -1) {
    throw new Error("Could not locate JSON payload in opengrep output");
  }
  const jsonText = lines.slice(start).join("\n").trim();
  return JSON.parse(jsonText);
}

async function runOpengrepScan({ language, code }) {
  const normalizedLang = String(language || "python").trim().toLowerCase();
  const ext = LANG_TO_EXT[normalizedLang] || ".txt";

  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "opengrep-scan-"));
  const tempName = ext.startsWith(".") ? `input${ext}` : ext;
  const targetPath = path.join(tempDir, tempName);

  await fsp.writeFile(targetPath, String(code || ""), "utf-8");

  const configs = await resolveConfigPaths(normalizedLang, code);
  if (configs.length === 0) {
    return {
      exitCode: 0,
      parsed: { results: [], errors: [], paths: { scanned: [] } },
      stderr: "",
      configs: [],
    };
  }
  const args = ["scan", "--json", "--quiet", "--error", "--timeout", "3"];
  for (const cfg of configs) {
    args.push("--config", cfg);
  }
  args.push(targetPath);

  const result = await new Promise((resolve, reject) => {
    const child = spawn(OPENGREP_BIN, args, { cwd: REPO_ROOT });
    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const killer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, OPENGREP_PROCESS_TIMEOUT_MS);

    child.stdout.on("data", (d) => {
      stdout += d.toString("utf-8");
    });
    child.stderr.on("data", (d) => {
      stderr += d.toString("utf-8");
    });

    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(killer);
      resolve({ exitCode: code ?? 1, stdout, stderr, timedOut });
    });
  });

  await fsp.rm(tempDir, { recursive: true, force: true });

  if (result.timedOut) {
    throw new Error(`OpenGrep timed out after ${OPENGREP_PROCESS_TIMEOUT_MS}ms`);
  }

  const parsed = extractJsonFromCli(result.stdout);
  return {
    exitCode: result.exitCode,
    parsed,
    stderr: result.stderr,
    configs,
  };
}

async function handleApiScan(req, res) {
  let body;
  try {
    body = await parseBody(req);
  } catch (err) {
    return sendJson(res, 400, { error: `Invalid JSON body: ${err.message}` });
  }

  const language = String(body.language || "python");
  const code = String(body.code || "");

  if (!code.trim()) {
    return sendJson(res, 400, { error: "Code is empty" });
  }

  try {
    const scan = await runOpengrepScan({ language, code });
    const findings = (scan.parsed.results || []).map((result) => ({
      id: result.check_id || "unknown-check",
      severity: result.extra?.severity || "INFO",
      message: result.extra?.message || "No message",
      sourceFile: result.path || "<stdin>",
      line: result.start?.line || null,
      snippet: result.extra?.lines || "",
    }));

    return sendJson(res, 200, {
      engine: "opengrep",
      configs: scan.configs,
      findingCount: findings.length,
      findings,
      errors: scan.parsed.errors || [],
      exitCode: scan.exitCode,
    });
  } catch (err) {
    return sendJson(res, 500, { error: `OpenGrep scan failed: ${err.message}` });
  }
}

async function serveStatic(req, res) {
  const urlPath = decodeURIComponent(req.url.split("?")[0]);
  const relative = urlPath === "/" ? "/index.html" : urlPath;
  const filePath = safeJoin(ROOT, relative);

  if (!filePath) {
    res.writeHead(403);
    return res.end("Forbidden");
  }

  try {
    const data = await fsp.readFile(filePath);
    res.writeHead(200, {
      "Content-Type": contentTypeFor(filePath),
      "Cache-Control": "no-store",
    });
    res.end(data);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

const server = http.createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/healthz") {
    return sendJson(res, 200, { ok: true });
  }

  if (req.method === "POST" && req.url === "/api/scan") {
    return handleApiScan(req, res);
  }

  if (req.method === "GET") {
    return serveStatic(req, res);
  }

  res.writeHead(405, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("Method Not Allowed");
});

server.listen(PORT, HOST, () => {
  console.log(`OpenGrep scanner server running at http://${HOST}:${PORT}`);
});
