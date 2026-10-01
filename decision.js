import fs from "node:fs/promises";
import path from "node:path";
import dns from "node:dns/promises";
import net from "node:net";

const MAX_EVIDENCE = 40;
const MAX_MEMORY = 80;

function clip(value, limit = 5000) {
  const s = String(value ?? "");
  return s.length > limit ? s.slice(0, limit) + "\n…[truncated]" : s;
}

function tokens(text) {
  return new Set(
    String(text ?? "")
      .toLowerCase()
      .replace(/[^a-z0-9_+-]+/g, " ")
      .split(/\s+/)
      .filter(x => x.length >= 3)
  );
}

function overlap(a, b) {
  const aa = tokens(a);
  const bb = tokens(b);
  if (!aa.size || !bb.size) return 0;
  let hit = 0;
  for (const t of aa) if (bb.has(t)) hit++;
  return hit / Math.max(1, aa.size);
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map(v => Math.exp(Math.max(-12, Math.min(12, v - max))));
  const total = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map(v => v / total);
}

function normalizeProbabilities(values) {
  const entries = Object.entries(values || {}).filter(([, v]) => Number.isFinite(Number(v)));
  if (!entries.length) return {};
  const positive = entries.map(([k, v]) => [k, Math.max(0, Number(v))]);
  const total = positive.reduce((sum, [, v]) => sum + v, 0) || 1;
  return Object.fromEntries(positive.map(([k, v]) => [k, Number((v / total).toFixed(4))]));
}

function normalizeDecision(raw, fallback) {
  const out = raw && typeof raw === "object" ? raw : {};
  return {
    type: fallback.type,
    question: fallback.question,
    selected: out.selected ?? fallback.selected ?? null,
    probabilities: normalizeProbabilities(out.probabilities || fallback.probabilities),
    confidence: Number.isFinite(Number(out.confidence))
      ? Math.max(0, Math.min(1, Number(out.confidence)))
      : fallback.confidence,
    rationale: clip(out.rationale || fallback.rationale || "", 700),
    evidence: Array.isArray(out.evidence)
      ? out.evidence.slice(0, MAX_EVIDENCE).map((x, i) => ({
          id: String(x?.id || "E" + String(i + 1).padStart(2, "0")),
          status: ["observed", "web", "inferred", "proposed", "unknown"].includes(x?.status) ? x.status : "unknown",
          claim: clip(x?.claim || "", 400),
          source: clip(x?.source || "", 500)
        }))
      : (fallback.evidence || [])
  };
}

function heuristicChoice(state, question, options) {
  const scores = options.map((option, index) => {
    const text = String(option);
    const base = overlap(state, text) + overlap(question, text) * 0.65;
    return base + (1 / (index + 2)) * 0.01;
  });
  const probs = softmax(scores.map(v => v * 5));
  const best = probs.indexOf(Math.max(...probs));
  const sorted = [...probs].sort((a, b) => b - a);
  const confidence = Math.max(0, Math.min(1, 0.5 + (sorted[0] - (sorted[1] || 0)) * 1.8));
  const probabilities = Object.fromEntries(options.map((option, i) => [option, Number(probs[i].toFixed(4))]));
  return {
    selected: options[best] ?? null,
    probabilities,
    confidence: Number(confidence.toFixed(4)),
    rationale: "Local fallback compared the available options using lightweight semantic token overlap; treat the result as a heuristic, not a calibrated model prediction."
  };
}

function heuristicScore(state, question, rungs) {
  const raw = rungs.map(r => overlap(state, r.label) + overlap(question, r.label) * 0.5);
  const probs = softmax(raw.map(v => v * 5));
  const index = probs.indexOf(Math.max(...probs));
  const chosen = rungs[index];
  const sorted = [...probs].sort((a, b) => b - a);
  return {
    selected: chosen?.value ?? null,
    probabilities: Object.fromEntries(rungs.map((r, i) => [String(r.value), Number(probs[i].toFixed(4))])),
    confidence: Number(Math.max(0, Math.min(1, 0.5 + (sorted[0] - (sorted[1] || 0)) * 1.8)).toFixed(4)),
    rationale: "Local fallback selected the rubric rung with the strongest lexical fit. This is a heuristic and is not calibrated to Jev."
  };
}

function heuristicNoul(state, question) {
  const text = (String(state) + " " + String(question)).toLowerCase();
  const positive = ["yes", "safe", "supported", "correct", "valid", "approved", "done", "works", "true"].reduce((n, k) => n + (text.includes(k) ? 1 : 0), 0);
  const negative = ["no", "unsafe", "unsupported", "incorrect", "invalid", "blocked", "failed", "false", "unknown", "risky"].reduce((n, k) => n + (text.includes(k) ? 1 : 0), 0);
  const p = positive === negative ? 0.5 : positive / (positive + negative);
  const yes = Number(Math.max(0.01, Math.min(0.99, p)).toFixed(4));
  return {
    selected: yes >= 0.5,
    probabilities: {true: yes, false: Number((1 - yes).toFixed(4))},
    confidence: Number((Math.abs(yes - 0.5) * 2).toFixed(4)),
    rationale: "Local fallback used explicit positive/negative cues in the shared state. This is a heuristic, not a calibrated model prediction."
  };
}

async function providerDecision({type, state, question, options, rungs, cfg}) {
  const shape = type === "choice"
    ? {type, options}
    : type === "score"
      ? {type, rungs}
      : {type, answer_space: ["true", "false"]};

  const prompt = [
    "You are Dev's typed decision evaluator.",
    "Evaluate ONE atomic decision from shared application state.",
    "Do not write chain-of-thought. Return only JSON.",
    "Use the provided answer space exactly.",
    "Probabilities must be numbers between 0 and 1 and sum to approximately 1.",
    "Confidence is your estimate that the selected answer is appropriate for this exact state, not a probability of truth.",
    "Cite only evidence actually present in the supplied state; never invent sources.",
    "",
    JSON.stringify({decision_type: type, question, state, answer_space: shape}, null, 2),
    "",
    'Output schema: {"selected": ..., "probabilities": {"option": 0.0}, "confidence": 0.0, "rationale": "brief", "evidence": [{"id":"E01","status":"observed|web|inferred|proposed|unknown","claim":"brief","source":"optional"}]}'
  ].join("\n");

  const res = await fetch(cfg.url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": "Bearer " + process.env.DEV_MODEL_API_KEY
    },
    body: JSON.stringify({
      model: cfg.model,
      messages: [
        {role: "system", content: "Return strict JSON only. Never reveal hidden reasoning."},
        {role: "user", content: prompt}
      ],
      temperature: 0,
      max_tokens: 1100
    }),
    signal: AbortSignal.timeout(30000)
  });

  const raw = await res.text();
  if (!res.ok) throw new Error("decision provider HTTP " + res.status + ": " + clip(raw, 1000));

  let data;
  try { data = JSON.parse(raw); } catch { throw new Error("decision provider returned invalid JSON"); }
  const content = String(data?.choices?.[0]?.message?.content || "").trim();
  const fence = String.fromCharCode(96).repeat(3);
  const fenced = content.match(new RegExp(fence + "(?:json)?\\s*([\\s\\S]*?)" + fence, "i"));
  const source = fenced ? fenced[1].trim() : content;
  let parsed;
  try { parsed = JSON.parse(source); } catch { throw new Error("decision provider returned non-JSON content"); }

  const fallback = type === "choice"
    ? heuristicChoice(state, question, options)
    : type === "score"
      ? heuristicScore(state, question, rungs)
      : heuristicNoul(state, question);

  return normalizeDecision(parsed, {type, question, ...fallback});
}

export async function evaluateDecision(input) {
  const type = input.type;
  if (!["choice", "score", "noul"].includes(type)) throw new Error("decision type must be choice, score, or noul");
  if (!String(input.question || "").trim()) throw new Error("decision question is required");
  const state = clip(input.state || "", 14000);

  if (type === "choice") {
    const options = Array.isArray(input.options) ? input.options.map(String).filter(Boolean).slice(0, 8) : [];
    if (options.length < 2) throw new Error("choice decisions require 2-8 options");
    const fallback = heuristicChoice(state, input.question, options);
    if (input.cfg?.configured) return providerDecision({type, state, question: input.question, options, cfg: input.cfg});
    return normalizeDecision({probabilities: fallback.probabilities}, {type, question: input.question, ...fallback});
  }

  if (type === "score") {
    const rungs = Array.isArray(input.rungs)
      ? input.rungs.slice(0, 8).map((r, i) => ({value: Number.isFinite(Number(r?.value)) ? Number(r.value) : i, label: String(r?.label || "")})).filter(r => r.label)
      : [];
    if (rungs.length < 2) throw new Error("score decisions require 2-8 ordered rungs");
    const fallback = heuristicScore(state, input.question, rungs);
    if (input.cfg?.configured) return providerDecision({type, state, question: input.question, rungs, cfg: input.cfg});
    return normalizeDecision({probabilities: fallback.probabilities}, {type, question: input.question, ...fallback});
  }

  const fallback = heuristicNoul(state, input.question);
  if (input.cfg?.configured) return providerDecision({type, state, question: input.question, cfg: input.cfg});
  return normalizeDecision({probabilities: fallback.probabilities}, {type, question: input.question, ...fallback});
}

async function fetchJson(url, headers = {}) {
  const res = await fetch(url, {
    headers: {"accept": "application/vnd.github+json", "user-agent": "Dev-Agent/0.3.0", ...headers},
    signal: AbortSignal.timeout(12000)
  });
  const body = await res.text();
  if (!res.ok) throw new Error("research HTTP " + res.status + ": " + clip(body, 1000));
  try { return JSON.parse(body); } catch { return body; }
}

export async function researchGithub(query, kind = "repositories") {
  const q = String(query || "").trim();
  if (!q) throw new Error("search query is required");
  const safeKind = ["repositories", "code"].includes(kind) ? kind : "repositories";
  const endpoint = safeKind === "repositories"
    ? "https://api.github.com/search/repositories?q=" + encodeURIComponent(q) + "&sort=stars&order=desc&per_page=8"
    : "https://api.github.com/search/code?q=" + encodeURIComponent(q) + "&per_page=8";
  const data = await fetchJson(endpoint);
  if (safeKind === "repositories") {
    return (data.items || []).map(x => ({
      name: x.full_name,
      description: clip(x.description || "", 280),
      stars: x.stargazers_count,
      language: x.language,
      license: x.license?.spdx_id || null,
      url: x.html_url
    }));
  }
  return (data.items || []).map(x => ({
    name: x.name,
    repository: x.repository?.full_name || null,
    path: x.path,
    url: x.html_url
  }));
}

function htmlToText(html) {
  return clip(String(html)
    .replace(/<script[\\s\\S]*?<\\/script>/gi, " ")
    .replace(/<style[\\s\\S]*?<\\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/\\s+/g, " ").trim(), 12000);
}

function isPrivateIp(address) {
  if (net.isIP(address) === 4) {
    const parts = address.split(".").map(Number);
    const [a,b]=parts;
    return a===10 || a===127 || (a===169 && b===254) || (a===172 && b>=16 && b<=31) ||
      (a===192 && b===168) || (a===100 && b>=64 && b<=127) || (a===198 && b===18) ||
      (a===198 && b===19) || (a===0) || (a===192 && b===0) || (a===203 && b===0 && parts[2]===113);
  }
  if (net.isIP(address) === 6) {
    const normalized=address.toLowerCase();
    return normalized==="::1" || normalized.startsWith("fc") || normalized.startsWith("fd") ||
      normalized.startsWith("fe8") || normalized.startsWith("fe9") || normalized.startsWith("fea") ||
      normalized.startsWith("feb") || normalized.startsWith("::ffff:127.") || normalized.startsWith("::ffff:10.") ||
      normalized.startsWith("::ffff:192.168.") || normalized.startsWith("::ffff:172.16.");
  }
  return true;
}

async function assertPublicHost(value) {
  let parsed;
  try { parsed = value instanceof URL ? value : new URL(value); } catch { throw new Error("valid http(s) URL required"); }
  if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("only http(s) URLs are allowed");
  const host = parsed.hostname.toLowerCase();
  if (host.endsWith(".local") || host === "localhost" || host === "metadata.google.internal") {
    throw new Error("local network targets are not allowed");
  }
  if (net.isIP(host)) {
    if (isPrivateIp(host)) throw new Error("private or link-local network targets are not allowed");
    return parsed;
  }
  let addresses;
  try {
    addresses = await dns.lookup(host, {all:true, verbatim:true});
  } catch {
    throw new Error("target hostname could not be resolved");
  }
  if (!addresses.length || addresses.some(x => isPrivateIp(x.address))) {
    throw new Error("target hostname resolves to a private or link-local network");
  }
  return parsed;
}

export async function fetchResearchUrl(rawUrl) {
  let next = await assertPublicHost(String(rawUrl || "").trim());

  for (let hop=0; hop<4; hop++) {
    const res = await fetch(next, {
      headers: {"user-agent": "Dev-Agent/0.3.0", "accept": "text/html,text/plain,application/json,*/*"},
      signal: AbortSignal.timeout(12000),
      redirect: "manual"
    });

    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get("location");
      if (!location) throw new Error("redirect response had no location");
      next = await assertPublicHost(new URL(location, next));
      continue;
    }

    const body = await res.text();
    if (!res.ok) throw new Error("fetch HTTP " + res.status + ": " + clip(body, 1000));
    const contentType = String(res.headers.get("content-type") || "");
    return {
      url: res.url || next.toString(),
      status: res.status,
      contentType,
      text: contentType.includes("html") ? htmlToText(body) : clip(body, 12000)
    };
  }

  throw new Error("too many redirects");
}

async function memoryPath(root) {
  const dir = path.join(root, ".dev");
  await fs.mkdir(dir, {recursive: true});
  return path.join(dir, "research-memory.json");
}

export async function rememberFinding(root, finding) {
  const file = await memoryPath(root);
  let current = [];
  try { current = JSON.parse(await fs.readFile(file, "utf8")); } catch {}
  current.push({
    id: "F" + String(current.length + 1).padStart(3, "0"),
    at: new Date().toISOString(),
    status: ["observed", "web", "inferred", "proposed", "unknown"].includes(finding?.status) ? finding.status : "unknown",
    claim: clip(finding?.claim || "", 700),
    source: clip(finding?.source || "", 700),
    alternative: clip(finding?.alternative || "", 300)
  });
  current = current.slice(-MAX_MEMORY);
  await fs.writeFile(file, JSON.stringify(current, null, 2), "utf8");
  return current[current.length - 1];
}

export async function recallFindings(root, query = "") {
  const file = await memoryPath(root);
  let current = [];
  try { current = JSON.parse(await fs.readFile(file, "utf8")); } catch { return []; }
  const q = String(query || "").trim().toLowerCase();
  if (!q) return current.slice(-MAX_MEMORY);
  return current.filter(x => (x.claim + " " + x.source + " " + x.alternative).toLowerCase().includes(q)).slice(-MAX_MEMORY);
}
