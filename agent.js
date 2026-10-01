
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pipeline, env } from "@huggingface/transformers";
import { evaluateDecision, researchGithub, fetchResearchUrl, rememberFinding, recallFindings } from "./decision.js";

const execFileAsync = promisify(execFile);
const MAX_OUTPUT = 7000;
const MAX_STEPS = 8;
const WORK_ROOT = "/tmp/dev-workspaces";
const LOCAL_MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";
const LOCAL_MODEL_DTYPE = "q4f16";
const LOCAL_DEVICE = "cpu";
const LOCAL_MAX_NEW_TOKENS = 192;
const LOCAL_MAX_STEPS = 4;
const LOCAL_CACHE_DIR = "/tmp/dev-model-cache";

env.cacheDir = LOCAL_CACHE_DIR;
env.useFSCache = true;
env.useBrowserCache = false;
env.allowRemoteModels = true;
env.allowLocalModels = false;
env.logLevel = 40;

function clip(s, n = MAX_OUTPUT) {
  s = String(s ?? "");
  return s.length > n ? s.slice(0, n) + "\\n…[truncated]" : s;
}

function safePath(root, p) {
  const abs = path.resolve(root, p || ".");
  const base = path.resolve(root) + path.sep;
  if (abs !== path.resolve(root) && !abs.startsWith(base)) throw new Error("Path escapes workspace");
  return abs;
}

const COMMANDS = {
  "git status": ["git",["status","--short"]],
  "git diff": ["git",["diff","--"]],
  "git diff --stat": ["git",["diff","--stat"]],
  "npm test": ["npm",["test"]],
  "npm run build": ["npm",["run","build"]],
  "npm run typecheck": ["npm",["run","typecheck"]],
  "node --check": ["node",["--check"]],
  "python -m pytest": ["python",["-m","pytest"]],
  "pytest": ["pytest",[]],
  "python -m compileall": ["python",["-m","compileall","."]]
};

export function providerConfig() {
  return {
    url: process.env.DEV_MODEL_URL || "",
    model: process.env.DEV_MODEL_NAME || "",
    configured: !!(process.env.DEV_MODEL_URL && process.env.DEV_MODEL_NAME && process.env.DEV_MODEL_API_KEY),
    localModel: {
      id: LOCAL_MODEL_ID,
      dtype: LOCAL_MODEL_DTYPE,
      available: true
    }
  };
}

async function tool(name, args, root, cfg = providerConfig()) {
  if (name === "list_dir") {
    const dir = safePath(root, args.path || ".");
    const entries = await fs.readdir(dir, {withFileTypes:true});
    return entries.slice(0,120).map(e => (e.isDirectory() ? "d " : "f ") + e.name).join("\\n");
  }
  if (name === "read_file") {
    const file = safePath(root, args.path);
    const text = await fs.readFile(file,"utf8");
    const lines = text.split("\\n");
    const start = Math.max(0, Number(args.start || 1)-1);
    const end = Math.min(lines.length, start + Math.min(Number(args.lines || 120), 180));
    return lines.slice(start,end).map((v,i)=>(start+i+1) + ": " + v).join("\\n");
  }
  if (name === "search_text") {
    const q = String(args.query || "");
    if (!q) throw new Error("query required");
    try {
      const {stdout} = await execFileAsync("grep",["-RIn","--exclude-dir=.git","--exclude-dir=node_modules","-F",q,root],{timeout:8000,maxBuffer:30000});
      return clip(stdout || "(no matches)");
    } catch (e) {
      if (e?.code === 1) return "(no matches)";
      throw e;
    }
  }
  if (name === "write_file") {
    const file = safePath(root, args.path);
    const content = String(args.content ?? "");
    if (content.length > 30000) throw new Error("write exceeds 30k character limit");
    await fs.mkdir(path.dirname(file),{recursive:true});
    await fs.writeFile(file,content,"utf8");
    return "wrote " + path.relative(root,file) + " (" + content.length + " chars)";
  }
  if (name === "git_status") {
    const {stdout,stderr} = await execFileAsync("git",["-C",root,"status","--short"],{timeout:8000,maxBuffer:20000});
    return clip(stdout || stderr || "(clean)");
  }
  if (name === "git_diff") {
    const {stdout} = await execFileAsync("git",["-C",root,"diff","--"],{timeout:8000,maxBuffer:40000});
    return clip(stdout || "(no diff)");
  }
  if (name === "run_check") {
    const key = String(args.command || "");
    const spec = COMMANDS[key];
    if (!spec) throw new Error("command not allowed; choose one of: " + Object.keys(COMMANDS).join(", "));
    const {stdout,stderr} = await execFileAsync(spec[0],spec[1],{cwd:root,timeout:20000,maxBuffer:30000});
    return clip((stdout || "") + (stderr ? "\\nSTDERR:\\n" + stderr : "") || "(success, no output)");
  }
  if (name === "decision_choice") return JSON.stringify(await evaluateDecision({type:"choice",state:args.state,question:args.question,options:args.options,cfg}), null, 2);
  if (name === "decision_score") return JSON.stringify(await evaluateDecision({type:"score",state:args.state,question:args.question,rungs:args.rungs,cfg}), null, 2);
  if (name === "decision_noul") return JSON.stringify(await evaluateDecision({type:"noul",state:args.state,question:args.question,cfg}), null, 2);
  if (name === "search_github") return JSON.stringify(await researchGithub(args.query,args.kind || "repositories"), null, 2);
  if (name === "fetch_url") return JSON.stringify(await fetchResearchUrl(args.url), null, 2);
  if (name === "remember_finding") return JSON.stringify(await rememberFinding(root,args), null, 2);
  if (name === "recall_findings") return JSON.stringify(await recallFindings(root,args.query || ""), null, 2);
  throw new Error("unknown tool: " + name);
}

export const TOOLS = [
  {type:"function",function:{name:"list_dir",description:"List a workspace directory. Use before reading unknown files.",parameters:{type:"object",properties:{path:{type:"string"}},required:[]}}},
  {type:"function",function:{name:"read_file",description:"Read a bounded slice of a text file with line numbers.",parameters:{type:"object",properties:{path:{type:"string"},start:{type:"integer"},lines:{type:"integer"}},required:["path"]}}},
  {type:"function",function:{name:"search_text",description:"Search workspace text for an exact string. Returns concise matching lines.",parameters:{type:"object",properties:{query:{type:"string"}},required:["query"]}}},
  {type:"function",function:{name:"write_file",description:"Create or replace a text file in the workspace. Prefer targeted files and preserve unrelated code.",parameters:{type:"object",properties:{path:{type:"string"},content:{type:"string"}},required:["path","content"]}}},
  {type:"function",function:{name:"git_status",description:"Show changed files in the workspace.",parameters:{type:"object",properties:{},required:[]}}},
  {type:"function",function:{name:"git_diff",description:"Show the current patch.",parameters:{type:"object",properties:{},required:[]}}},
  {type:"function",function:{name:"run_check",description:"Run one allowlisted verification command after edits.",parameters:{type:"object",properties:{command:{type:"string",enum:Object.keys(COMMANDS)}},required:["command"]}}},
  {type:"function",function:{name:"decision_choice",description:"Choose one option from 2-8 alternatives using shared state.",parameters:{type:"object",properties:{state:{type:"string"},question:{type:"string"},options:{type:"array",items:{type:"string"},minItems:2,maxItems:8}},required:["state","question","options"]}}},
  {type:"function",function:{name:"decision_score",description:"Score shared state against 2-8 ordered rubric rungs.",parameters:{type:"object",properties:{state:{type:"string"},question:{type:"string"},rungs:{type:"array",items:{type:"object",properties:{value:{type:"number"},label:{type:"string"}},required:["value","label"]},minItems:2,maxItems:8}},required:["state","question","rungs"]}}},
  {type:"function",function:{name:"decision_noul",description:"Answer one atomic yes/no question from shared state.",parameters:{type:"object",properties:{state:{type:"string"},question:{type:"string"}},required:["state","question"]}}},
  {type:"function",function:{name:"search_github",description:"Search public GitHub repositories or code for alternative implementations.",parameters:{type:"object",properties:{query:{type:"string"},kind:{type:"string",enum:["repositories","code"]}},required:["query"]}}},
  {type:"function",function:{name:"fetch_url",description:"Fetch a public HTTP(S) documentation or research URL.",parameters:{type:"object",properties:{url:{type:"string"}},required:["url"]}}},
  {type:"function",function:{name:"remember_finding",description:"Store a concise research finding for the current workspace.",parameters:{type:"object",properties:{status:{type:"string",enum:["observed","web","inferred","proposed","unknown"]},claim:{type:"string"},source:{type:"string"},alternative:{type:"string"}},required:["claim","status"]}}},
  {type:"function",function:{name:"recall_findings",description:"Recall research findings already stored in the current workspace.",parameters:{type:"object",properties:{query:{type:"string"}},required:[]}}}
];

const SYSTEM = "You are Dev, DEMO MCP's coding and research agent. Your loop is Jev-inspired: build shared state, research alternatives when useful, make explicit typed decisions, act, verify, and reassess. Inspect before editing. When multiple approaches are plausible, research at least two alternatives when practical using public GitHub sources and documentation. Label evidence as observed, web, inferred, proposed, or unknown. Use decision_choice for mutually exclusive approaches, decision_score for ordered rubrics, and decision_noul for atomic yes/no gates. Never invent calibrated probabilities. Compare alternatives by behavior, tradeoffs, compatibility, maintenance, complexity, and evidence quality. Store concise findings and recall them during the task. Before destructive or hard-to-reverse changes, use a safety gate and prefer reversible changes. After edits, run verification and fix failures. Never expose secrets and never reveal hidden chain-of-thought; report concise conclusions and evidence instead.";

const LOCAL_SYSTEM = "You are Dev, a lightweight self-hosted coding and research assistant. Emulate a Jev-inspired workflow without claiming to reproduce Jev: shared state -> research -> typed decision -> action -> verification -> reassessment. Respond with exactly one JSON object and no markdown. Available actions: list_dir, read_file, search_text, write_file, git_status, git_diff, run_check, decision_choice, decision_score, decision_noul, search_github, fetch_url, remember_finding, recall_findings, final. Use research when it can materially improve a choice. Compare alternatives instead of blindly choosing the first idea. Use evidence labels and verify edits when possible. Never reveal hidden reasoning."

let localGeneratorPromise = null;
let localGenerationQueue = Promise.resolve();

async function getLocalGenerator() {
  if (!localGeneratorPromise) {
    localGeneratorPromise = pipeline("text-generation", LOCAL_MODEL_ID, {
      dtype: LOCAL_MODEL_DTYPE,
      device: LOCAL_DEVICE
    }).catch(err => {
      localGeneratorPromise = null;
      throw err;
    });
  }
  return localGeneratorPromise;
}

async function generateLocal(generator, messages) {
  const task = async () => generator(messages, {
    max_new_tokens: LOCAL_MAX_NEW_TOKENS,
    do_sample: false,
    return_full_text: false
  });
  const run = localGenerationQueue.then(task, task);
  localGenerationQueue = run.catch(() => {});
  return run;
}

function extractGeneratedText(output) {
  const generated = output?.[0]?.generated_text;
  if (Array.isArray(generated)) {
    const last = generated[generated.length - 1];
    return String(last?.content ?? last?.text ?? "");
  }
  return String(generated ?? "");
}

function extractJsonObject(text) {
  const trimmed = String(text || "").trim();
  const fence = String.fromCharCode(96).repeat(3);
  const fenced = trimmed.match(new RegExp(fence + "(?:json)?\\s*([\\s\\S]*?)" + fence, "i"));
  const source = fenced ? fenced[1].trim() : trimmed;
  try { return JSON.parse(source); } catch {}
  const first = source.indexOf("{");
  const last = source.lastIndexOf("}");
  if (first >= 0 && last > first) {
    try { return JSON.parse(source.slice(first, last + 1)); } catch {}
  }
  return null;
}

async function runProviderAgent(message, workspace) {
  const cfg = providerConfig();
  const messages = [{role:"system",content:SYSTEM},{role:"user",content:message}];
  const trace = [];
  for (let step=0; step<MAX_STEPS; step++) {
    const body = {model:cfg.model,messages,tools:TOOLS,tool_choice:"auto",temperature:0.15};
    const headers = {"content-type":"application/json","authorization":"Bearer " + process.env.DEV_MODEL_API_KEY};
    const res = await fetch(cfg.url,{method:"POST",headers,body:JSON.stringify(body),signal:AbortSignal.timeout(45000)});
    const raw = await res.text();
    if (!res.ok) throw new Error("model provider HTTP " + res.status + ": " + clip(raw,1000));
    const data = JSON.parse(raw);
    const msg = data?.choices?.[0]?.message;
    if (!msg) throw new Error("model provider returned no message");
    messages.push(msg);
    if (!msg.tool_calls?.length) return {answer:String(msg.content || ""),trace,engine:"external"};
    for (const call of msg.tool_calls.slice(0,4)) {
      const name = call.function?.name;
      let args = {};
      try { args = JSON.parse(call.function?.arguments || "{}"); } catch { args = {}; }
      try {
        const result = await tool(name,args,workspace,cfg);
        trace.push({tool:name,ok:true});
        messages.push({role:"tool",tool_call_id:call.id,content:clip(result)});
      } catch (e) {
        trace.push({tool:name,ok:false});
        messages.push({role:"tool",tool_call_id:call.id,content:"ERROR: " + clip(e?.message || e,2000)});
      }
    }
  }
  return {answer:"Dev reached its bounded tool-step budget. Review the verified changes and continue with another request if needed.",trace,engine:"external"};
}

async function runLocalAgent(message, workspace) {
  const generator = await getLocalGenerator();
  const messages = [{role:"system",content:LOCAL_SYSTEM},{role:"user",content:message}];
  const trace = [];

  for (let step = 0; step < LOCAL_MAX_STEPS; step++) {
    const output = await generateLocal(generator, messages);
    const raw = extractGeneratedText(output);
    const action = extractJsonObject(raw);

    if (!action || typeof action !== "object") {
      return {answer:raw || "I could not produce a structured response.",trace,engine:"local"};
    }

    if (action.action === "final") {
      return {answer:String(action.answer || ""),trace,engine:"local"};
    }

    const allowed = new Set(["list_dir","read_file","search_text","write_file","git_status","git_diff","run_check","decision_choice","decision_score","decision_noul","search_github","fetch_url","remember_finding","recall_findings"]);
    if (!allowed.has(action.action)) {
      return {answer:"The local engine returned an unsupported action. Please retry with a more specific request.",trace,engine:"local"};
    }

    try {
      const args = action.action === "git_status" || action.action === "git_diff" ? {} : action;
      const result = await tool(action.action,args,workspace);
      trace.push({tool:action.action,ok:true});
      messages.push({role:"assistant",content:raw});
      messages.push({role:"user",content:"Tool result for " + action.action + ":\\n" + clip(result) + "\\nContinue. Return exactly one JSON object."});
    } catch (e) {
      trace.push({tool:action.action,ok:false});
      messages.push({role:"assistant",content:raw});
      messages.push({role:"user",content:"Tool error for " + action.action + ": " + clip(e?.message || e,2000) + "\\nRecover or return a final JSON answer. Return exactly one JSON object."});
    }
  }

  return {answer:"Dev reached its local tool-step budget. Continue with another request to keep working.",trace,engine:"local"};
}

export async function runAgent({message, workspace}) {
  const cfg = providerConfig();
  if (cfg.configured) return runProviderAgent(message, workspace);
  return runLocalAgent(message, workspace);
}

export async function createWorkspace(repoUrl) {
  await fs.mkdir(WORK_ROOT,{recursive:true});
  const id = crypto.randomBytes(10).toString("hex");
  const root = path.join(WORK_ROOT,id);
  await fs.mkdir(root,{recursive:true});
  if (!repoUrl) return {id,root};
  if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\.git)?\/?$/.test(repoUrl)) throw new Error("Only public GitHub repository URLs are accepted for hosted workspaces.");
  await execFileAsync("git",["clone","--depth","1",repoUrl,root],{timeout:45000,maxBuffer:10000});
  return {id,root};
}
