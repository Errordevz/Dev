import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
const MAX_OUTPUT = 7000;
const MAX_STEPS = 8;
const WORK_ROOT = '/tmp/dev-workspaces';

function clip(s, n = MAX_OUTPUT) {
  s = String(s ?? '');
  return s.length > n ? s.slice(0, n) + '\n…[truncated]' : s;
}
function safePath(root, p) {
  const abs = path.resolve(root, p || '.');
  const base = path.resolve(root) + path.sep;
  if (abs !== path.resolve(root) && !abs.startsWith(base)) throw new Error('Path escapes workspace');
  return abs;
}

const COMMANDS = {
  'git status': ['git',['status','--short']],
  'git diff': ['git',['diff','--']],
  'git diff --stat': ['git',['diff','--stat']],
  'npm test': ['npm',['test']],
  'npm run build': ['npm',['run','build']],
  'npm run typecheck': ['npm',['run','typecheck']],
  'node --check': ['node',['--check']],
  'python -m pytest': ['python',['-m','pytest']],
  'pytest': ['pytest',[]],
  'python -m compileall': ['python',['-m','compileall','.']]
};

export function providerConfig() {
  return {
    url: process.env.DEV_MODEL_URL || '',
    model: process.env.DEV_MODEL_NAME || '',
    configured: !!(process.env.DEV_MODEL_URL && process.env.DEV_MODEL_NAME && process.env.DEV_MODEL_API_KEY)
  };
}

async function tool(name, args, root) {
  if (name === 'list_dir') {
    const dir = safePath(root, args.path || '.');
    const entries = await fs.readdir(dir, {withFileTypes:true});
    return entries.slice(0,120).map(e => `${e.isDirectory() ? 'd' : 'f'} ${e.name}`).join('\n');
  }
  if (name === 'read_file') {
    const file = safePath(root, args.path);
    const text = await fs.readFile(file,'utf8');
    const lines = text.split('\n');
    const start = Math.max(0, Number(args.start || 1)-1);
    const end = Math.min(lines.length, start + Math.min(Number(args.lines || 120), 180));
    return lines.slice(start,end).map((v,i)=>`${start+i+1}: ${v}`).join('\n');
  }
  if (name === 'search_text') {
    const q = String(args.query || '');
    if (!q) throw new Error('query required');
    const {stdout} = await execFileAsync('grep',['-RIn','--exclude-dir=.git','--exclude-dir=node_modules','-F',q,root],{timeout:8000,maxBuffer:30000});
    return clip(stdout);
  }
  if (name === 'write_file') {
    const file = safePath(root, args.path);
    const content = String(args.content ?? '');
    if (content.length > 30000) throw new Error('write exceeds 30k character limit');
    await fs.mkdir(path.dirname(file),{recursive:true});
    await fs.writeFile(file,content,'utf8');
    return `wrote ${path.relative(root,file)} (${content.length} chars)`;
  }
  if (name === 'git_status') {
    const {stdout,stderr} = await execFileAsync('git',['-C',root,'status','--short'],{timeout:8000,maxBuffer:20000});
    return clip(stdout || stderr || '(clean)');
  }
  if (name === 'git_diff') {
    const {stdout} = await execFileAsync('git',['-C',root,'diff','--'],{timeout:8000,maxBuffer:40000});
    return clip(stdout || '(no diff)');
  }
  if (name === 'run_check') {
    const key = String(args.command || '');
    const spec = COMMANDS[key];
    if (!spec) throw new Error(`command not allowed; choose one of: ${Object.keys(COMMANDS).join(', ')}`);
    const {stdout,stderr} = await execFileAsync(spec[0],spec[1],{cwd:root,timeout:20000,maxBuffer:30000});
    return clip((stdout || '') + (stderr ? `\nSTDERR:\n${stderr}` : '') || '(success, no output)');
  }
  throw new Error(`unknown tool: ${name}`);
}

export const TOOLS = [
  {type:'function',function:{name:'list_dir',description:'List a workspace directory. Use before reading unknown files.',parameters:{type:'object',properties:{path:{type:'string'}},required:[]}}},
  {type:'function',function:{name:'read_file',description:'Read a bounded slice of a text file with line numbers.',parameters:{type:'object',properties:{path:{type:'string'},start:{type:'integer'},lines:{type:'integer'}},required:['path']}}},
  {type:'function',function:{name:'search_text',description:'Search workspace text for an exact string. Returns concise matching lines.',parameters:{type:'object',properties:{query:{type:'string'}},required:['query']}}},
  {type:'function',function:{name:'write_file',description:'Create or replace a text file in the workspace. Prefer targeted files and preserve unrelated content.',parameters:{type:'object',properties:{path:{type:'string'},content:{type:'string'}},required:['path','content']}}},
  {type:'function',function:{name:'git_status',description:'Show changed files in the workspace.',parameters:{type:'object',properties:{},required:[]}}},
  {type:'function',function:{name:'git_diff',description:'Show the current patch.',parameters:{type:'object',properties:{},required:[]}}},
  {type:'function',function:{name:'run_check',description:'Run one allowlisted verification command after edits.',parameters:{type:'object',properties:{command:{type:'string',enum:Object.keys(COMMANDS)}},required:['command']}}}
];

const SYSTEM = `You are Dev, DEMO MCP's coding-focused agent. You are concise, practical, and verification-driven. You can also answer general reasoning questions, but when a task involves code, repository files, debugging, architecture, or tests, behave as a software engineer.

Rules:
1. Inspect before editing. Do not invent file contents.
2. Prefer targeted edits and preserve unrelated code.
3. Use the smallest useful amount of context.
4. After edits, run an appropriate verification command when possible.
5. If a check fails, inspect the failure and fix it rather than declaring success.
6. Never claim a file was changed unless the write tool succeeded.
7. Never claim tests passed unless a check tool returned success.
8. Explain what changed and what remains uncertain.
9. Do not expose provider credentials or secrets.
10. Do not perform destructive operations.`;

export async function runAgent({message, workspace}) {
  const cfg = providerConfig();
  if (!cfg.configured) throw new Error('Dev model provider is not configured on this deployment. Set DEV_MODEL_URL, DEV_MODEL_NAME and DEV_MODEL_API_KEY on the server.');
  const messages = [{role:'system',content:SYSTEM},{role:'user',content:message}];
  const trace = [];
  for (let step=0; step<MAX_STEPS; step++) {
    const body = {model:cfg.model,messages,tools:TOOLS,tool_choice:'auto',temperature:0.15};
    const headers = {'content-type':'application/json','authorization':`Bearer ${process.env.DEV_MODEL_API_KEY}`};
    const res = await fetch(cfg.url,{method:'POST',headers,body:JSON.stringify(body),signal:AbortSignal.timeout(45000)});
    const raw = await res.text();
    if (!res.ok) throw new Error(`model provider HTTP ${res.status}: ${clip(raw,1000)}`);
    const data = JSON.parse(raw);
    const msg = data?.choices?.[0]?.message;
    if (!msg) throw new Error('model provider returned no message');
    messages.push(msg);
    if (!msg.tool_calls?.length) return {answer:String(msg.content || ''),trace};
    for (const call of msg.tool_calls.slice(0,4)) {
      const name = call.function?.name;
      let args = {};
      try { args = JSON.parse(call.function?.arguments || '{}'); } catch { args = {}; }
      try {
        const result = await tool(name,args,workspace);
        trace.push({tool:name,ok:true});
        messages.push({role:'tool',tool_call_id:call.id,content:clip(result)});
      } catch (e) {
        trace.push({tool:name,ok:false});
        messages.push({role:'tool',tool_call_id:call.id,content:`ERROR: ${clip(e?.message || e,2000)}`});
      }
    }
  }
  return {answer:'Dev reached its bounded tool-step budget. Review the verified changes and continue with another request if needed.',trace};
}

export async function createWorkspace(repoUrl) {
  await fs.mkdir(WORK_ROOT,{recursive:true});
  const id = crypto.randomBytes(10).toString('hex');
  const root = path.join(WORK_ROOT,id);
  await fs.mkdir(root,{recursive:true});
  if (!repoUrl) return {id,root};
  if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\.git)?\/?$/.test(repoUrl)) throw new Error('Only public GitHub repository URLs are accepted for hosted workspaces.');
  await execFileAsync('git',['clone','--depth','1',repoUrl,root],{timeout:45000,maxBuffer:10000});
  return {id,root};
}
