
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {runAgent,createWorkspace,providerConfig} from "./agent.js";

const PORT = Number(process.env.PORT || 10000);
const PUBLIC_DIR = path.resolve("./web");
const KEY_SECRET = crypto.createHash("sha256").update(process.env.DEV_KEY_SECRET || "dev-local-secret-change-me").digest();
const rate = new Map();

function json(res,status,obj){res.writeHead(status,{"content-type":"application/json; charset=utf-8","cache-control":"no-store"});res.end(JSON.stringify(obj));}
function body(req){return new Promise((resolve,reject)=>{let s="";req.on("data",c=>{s+=c;if(s.length>1000000)req.destroy();});req.on("end",()=>{try{resolve(s?JSON.parse(s):{})}catch(e){reject(e)}});req.on("error",reject)})}
function limiter(req,kind,limit=30){const key=kind+":"+ (req.socket.remoteAddress||"unknown");const now=Date.now();const a=(rate.get(key)||[]).filter(t=>now-t<60000);if(a.length>=limit)return false;a.push(now);rate.set(key,a);return true;}
function encryptToken(){const plain=crypto.randomBytes(32);const iv=crypto.randomBytes(12);const c=crypto.createCipheriv("aes-256-gcm",KEY_SECRET,iv);const enc=Buffer.concat([c.update(plain),c.final()]);const tag=c.getAuthTag();return "DEV_"+Buffer.concat([iv,tag,enc]).toString("base64url");}
function verifyToken(req){const h=req.headers.authorization||"";if(!h.startsWith("Bearer "))return false;const token=h.slice(7);if(!token.startsWith("DEV_"))return false;try{const b=Buffer.from(token.slice(4),"base64url");if(b.length<28)return false;const d=crypto.createDecipheriv("aes-256-gcm",KEY_SECRET,b.subarray(0,12));d.setAuthTag(b.subarray(12,28));d.update(b.subarray(28));d.final();return true;}catch{return false;}}
async function serveFile(res,file,type){try{const data=await fs.readFile(file);res.writeHead(200,{"content-type":type,"cache-control":"no-cache"});res.end(data)}catch{json(res,404,{error:"not_found"})}}

const server=http.createServer(async(req,res)=>{
  try {
    const cfg = providerConfig();
    if(req.method==="GET" && req.url==="/health") return json(res,200,{ok:true,name:"Dev",version:"0.2.0",engine:cfg.configured?"external":"local",providerConfigured:cfg.configured,localModel:cfg.localModel});
    if(req.method==="GET" && req.url==="/api/status") return json(res,200,{name:"Dev",version:"0.2.0",codingFocused:true,loginRequired:false,publicChat:true,engine:cfg.configured?"external":"local",providerConfigured:cfg.configured,localModel:cfg.localModel});
    if(req.method==="POST" && req.url==="/api/key") {
      if(!limiter(req,"key",5)) return json(res,429,{error:"rate_limited"});
      return json(res,201,{name:"DEV_API_KEY",value:encryptToken(),warning:"This secret is shown only in this response. Copy it now."});
    }
    if(req.method==="POST" && req.url==="/api/chat") {
      if(!limiter(req,"chat",20)) return json(res,429,{error:"rate_limited"});
      const b=await body(req); if(typeof b.message!=="string"||!b.message.trim()) return json(res,400,{error:"message_required"});
      const repo=typeof b.repo==="string"&&b.repo.trim()?b.repo.trim():null;
      const ws=await createWorkspace(repo);
      const result=await runAgent({message:b.message.trim(),workspace:ws.root});
      return json(res,200,{...result,workspace:repo?{repository:repo}:undefined});
    }
    if(req.method==="POST" && req.url==="/api/v1/chat") {
      if(!verifyToken(req)) return json(res,401,{error:"DEV_API_KEY required"});
      if(!limiter(req,"api",60)) return json(res,429,{error:"rate_limited"});
      const b=await body(req); if(typeof b.message!=="string"||!b.message.trim()) return json(res,400,{error:"message_required"});
      const ws=await createWorkspace(typeof b.repo==="string"&&b.repo.trim()?b.repo.trim():null);
      return json(res,200,await runAgent({message:b.message.trim(),workspace:ws.root}));
    }
    if(req.method==="GET") {
      const p=new URL(req.url,"http://localhost").pathname;
      if(p==="/"||p==="/index.html") return serveFile(res,path.join(PUBLIC_DIR,"index.html"),"text/html; charset=utf-8");
      if(p==="/app.js") return serveFile(res,path.join(PUBLIC_DIR,"app.js"),"text/javascript; charset=utf-8");
      if(p==="/styles.css") return serveFile(res,path.join(PUBLIC_DIR,"styles.css"),"text/css; charset=utf-8");
    }
    json(res,404,{error:"not_found"});
  } catch(e){ json(res,500,{error:e?.message||"server_error"}); }
});
server.listen(PORT,()=>console.log("Dev listening on "+PORT));
