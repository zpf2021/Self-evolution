import http from 'node:http';
import {createHash} from 'node:crypto';
import {createReadStream, readFileSync} from 'node:fs';
import {getLlama} from 'node-llama-cpp';
const spec=JSON.parse(readFileSync(new URL('./model.json',import.meta.url)));
const modelPath=process.env.UNIFIED_MODEL_PATH;
if(!modelPath) throw new Error('UNIFIED_MODEL_PATH is required; models are never downloaded at runtime');
const hash=createHash('sha256');
for await(const chunk of createReadStream(modelPath)) hash.update(chunk);
if(hash.digest('hex')!==spec.sha256) throw new Error('Model SHA-256 mismatch');
const llama=await getLlama({gpu:false,build:"never",skipDownload:true,maxThreads:Number(process.env.UNIFIED_EMBEDDING_THREADS||4)});
const model=await llama.loadModel({modelPath});
const context=await model.createEmbeddingContext({contextSize:2048});
const modelId='embeddinggemma-300M-Q8_0';
let ready=false;
async function embed(text) {
  if(typeof text!=='string'||!text.trim()) throw new Error('Each input must be nonempty text');
  // Token-bounded document chunks preserve all content. Weighted mean pooling is fingerprinted.
  const prefix=text.startsWith('task: search result | query: ') ? 'task: search result | query: ' : text.startsWith('title: none | text: ') ? 'title: none | text: ' : '';
  const tokens=model.tokenize(text.slice(prefix.length));
  if(tokens.length>100000) throw new Error('Input too large; split into separate memory records');
  const vector=new Array(spec.native_dimension).fill(0);
  for(let i=0;i<tokens.length;i+=1900){
    const part=tokens.slice(i,i+1900);
    const values=Array.from((await context.getEmbeddingFor(prefix+model.detokenize(part))).vector);
    if(values.length!==spec.native_dimension||!values.every(Number.isFinite)) throw new Error('Invalid native embedding dimension or values');
    for(let j=0;j<vector.length;j++) vector[j]+=values[j]*part.length;
  }
  const norm=Math.sqrt(vector.reduce((s,x)=>s+x*x,0));
  if(!norm) throw new Error('Zero embedding');
  return vector.map(x=>x/norm);
}
await embed('task: search result | query: startup health probe');
ready=true;
const fingerprint=createHash('sha256').update(JSON.stringify(spec)).digest('hex');
let tail=Promise.resolve(); let pending=0;
function serialize(work) {
  if(pending>=64) return Promise.reject(new Error('Embedding queue full'));
  pending++;
  const result=tail.then(work);
  tail=result.catch(()=>{}).finally(()=>pending--);
  return result;
}
const server=http.createServer(async(req,res)=>{
  function reply(code,body){res.writeHead(code,{'content-type':'application/json'});res.end(JSON.stringify(body));}
  if(req.method==='GET'&&req.url==='/health') return reply(ready?200:503,{status:ready?'ok':'unhealthy',dimension:spec.native_dimension,model:modelId,sha256:spec.sha256,fingerprint,queued:pending});
  if(req.method==='GET'&&req.url==='/v1/models') return reply(200,{object:'list',data:[{id:modelId,object:'model',owned_by:'local'}]});
  if(req.method!=='POST'||req.url!=='/v1/embeddings') return reply(404,{error:{message:'Not found'}});
  try {
    let raw='';
    for await(const chunk of req){raw+=chunk.toString();if(Buffer.byteLength(raw)>1048576) throw new Error('Request too large');}
    const data=JSON.parse(raw);
    if(data.model!==modelId) throw new Error('Unknown model');
    if(data.dimensions!==undefined&&data.dimensions!==spec.native_dimension) throw new Error('Only native dimensions supported');
    if(data.encoding_format && data.encoding_format!=='float') throw new Error('Only float encoding supported');
    const inputs=typeof data.input==='string'?[data.input]:data.input;
    if(!Array.isArray(inputs)||inputs.length<1||inputs.length>64) throw new Error('Expected 1..64 texts');
    const vectors=await serialize(async()=>{
      const values=[];for(const text of inputs) values.push(await embed(text));return values;
    });
    reply(200,{object:'list',model:modelId,data:vectors.map((embedding,index)=>({object:'embedding',index,embedding})),usage:{prompt_tokens:inputs.reduce((n,t)=>n+model.tokenize(t).length,0),total_tokens:inputs.reduce((n,t)=>n+model.tokenize(t).length,0)}});
  } catch(e){reply(400,{error:{message:String(e.message),type:'invalid_request_error'}});}
});
server.listen(Number(process.env.UNIFIED_EMBEDDING_PORT||8125),'127.0.0.1');
async function stop(){ready=false;server.close();await tail;await context.dispose();await model.dispose();await llama.dispose();process.exit(0);}
process.on('SIGTERM',stop);process.on('SIGINT',stop);
