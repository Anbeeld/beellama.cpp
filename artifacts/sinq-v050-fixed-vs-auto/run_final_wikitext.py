from __future__ import annotations
import argparse,csv,hashlib,json,math,os,re,subprocess,sys,time,threading,ctypes
from ctypes import wintypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
from tools.sinq.optimizer.resources import file_digest
parser=argparse.ArgumentParser();parser.add_argument('recipe',choices=['T340','T425']);parser.add_argument('gpu',type=int,choices=[0,1]);parser.add_argument('--models-manifest',type=Path);parser.add_argument('--output-directory',type=Path);parser.add_argument('--execution-lock',type=Path);args=parser.parse_args()
if any((args.models_manifest,args.output_directory,args.execution_lock)) and not all((args.models_manifest,args.output_directory,args.execution_lock)):raise ValueError('external frozen models require manifest, output directory and execution lock')
RECIPE=args.recipe;GPU_INDEX=args.gpu;LOCK=json.loads((ROOT/'locks/hardware-build-lock.json').read_text())
UUIDS=[x.split(',')[0] for x in LOCK['gpus']];UUID=UUIDS[GPU_INDEX]
SOURCE=Path(r'C:\models\Qwen3.5-9B-bf16\Qwen3.5-9B-text-BF16.gguf');REFERENCE=Path(r'D:\Qwen3.5-9B-wikitext2-64k-bf16.logits')
INPUT_LOCK=ROOT/'locks/final-wikitext-input-lock.json'
CORPUS=Path(r'C:\Users\anbee\projects\beellama.cpp\artifacts\wikitext-2-raw-v1\wikitext-2-raw\wiki.test.raw')
TOKENS=ROOT/'data/final_test.u32';EXEC=REPO/'build-sinq-clean/bin/llama-perplexity.exe'
AUDIT=json.loads((ROOT/'report/fixed-model-audit.json').read_text());RANK=json.loads((ROOT/'report/fixed-selection-ranking.json').read_text());AUTO_AUDIT=json.loads((ROOT/'report/auto-model-audit.json').read_text());SCORE_AUDIT=json.loads((ROOT/'report/surrogate-score-matrix-audit.json').read_text());COST_INDEX=json.loads((ROOT/'report/selector-costs-index.json').read_text());FIXED_RUNTIME=json.loads((ROOT/'report/fixed-model-runtime-validation.json').read_text());AUTO_RUNTIME=json.loads((ROOT/'report/auto-model-runtime-validation.json').read_text())
if AUDIT.get('status')!='PASS' or AUTO_AUDIT.get('status')!='PASS' or len(AUTO_AUDIT.get('models',[]))!=10:raise ValueError('full-model audit gate incomplete')
if FIXED_RUNTIME.get('status')!='PASS' or FIXED_RUNTIME.get('validated_arms')!=24 or AUTO_RUNTIME.get('status')!='PASS' or AUTO_RUNTIME.get('validated_models')!=10:raise ValueError('CPU/CUDA runtime validation gate incomplete')
if SCORE_AUDIT.get('status')!='PASS' or SCORE_AUDIT.get('audited_score_files')!=8 or COST_INDEX.get('status')!='PASS' or len(COST_INDEX.get('records',[]))!=10:raise ValueError('scorer coverage or selector cost gate incomplete')
if any(not Path(x['path']).is_file() or file_digest(x['path'])!=x['sha256'] for x in COST_INDEX['records']):raise ValueError('selector cost records changed after lock')
fixed_runtime={r['arm']:r for r in FIXED_RUNTIME.get('results',[])};fixed_audit={r['arm']:r['model_sha256'] for r in AUDIT['records']}
if set(fixed_runtime)!=set(fixed_audit) or any(r.get('status')!='PASS' or r['model_sha256_before']!=r['model_sha256_after'] or r['model_sha256_after']!=fixed_audit.get(arm) for arm,r in fixed_runtime.items()):raise ValueError('fixed runtime validation closure mismatch')
auto_runtime={(r['recipe'],r['label']):r for r in AUTO_RUNTIME.get('results',[])};auto_audit={(r['recipe'],r['label']):r['model_sha256'] for r in AUTO_AUDIT['models']}
if set(auto_runtime)!=set(auto_audit) or any(r.get('status')!='PASS' or r['model_sha256_before']!=r['model_sha256_after'] or r['model_sha256_after']!=auto_audit.get(key) for key,r in auto_runtime.items()):raise ValueError('automatic runtime validation closure mismatch')
for r in fixed_runtime.values():
 if any(r[s].get('exit_code')!=0 or r[s].get('status')!='PASS' for s in ('load_only','cpu_forward','cuda_forward')):raise ValueError('fixed model CPU/CUDA runtime stage failed')
for r in auto_runtime.values():
 if len(r.get('stages',[]))!=3 or any(s.get('exit_code')!=0 for s in r['stages']):raise ValueError('automatic model CPU/CUDA runtime stage failed')
OUT=(args.output_directory.resolve()/RECIPE if args.output_directory else ROOT/'evaluation/wikitext2'/RECIPE);OUT.mkdir(parents=True,exist_ok=True);PROGRESS=OUT/'progress.json'
external_lock=json.loads(args.execution_lock.read_text()) if args.execution_lock else None
if external_lock and (external_lock.get('status')!='FROZEN_BEFORE_HELD_OUT_EVALUATION' or external_lock.get('models_manifest_sha256')!=file_digest(args.models_manifest) or external_lock.get('executable_sha256')!=file_digest(EXEC) or external_lock.get('source_sha256')!='da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af' or external_lock.get('wikitext_input_lock_sha256')!=file_digest(INPUT_LOCK)):raise ValueError('external final-evaluation lock mismatch')
EXPECTED={'source':'da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af','reference':'b0963bb182cfb65b9f317aab36203718d5730c21f82eb57ca401bdb89969d89b','corpus':'173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08','targets':297193,'tokens':297194,'vocabulary':248320,'context':65536}
input_lock=json.loads(INPUT_LOCK.read_text());parity=json.loads((ROOT/'report/evaluator-parity.json').read_text())
if parity.get('status')!='PASS':raise ValueError('current CPU/CUDA evaluator parity gate has not passed')
if input_lock.get('status')!='PASS' or input_lock.get('source_sha256')!=EXPECTED['source'] or input_lock.get('reference_sha256')!=EXPECTED['reference'] or input_lock.get('corpus_sha256')!=EXPECTED['corpus']:raise ValueError('final-test input lock is absent or mismatched')
if file_digest(SOURCE)!=EXPECTED['source'] or file_digest(CORPUS)!=EXPECTED['corpus']:raise ValueError('locked source or WikiText corpus identity mismatch')
if REFERENCE.stat().st_size!=147601497864 or file_digest(REFERENCE)!=EXPECTED['reference']:raise ValueError('locked full BF16 WikiText reference identity/size mismatch')
expected_executable=external_lock['executable_sha256'] if external_lock else input_lock.get('evaluator_sha256')
if file_digest(EXEC)!=expected_executable:raise ValueError('full-test evaluator differs from its lock')
ids=np.fromfile(TOKENS,dtype='<u4')
if len(ids)!=EXPECTED['tokens'] or hashlib.sha256(TOKENS.read_bytes()).hexdigest()!=json.loads((ROOT/'locks/corpus-source-provenance.json').read_text())['token_streams']['final_test']['sha256']:raise ValueError('frozen full-test token stream mismatch')
if hashlib.sha256(ids.astype('<u4',copy=False).tobytes()).hexdigest()!=json.loads((ROOT/'locks/corpus-source-provenance.json').read_text())['token_streams']['final_test']['sha256']:raise ValueError('final token stream identity mismatch')
models=[]
for row in AUDIT['records']:
 if row['recipe']==RECIPE:
  arm=row['arm'];models.append({'label':'FIXED-'+arm,'path':ROOT/'fixed'/RECIPE/arm/'model.gguf','expected_sha256':row['model_sha256'],'kind':'fixed','arm':arm})
for label in ('AUTO-LOCAL-L1','AUTO-LOCAL-MSE','AUTO-BLOCK-MSE','AUTO-FISHER','AUTO-GROUP-KLD-BUDGETED'):
 d=ROOT/'selected'/RECIPE/label;sel=json.loads((d/'selection.json').read_text());
 if sel.get('status')!='PASS' or sel.get('best_gguf_sha256')!=file_digest(d/'best.gguf'):raise ValueError('automatic selection output/manifest not complete: '+label)
 models.append({'label':label,'path':d/'best.gguf','expected_sha256':sel['best_gguf_sha256'],'kind':'auto','arm':label,'selection_sha256':file_digest(d/'selection.json')})
if len(models)!=17 or len({m['label'] for m in models})!=17:raise ValueError('expected exactly 12 fixed and 5 automatic outputs')
if args.models_manifest:
 frozen=json.loads(args.models_manifest.read_text())
 if frozen.get('status')!='FROZEN_BEFORE_HELD_OUT_EVALUATION':raise ValueError('external models not frozen')
 models=[dict(m,path=Path(m['path'])) for m in frozen['models'] if m['recipe']==RECIPE]
 if not models or len({m['label'] for m in models})!=len(models):raise ValueError('invalid external final models')
 for m in models:
  if m['kind']=='fixed':
   if fixed_audit.get(m['arm'])!=m['expected_sha256']:raise ValueError('fixed control audit mismatch')
   continue
  if m['kind']=='stock':
   control=json.loads((ROOT/'locks/historical-controls.json').read_text())['historical_controls'][RECIPE+'-stock']
   if control['sha256']!=m['expected_sha256']:raise ValueError('stock control audit mismatch')
   continue
  s=Path(m['path']).parent/'selection.json';selection=json.loads(s.read_text())
  if file_digest(s)!=m['selection_sha256'] or selection.get('status')!='PASS' or selection.get('best_gguf_sha256')!=m['expected_sha256']:raise ValueError('external selection/model closure mismatch')
EXPECTED_MODELS=len(models)
models.sort(key=lambda m:m['label'])

def current_config_identity():
 policy={'source':EXPECTED['source'],'reference':EXPECTED['reference'],'corpus':EXPECTED['corpus'],'executable':file_digest(EXEC),'threads_per_process':16,'context':65536,'batch':512,'microbatch':256}
 if external_lock:policy.update(external_lock_sha256=file_digest(args.execution_lock),models_manifest_sha256=file_digest(args.models_manifest))
 return hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()

# Existing completed immutable rows may be reused only after full closure checks.
results={};failed_attempts=[]
if PROGRESS.exists():
 old=json.loads(PROGRESS.read_text())
 if old.get('recipe')!=RECIPE or old.get('uuid')!=UUID or old.get('config_identity')!=current_config_identity():raise ValueError('existing evaluation progress uses a different immutable run identity')
 for row in old.get('results',[]):
  rp=Path(row['metrics_path']);tp=Path(row['token_tsv_path'])
  if row.get('status')!='PASS' or not rp.is_file() or not tp.is_file() or file_digest(rp)!=row.get('record_sha256') or file_digest(tp)!=row.get('token_tsv_sha256'):raise ValueError('existing evaluation result closure mismatch')
  results[row['label']]=row
 failed_attempts=list(old.get('failed_attempts',[]))

class _PMC(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t),('PrivateUsage',ctypes.c_size_t)]
_K=ctypes.WinDLL('kernel32',use_last_error=True);_P=ctypes.WinDLL('psapi',use_last_error=True)
_K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];_K.OpenProcess.restype=wintypes.HANDLE
_K.CloseHandle.argtypes=[wintypes.HANDLE];_P.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(_PMC),wintypes.DWORD];_P.GetProcessMemoryInfo.restype=wintypes.BOOL

def peak_working_set(pid):
 h=_K.OpenProcess(0x0400|0x0010,False,pid)
 if not h:return 0
 try:
  c=_PMC();c.cb=ctypes.sizeof(c)
  return int(c.PeakWorkingSetSize) if _P.GetProcessMemoryInfo(h,ctypes.byref(c),c.cb) else 0
 finally:_K.CloseHandle(h)

def gpu_usage():
 try:
  x=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
  return {a.strip():int(b.strip()) for a,b in (line.split(',',1) for line in x.stdout.splitlines())}
 except Exception:return {}

def quantile(a,q):return float(np.quantile(a,q,method='linear'))

def summarize(tsv):
 fields={'kld':[],'base_nll':[],'candidate_nll':[],'top1_agree':[]};depth={};worst=[];count=0
 with tsv.open(encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f,delimiter='\t');expected=['input_index','context_position','target_token','kld','base_nll','candidate_nll','top1_agree']
  if reader.fieldnames not in (expected,expected+['candidate_top1']):raise ValueError('token metric TSV schema mismatch')
  for i,row in enumerate(reader):
   if i>=EXPECTED['targets'] or int(row['input_index'])!=i or int(row['context_position'])!=i%EXPECTED['context'] or int(row['target_token'])!=int(ids[i+1]):raise ValueError('full-test target sequence/alignment mismatch at '+str(i))
   vals={k:float(row[k]) for k in fields}
   if not all(math.isfinite(v) for v in vals.values()):raise ValueError('nonfinite per-token metric at '+str(i))
   for k,v in vals.items():fields[k].append(v)
   layer=int(row['context_position'])//4096;slot=depth.setdefault(layer,{'kld':[],'base_nll':[],'candidate_nll':[]});slot['kld'].append(vals['kld']);slot['base_nll'].append(vals['base_nll']);slot['candidate_nll'].append(vals['candidate_nll'])
   delta=vals['candidate_nll']-vals['base_nll'];worst.append((vals['kld'],delta,i,int(row['target_token']),int(row['context_position'])))
   count+=1
 if count!=EXPECTED['targets']:raise ValueError(f'expected {EXPECTED["targets"]} target rows, got {count}')
 arrays={k:np.asarray(v,dtype=np.float64) for k,v in fields.items()};nll_delta=arrays['candidate_nll']-arrays['base_nll'];kld=arrays['kld']
 out={'targets':count,'mean_test_KL':math.fsum(arrays['kld'].tolist())/count,'WikiPPL':math.exp(math.fsum(arrays['candidate_nll'].tolist())/count),'mean_candidate_NLL':math.fsum(arrays['candidate_nll'].tolist())/count,'mean_reference_NLL':math.fsum(arrays['base_nll'].tolist())/count,'top1_agreement':math.fsum(arrays['top1_agree'].tolist())/count,'tails':{key:{'mean':float(np.mean(a)),'median':quantile(a,.5),'p90':quantile(a,.9),'p95':quantile(a,.95),'p99':quantile(a,.99),'p99_9':quantile(a,.999),'maximum':float(np.max(a))} for key,a in [('KL',kld),('candidate_NLL',arrays['candidate_nll']),('paired_NLL_delta',nll_delta)]},'fraction_better_NLL':float(np.mean(nll_delta<0)),'fraction_worse_NLL':float(np.mean(nll_delta>0)),'depth_bins':{str(k):{'count':len(v['kld']),'mean_KL':math.fsum(v['kld'])/len(v['kld']),'mean_candidate_NLL':math.fsum(v['candidate_nll'])/len(v['candidate_nll']),'mean_reference_NLL':math.fsum(v['base_nll'])/len(v['base_nll'])} for k,v in sorted(depth.items())},'worst_KL_examples':[{'input_index':i,'target_token':tok,'context_position':pos,'KL':kl,'paired_NLL_delta':d} for kl,d,i,tok,pos in sorted(worst,reverse=True)[:20]],'worst_NLL_regressions':[{'input_index':i,'target_token':tok,'context_position':pos,'KL':kl,'paired_NLL_delta':d} for kl,d,i,tok,pos in sorted(worst,key=lambda r:(r[1],r[2]),reverse=True)[:20]]}
 return out

config_identity=current_config_identity()
def save_progress(status):
 obj={'schema_version':1,'status':status,'recipe':RECIPE,'uuid':UUID,'config_identity':config_identity,'expected_models':EXPECTED_MODELS,'completed_models':len(results),'failed_models':sorted({r['label'] for r in failed_attempts if r.get('status')=='FAIL'}-set(results)),'failed_attempts':failed_attempts,'results':list(results.values())}
 tmp=PROGRESS.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');os.replace(tmp,PROGRESS)

for m in models:
 if m['label'] in results:continue
 model=m['path'];expected=m['expected_sha256'];before=file_digest(model)
 if before!=expected:raise ValueError('model SHA differs from locked expected hash: '+m['label'])
 arm_root=OUT/m['label'];arm_root.mkdir(exist_ok=True);occupied=lambda d:any((d/n).exists() for n in ('tokens.tsv','combined.log','metrics.json','failure.json'))
 if not occupied(arm_root):arm_dir=arm_root
 else:
  number=1
  while (arm_root/f'attempt-{number:02d}').exists():number+=1
  arm_dir=arm_root/f'attempt-{number:02d}';arm_dir.mkdir()
 tsv=arm_dir/'tokens.tsv';stdout=arm_dir/'combined.log';record_path=arm_dir/'metrics.json'
 command=[str(EXEC),'-m',str(model.resolve()),'-f',str(CORPUS.resolve()),'--ppl-full','-c','65536','-b','512','-ub','256','-ctk','bf16','-ctv','bf16','-fa','on','-t','16','--no-warmup','--split-mode','none','--fit','off','--verbose','--kld-token-output',str(tsv.resolve()),'-ngl','999','--device','CUDA0','--kl-divergence','--kl-divergence-base',str(REFERENCE.resolve())]
 env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_ARG_','GGML_','CUDA_','CUBLAS_','NVIDIA_TF32_'))};env.update(CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto',GGML_CUDA_MMQ_PREC='auto')
 base_gpu=gpu_usage().get(UUID);peak_gpu=base_gpu or 0;peak_rss=0;start=time.monotonic();proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); chunks=[]
 def drain():
  while True:
   part=proc.stdout.read(1<<20)
   if not part:break
   chunks.append(part)
 reader=threading.Thread(target=drain,daemon=True);reader.start()
 while proc.poll() is None:
  peak_rss=max(peak_rss,peak_working_set(proc.pid))
  usage=gpu_usage().get(UUID)
  if usage is not None:peak_gpu=max(peak_gpu,usage)
  time.sleep(1)
 reader.join();elapsed=time.monotonic()-start;log=b''.join(chunks);stdout.write_bytes(log)
 if proc.returncode!=0:
  result={'status':'FAIL','label':m['label'],'stage':'native_evaluation','attempt_directory':str(arm_dir.resolve()),'exit_code':proc.returncode,'elapsed_seconds':elapsed,'model_sha256_before':before,'command':command,'log_sha256':file_digest(stdout),'log_path':str(stdout.resolve()),'token_tsv_sha256':file_digest(tsv) if tsv.is_file() else None}
  (arm_dir/'failure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');failed_attempts.append(result);save_progress('INCOMPLETE');print(json.dumps({'status':'FAIL','label':m['label'],'exit_code':proc.returncode,'attempt':str(arm_dir)}),flush=True);continue
 if external_lock:
  text=log.decode(errors='replace');offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',text)
  if not offload or offload[-1]!=('33','33') or external_lock['gpus'][str(GPU_INDEX)]['pci_bus_id'] not in text:raise RuntimeError('held-out CUDA execution evidence mismatch')
 try:metrics=summarize(tsv)
 except Exception as error:
  result={'status':'FAIL','label':m['label'],'stage':'token_alignment_or_metric_validation','attempt_directory':str(arm_dir.resolve()),'error_type':type(error).__name__,'error':str(error),'elapsed_seconds':elapsed,'model_sha256_before':before,'command':command,'log_sha256':file_digest(stdout),'log_path':str(stdout.resolve()),'token_tsv_sha256':file_digest(tsv) if tsv.is_file() else None}
  (arm_dir/'failure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');failed_attempts.append(result);save_progress('INCOMPLETE');print(json.dumps({'status':'FAIL','label':m['label'],'stage':result['stage']}),flush=True);continue
 after=file_digest(model)
 if after!=before:raise ValueError('model changed during final evaluation: '+m['label'])
 if file_digest(CORPUS)!=EXPECTED['corpus']:raise ValueError('frozen final corpus changed during evaluation')
 record={'schema_version':1,'status':'PASS','label':m['label'],'recipe':RECIPE,'kind':m['kind'],'arm':m['arm'],'model_path':str(model.resolve()),'model_sha256_before':before,'model_sha256_after':after,'selection_manifest_sha256':m.get('selection_sha256'),'source_sha256':EXPECTED['source'],'corpus_path':str(CORPUS.resolve()),'corpus_sha256':EXPECTED['corpus'],'reference_path':str(REFERENCE.resolve()),'reference_sha256':EXPECTED['reference'],'reference_targets':EXPECTED['targets'],'context':EXPECTED['context'],'batch':512,'microbatch':256,'cache_type_k':'bf16','cache_type_v':'bf16','flash_attention':'on','gpu_uuid':UUID,'visible_cuda_devices':1,'threads':16,'command':command,'executable_sha256':file_digest(EXEC),'elapsed_seconds':elapsed,'peak_process_working_set_bytes':peak_rss,'gpu_global_memory_baseline_mib':base_gpu,'gpu_global_memory_peak_mib':peak_gpu,'gpu_global_memory_peak_includes_other_processes':True,'token_tsv_sha256':file_digest(tsv),'token_tsv_path':str(tsv.resolve()),'combined_log_sha256':file_digest(stdout),'combined_log_path':str(stdout.resolve()),'metrics':metrics}
 record_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');row={'label':m['label'],'status':'PASS','metrics_path':str(record_path.resolve()),'record_sha256':file_digest(record_path),'token_tsv_path':str(tsv.resolve()),'token_tsv_sha256':file_digest(tsv),'model_sha256':after,'mean_test_KL':metrics['mean_test_KL'],'WikiPPL':metrics['WikiPPL'],'elapsed_seconds':elapsed,'peak_process_working_set_bytes':peak_rss,'gpu_global_memory_peak_mib':peak_gpu};results[m['label']]=row;save_progress('IN_PROGRESS' if len(results)<EXPECTED_MODELS else 'PASS');print(json.dumps({'status':'PASS','completed':len(results),'expected':EXPECTED_MODELS,'label':m['label'],'mean_test_KL':metrics['mean_test_KL'],'WikiPPL':metrics['WikiPPL'],'seconds':elapsed}),flush=True)
corpus_after=file_digest(CORPUS)
if corpus_after!=EXPECTED['corpus']:raise ValueError('frozen final corpus changed during evaluation')
if len(results)==EXPECTED_MODELS and all(r['status']=='PASS' for r in results.values()):save_progress('PASS')
else:save_progress('INCOMPLETE')
