from __future__ import annotations
import argparse,csv,ctypes,hashlib,json,math,os,re,subprocess,sys,time
from ctypes import wintypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];sys.path.insert(0,str(REPO))
from tools.sinq.optimizer.resources import file_digest
from tools.sinq.optimizer.kl import NativeKLEvaluator
parser=argparse.ArgumentParser();parser.add_argument('--output-directory',type=Path);parser.add_argument('--model',type=Path);parser.add_argument('--expected-model-sha256');parser.add_argument('--reference',type=Path);parser.add_argument('--cpu-executable',type=Path);parser.add_argument('--cache-type',choices=['bf16','f16','f32'],default='bf16');parser.add_argument('--flash-attention',choices=['on','off'],default='on');parser.add_argument('--compute',choices=['auto','f32','f16','bf16'],default='auto');parser.add_argument('--cuda-threads',type=int,default=16);options=parser.parse_args()
LOCK=json.loads((ROOT/'locks/evaluator-parity-preregistration.json').read_text());TOL=json.loads((ROOT/'locks/evaluator-parity-tolerance-addendum.json').read_text());CAT=json.loads((ROOT/'locks/evaluator-parity-categorical-addendum.json').read_text());DIV=json.loads((ROOT/'locks/evaluator-parity-divergence-policy-addendum.json').read_text());OUT=ROOT/'report/evaluator-parity.json';RUNS=ROOT/'evaluation/evaluator-parity'
if TOL.get('status')!='PASS' or not TOL.get('registered_before_the_parity_run_and_before_final_TEST') or CAT.get('status')!='PASS' or not CAT.get('registered_before_the_parity_run_and_before_final_TEST') or DIV.get('status')!='PASS' or not DIV.get('registered_before_the_parity_run_and_before_final_TEST'):raise ValueError('parity addendum invalid')
if options.output_directory:
 OUT=options.output_directory.resolve()/'parity.json';RUNS=options.output_directory.resolve()
if OUT.exists():raise FileExistsError('parity report already exists')
RUNS.mkdir(parents=True,exist_ok=True);attempt=max((int(p.name.split('-')[-1]) for p in RUNS.glob('attempt-*') if p.is_dir() and p.name.split('-')[-1].isdigit()),default=0)+1;WORK=RUNS/f'attempt-{attempt:02d}';WORK.mkdir()
model=ROOT/LOCK['model'];corpus=ROOT/LOCK['corpus'];reference=ROOT/LOCK['reference'];cpu=REPO/'build-sinq-optimizer-cpu-clean/bin/llama-perplexity.exe';cuda=REPO/'build-sinq-clean/bin/llama-perplexity.exe';hardware=json.loads((ROOT/'locks/hardware-build-lock.json').read_text());gpu=hardware['gpus'][0].split(',')[0]
fixed=json.loads((ROOT/'report/fixed-model-audit.json').read_text());fixed_model=next(x for x in fixed['records'] if x['recipe']=='T340' and x['arm']=='T340-W128-S100')
if options.model:
 model=options.model.resolve()
 if not options.expected_model_sha256 or file_digest(model)!=options.expected_model_sha256:raise ValueError('explicit parity model SHA mismatch')
elif file_digest(model)!=fixed_model['model_sha256']:raise ValueError('parity model differs from fixed audit')
if options.reference:reference=options.reference.resolve()
if options.cpu_executable:cpu=options.cpu_executable.resolve()
for p in (corpus,reference,cpu,cuda):
 if not p.is_file():raise FileNotFoundError(p)
class PMC(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),('peak',ctypes.c_size_t),('working',ctypes.c_size_t),('paged_peak',ctypes.c_size_t),('paged',ctypes.c_size_t),('nonpaged_peak',ctypes.c_size_t),('nonpaged',ctypes.c_size_t),('pagefile',ctypes.c_size_t),('pagefile_peak',ctypes.c_size_t),('private',ctypes.c_size_t)]
K=ctypes.WinDLL('kernel32',use_last_error=True);P=ctypes.WinDLL('psapi',use_last_error=True);K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];K.OpenProcess.restype=wintypes.HANDLE;K.CloseHandle.argtypes=[wintypes.HANDLE];P.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];P.GetProcessMemoryInfo.restype=wintypes.BOOL
def peak_rss(pid):
 h=K.OpenProcess(0x0400|0x0010,False,pid)
 if not h:return 0
 try:x=PMC();x.cb=ctypes.sizeof(x);return int(x.peak) if P.GetProcessMemoryInfo(h,ctypes.byref(x),x.cb) else 0
 finally:K.CloseHandle(h)
def run(name,exe,ngl,threads,uuid):
 tsv=WORK/f'{name}.tokens.tsv';log=WORK/f'{name}.combined.log';args=[str(exe),'-m',str(model.resolve()),'-f',str(corpus.resolve()),'--ppl-full','-c','65536','-b','512','-ub','256','-ctk',options.cache_type,'-ctv',options.cache_type,'-fa',options.flash_attention,'-t',str(threads),'-tb',str(threads),'--no-warmup','--split-mode','none','--fit','off','--verbose','--kld-token-output',str(tsv.resolve()),'-ngl',str(ngl)]
 if uuid:args.extend(['--device','CUDA0'])
 else:args.extend(['--device','none','--no-op-offload'])
 args.extend(['--kl-divergence','--kl-divergence-base',str(reference.resolve())]);policy=dict(backend='cuda' if uuid else 'cpu',cache_type_k=options.cache_type,cache_type_v=options.cache_type,flash_attention=options.flash_attention)
 if uuid:policy.update(cuda_device=0,cublas_compute_type=options.compute)
 evaluator=NativeKLEvaluator(exe,context=65536,batch=512,microbatch=256,max_tokens=4096,threads=threads,execution=policy)
 env=evaluator._environment();start=time.monotonic();peak=0
 with log.open('xb') as f:
  p=subprocess.Popen(args,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:peak=max(peak,peak_rss(p.pid));time.sleep(.25)
 elapsed=time.monotonic()-start
 if p.returncode!=0:raise RuntimeError(f'{name} evaluator exited {p.returncode}; see {log}')
 text=log.read_text(errors='replace')
 if not uuid and re.search(r'(using device CUDA|CUDA\d+ compute buffer size)',text):raise RuntimeError('CPU run unexpectedly uses CUDA')
 offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',text)
 if uuid and (not offload or offload[-1][0]!=offload[-1][1] or int(offload[-1][0])<=0 or evaluator.gpu['pci_bus_id'] not in text):raise RuntimeError('CUDA execution evidence missing')
 return {'policy':evaluator.policy,'actual_offload':offload[-1] if offload else None,'timings':[line for line in text.splitlines() if any(s in line for s in ['ppl-full timing:','load time =','eval time =','total time ='])],'name':name,'path':str(exe.resolve()),'sha256':file_digest(exe),'command':args,'threads':threads,'ngl':ngl,'gpu_uuid':uuid,'elapsed_seconds':elapsed,'peak_process_working_set_bytes':peak,'token_tsv':str(tsv.resolve()),'token_tsv_sha256':file_digest(tsv),'log_path':str(log.resolve()),'log_sha256':file_digest(log)}
def read_tsv(path):
 out={k:[] for k in ('input_index','context_position','target_token','kld','base_nll','candidate_nll','top1_agree')}
 with Path(path).open(encoding='utf-8',newline='') as f:
  rd=csv.DictReader(f,delimiter='\t');
  if rd.fieldnames==list(out)+['candidate_top1']:out['candidate_top1']=[]
  elif rd.fieldnames!=list(out):raise ValueError('parity output token TSV schema mismatch')
  for i,r in enumerate(rd):
   for k in ('input_index','context_position','target_token','top1_agree'):out[k].append(int(r[k]))
   for k in ('kld','base_nll','candidate_nll'):
    value=float(r[k])
    if not math.isfinite(value):raise ValueError('nonfinite parity token metric')
    out[k].append(value)
   if 'candidate_top1' in out:out['candidate_top1'].append(int(r['candidate_top1']))
 if len(out['input_index'])!=LOCK['expected_targets']:raise ValueError('parity output target count mismatch')
 return {k:np.asarray(v,dtype=np.int64 if k in ('input_index','context_position','target_token','top1_agree','candidate_top1') else np.float64) for k,v in out.items()}
model_before=file_digest(model);input_hashes={'corpus':file_digest(corpus),'reference':file_digest(reference)};reference_run=json.loads((ROOT/'locks/selection-reference-run.json').read_text())
expected_reference=reference_run['reference_sha256']
if options.reference:
 sidecar=json.loads(Path(str(reference)+'.json').read_text());expected_reference=sidecar['reference_sha256']
 if sidecar['corpus_sha256']!=input_hashes['corpus']:raise ValueError('selection reference corpus mismatch')
if input_hashes['corpus']!=reference_run['selection_corpus_sha256'] or input_hashes['reference']!=expected_reference:raise ValueError('selection input hash mismatch')
runs=[run('cpu',cpu,0,32,None),run('cuda',cuda,999,options.cuda_threads,gpu)];a=read_tsv(runs[0]['token_tsv']);b=read_tsv(runs[1]['token_tsv']);
for k in ('input_index','context_position','target_token'):
 if not np.array_equal(a[k],b[k]):raise ValueError(f'CPU/CUDA parity identity mismatch: {k}')
top1_field='candidate_top1' if 'candidate_top1' in a and 'candidate_top1' in b else 'top1_agree'
top1_mismatch_count=int(np.count_nonzero(a[top1_field]!=b[top1_field]));top1_mismatch_rate=top1_mismatch_count/len(a['top1_agree'])
metrics={}
for k in ('kld','base_nll','candidate_nll'):
 delta=b[k]-a[k];metrics[k]={'max_abs_delta':float(np.max(np.abs(delta))),'mean_abs_delta':float(np.mean(np.abs(delta))),'mean_cpu':float(np.mean(a[k])),'mean_cuda':float(np.mean(b[k])),'mean_delta':float(np.mean(delta)),'within_tolerance':bool(np.allclose(a[k],b[k],rtol=TOL['relative_tolerance'],atol=TOL['absolute_tolerance']))}
numerical_parity_status='PASS' if all(x['within_tolerance'] for x in metrics.values()) and top1_mismatch_rate<=CAT['maximum_top1_mismatch_rate'] else 'DIVERGENT'
model_after=file_digest(model);input_after={'corpus':file_digest(corpus),'reference':file_digest(reference)}
if model_after!=model_before or input_after!=input_hashes:raise ValueError('parity evaluation mutated a locked input')
result={'schema_version':1,'status':'PASS','numerical_parity_status':numerical_parity_status,'divergence_policy_sha256':file_digest(ROOT/'locks/evaluator-parity-divergence-policy-addendum.json'),'model_path':str(model.resolve()),'model_sha256_before':model_before,'model_sha256_after':model_after,'corpus_sha256':input_hashes['corpus'],'reference_sha256':input_hashes['reference'],'targets':LOCK['expected_targets'],'evaluator_runs':runs,'exact_alignment_fields':['input_index','context_position','target_token'],'top1_comparison_field':top1_field,'top1_agree_comparison':{'mismatch_count':top1_mismatch_count,'mismatch_rate':top1_mismatch_rate,'maximum_allowed_rate':CAT['maximum_top1_mismatch_rate'],'categorical_addendum_sha256':file_digest(ROOT/'locks/evaluator-parity-categorical-addendum.json')},'numerical_tolerances':{'absolute_tolerance':TOL['absolute_tolerance'],'relative_tolerance':TOL['relative_tolerance'],'addendum_sha256':file_digest(ROOT/'locks/evaluator-parity-tolerance-addendum.json')},'metric_comparison':metrics,'CPU_WikiPPL':math.exp(float(np.mean(a['candidate_nll']))),'CUDA_WikiPPL':math.exp(float(np.mean(b['candidate_nll']))),'provenance_note':'Validated current CUDA and CPU build paths on the frozen disjoint selection stream. The archived prior evaluator binary (3022...) is unavailable, so this does not prove byte-identical behavior to that historical executable.'}
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'status':'PASS','numerical_parity_status':numerical_parity_status,'targets':LOCK['expected_targets'],'metric_comparison':metrics,'report':str(OUT.resolve())},indent=2))
