from __future__ import annotations
import argparse,ctypes,hashlib,json,os,subprocess,sys,time,threading
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];sys.path.insert(0,str(REPO))
from tools.sinq.optimizer.resources import file_digest,observe_performance,observe_caches
from tools.sinq.optimizer.schemas import load_config
import tools.sinq.optimizer.assembly as assembly
import tools.sinq.optimizer.kl as kl
import tools.sinq.optimizer.stages as stages
parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--report-directory',type=Path,default=ROOT/'report');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
REPORT=args.report_directory.resolve();REPORT.mkdir(parents=True,exist_ok=True)
CONFIG=Path(args.config).resolve();cfg=load_config(CONFIG,'select');label=Path(cfg['output_directory']).name;recipe=next((part for part in Path(cfg['output_directory']).parts if part in ('T340','T425')),None)
if recipe is None:raise ValueError('measured A/B output path must identify T340 or T425')
BASE_OUT=REPORT/f'{recipe}-{label}-select-runtime.json';BASE_LOG=REPORT/f'{recipe}-{label}-select.log';OUT=BASE_OUT;LOG=BASE_LOG;retry=0;run_dir=Path(cfg['output_directory'])
while OUT.exists() or LOG.exists():
 retry+=1;OUT=REPORT/f'{recipe}-{label}-select-runtime-retry{retry}.json';LOG=REPORT/f'{recipe}-{label}-select-retry{retry}.log'
if run_dir.exists() and not args.resume:raise FileExistsError('selection output directory already exists; explicit --resume is required')
threads=cfg['threads'];os.environ.update(OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),CUDA_VISIBLE_DEVICES='')
class PMC(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),('peak',ctypes.c_size_t),('working',ctypes.c_size_t),('paged_peak',ctypes.c_size_t),('paged',ctypes.c_size_t),('nonpaged_peak',ctypes.c_size_t),('nonpaged',ctypes.c_size_t),('pagefile',ctypes.c_size_t),('pagefile_peak',ctypes.c_size_t),('private',ctypes.c_size_t)]
K=ctypes.WinDLL('kernel32',use_last_error=True);P=ctypes.WinDLL('psapi',use_last_error=True);K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];K.OpenProcess.restype=wintypes.HANDLE;K.CloseHandle.argtypes=[wintypes.HANDLE];P.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];P.GetProcessMemoryInfo.restype=wintypes.BOOL
def peak_rss(pid):
 h=K.OpenProcess(0x0400|0x0010,False,pid)
 if not h:return 0
 try:x=PMC();x.cb=ctypes.sizeof(x);return int(x.peak) if P.GetProcessMemoryInfo(h,ctypes.byref(x),x.cb) else 0
 finally:K.CloseHandle(h)
subprocess_calls=[];assembly_calls=[];validation_calls=[];evaluation_calls=[];active_stage='';original_run=subprocess.run;original_assemble=assembly.assemble;original_evaluate=kl.NativeKLEvaluator.evaluate
vram_peak={};vram_start={};monitor_errors=[];stop_monitor=threading.Event()
def monitor_vram():
 while not stop_monitor.is_set():
  try:
   result=original_run(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,timeout=10,check=True)
   for line in result.stdout.decode().splitlines():
    uuid,used=line.split(',');used=int(used.strip())*1048576;vram_start.setdefault(uuid,used);vram_peak[uuid]=max(vram_peak.get(uuid,0),used)
  except Exception as error:monitor_errors.append(type(error).__name__)
  stop_monitor.wait(2)
monitor=threading.Thread(target=monitor_vram,daemon=True)

def measured_run(args,**kwargs):
 global active_stage
 capture=kwargs.pop('capture_output',False);timeout=kwargs.pop('timeout',None);check=kwargs.pop('check',False);input_data=kwargs.pop('input',None)
 if capture:
  if 'stdout' in kwargs or 'stderr' in kwargs:raise ValueError('capture_output conflicts with stdout/stderr')
  kwargs.update(stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if input_data is not None and 'stdin' in kwargs:raise ValueError('input and stdin conflict')
 if input_data is not None:kwargs['stdin']=subprocess.PIPE
 start=time.monotonic();proc=subprocess.Popen(args,**kwargs);peak=0;out=err=None
 while True:
  try:out,err=proc.communicate(input=input_data if not proc._communication_started else None,timeout=.5);break
  except subprocess.TimeoutExpired:
   peak=max(peak,peak_rss(proc.pid))
   if timeout is not None and time.monotonic()-start>=timeout:
    proc.kill();out,err=proc.communicate();raise subprocess.TimeoutExpired(args,timeout,output=out,stderr=err)
 wall=time.monotonic()-start;blob=(out if isinstance(out,bytes) else (out or '').encode())+(err if isinstance(err,bytes) else (err or '').encode());row={'stage':active_stage,'command':[str(x) for x in (args if isinstance(args,(tuple,list)) else [args])],'wall_seconds':wall,'peak_child_working_set_bytes':peak,'return_code':proc.returncode,'captured_log_sha256':hashlib.sha256(blob).hexdigest()};subprocess_calls.append(row)
 if active_stage=='native_validation':validation_calls.append(row)
 if active_stage=='native_evaluator':evaluation_calls.append(row)
 result=subprocess.CompletedProcess(args,proc.returncode,out,err)
 if check and proc.returncode:raise subprocess.CalledProcessError(proc.returncode,args,output=out,stderr=err)
 return result

def measured_assemble(*args,**kwargs):
 global active_stage
 validation=kwargs.get('native_validate')
 if validation is not None:
  def timed_validate(path):
   global active_stage
   prior=active_stage;active_stage='native_validation';t=time.monotonic()
   try:return validation(path)
   finally:active_stage=prior
  kwargs['native_validate']=timed_validate
 prior=active_stage;active_stage='assembly';start=time.monotonic()
 try:result=original_assemble(*args,**kwargs)
 finally:wall=time.monotonic()-start;active_stage=prior
 path=Path(args[1]);assembly_calls.append({'output_path':str(path),'wall_seconds':wall,'output_bytes':path.stat().st_size if path.is_file() else None,'output_sha256':result if isinstance(result,str) else None,'native_validation_calls':len(validation_calls)})
 return result

def measured_evaluate(self,*args,**kwargs):
 global active_stage
 prior=active_stage;active_stage='native_evaluator';start=time.monotonic()
 try:score=original_evaluate(self,*args,**kwargs)
 finally:wall=time.monotonic()-start;active_stage=prior
 evaluation_calls.append({'stage':'native_kl_evaluate','wall_seconds':wall,'score':score if 'score' in locals() else None,'split':kwargs.get('split'),'execution_evidence':self.last_execution});return score

def main():
 global active_stage
 assembly.assemble=measured_assemble;kl.NativeKLEvaluator.evaluate=measured_evaluate;subprocess.run=measured_run;start=time.monotonic();monitor.start()
 try:
  active_stage='selection'
  with observe_performance() as performance,observe_caches() as caches:
   result=stages.select_config(cfg,resume=run_dir if args.resume else None)
  elapsed=time.monotonic()-start
 finally:
  stop_monitor.set();monitor.join(timeout=15)
  assembly.assemble=original_assemble;kl.NativeKLEvaluator.evaluate=original_evaluate;subprocess.run=original_run;active_stage=''
 selection_path=run_dir/'selection.json';best=run_dir/'best.gguf'
 if result.get('status')!='PASS' or not selection_path.is_file() or not best.is_file():raise RuntimeError('select stage did not publish closed PASS output')
 sel=json.loads(selection_path.read_text());identity=json.loads((run_dir/'run-identity.json').read_text());
 if file_digest(best)!=sel.get('best_gguf_sha256') or sel.get('run_identity_sha256')!=identity.get('checksum'):raise ValueError('selected output closure mismatch')
 score_calls=[x for x in evaluation_calls if x.get('stage')=='native_kl_evaluate'];native_eval=sum(x['wall_seconds'] for x in evaluation_calls if 'native_evaluator_process_wall_seconds' in x) # process runs are separately represented below
 assembly_inclusive_validation=sum(x['wall_seconds'] for x in assembly_calls);validation_total=sum(x['wall_seconds'] for x in validation_calls)
 record={'schema_version':1,'status':'PASS','recipe':recipe,'method':label,'config_path':str(CONFIG),'config_sha256':file_digest(CONFIG),'normalized_config_sha256':identity.get('config_sha256'),'selection_run_identity_sha256':file_digest(run_dir/'run-identity.json'),'best_model_sha256':file_digest(best),'selection_sha256':file_digest(selection_path),'selection_total_time_sec':elapsed,'assembly_time_sec':max(0.0,assembly_inclusive_validation-validation_total),'assembly_including_validation_sec':assembly_inclusive_validation,'native_validation_time_sec':validation_total,'native_kl_evaluation_time_sec':sum(x['wall_seconds'] for x in evaluation_calls if x.get('stage')=='native_kl_evaluate'),'native_evaluator_process_time_sec':sum(x['wall_seconds'] for x in subprocess_calls if x['stage']=='native_evaluator'),'native_evaluations':len([x for x in evaluation_calls if x.get('stage')=='native_kl_evaluate']),'assembly_calls':assembly_calls,'native_validation_calls':validation_calls,'native_evaluator_process_calls':[x for x in subprocess_calls if x['stage']=='native_evaluator'],'selection_evaluations':sel.get('evaluations',0),'budget_exhausted':sel.get('budget_exhausted',False),'trace_events':len(sel.get('trace',[])),'threads':threads,'cpu_only':json.loads(Path(cfg['evaluator']).read_text()).get('execution',{}).get('backend','cpu')=='cpu','performance':performance,'cache':caches,'resume':args.resume,'parent_peak_working_set_bytes':peak_rss(os.getpid()),'vram_start_bytes':vram_start,'vram_peak_bytes':vram_peak,'vram_sampling_errors':monitor_errors,'vram_scope':'device memory.used sampled every 2 s, includes other allocations; not exact transient/process peak','selected_model_bytes':best.stat().st_size,'selected_model_recorded_sha256':sel['best_gguf_sha256'],'subprocess_calls':subprocess_calls}
 OUT.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');LOG.write_text(json.dumps({'command':['python','run_measured_selection.py',str(CONFIG)],'status':'PASS','result':record},indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'status':'PASS','recipe':recipe,'method':label,'selection_total_time_sec':elapsed,'assembly_time_sec':record['assembly_time_sec'],'evaluations':record['selection_evaluations'],'best_model_sha256':record['best_model_sha256']},indent=2))
try:main()
except Exception as error:
 assembly.assemble=original_assemble;kl.NativeKLEvaluator.evaluate=original_evaluate;subprocess.run=original_run
 if not OUT.exists():OUT.write_text(json.dumps({'schema_version':1,'status':'FAIL','recipe':recipe,'method':label,'config_sha256':file_digest(CONFIG),'error_type':type(error).__name__,'error':str(error),'assembly_calls':assembly_calls,'validation_calls':validation_calls,'evaluation_calls':evaluation_calls,'subprocess_calls':subprocess_calls},indent=2)+'\n',encoding='utf-8',newline='\n')
 raise
