"""Aligned full-vocabulary KL(p_reference || p_candidate), including native ppl-full."""
import csv
import json
import math
import os
import re
import time
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np
import gguf
from .resources import atomic_write, canonical_json, digest, file_digest, reject_alias, record_cache, observe_phase
from .schemas import read_json


def validate_split(split):
    if split not in ('selection','research-selection'):
        raise ValueError('optimization requires a designated research selection split, never final-test')


def full_vocabulary_kl(reference_logits,candidate_logits,reference_targets,candidate_targets):
    p=np.asarray(reference_logits,dtype=np.float64); q=np.asarray(candidate_logits,dtype=np.float64)
    if p.ndim!=2 or p.shape!=q.shape or min(p.shape)<1 or len(reference_targets)!=p.shape[0] or not np.array_equal(reference_targets,candidate_targets):
        raise ValueError('logit vocabulary/target alignment mismatch')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError('nonfinite logits')
    def logsoftmax(x):
        x=x-x.max(axis=1,keepdims=True)
        return x-np.log(np.exp(x).sum(axis=1,keepdims=True))
    lp,lq=logsoftmax(p),logsoftmax(q)
    rows=[math.fsum((np.exp(a)*(a-b)).tolist()) for a,b in zip(lp,lq)]
    value=math.fsum(rows)/len(rows)
    if value < -1e-12: raise ValueError('negative KL beyond rounding tolerance')
    return max(0.0,value)


def _model_identity(path, reference=False):
    r=gguf.GGUFReader(path)
    try:
        if reference and any(t.tensor_type not in (gguf.GGMLQuantizationType.BF16,gguf.GGMLQuantizationType.F32) for t in r.tensors):
            raise ValueError('reference must contain BF16/F32 original weights, not quantized/F16 weights')
        architecture=r.fields['general.architecture'].contents()
        return digest({k:v.contents() for k,v in r.fields.items()
            if k.startswith('tokenizer.') or k.startswith(architecture+'.') or k=='general.architecture'})
    finally:
        r.data._mmap.close()


def _header(path, max_tokens):
    with open(path,'rb') as f:
        data=f.read(24)
        if len(data)!=24: raise ValueError('truncated native logits header')
        magic,ctx,vocab,count=struct.unpack('<8sIIQ',data)
        if magic!=b'_logitF1' or not ctx or not vocab or count<2 or count>max_tokens:
            raise ValueError('invalid native full-corpus logits header/budget')
        expected=24+count*4+(count-1)*(2*((vocab+1)//2)+4)*2
        if Path(path).stat().st_size!=expected: raise ValueError('native reference size mismatch')
        tokens=np.frombuffer(f.read(count*4),dtype='<i4').tolist()
        if any(t<0 or t>=vocab for t in tokens): raise ValueError('invalid reference token IDs')
        return {'context':ctx,'vocabulary':vocab,'tokens':tokens}


class NativeKLEvaluator:
    """Explicit CPU/CUDA native full-corpus wrapper; timeout is a hard budget.

    Native _logitF1 stores the established compressed full-vocabulary reference
    log-probabilities, NOT target-only NLL. The binary digest binds that encoding.
    """
    def __init__(self, executable, *, context=32, batch=24, microbatch=24, max_tokens=32, timeout=120, threads=None, execution=None):
        self.executable=Path(executable).resolve()
        threads=(os.cpu_count() or 1) if threads is None else threads
        if type(threads) is not int or not 1<=threads<=1024:
            raise ValueError('threads must be an integer in 1..1024')
        if any(type(v) is not int or v<=0 for v in (context,batch,microbatch,max_tokens)) or type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0 or microbatch>batch:
            raise ValueError('invalid native evaluator budgets')
        self.context=context; self.batch=batch; self.microbatch=microbatch
        self.max_tokens=max_tokens; self.timeout=timeout; self.threads=threads
        from .schemas import native_kl_execution
        from .implementation import native_runtime_fingerprint
        self.execution=native_kl_execution({} if execution is None else execution)
        self.gpu=None; self.last_execution=None
        self.runtime=native_runtime_fingerprint({'evaluator':self.executable})
        if self.execution['backend']=='cuda':
            try:
                query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,driver_version','--format=csv,noheader'],capture_output=True,timeout=15,check=True)
                devices=list(csv.reader(query.stdout.decode('utf-8').splitlines(),skipinitialspace=True))
                selected=[r for r in devices if len(r)==5 and r[0]==str(self.execution['cuda_device'])]
                if len(selected)!=1:raise RuntimeError('requested CUDA GPU unavailable')
                index,uuid,name,pci,driver=selected[0]
                if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',uuid):raise RuntimeError('invalid CUDA GPU identity')
                self.gpu=dict(uuid=uuid,name=name,pci_bus_id=pci.lower()[-12:],driver=driver)
                probe=subprocess.run([str(self.executable),'--list-devices'],env=self._environment(),capture_output=True,timeout=30,check=True)
                if not re.search(rb'CUDA0:',probe.stdout+probe.stderr):raise RuntimeError('native executable has no CUDA support/device')
            except (OSError,subprocess.SubprocessError) as error:
                raise RuntimeError('CUDA discovery/support unavailable') from error
        self.policy={'context':context,'batch':batch,'microbatch':microbatch,'threads':threads,'device':self.execution['backend'],'execution':self.execution,'gpu':self.gpu,'warmup':False,'fit':False,'split_mode':'none','op_offload':bool(self.gpu),'runtime':self.runtime,'reset':'every-context-no-overlap','format':'_logitF1','binary_sha256':file_digest(self.executable)}

    def _environment(self):
        # Numerical environment overrides must not leak in from the launching shell.
        env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_ARG_','GGML_','CUDA_','CUBLAS_','NVIDIA_TF32_'))}
        env.update(OMP_NUM_THREADS=str(self.threads),MKL_NUM_THREADS=str(self.threads),
            CUDA_VISIBLE_DEVICES=self.gpu['uuid'] if self.gpu else '')
        if self.gpu:
            env.update(GGML_CUDA_CUBLAS_COMPUTE_TYPE=self.execution['cublas_compute_type'],
                GGML_CUDA_MMQ_PREC=self.execution['mmq_precision'])
        return env

    def _check_runtime(self):
        from .implementation import native_runtime_fingerprint
        if native_runtime_fingerprint({'evaluator':self.executable})!=self.runtime:
            raise ValueError('native executable/runtime changed during evaluation')

    def _run(self,model,corpus,reference,token_output,generate=False):
        self._check_runtime()
        args=[str(self.executable),'-m',str(Path(model).resolve()),'-f',str(Path(corpus).resolve()),'--ppl-full','-c',str(self.context),'-b',str(self.batch),'-ub',str(self.microbatch),'-t',str(self.threads),'-tb',str(self.threads),'-ngl','999' if self.gpu else '0','--cache-type-k',self.execution['cache_type_k'],'--cache-type-v',self.execution['cache_type_v'],'--flash-attn',self.execution['flash_attention'],'--kld-token-output',str(token_output), '--save-all-logits' if generate else '--kl-divergence-base',str(reference)]
        args.extend(['--no-warmup','--split-mode','none','--fit','off','--verbose'])
        if self.gpu:args.extend(['--device','CUDA0'])
        else:args.extend(['--device','none','--no-op-offload'])
        if not generate:
            args.append('--kl-divergence')
        started=time.perf_counter()
        with observe_phase('native_kl_process'):
            result=subprocess.run(args,env=self._environment(),capture_output=True,timeout=self.timeout,check=False)
        if result.returncode:
            # Avoid copying arbitrary model/corpus data from tool logs to artifacts.
            raise RuntimeError(f'native KL process failed with exit code {result.returncode} (backend={self.execution["backend"]}; check GPU capacity/support)')
        log=(getattr(result,'stdout',b'') or b'')+(getattr(result,'stderr',b'') or b'')
        text=log.decode('utf-8',errors='replace')
        offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',text)
        actual=tuple(map(int,offload[-1])) if offload else (0,0)
        if not self.gpu and (actual[0] or re.search(r'(using device CUDA|CUDA\d+ compute buffer size)',text)):
            raise RuntimeError('CPU execution unexpectedly used CUDA')
        if self.gpu:
            devices=re.findall(r'using device (CUDA\d+) \([^\n]+?\) \(([0-9a-fA-F:.]+)\)',text)
            if actual[0]<=0 or actual[0]!=actual[1] or devices!=[('CUDA0',self.gpu['pci_bus_id'])]:
                raise RuntimeError('CUDA full offload/physical GPU evidence missing or mismatched; CPU fallback rejected')
        self.last_execution=dict(command=args,wall_seconds=time.perf_counter()-started,
            offloaded_layers=actual[0],total_layers=actual[1],gpu=self.gpu,
            native_log_sha256=__import__('hashlib').sha256(log).hexdigest(),
            timings=[line for line in text.splitlines() if re.search(r'(load time|eval time|total time|ppl-full timing:).*',line)])
        self._check_runtime()

    def create_reference(self,model,corpus,output,*,expected_tokens):
        output=Path(output); model=Path(model); corpus=Path(corpus)
        reject_alias(output,[model,corpus,self.executable])
        if output.exists() or output.with_suffix(output.suffix+'.json').exists(): raise ValueError('reference output exists')
        if not 2<=len(expected_tokens)<=self.max_tokens: raise ValueError('token budget exceeded')
        if corpus.stat().st_size>self.max_tokens*1024:
            raise ValueError('corpus byte budget exceeded')
        tokenizer=_model_identity(model,reference=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            ref=Path(tmp)/'reference.bin'; rows=Path(tmp)/'tokens.tsv'
            self._run(model,corpus,ref,rows,generate=True)
            header=_header(ref,self.max_tokens)
            if header['tokens']!=list(expected_tokens) or header['context']!=self.context:
                raise ValueError('reference tokenization/context mismatch')
            manifest={'schema_version':2,'execution_evidence':self.last_execution,'source_sha256':file_digest(model),'corpus_sha256':file_digest(corpus),'reference_sha256':file_digest(ref),'tokenizer':tokenizer,'policy':self.policy,'alignment':header}
            os.link(ref,output)
            atomic_write(output.with_suffix(output.suffix+'.json'),canonical_json(manifest),max_bytes=1 << 20)
        return manifest

    def evaluate(self,model,corpus,reference,*,split='selection',cache_directory=None):
        validate_split(split)
        self._check_runtime()
        reference=Path(reference)
        sidecar=reference.with_suffix(reference.suffix+'.json')
        if sidecar.stat().st_size>1 << 20 or Path(corpus).stat().st_size>self.max_tokens*1024:
            raise ValueError('reference manifest/corpus byte budget exceeded')
        manifest=read_json(sidecar)
        header=_header(reference,self.max_tokens)
        if manifest.get('schema_version')!=2 or manifest['policy']!=self.policy or manifest['reference_sha256']!=file_digest(reference) or manifest['corpus_sha256']!=file_digest(corpus) or manifest['alignment']!=header or _model_identity(model)!=manifest['tokenizer']:
            raise ValueError('native KL reference/model/corpus/policy mismatch')
        expected_offload=manifest.get('execution_evidence',{})
        if self.gpu and (expected_offload.get('gpu')!=self.gpu or not expected_offload.get('offloaded_layers') or expected_offload.get('offloaded_layers')!=expected_offload.get('total_layers')):
            raise ValueError('CUDA reference execution evidence mismatch')
        identity={'model_sha256':file_digest(model),'reference_manifest':digest(manifest),'policy':self.policy,'split':split}
        key=digest(identity)
        cached=Path(cache_directory)/f'{key}.json' if cache_directory else None
        if cached and cached.exists():
            record=read_json(cached)
            if record.get('identity')!=identity or record.get('checksum')!=digest({k:v for k,v in record.items() if k!='checksum'}): raise ValueError('KL cache identity/checksum mismatch')
            score=record['score']
            if not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:
                raise ValueError('invalid cached KL score')
            record_cache('model_kl',True)
            return score
        record_cache('model_kl',False)
        with tempfile.TemporaryDirectory() as tmp:
            rows=Path(tmp)/'tokens.tsv'
            self._run(model,corpus,reference,rows)
            if file_digest(reference)!=manifest['reference_sha256']:
                raise ValueError('native evaluator mutated the locked reference')
            if self.gpu and self.last_execution['total_layers']!=expected_offload['total_layers']:
                raise ValueError('CUDA actual offload layout differs from reference')
            values=[]
            with rows.open(newline='') as f:
                for i,row in enumerate(csv.DictReader(f,delimiter='\t')):
                    if i>=len(header['tokens'])-1 or int(row['input_index'])!=i or int(row['context_position'])!=i%self.context or int(row['target_token'])!=header['tokens'][i+1]:
                        raise ValueError('native KL output token alignment mismatch')
                    value=float(row['kld'])
                    if not math.isfinite(value) or value < -1e-7: raise ValueError('invalid native token KL')
                    values.append(value)
            if len(values)!=len(header['tokens'])-1: raise ValueError('native KL target count mismatch')
            score=max(0.0,math.fsum(values)/len(values))
        if cached:
            record={'identity':identity,'score':score,'execution_evidence':self.last_execution}; record['checksum']=digest(record)
            atomic_write(cached,canonical_json(record),max_bytes=1 << 20)
        return score
