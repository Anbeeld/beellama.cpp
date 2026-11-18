"""Minimal Python owner for the opt-in native persistent CUDA evaluator.

The canonical assembler and original NativeKLEvaluator process remain independent
oracles. This path never decodes quantized weights or implements model math here.
"""
import csv
from contextlib import ExitStack
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import gguf
from .resources import canonical_json, digest, file_digest, atomic_write, observe_phase, peak_rss_bytes
from .implementation import native_runtime_fingerprint
from .prepared import VerifiedInputs


def token_metrics(path, tokens, context):
    rows=[]
    with Path(path).open(newline='') as stream:
        for i,row in enumerate(csv.DictReader(stream,delimiter='\t')):
            if (i>=len(tokens)-1 or int(row['input_index'])!=i or int(row['context_position'])!=i%context
                    or int(row['target_token'])!=tokens[i+1]):raise ValueError('prepared KL target alignment mismatch')
            values={k:float(row[k]) for k in ('kld','base_nll','candidate_nll','top1_agree')}
            if not all(math.isfinite(v) for v in values.values()) or values['kld'] < -1e-7 or values['top1_agree'] not in (0,1):
                raise ValueError('invalid prepared per-token metrics')
            values['candidate_top1']=int(row['candidate_top1'])
            rows.append(values)
    if len(rows)!=len(tokens)-1:raise ValueError('prepared KL target count mismatch')
    return max(0.0,math.fsum(row['kld'] for row in rows)/len(rows)),rows


def validate_reference_evidence(manifest, *, policy, header, source_sha256, tokenizer):
    """Retain the canonical evaluator's reference/model and CUDA-origin gates."""
    if (manifest.get('schema_version')!=2 or manifest.get('policy')!=policy or
            manifest.get('alignment')!=header or manifest.get('source_sha256')!=source_sha256 or
            manifest.get('tokenizer')!=tokenizer):
        raise ValueError('prepared reference source/tokenizer/alignment/policy mismatch')
    evidence=manifest.get('execution_evidence',{})
    if (not policy.get('gpu') or evidence.get('gpu')!=policy['gpu'] or
            evidence.get('offloaded_layers')!=33 or evidence.get('total_layers')!=33):
        raise ValueError('prepared CUDA reference origin lacks matching 33/33 GPU attestation')


class PreparedGroupwiseEvaluator:
    def __init__(self, oracle, executable, corpus, reference, *, native_loader, reference_policy=None):
        if oracle.gpu is None:raise ValueError('prepared groupwise evaluator requires CUDA; CPU fallback forbidden')
        self.oracle=oracle
        self.native_loader=native_loader
        self.executable=Path(executable).resolve()
        self.corpus=Path(corpus).resolve();self.reference=Path(reference).resolve()
        self.runtime=native_runtime_fingerprint({'prepared_evaluator':self.executable})
        self.reference_policy=oracle.policy if reference_policy is None else reference_policy
        self.policy={**oracle.policy,'reference_origin_policy_sha256':digest(self.reference_policy),'prepared_runtime':self.runtime,
            'prepared_semantics':'private-f32-diagonals-canonical-identity-elision-context-rebuilt-per-score-v2',
            'oracle_policy':oracle.policy}
        self.stack=None;self.proc=None;self.last_execution=None

    def start(self, prepared, store, assignment, output, *, max_output_bytes, verified_inputs=None):
        from .assembly import assemble
        from .kl import _header, _model_identity
        if self.stack is not None:raise RuntimeError('prepared evaluator already active')
        self.stack=ExitStack()
        try:
            self.inputs=self.stack.enter_context(VerifiedInputs(verified_inputs))
            self.prepared=prepared;self.store=store
            self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
            # The small selection reference is freshly authenticated, once per
            # guarded session. No final-test distribution is accepted here.
            self.inputs.fingerprint(self.reference)
            self.inputs.fingerprint(self.reference.with_suffix(self.reference.suffix+'.json'))
            self.inputs.fingerprint(self.corpus)
            self.inputs.fingerprint(self.executable)
            for role in self.runtime.values():
                for name,expected in role.items():
                    path=(self.executable if name=='executable' else self.executable.parent.parent/name)
                    if name.startswith('bin/'):path=self.executable.parent/name[4:]
                    self.inputs.fingerprint(path,expected)
            self.manifest=json.loads(self.reference.with_suffix(self.reference.suffix+'.json').read_text())
            self.header=_header(self.reference,self.oracle.max_tokens)
            validate_reference_evidence(self.manifest,policy=self.reference_policy,header=self.header,
                source_sha256=prepared.manifest['source_sha256'],tokenizer=_model_identity(prepared.source,reference=True,reader=prepared.reader))
            if (self.inputs.fingerprint(self.reference)!=self.manifest['reference_sha256']
                    or self.inputs.fingerprint(self.corpus)!=self.manifest['corpus_sha256']):
                raise ValueError('prepared reference policy/content mismatch')
            self.active=dict(assignment)
            starting=(self.output/'starting.gguf').resolve()
            # Never overwrite a prior attempt: canonical starting output is new.
            if starting.exists():raise FileExistsError('prepared runtime attempt directory already used')
            assemble(prepared.source,starting,store,assignment,stock_baseline=prepared.stock_baseline,
                prepared=prepared,native_validate=self.validator,threads=prepared.threads,max_output_bytes=max_output_bytes)
            self.log=(self.output/'native.log').open('wb')
            self.stack.callback(self.log.close)
            args=[str(self.executable),'-m',str(starting),'-f',str(self.corpus),'--ppl-full',
                '-c',str(self.oracle.context),'-b',str(self.oracle.batch),'-ub',str(self.oracle.microbatch),
                '-t',str(self.oracle.threads),'-tb',str(self.oracle.threads),'-ngl','999',
                '-ctk','bf16','-ctv','bf16','-fa','on','--no-warmup','--split-mode','none',
                '--fit','off','--verbose','--device','CUDA0','--kl-divergence',
                '--kl-divergence-base',str(self.reference)]
            self.command=args;self.messages=queue.Queue()
            self.proc=subprocess.Popen(args,env=self.oracle._environment(),stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
            self.stack.callback(self._stop)
            def read_messages():
                try:
                    for line in self.proc.stdout:
                        if line.startswith(b'SINQ_RESPONSE '):self.messages.put(json.loads(line[14:]))
                except Exception as error:self.messages.put(error)
                finally:self.messages.put(EOFError('native prepared evaluator exited'))
            self.reader=threading.Thread(target=read_messages,daemon=True);self.reader.start()
            self._send({'tensors':sorted(assignment)})
            message=self._receive()
            if (message.get('status')!='READY' or message.get('mutable_tensors')!=len(assignment) or
                    message.get('topology')!='private-f32-diagonals-canonical-identity-elision-context-rebuilt-per-score'):
                raise RuntimeError('native setup/graph semantics protocol mismatch')
            self.log.flush()
            text=(self.output/'native.log').read_text(errors='replace')
            offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',text)
            devices=re.findall(r'using device (CUDA\d+) \([^\n]+?\) \(([0-9a-fA-F:.]+)\)',text)
            if not offload or offload[-1]!=('33','33') or devices!=[('CUDA0',self.oracle.gpu['pci_bus_id'])]:
                raise RuntimeError('prepared CUDA full-offload/physical GPU attestation mismatch')
            return self
        except BaseException:
            self.close();raise

    @property
    def validator(self):
        from .stages import _validator
        # Independent ordinary native loader, not the persistent adapter.
        return _validator(self.native_loader,threads=self.oracle.threads)

    def _send(self,message):
        self.proc.stdin.write(canonical_json(message)+b'\n');self.proc.stdin.flush()

    def _receive(self):
        try:message=self.messages.get(timeout=self.oracle.timeout)
        except queue.Empty as error:raise TimeoutError('prepared native score exceeded timeout') from error
        if isinstance(message,Exception):raise RuntimeError('prepared native process failed; see run-owned native.log') from message
        return message

    def evaluate(self,state):
        if self.stack is None:raise RuntimeError('prepared evaluator not active')
        self.inputs.check();self.prepared.check()
        assignment=dict(state.assignment)
        if assignment.keys()!=self.active.keys():raise ValueError('mutable assignment allocation changed')
        updates=[]
        for name,key in sorted(assignment.items()):
            if self.active[name]==key:continue
            c=self.store.get(key,expected={'source_sha256':self.prepared.manifest['source_sha256'],'tensor_name':name})
            allocation=self.prepared.manifest['tensors'][name]
            if (c.provenance['qtype']!=allocation['qtype'] or c.provenance['shape']!=allocation['shape']
                    or c.provenance['recipe_sha256']!=self.prepared.manifest['recipe_sha256']
                    or c.provenance['imatrix_sha256']!=self.prepared.manifest['imatrix_sha256']
                    or c.provenance['input_sha256']!=allocation['source_sha256']):
                raise ValueError('prepared candidate allocation/provenance mismatch')
            updates.append({'name':name,'payload':str(c.payload.resolve()),
                'scales':str(c.scales.resolve()) if c.scales else None,
                'qtype':int(gguf.GGMLQuantizationType[c.provenance['qtype']])})
        rows=self.output/(state.state_id+'.tsv');started=time.perf_counter()
        with observe_phase('prepared_native_candidate'):
            self._send({'operation':'score','request_id':state.state_id,'updates':updates,'token_output':str(rows.resolve())})
            result=self._receive()
        if result.get('status')!='PASS' or result.get('request_id')!=state.state_id:raise RuntimeError('prepared native result identity mismatch')
        self.inputs.check();self.prepared.check()
        self.active=assignment
        score,metrics=token_metrics(rows,self.header['tokens'],self.oracle.context)
        self.last_execution={'command':self.command,'gpu':self.oracle.gpu,'offloaded_layers':33,'total_layers':33,
            'wall_seconds':time.perf_counter()-started,'changed_tensors':len(updates),
            'native_process_peak_working_set_bytes':peak_rss_bytes(self.proc.pid),
            'reference_inference_bytes_read_by_native_format':self.reference.stat().st_size,
            'reference_sha256_new_bytes_read':0,
            'candidate_payload_bytes_read':sum(Path(u['payload']).stat().st_size+(Path(u['scales']).stat().st_size if u['scales'] else 0) for u in updates),
            'token_output':str(rows.resolve()),'token_output_sha256':file_digest(rows),'score':score}
        record={'state':state.state_id,'assignment':assignment,'policy':self.policy,'reference_manifest':digest(self.manifest),
            'execution':self.last_execution,'score':score}
        record['checksum']=digest(record)
        atomic_write(self.output/(state.state_id+'.json'),canonical_json(record),max_bytes=1<<20,immutable=False)
        return score

    def _stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self._send({'operation':'close'});self.proc.wait(timeout=10)
                except (OSError,subprocess.TimeoutExpired):self.proc.kill();self.proc.wait()
            for stream in (self.proc.stdin,self.proc.stdout):
                if stream:stream.close()
            self.reader.join(timeout=5)
            self.proc=None

    def close(self):
        if self.stack is not None:self.stack.close();self.stack=None

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
