from __future__ import annotations
import argparse,hashlib,json,struct,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
from tools.sinq.optimizer.kl import NativeKLEvaluator

SOURCE=Path(r'C:\models\Qwen3.5-9B-bf16\Qwen3.5-9B-text-BF16.gguf')
CORPUS=ROOT/'data/selection.valid-prefix.native-prompt.raw'
TOKENS=ROOT/'data/selection.u32'
parser=argparse.ArgumentParser();parser.add_argument('--output-directory',type=Path,default=ROOT/'locks');parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu');parser.add_argument('--threads',type=int,default=32);parser.add_argument('--cuda-device',type=int,default=0);args=parser.parse_args()
DEST=args.output_directory.resolve();DEST.mkdir(parents=True,exist_ok=True)
REF=DEST/'selection-reference.logits'
EVALUATOR=DEST/'evaluator.json'
SELECTION=DEST/'selection-manifest.json'
execution=dict(backend=args.backend,cache_type_k='bf16',cache_type_v='bf16',flash_attention='on')
if args.backend=='cuda':execution.update(cuda_device=args.cuda_device,gpu_layers='all')
EXEC=REPO/'build-sinq-clean/bin/llama-perplexity.exe'
LOADER=REPO/'build-sinq-clean/bin/llama-sinq-model.exe'
if any(p.exists() for p in (REF,REF.with_suffix(REF.suffix+'.json'),EVALUATOR,SELECTION)):
    raise FileExistsError('selection reference lock already exists; refusing overwrite')
raw=TOKENS.read_bytes()
if len(raw)%4: raise ValueError('selection token source misaligned')
expected=list(struct.unpack('<'+'I'*(len(raw)//4),raw))
if len(expected)<2: raise ValueError('selection partition needs at least two tokens')
evaluator=NativeKLEvaluator(EXEC,context=65536,batch=512,microbatch=256,max_tokens=4096,timeout=3600,threads=args.threads,execution=execution)
start=time.time()
reference=evaluator.create_reference(SOURCE,CORPUS,REF,expected_tokens=expected)
e_json={'schema_version':2,'executable':str(EXEC.resolve()),'reference':str(REF.resolve()),'corpus':str(CORPUS.resolve()),'native_loader':str(LOADER.resolve()),'context':65536,'batch':512,'microbatch':256,'max_tokens':4096,'timeout':3600,'execution':execution}
EVALUATOR.write_text(json.dumps(e_json,indent=2)+'\n')
s_json={'schema_version':2,'split':'selection','corpus_sha256':hashlib.sha256(CORPUS.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'token_count':len(expected),'token_sha256':hashlib.sha256(raw).hexdigest(),'reference_manifest_sha256':hashlib.sha256(REF.with_suffix(REF.suffix+'.json').read_bytes()).hexdigest(),'context':65536,'vocabulary':248320,'targets':len(expected)-1}
SELECTION.write_text(json.dumps(s_json,indent=2)+'\n')
record={'status':'PASS','threads':args.threads,'elapsed_seconds':time.time()-start,'source_sha256':reference['source_sha256'],'selection_corpus_sha256':s_json['corpus_sha256'],'selection_tokens':len(expected),'selection_targets':len(expected)-1,'reference_sha256':s_json['reference_sha256'],'reference_bytes':REF.stat().st_size,'reference_policy':reference['policy'],'reference_manifest_sha256':s_json['reference_manifest_sha256']}
(DEST/'selection-reference-run.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
