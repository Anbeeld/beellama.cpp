import sys, tempfile, unittest, math, os, subprocess, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
from tools.sinq.optimizer.selector import coordinate_descent, Group, State
from tools.sinq.optimizer.kl import full_vocabulary_kl, validate_split, NativeKLEvaluator
import gguf

def native_fixture(path):
    spec=importlib.util.spec_from_file_location('sinq_activation_parity',ROOT/'tests/test-sinq-activation-parity.py')
    parity=importlib.util.module_from_spec(spec); spec.loader.exec_module(parity)
    directory=path.parent/'checkpoint'
    parity.fixture(directory,'bf16')
    import torch
    torch.set_num_threads(32)
    # Use the proven recurrent/full-attention model and BF16-rounded F32 payloads.
    # Replace all tokenizer metadata so the corpus has a known native tokenization.
    reader=gguf.GGUFReader(directory/'model.gguf')
    w=gguf.GGUFWriter(path,reader.fields['general.architecture'].contents())
    for key,field in reader.fields.items():
        if key.startswith(('GGUF.','tokenizer.')) or key=='general.architecture':
            continue
        w.add_key_value(key,field.contents(),field.types[0],field.types[1] if len(field.types)>1 else None)
    w.add_tokenizer_model('llama'); w.add_token_list(['<unk>','<s>','</s>']+list('abcdefghijklmnopqrstuvwxyz')+[' ','.','!'])
    w.add_token_scores([0.0]*32); w.add_token_types([2,3,3]+[1]*29)
    w.add_unk_token_id(0); w.add_bos_token_id(1); w.add_eos_token_id(2)
    w.add_add_bos_token(True); w.add_add_eos_token(False); w.add_add_space_prefix(False)
    for tensor in reader.tensors:
        w.add_tensor_info(tensor.name,list(reversed(tensor.shape.tolist())),np.dtype('float32'),tensor.data.nbytes,raw_dtype=tensor.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for tensor in reader.tensors:
        w.write_tensor_data(tensor.data)
    w.close(); reader.data._mmap.close()

class SelectionTest(unittest.TestCase):
    def test_kl_direction_and_alignment(self):
        p=np.log([[0.75,0.25]]); q=np.log([[0.5,0.5]])
        self.assertAlmostEqual(full_vocabulary_kl(p,q,[1],[1]),.75*math.log(1.5)+.25*math.log(.5))
        with self.assertRaises(ValueError): full_vocabulary_kl(p,q,[1],[0])
        with self.assertRaises(ValueError): validate_split('final-test')
    def test_search_resume_ties_budget(self):
        with tempfile.TemporaryDirectory() as d:
            groups=[Group('a',('a',),({'a':'1'},{'a':'2'})),Group('b',('b',),({'b':'1'},))]
            scores={('0','0'):10,('1','0'):5,('2','0'):5,('1','1'):2,('2','1'):3,('0','1'):11}
            calls=[]
            def evaluate(s):
                calls.append(s.state_id)
                return scores[(dict(s.assignment)['a'],dict(s.assignment)['b'])]
            initial=State.create({'a':'0','b':'0'},{'source':'fixture'})
            r=coordinate_descent(initial,groups,evaluate,checkpoint=Path(d)/'state.json',max_evaluations=8,max_passes=3)
            self.assertEqual(dict(r.state.assignment),{'a':'1','b':'1'})
            n=len(calls)
            r2=coordinate_descent(initial,groups,evaluate,checkpoint=Path(d)/'state.json',max_evaluations=8,max_passes=3)
            self.assertEqual(r.state,r2.state); self.assertEqual(n,len(calls))
            self.assertLessEqual(r.evaluations,8)
    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'set SINQ_NATIVE_BIN for actual native CPU KL/assembly smoke')
    def test_native_cpu_kl_and_hybrid(self):
        from tools.sinq.optimizer.cache import CandidateStore
        from tools.sinq.optimizer.assembly import assemble
        from tools.sinq.optimizer.resources import file_digest
        binaries=Path(os.environ['SINQ_NATIVE_BIN']); suffix='.exe' if os.name=='nt' else ''
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); source=d/'source.gguf'; native_fixture(source)
            for count in (24,32):
                forward=subprocess.run([str(binaries/('llama-sinq-model'+suffix)),'--forward',str(source),*map(str,[1]+[3]*(count-1))],capture_output=True,timeout=120)
                self.assertEqual(forward.returncode,0,forward.stderr.decode(errors='replace')[-1500:])
                self.assertTrue(np.isfinite(json.loads(forward.stdout)['logits']).all())
            rejected=subprocess.run([str(binaries/('llama-sinq-model'+suffix)),'--forward',str(source),*map(str,[1]+[3]*32)],capture_output=True,timeout=120)
            self.assertEqual(rejected.returncode,1)
            self.assertIn(b'forward token budget',rejected.stderr)
            corpus=d/'tokens.txt'; corpus.write_text('abcdefg',encoding='utf-8')
            evaluator=NativeKLEvaluator(binaries/('llama-perplexity'+suffix),context=32,batch=24,microbatch=24,max_tokens=32)
            ref=d/'reference.bin'; manifest=evaluator.create_reference(source,corpus,ref,expected_tokens=[1,3,4,5,6,7,8,9])
            self.assertEqual(manifest['alignment']['vocabulary'],32)
            identity_score=evaluator.evaluate(source,corpus,ref,cache_directory=d/'kl-cache')
            # The established native reference format is compressed, not exact FP64 logits.
            self.assertLess(identity_score,1e-4)
            name='blk.0.ffn_up.weight'; selected=d/'selected.txt'; selected.write_text(name+'\n')
            quantized=d/'quantized.gguf'
            config=d/'direct.json';config.write_text(json.dumps(dict(schema_version=2,defaults=dict(generator='sinq',strip_width=64,iterations=16,strength=1.0),targets=dict(include=[name]))))
            args=[str(binaries/('llama-quantize'+suffix)),'--pure','--sinq-config',str(config),str(source),str(quantized),'Q5_0','32']
            result=subprocess.run(args,env=dict(os.environ,OMP_NUM_THREADS='32',MKL_NUM_THREADS='32',CUDA_VISIBLE_DEVICES=''),capture_output=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')[-1500:])
            r=gguf.GGUFReader(quantized); original=gguf.GGUFReader(source)
            tensor=next(t for t in r.tensors if t.name==name); scale=next(t for t in r.tensors if t.name==name+'.sinq_scale2')
            h=file_digest(source)
            p=dict(source_sha256=h,tensor_name=name,shape=[512,256],source_qtype='F32',qtype='Q5_0',input_sha256=hashlib.sha256(next(t for t in original.tensors if t.name==name).data.tobytes()).hexdigest(),recipe_sha256=h,imatrix_sha256=None,codec={'implementation':'ggml','chunk_rows':512},sinq={'strip_width':64,'iterations':16,'strength':1.0,'algorithm':'sinkhorn_log','numerics':'serialized-f16-diagonal'},generator={'generator':'sinq','strip_width':64,'iterations':16,'strength':1.0},activation=None,producer={'source_sha256':h,'binary_sha256':file_digest(binaries/('llama-quantize'+suffix))})
            store=CandidateStore(d/'cache'); key=store.put(p,tensor.data.tobytes(),scale.data.tobytes())
            r.data._mmap.close(); original.data._mmap.close()
            output=d/'hybrid.gguf'
            assemble(source,output,store,{name:key},native_validate=lambda path:evaluator.evaluate(path,corpus,ref))
            score=evaluator.evaluate(output,corpus,ref,cache_directory=d/'kl-cache')
            self.assertTrue(math.isfinite(score)); self.assertGreaterEqual(score,0)
            self.assertEqual(score,evaluator.evaluate(output,corpus,ref,cache_directory=d/'kl-cache'))

    def test_budget_and_failure_recovery(self):
        state=State.create({'x':'0'},{'reference':'fixture'})
        groups=[Group('x',('x',),({'x':'1'},{'x':'2'}))]
        result=coordinate_descent(state,groups,lambda s:3,max_evaluations=1)
        self.assertEqual(result.state,state); self.assertTrue(result.budget_exhausted)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'resume.json'; calls=[]
            def failing(s):
                value=dict(s.assignment)['x']; calls.append(value)
                if value=='2': raise TimeoutError('synthetic evaluator timeout')
                return 3-int(value)
            with self.assertRaises(TimeoutError): coordinate_descent(state,groups,failing,checkpoint=path)
            result=coordinate_descent(state,groups,lambda s:3-int(dict(s.assignment)['x']),checkpoint=path)
            self.assertEqual(dict(result.state.assignment),{'x':'2'})
            self.assertEqual(calls.count('0'),1); self.assertEqual(calls.count('1'),1)
            with self.assertRaises(ValueError): coordinate_descent(State.create({'x':'0'},{'reference':'different'}),groups,lambda s:0,checkpoint=path)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            coordinate_descent(State.create({'a':'0'},{}),[Group('x',('a',),()),Group('y',('a',),())],lambda s:0)

if __name__=='__main__': unittest.main()
