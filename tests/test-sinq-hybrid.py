import sys, tempfile, unittest, importlib.util, hashlib, itertools, os, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
from tools.sinq.research.cache import CandidateStore
from tools.sinq.research.hybrid import assemble
spec=importlib.util.spec_from_file_location('cache_test',Path(__file__).with_name('test-sinq-candidate-cache.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from tools.sinq.research.resources import file_digest

class HybridTest(unittest.TestCase):
    def test_bytes_namespace_stock_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); base=d/'base.gguf'; name='blk.0.ffn_down.weight'
            w=gguf.GGUFWriter(base,'qwen35'); w.add_file_type(gguf.LlamaFileType.MOSTLY_F16); w.add_tensor(name,np.ones((2,256),dtype=np.float16)); w.add_tensor('norm',np.ones(4,dtype=np.float32)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            p=m.provenance(name); p['source_sha256']=file_digest(base)
            p['input_sha256']=hashlib.sha256(np.ones((2,256),dtype='<f2').tobytes()).hexdigest()
            s=CandidateStore(d/'cache'); packed=bytes(352)
            key=s.put(p,packed,np.ones(256,dtype='<f2').tobytes())
            out=d/'hybrid.gguf'; assemble(base,out,s,{name:key},native_validate=lambda path: None)
            r=gguf.GGUFReader(out); self.assertEqual(len([k for k in r.fields if k.startswith('beellama.qk_sinq.')]),7)
            self.assertEqual(r.tensors[0].data.tobytes(),packed)
            self.assertNotIn('general.file_type',r.fields)
            self.assertEqual(r.fields['general.quantization_version'].contents(),gguf.GGML_QUANT_VERSION)
            stock=dict(p); stock['sinq']=None; stock['generator']={'family':'stock'}
            k=s.put(stock,packed)
            assemble(base,d/'stock.gguf',s,{name:k},native_validate=lambda path: None)
            r=gguf.GGUFReader(d/'stock.gguf'); self.assertFalse(any(k.startswith('beellama.qk_sinq.') for k in r.fields))
            with self.assertRaises(ValueError): assemble(base,base,s,{name:key},native_validate=lambda path: None)
            del r

    def test_three_groups_all_stock_ordinary_awq_states(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); base=d/'base.gguf'; names=['blk.0.ffn_down.weight','blk.0.attn_q.weight','blk.0.ssm_out.weight']
            values=np.ones((2,256),dtype='<f2')
            w=gguf.GGUFWriter(base,'qwen35')
            for name in names: w.add_tensor(name,values)
            w.add_tensor('norm',np.ones(4,dtype='<f4')); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            before=file_digest(base); s=CandidateStore(d/'cache'); choices={}
            for name,q in zip(names,['Q3_K','IQ3_S','IQ4_XS']):
                choices[name]=[]
                p=m.provenance(name); p.update(source_sha256=before,input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),qtype=q)
                block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[q]]; packed=bytes(512//block*size)
                for state in range(3):
                    spec=dict(p)
                    if state==0: spec.update(sinq=None,generator={'family':'stock'})
                    elif state==2: spec.update(generator={'family':'awq_weight_activation','alpha':0.5,'lambda':1.0},activation={'source_sha256':before,'split_sha256':before})
                    choices[name].append(s.put(spec,packed,None if state==0 else np.full(256,1+state/4,dtype='<f2').tobytes()))
            for number,states in enumerate(itertools.product(range(3),repeat=3)):
                assignment={name:choices[name][state] for name,state in zip(names,states)}
                out=d/f'{number}.gguf'; assemble(base,out,s,assignment,native_validate=lambda path:None)
                r=gguf.GGUFReader(out)
                self.assertEqual({t.name for t in r.tensors if t.name.endswith('.sinq_scale2')},{n+'.sinq_scale2' for n,state in zip(names,states) if state})
                r.data._mmap.close()
            self.assertEqual(file_digest(base),before)
            p=m.provenance(names[0],128); p.update(source_sha256=before,input_sha256=hashlib.sha256(values.tobytes()).hexdigest())
            wide=s.put(p,bytes(352),np.ones(256,dtype='<f2').tobytes())
            with self.assertRaisesRegex(ValueError,'mixed global'):
                assemble(base,d/'mixed.gguf',s,{names[0]:wide,names[1]:choices[names[1]][1],names[2]:choices[names[2]][0]},native_validate=lambda path:None)
            self.assertFalse((d/'mixed.gguf').exists())
            def reject(path): raise RuntimeError('native validation failure fixture')
            with self.assertRaises(RuntimeError):
                assemble(base,d/'failed.gguf',s,{n:choices[n][0] for n in names},native_validate=reject)
            self.assertFalse((d/'failed.gguf').exists())

    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'set SINQ_NATIVE_BIN for actual native codec/loader smoke')
    def test_actual_native_payload_and_loader(self):
        spec=importlib.util.spec_from_file_location('kl_fixture',Path(__file__).with_name('test-sinq-groupwise-kl.py'))
        fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        binaries=Path(os.environ['SINQ_NATIVE_BIN']); suffix='.exe' if os.name=='nt' else ''
        env=dict(os.environ,OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',CUDA_VISIBLE_DEVICES='')
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); source=d/'source.gguf'; fixture.native_fixture(source)
            name='blk.0.ffn_up.weight'; selected=d/'selected.txt'; selected.write_text(name+'\n')
            store=CandidateStore(d/'cache')
            for q in ('Q3_K','IQ3_S','IQ4_XS','Q5_0','Q6_K'):
                output=d/(q+'.gguf')
                cmd=[str(binaries/('llama-quantize'+suffix)),'--pure','--sinq-precondition','sidecar','--sinq-tensor-list',str(selected),'--sinq-strip-width','64',str(source),str(output),q,'24']
                r=subprocess.run(cmd,env=env,capture_output=True,timeout=120)
                self.assertEqual(r.returncode,0,r.stderr.decode(errors='replace')[-1500:])
                reader=gguf.GGUFReader(output); original=gguf.GGUFReader(source)
                tensor=next(t for t in reader.tensors if t.name==name); scales=next(t for t in reader.tensors if t.name==name+'.sinq_scale2')
                p=m.provenance(name); p.update(source_sha256=file_digest(source),input_sha256=hashlib.sha256(next(t for t in original.tensors if t.name==name).data.tobytes()).hexdigest(),shape=[512,256],source_qtype='F32',qtype=q)
                p['sinq'].update(strength=1.0,numerics=reader.fields['beellama.qk_sinq.numerics'].contents())
                p['producer']['binary_sha256']=file_digest(binaries/('llama-quantize'+suffix))
                key=store.put(p,tensor.data.tobytes(),scales.data.tobytes())
                reader.data._mmap.close(); original.data._mmap.close()
                def load(path):
                    r=subprocess.run([str(binaries/('test-qk-sinq-model'+suffix)),'--load-only',str(path)],env=env,capture_output=True,timeout=120)
                    if r.returncode: raise RuntimeError('native hybrid loader failed')
                assemble(source,d/(q+'-hybrid.gguf'),store,{name:key},native_validate=load)

if __name__=='__main__': unittest.main()
