"""Source-based assembly, exact baseline bytes and native loader closure."""
import sys, tempfile, unittest, importlib.util, hashlib, itertools, os, subprocess, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
from tools.sinq.optimizer.cache import CandidateStore
from tools.sinq.optimizer.assembly import assemble
from tools.sinq.optimizer.resources import file_digest
spec=importlib.util.spec_from_file_location('cache_test',Path(__file__).with_name('test-sinq-candidate-cache.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def inventory(path):
    r=gguf.GGUFReader(path)
    try:return {t.name:(t.tensor_type.name,t.shape.tolist(),t.data.tobytes()) for t in r.tensors}
    finally:r.data._mmap.close()


class AssemblyTest(unittest.TestCase):
    def test_parallel_hashes_match_independent_digests(self):
        from tools.sinq.optimizer.resources import tensor_hashes
        arrays=[(str(i),np.arange((1<<20)+i,dtype=np.float32)) for i in range(5)]
        expected={name:hashlib.sha256(data.tobytes()).hexdigest() for name,data in arrays}
        for threads in (1,2,4):
            self.assertEqual(tensor_hashes(arrays,threads=threads),expected)
            self.assertEqual(list(tensor_hashes(arrays,threads=threads)),list(expected))
        self.assertEqual(tensor_hashes([],threads=4),{})
        for threads in (0,-1,True,1.5):
            with self.assertRaises(ValueError):tensor_hashes(arrays,threads=threads)
        with self.assertRaises(AttributeError):tensor_hashes([('valid',arrays[0][1]),('invalid',None)],threads=2)

    def test_parallel_file_digests_preserve_sha256_and_errors(self):
        from tools.sinq.optimizer.resources import file_digests
        with tempfile.TemporaryDirectory() as directory:
            paths={}
            for i in range(3):
                path=Path(directory)/str(i);path.write_bytes(bytes([i])*((1<<20)+i));paths[str(i)]=path
            expected={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
            for threads in (1,2,4):self.assertEqual(file_digests(paths,threads=threads),expected)
            for threads in (0,-1,True,1.5):
                with self.assertRaises(ValueError):file_digests(paths,threads=threads)
            with self.assertRaises(FileNotFoundError):file_digests({**paths,'missing':Path(directory)/'missing'},threads=4)
            self.assertEqual(file_digests({},threads=4),{})

    def test_removed_anchor_inputs_are_not_public_assembly_config(self):
        from tools.sinq.optimizer.schemas import validate_config
        c=dict(schema_version=2,inputs=dict(source_gguf='s',candidate_manifest='c',native_loader='l'),
            selection=dict(assignment={'a':'id'}),limits=dict(workspace_bytes=1024,max_artifact_bytes=1024,max_files=16),
            output=dict(gguf='assembled.gguf'))
        for key in ('anchor_gguf','anchor_manifest','allocation_audit'):
            bad={**c,'inputs':{**c['inputs'],key:'old.json'}}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(bad,'assemble')

    def test_bytes_namespace_stock_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);base=d/'base.gguf';name='blk.0.ffn_down.weight'
            w=gguf.GGUFWriter(base,'qwen35');w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
            w.add_tensor(name,np.ones((2,256),dtype=np.float16));w.add_tensor('norm',np.ones(4,dtype=np.float32))
            w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
            p=m.provenance(name);p.update(source_sha256=file_digest(base),input_sha256=hashlib.sha256(np.ones((2,256),dtype='<f2').tobytes()).hexdigest())
            s=CandidateStore(d/'cache');packed=bytes(352);key=s.put(p,packed,np.ones(256,dtype='<f2').tobytes())
            out=d/'assembled.gguf';assemble(base,out,s,{name:key},native_validate=lambda path:None)
            r=gguf.GGUFReader(out)
            self.assertEqual({k for k in r.fields if k.startswith('beellama.qk_sinq.')},{'beellama.qk_sinq.'+k for k in ('version','transform','tensors','recipes','producer')})
            self.assertEqual(r.tensors[0].data.tobytes(),packed);self.assertNotIn('general.file_type',r.fields)
            self.assertEqual(r.fields['general.quantization_version'].contents(),gguf.GGML_QUANT_VERSION);r.data._mmap.close()
            stock=dict(p,sinq=None,generator={'generator':'stock'});k=s.put(stock,packed)
            assemble(base,d/'stock.gguf',s,{name:k},native_validate=lambda path:None)
            r=gguf.GGUFReader(d/'stock.gguf');self.assertFalse(any(k.startswith('beellama.qk_sinq.') for k in r.fields));r.data._mmap.close()
            with self.assertRaises(ValueError):assemble(base,base,s,{name:key},native_validate=lambda path:None)

    def test_three_groups_all_stock_sinq_awq_states(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);base=d/'base.gguf';names=['blk.0.ffn_down.weight','blk.0.attn_q.weight','blk.0.ssm_out.weight']
            values=np.ones((2,256),dtype='<f2');w=gguf.GGUFWriter(base,'qwen35')
            for name in names:w.add_tensor(name,values)
            w.add_tensor('norm',np.ones(4,dtype='<f4'));w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
            before=file_digest(base);s=CandidateStore(d/'cache');choices={}
            for name,q in zip(names,['Q3_K','IQ3_S','IQ4_XS']):
                choices[name]=[];p=m.provenance(name)
                p.update(source_sha256=before,input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),qtype=q)
                block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[q]];packed=bytes(512//block*size)
                for state in range(3):
                    spec=dict(p)
                    if state==0:spec.update(sinq=None,generator={'generator':'stock'})
                    elif state==2:spec.update(generator={**p['generator'],'generator':'awq','correction':'weight_activation','alpha':.5,'lambda':1.},activation={'source_sha256':before,'split_sha256':before})
                    choices[name].append(s.put(spec,packed,None if state==0 else np.full(256,1+state/4,dtype='<f2').tobytes()))
            for number,states in enumerate(itertools.product(range(3),repeat=3)):
                assignment={name:choices[name][state] for name,state in zip(names,states)}
                out=d/f'{number}.gguf';assemble(base,out,s,assignment,native_validate=lambda path:None)
                self.assertEqual({n for n in inventory(out) if n.endswith('.sinq_scale2')},{n+'.sinq_scale2' for n,state in zip(names,states) if state})
                parallel=d/f'{number}-parallel.gguf';assemble(base,parallel,s,assignment,native_validate=lambda path:None,threads=4)
                self.assertEqual(out.read_bytes(),parallel.read_bytes())
            self.assertEqual(file_digest(base),before)
            p=m.provenance(names[0],128);p.update(source_sha256=before,input_sha256=hashlib.sha256(values.tobytes()).hexdigest())
            wide=s.put(p,bytes(352),np.ones(256,dtype='<f2').tobytes())
            assemble(base,d/'mixed.gguf',s,{names[0]:wide,names[1]:choices[names[1]][1],names[2]:choices[names[2]][0]},native_validate=lambda path:None)
            mixed=gguf.GGUFReader(d/'mixed.gguf')
            try:
                recipes=dict(zip(mixed.fields['beellama.qk_sinq.tensors'].contents(),mixed.fields['beellama.qk_sinq.recipes'].contents()))
                self.assertEqual(json.loads(recipes[names[0]])['strip_width'],128)
                self.assertEqual(json.loads(recipes[names[1]])['strip_width'],64)
                self.assertNotIn(names[2],recipes)
            finally:mixed.data._mmap.close()
            def reject(path):raise RuntimeError('native validation failure fixture')
            with self.assertRaises(RuntimeError):assemble(base,d/'failed.gguf',s,{n:choices[n][0] for n in names},native_validate=reject)
            self.assertFalse((d/'failed.gguf').exists())

    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'requires native CPU binaries')
    def test_complete_stock_baseline_preserves_unselected_quantized_bytes(self):
        import re
        from tools.sinq.optimizer.baseline import lock_baseline
        spec=importlib.util.spec_from_file_location('baseline_fixture',ROOT/'tests/test-sinq-groupwise-kl.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        binaries=Path(os.environ['SINQ_NATIVE_BIN']);suffix='.exe' if os.name=='nt' else ''
        quantizer=binaries/('llama-quantize'+suffix);env=dict(os.environ,OMP_NUM_THREADS='32',MKL_NUM_THREADS='32',CUDA_VISIBLE_DEVICES='')
        def run(args):
            result=subprocess.run([str(x) for x in args],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')[-2000:])
        def load(path):run([binaries/('test-qk-sinq-model'+suffix),'--load-only',path])
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);source=d/'source.gguf';tied=d/'tied.gguf';fixture.native_fixture(tied)
            r=gguf.GGUFReader(tied);w=gguf.GGUFWriter(source,'qwen35')
            for key,field in r.fields.items():
                if key.startswith('GGUF.') or key=='general.architecture':continue
                w.add_key_value(key,field.contents(),field.types[0],field.types[1] if len(field.types)>1 else None)
            for tensor in r.tensors:w.add_tensor(tensor.name,tensor.data)
            w.add_tensor('output.weight',next(t for t in r.tensors if t.name=='token_embd.weight').data)
            w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close();r.data._mmap.close()
            original=inventory(source);recipe={n:('Q5_0' if len(shape)>1 and shape[0]%32==0 else q) for n,(q,shape,data) in original.items()}
            types=d/'types.txt';types.write_text('\n'.join('^'+re.escape(n)+'$='+q.lower() for n,q in sorted(recipe.items())))
            common=[quantizer,'--pure','--tensor-type-file',types,'--max-buffer-size','64'];stock=d/'stock.gguf'
            run(common+[source,stock,'Q8_0','32']);lock=d/'baseline.json'
            manifest=lock_baseline(source,stock,recipe,lock,quantizer=quantizer,native_validate=load)
            selected=['blk.0.ffn_up.weight','blk.0.ssm_out.weight','blk.1.attn_output.weight']
            names=d/'selected.txt';names.write_text('\n'.join(selected));conditioned=d/'conditioned.gguf'
            config=d/'direct.json';config.write_text(json.dumps(dict(schema_version=2,defaults=dict(generator='sinq',strip_width=64,iterations=16,strength=1.0),targets=dict(include=selected))))
            run(common+['--sinq-config',config,source,conditioned,'Q8_0','32'])
            r=gguf.GGUFReader(conditioned);data={t.name:t for t in r.tensors};expected=inventory(stock)
            for name in ['token_embd.weight','output.weight']+selected:self.assertEqual(expected[name][0],'Q5_0')
            store=CandidateStore(d/'cache');assignment={}
            for name in selected:
                q,shape,raw=original[name];p=m.provenance(name)
                p.update(source_sha256=file_digest(source),input_sha256=hashlib.sha256(raw).hexdigest(),source_qtype=q,
                    shape=list(reversed(shape)),qtype=recipe[name],recipe_sha256=manifest['recipe_sha256'])
                p['sinq'].update(strength=1.,numerics='serialized-f16-diagonal');p['generator']['strength']=1.
                p['producer']['binary_sha256']=file_digest(quantizer)
                assignment[name]=store.put(p,data[name].data.tobytes(),data[name+'.sinq_scale2'].data.tobytes())
            r.data._mmap.close();output=d/'assembled.gguf'
            assemble(source,output,store,assignment,native_validate=load,stock_baseline=lock)
            actual=inventory(output)
            self.assertEqual({n:v for n,v in actual.items() if n not in selected and not n.endswith('.sinq_scale2')},
                             {n:v for n,v in expected.items() if n not in selected})
            run([binaries/('llama-completion'+suffix),'-m',output,'-p','abc','-n','2','-c','32','-b','24','-ub','24','-t','24','-ngl','0','--no-warmup','--seed','0'])
            name=selected[0];p=dict(store.get(assignment[name]).provenance);p.update(sinq=None,generator={'generator':'stock'})
            payload=bytearray(expected[name][2]);payload[0]^=1;bad_store=CandidateStore(d/'bad-cache');bad=bad_store.put(p,bytes(payload))
            with self.assertRaisesRegex(ValueError,'stock candidate differs'):assemble(source,d/'bad.gguf',bad_store,{name:bad},native_validate=load,stock_baseline=lock)
            self.assertFalse((d/'bad.gguf').exists())
            good=store.put(p,expected[name][2]);stock_copy=d/'stock-copy.gguf'
            assemble(source,stock_copy,store,{name:good},native_validate=load,stock_baseline=lock)
            self.assertEqual(inventory(stock_copy),expected)
            incompatible=dict(p,imatrix_sha256='0'*64);wrong=store.put(incompatible,expected[name][2])
            with self.assertRaisesRegex(ValueError,'recipe/imatrix'):assemble(source,d/'imatrix.gguf',store,{name:wrong},native_validate=load,stock_baseline=lock)
            incompatible=dict(p,qtype='Q8_0');block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
            wrong=store.put(incompatible,bytes(int(np.prod(p['shape']))//block*size))
            with self.assertRaisesRegex(ValueError,'allocation'):assemble(source,d/'allocation.gguf',store,{name:wrong},native_validate=load,stock_baseline=lock)

    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'requires native CPU binaries')
    def test_actual_native_payload_and_loader(self):
        spec=importlib.util.spec_from_file_location('kl_fixture',Path(__file__).with_name('test-sinq-groupwise-kl.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        binaries=Path(os.environ['SINQ_NATIVE_BIN']);suffix='.exe' if os.name=='nt' else ''
        env=dict(os.environ,OMP_NUM_THREADS='32',MKL_NUM_THREADS='32',CUDA_VISIBLE_DEVICES='')
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'source.gguf';fixture.native_fixture(source);name='blk.0.ffn_up.weight'
            selected=d/'direct.json';selected.write_text(json.dumps(dict(schema_version=2,defaults=dict(generator='sinq',strip_width=64,iterations=16,strength=1.0),targets=dict(include=[name]))));store=CandidateStore(d/'cache')
            for q in ('Q3_K','IQ3_S','IQ4_XS','Q5_0','Q6_K'):
                output=d/(q+'.gguf');binary=binaries/('llama-quantize'+suffix)
                result=subprocess.run([str(binary),'--pure','--sinq-config',str(selected),str(source),str(output),q,'24'],env=env,capture_output=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')[-1500:])
                reader=gguf.GGUFReader(output);original=gguf.GGUFReader(source)
                tensor=next(t for t in reader.tensors if t.name==name);scales=next(t for t in reader.tensors if t.name==name+'.sinq_scale2')
                p=m.provenance(name);p.update(source_sha256=file_digest(source),input_sha256=hashlib.sha256(next(t for t in original.tensors if t.name==name).data.tobytes()).hexdigest(),shape=[512,256],source_qtype='F32',qtype=q)
                p['sinq'].update(strength=1.,numerics='serialized-f16-diagonal');p['generator']['strength']=1.
                p['producer']['binary_sha256']=file_digest(binary);key=store.put(p,tensor.data.tobytes(),scales.data.tobytes())
                reader.data._mmap.close();original.data._mmap.close()
                def load(path):
                    result=subprocess.run([str(binaries/('test-qk-sinq-model'+suffix)),'--load-only',str(path)],env=env,capture_output=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')[-1500:])
                assemble(source,d/(q+'-assembled.gguf'),store,{name:key},native_validate=load)


if __name__=='__main__':unittest.main()
