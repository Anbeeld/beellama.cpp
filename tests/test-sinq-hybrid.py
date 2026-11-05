import sys, tempfile, unittest, importlib.util, hashlib, itertools, json, os, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
from tools.sinq.research.cache import CandidateStore
from tools.sinq.research.hybrid import assemble, assemble_from_anchor
from tools.sinq.research.model_audit import audit_assembled_model
spec=importlib.util.spec_from_file_location('cache_test',Path(__file__).with_name('test-sinq-candidate-cache.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from tools.sinq.research.resources import file_digest

class HybridTest(unittest.TestCase):
    def test_imported_anchor_preserves_unassigned_quantized_payloads(self):
        from tools.sinq.research.resources import digest
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); source=d/'source.gguf'; anchor=d/'anchor.gguf'
            selected='blk.0.ffn_down.weight';selected2='blk.1.ffn_down.weight';untouched='output.weight'
            writer=gguf.GGUFWriter(source,'qwen35')
            writer.add_tensor(selected,np.ones((2,256),dtype='<f2'))
            writer.add_tensor(selected2,np.full((2,256),3,dtype='<f2'))
            writer.add_tensor(untouched,np.full((2,256),2,dtype='<f2'))
            writer.add_tensor('norm',np.ones(4,dtype='<f4'))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            original=gguf.GGUFReader(source);source_tensors={t.name:t for t in original.tensors}
            source_sha=file_digest(source);recipe=hashlib.sha256(b'recipe').hexdigest();imatrix=hashlib.sha256(b'imatrix').hexdigest()
            store=CandidateStore(d/'cache')
            p=m.provenance(selected);p.update(source_sha256=source_sha,input_sha256=hashlib.sha256(source_tensors[selected].data.tobytes()).hexdigest(),
                source_qtype='F16',shape=[2,256],qtype='Q5_0',recipe_sha256=recipe,imatrix_sha256=imatrix)
            p['sinq'].update(strip_width=64,strength=1.0,numerics='fp16-source-statistics;fp32-best-preupdate;serialized-f16-diagonal;original-W/t;imatrix*t^2')
            p['generator']={'family':'ordinary'}
            packed=bytes(2*8*22);scales=np.ones(256,dtype='<f2').tobytes();anchor_key=store.put(p,packed,scales)
            p2=m.provenance(selected2);p2.update(source_sha256=source_sha,input_sha256=hashlib.sha256(source_tensors[selected2].data.tobytes()).hexdigest(),
                source_qtype='F16',shape=[2,256],qtype='Q5_0',recipe_sha256=recipe,imatrix_sha256=imatrix)
            p2['sinq'].update(strip_width=64,strength=1.0,numerics='fp16-source-statistics;fp32-best-preupdate;serialized-f16-diagonal;original-W/t;imatrix*t^2')
            p2['generator']={'family':'ordinary'}
            anchor_key2=store.put(p2,bytes([1])*len(packed),scales)
            q=m.provenance(untouched);q.update(source_sha256=source_sha,input_sha256=hashlib.sha256(source_tensors[untouched].data.tobytes()).hexdigest(),
                source_qtype='F16',shape=[2,256],qtype='Q4_0',recipe_sha256=recipe,imatrix_sha256=imatrix)
            q.update(sinq=None,generator={'family':'stock'},activation=None)
            stock_key=store.put(q,bytes(2*8*18))
            assemble(source,anchor,store,{selected:anchor_key,selected2:anchor_key2,untouched:stock_key},native_validate=lambda path:None)
            original_anchor=gguf.GGUFReader(anchor);reordered=d/'anchor-reordered.gguf'
            rewrite=gguf.GGUFWriter(reordered,'qwen35')
            for key,field in original_anchor.fields.items():
                if key.startswith('GGUF.') or key=='general.architecture':continue
                value=field.contents()
                if key=='beellama.qk_sinq.tensors':value=list(reversed(value))
                rewrite.add_key_value(key,value,field.types[0],field.types[1] if len(field.types)>1 else None)
            for tensor in original_anchor.tensors:
                rewrite.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
            rewrite.write_header_to_file();rewrite.write_kv_data_to_file();rewrite.write_tensors_to_file();rewrite.close()
            original_anchor.data._mmap.close();anchor=reordered
            ar=gguf.GGUFReader(anchor);anchor_tensors={t.name:t for t in ar.tensors}
            records={}
            for name,src in source_tensors.items():
                target=anchor_tensors[name];companion=anchor_tensors.get(name+'.sinq_scale2')
                records[name]={'candidate_id':anchor_key if name==selected else anchor_key2 if name==selected2 else None,
                    'companion_bytes':0 if companion is None else companion.data.nbytes,
                    'companion_sha256':None if companion is None else hashlib.sha256(companion.data.tobytes()).hexdigest(),
                    'packed_bytes':target.data.nbytes,'packed_sha256':hashlib.sha256(target.data.tobytes()).hexdigest(),
                    'qtype':target.tensor_type.name,'shape':list(reversed(target.shape.tolist())),
                    'source_qtype':src.tensor_type.name,'source_tensor_sha256':hashlib.sha256(src.data.tobytes()).hexdigest()}
            ar.data._mmap.close();original.data._mmap.close()
            manifest={'schema_version':1,'kind':'fixture-anchor-lock','source_path':str(source.resolve()),
                'source_sha256':source_sha,'anchor_path':str(anchor.resolve()),'anchor_sha256':file_digest(anchor),
                'recipe_sha256':recipe,'imatrix_sha256':imatrix,'algorithm':'sinkhorn_log','width':64,
                'iterations':16,'numerics':p['sinq']['numerics'],'tensors':records}
            manifest['checksum']=digest(manifest);lock=d/'anchor-lock.json'
            lock.write_text(json.dumps(manifest,sort_keys=True,separators=(',',':')))
            alternate=dict(p);alternate['generator']={'family':'awq_weight_activation','alpha':0.35,'lambda':0.5}
            alternate['activation']={'source_sha256':source_sha,'split_sha256':source_sha}
            alternate_key=store.put(alternate,bytes([1])*len(packed),np.full(256,1.5,dtype='<f2').tobytes())
            candidate_run={'schema_version':1,'source_sha256':source_sha,'recipe_sha256':recipe,
                'imatrix_sha256':imatrix,'cache':'cache','tensors':{selected:{'stock':anchor_key,
                'anchor_candidate_id':anchor_key,'candidates':[{'candidate_id':alternate_key}]},
                selected2:{'stock':anchor_key2,'anchor_candidate_id':anchor_key2,'candidates':[]}}}
            candidate_run['checksum']=digest(candidate_run);run_path=d/'candidate-run.json'
            run_path.write_text(json.dumps(candidate_run,sort_keys=True,separators=(',',':')))
            no_op=d/'no-op.gguf'
            assemble_from_anchor(source,anchor,lock,run_path,no_op,store,{selected:anchor_key,selected2:anchor_key2},native_validate=lambda path:None)
            def inventory(path):
                r=gguf.GGUFReader(path)
                try:return {t.name:(t.tensor_type.name,t.shape.tolist(),t.data.tobytes()) for t in r.tensors}
                finally:r.data._mmap.close()
            self.assertEqual(inventory(no_op),inventory(anchor))
            no_op_reader=gguf.GGUFReader(no_op);anchor_reader=gguf.GGUFReader(anchor)
            self.assertEqual(no_op_reader.fields['beellama.qk_sinq.tensors'].contents(),anchor_reader.fields['beellama.qk_sinq.tensors'].contents())
            no_op_reader.data._mmap.close();anchor_reader.data._mmap.close()
            self.assertEqual(inventory(no_op)[untouched][0],'Q4_0')
            self.assertNotEqual(inventory(no_op)[untouched][0],source_tensors[untouched].tensor_type.name)
            no_op_audit=audit_assembled_model(no_op,anchor,lock,run_path,store,{selected:anchor_key,selected2:anchor_key2},recipe='T340')
            self.assertEqual(no_op_audit['tensor_inventory_sha256'],no_op_audit['anchor_tensor_inventory_sha256'])
            self.assertEqual(no_op_audit['changed_tensors'],[])
            changed=d/'changed.gguf'
            assemble_from_anchor(source,anchor,lock,run_path,changed,store,{selected:alternate_key,selected2:anchor_key2},native_validate=lambda path:None)
            before=inventory(anchor);after=inventory(changed)
            self.assertEqual({n for n in before if before[n]!=after[n]},{selected,selected+'.sinq_scale2'})
            self.assertEqual(after[untouched],before[untouched])
            self.assertEqual(after['norm'],before['norm'])
            changed_audit=audit_assembled_model(changed,anchor,lock,run_path,store,{selected:alternate_key,selected2:anchor_key2},recipe='T340')
            self.assertEqual(changed_audit['changed_tensors'],[selected])
            missing=d/'missing.gguf'
            with self.assertRaisesRegex(ValueError,'all conditioned anchor tensors'):
                assemble_from_anchor(source,anchor,lock,run_path,missing,store,{},native_validate=lambda path:None)
            self.assertFalse(missing.exists())

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

    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'requires native CPU binaries')
    def test_complete_stock_baseline_preserves_unselected_quantized_bytes(self):
        import re
        from tools.sinq.research.baseline import lock_baseline
        spec=importlib.util.spec_from_file_location('baseline_fixture',ROOT/'tests/test-sinq-groupwise-kl.py')
        fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        binaries=Path(os.environ['SINQ_NATIVE_BIN']); suffix='.exe' if os.name=='nt' else ''
        quantizer=binaries/('llama-quantize'+suffix)
        env=dict(os.environ,OMP_NUM_THREADS='32',MKL_NUM_THREADS='32',CUDA_VISIBLE_DEVICES='')
        def run(args):
            result=subprocess.run([str(x) for x in args],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')[-2000:])
        def load(path): run([binaries/('test-qk-sinq-model'+suffix),'--load-only',path])
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); source=d/'source.gguf'; tied=d/'tied.gguf'; fixture.native_fixture(tied)
            # The shared fixture ties output to embeddings; explicitly materialize
            # output so this regression covers both independently quantized tables.
            r=gguf.GGUFReader(tied); w=gguf.GGUFWriter(source,'qwen35')
            for key,field in r.fields.items():
                if key.startswith('GGUF.') or key=='general.architecture': continue
                w.add_key_value(key,field.contents(),field.types[0],field.types[1] if len(field.types)>1 else None)
            for tensor in r.tensors: w.add_tensor(tensor.name,tensor.data)
            embedding=next(t for t in r.tensors if t.name=='token_embd.weight')
            w.add_tensor('output.weight',embedding.data)
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); r.data._mmap.close()
            r=gguf.GGUFReader(source)
            recipe={t.name:('Q5_0' if len(t.shape)>1 and int(t.shape[0])%32==0 else t.tensor_type.name) for t in r.tensors}
            source_tensors={t.name:(list(reversed(t.shape.tolist())),t.tensor_type.name,hashlib.sha256(t.data.tobytes()).hexdigest()) for t in r.tensors}
            r.data._mmap.close()
            types=d/'types.txt'; types.write_text('\n'.join('^'+re.escape(n)+'$='+q.lower() for n,q in sorted(recipe.items())))
            stock=d/'stock.gguf'
            common=[quantizer,'--pure','--tensor-type-file',types,'--max-buffer-size','64']
            run(common+[source,stock,'Q8_0','32'])
            lock=d/'baseline.json'; manifest=lock_baseline(source,stock,recipe,lock,quantizer=quantizer,native_validate=load)
            selected=['blk.0.ffn_up.weight','blk.0.ssm_out.weight','blk.1.attn_output.weight']
            self.assertTrue(set(selected)<=set(recipe))
            names=d/'selected.txt'; names.write_text('\n'.join(selected))
            conditioned=d/'conditioned.gguf'
            run(common+['--sinq-precondition','sidecar','--sinq-tensor-list',names,'--sinq-strip-width','64',source,conditioned,'Q8_0','32'])
            r=gguf.GGUFReader(conditioned); stock_reader=gguf.GGUFReader(stock)
            expected={t.name:(t.tensor_type.name,t.shape.tolist(),t.data.tobytes()) for t in stock_reader.tensors}
            for name in ['token_embd.weight','output.weight']+selected:
                self.assertEqual(expected[name][0],'Q5_0')
            data={t.name:t for t in r.tensors}; store=CandidateStore(d/'cache'); assignment={}
            for name in selected:
                shape,source_qtype,input_sha256=source_tensors[name]
                p=m.provenance(name); p.update(source_sha256=file_digest(source),input_sha256=input_sha256,source_qtype=source_qtype,shape=shape,qtype=recipe[name],recipe_sha256=manifest['recipe_sha256'])
                p['sinq'].update(strength=1.0,numerics=r.fields['beellama.qk_sinq.numerics'].contents())
                p['producer']['binary_sha256']=file_digest(quantizer)
                assignment[name]=store.put(p,data[name].data.tobytes(),data[name+'.sinq_scale2'].data.tobytes())
            r.data._mmap.close(); stock_reader.data._mmap.close()
            output=d/'hybrid.gguf'; assemble(source,output,store,assignment,native_validate=load,stock_baseline=lock)
            r=gguf.GGUFReader(output)
            for t in r.tensors:
                if t.name not in selected and not t.name.endswith('.sinq_scale2'):
                    self.assertEqual((t.tensor_type.name,t.shape.tolist(),t.data.tobytes()),expected[t.name])
            self.assertEqual(len([k for k in r.fields if k.startswith('beellama.qk_sinq.')]),7)
            r.data._mmap.close()
            run([binaries/('llama-completion'+suffix),'-m',output,'-p','abc','-n','2','-c','32','-b','24','-ub','24','-t','24','-ngl','0','--no-warmup','--seed','0'])
            # A stock candidate is not allowed to claim arbitrary baseline bytes.
            name=selected[0]; p=dict(store.get(assignment[name]).provenance)
            p.update(sinq=None,generator={'family':'stock'})
            payload=bytearray(expected[name][2]); payload[0]^=1
            bad_store=CandidateStore(d/'bad-cache'); bad=bad_store.put(p,bytes(payload))
            with self.assertRaisesRegex(ValueError,'stock candidate differs'):
                assemble(source,d/'bad.gguf',bad_store,{name:bad},native_validate=load,stock_baseline=lock)
            self.assertFalse((d/'bad.gguf').exists())
            good=store.put(p,expected[name][2])
            stock_copy=d/'stock-copy.gguf'
            assemble(source,stock_copy,store,{name:good},native_validate=load,stock_baseline=lock)
            r=gguf.GGUFReader(stock_copy)
            self.assertFalse(any(k.startswith('beellama.qk_sinq.') for k in r.fields))
            self.assertEqual({t.name:(t.tensor_type.name,t.shape.tolist(),t.data.tobytes()) for t in r.tensors},expected)
            r.data._mmap.close()
            incompatible=dict(p); incompatible['imatrix_sha256']='0'*64
            wrong=store.put(incompatible,expected[name][2])
            with self.assertRaisesRegex(ValueError,'recipe/imatrix'):
                assemble(source,d/'imatrix.gguf',store,{name:wrong},native_validate=load,stock_baseline=lock)
            incompatible=dict(p); incompatible['qtype']='Q8_0'
            block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
            wrong=store.put(incompatible,bytes(int(np.prod(p['shape']))//block*size))
            with self.assertRaisesRegex(ValueError,'allocation'):
                assemble(source,d/'allocation.gguf',store,{name:wrong},native_validate=load,stock_baseline=lock)

    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'set SINQ_NATIVE_BIN for actual native codec/loader smoke')
    def test_actual_native_payload_and_loader(self):
        spec=importlib.util.spec_from_file_location('kl_fixture',Path(__file__).with_name('test-sinq-groupwise-kl.py'))
        fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        binaries=Path(os.environ['SINQ_NATIVE_BIN']); suffix='.exe' if os.name=='nt' else ''
        env=dict(os.environ,OMP_NUM_THREADS='32',MKL_NUM_THREADS='32',CUDA_VISIBLE_DEVICES='')
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
