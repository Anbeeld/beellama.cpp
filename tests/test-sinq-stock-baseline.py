import sys, tempfile, unittest, importlib.util, os, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
from tools.sinq.research.baseline import lock_baseline, load_baseline

class BaselineTest(unittest.TestCase):
    def fixture(self,path,value=1):
        w=gguf.GGUFWriter(path,'qwen35'); w.add_tensor('output.weight',np.full((2,32),value,dtype='<f4')); w.add_tensor('norm',np.ones(32,dtype='<f4'))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    def test_complete_recipe_and_float_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); source=d/'source.gguf'; stock=d/'stock.gguf'; self.fixture(source); self.fixture(stock)
            with self.assertRaisesRegex(ValueError,'complete recipe'):
                lock_baseline(source,stock,{'output.weight':'F32'},d/'lock.json',quantizer=Path(sys.executable),native_validate=lambda p:None)
            self.assertFalse((d/'lock.json').exists())
            with self.assertRaisesRegex(ValueError,'uppercase'):
                lock_baseline(source,stock,{'output.weight':'f32','norm':'F32'},d/'lock.json',quantizer=Path(sys.executable),native_validate=lambda p:None)
            with self.assertRaisesRegex(ValueError,'allocation'):
                lock_baseline(source,stock,{'output.weight':'Q8_0','norm':'F32'},d/'lock.json',quantizer=Path(sys.executable),native_validate=lambda p:None)
    def test_resource_bounds_before_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);source=d/'source.gguf';stock=d/'stock.gguf';self.fixture(source);self.fixture(stock)
            for limits in ({'timeout':float('nan')},{'workspace_bytes':1},{'threads':0},{'max_output_bytes':float('inf')}):
                with self.assertRaisesRegex(ValueError,'resource bounds'):
                    lock_baseline(source,stock,{'output.weight':'F32','norm':'F32'},d/'lock.json',
                        quantizer=Path(sys.executable),native_validate=lambda p:None,**limits)
            self.assertFalse((d/'lock.json').exists())

    @unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'requires native CPU binaries')
    def test_native_reencode_closure_and_mismatch(self):
        spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test-sinq-groupwise-kl.py'); fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        binaries=Path(os.environ['SINQ_NATIVE_BIN']); suffix='.exe' if os.name=='nt' else ''; quantizer=binaries/('llama-quantize'+suffix)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); source=d/'source.gguf'; fixture.native_fixture(source)
            r=gguf.GGUFReader(source); recipe={t.name:('Q5_0' if len(t.shape)>1 and int(t.shape[0])%32==0 else t.tensor_type.name) for t in r.tensors}; r.data._mmap.close()
            import re
            types=d/'types.txt'; types.write_text('\n'.join('^'+re.escape(n)+'$='+q.lower() for n,q in sorted(recipe.items())))
            stock=d/'stock.gguf'; env=dict(os.environ,OMP_NUM_THREADS='32',CUDA_VISIBLE_DEVICES='')
            result=subprocess.run([str(quantizer),'--pure','--tensor-type-file',str(types),'--max-buffer-size','64',str(source),str(stock),'Q8_0','32'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
            self.assertEqual(result.returncode,0)
            def validate(path):
                result=subprocess.run([str(binaries/('test-qk-sinq-model'+suffix)),'--load-only',str(path)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
                self.assertEqual(result.returncode,0)
            manifest=lock_baseline(source,stock,recipe,d/'lock.json',quantizer=quantizer,native_validate=validate)
            loaded,path=load_baseline(d/'lock.json',source); self.assertEqual(loaded,manifest); self.assertEqual(path,stock.resolve())
            self.assertEqual(manifest['verification'],'native-reencode-exact')
            # Verify real importance-aware full stock allocation as well.
            from tools.sinq.research.resources import file_digest
            importance=d/'imatrix.gguf';writer=gguf.GGUFWriter(importance,'imatrix')
            writer.add_array('imatrix.datasets',['synthetic-baseline-verification'])
            writer.add_uint32('imatrix.chunk_count',1);writer.add_uint32('imatrix.chunk_size',4)
            original=gguf.GGUFReader(source)
            for tensor in original.tensors:
                if len(tensor.shape)==2:
                    k=int(tensor.shape[0]);moments=(1+np.arange(k,dtype=np.float32)%7)**2
                    writer.add_tensor(tensor.name+'.in_sum2',moments*4)
                    writer.add_tensor(tensor.name+'.counts',np.array([4],dtype='<f4'))
            original.data._mmap.close()
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            im_recipe={**recipe,'blk.0.ffn_up.weight':'IQ2_XXS'}
            im_types=d/'im-types.txt';im_types.write_text('\n'.join('^'+re.escape(n)+'$='+q.lower() for n,q in sorted(im_recipe.items())))
            im_stock=d/'im-stock.gguf'
            result=subprocess.run([str(quantizer),'--pure','--tensor-type-file',str(im_types),'--imatrix',str(importance),
                '--max-buffer-size','64',str(source),str(im_stock),'Q8_0','32'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
            self.assertEqual(result.returncode,0)
            im_lock=lock_baseline(source,im_stock,im_recipe,d/'im-lock.json',quantizer=quantizer,native_validate=validate,imatrix=importance)
            self.assertEqual(im_lock['imatrix_sha256'],file_digest(importance))
            self.assertEqual(load_baseline(d/'im-lock.json',source)[0],im_lock)
            with self.assertRaisesRegex(ValueError,'native reencode'):
                lock_baseline(source,im_stock,im_recipe,d/'missing-imatrix.json',quantizer=quantizer,native_validate=validate)
            self.assertFalse((d/'missing-imatrix.json').exists())
            import json
            from tools.sinq.research.resources import digest
            corrupt=dict(manifest); corrupt['source_sha256']='0'*64
            (d/'corrupt.json').write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError,'checksum'): load_baseline(d/'corrupt.json',source)
            corrupt=dict(manifest); corrupt['recipe']=dict(recipe); corrupt['recipe']['token_embd.weight']='F32'
            corrupt['recipe_sha256']=digest(corrupt['recipe']); corrupt['checksum']=digest({k:v for k,v in corrupt.items() if k!='checksum'})
            (d/'allocation.json').write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError,'allocation'): load_baseline(d/'allocation.json',source)
            # Different floating checkpoint cannot reuse an otherwise valid lock.
            changed=d/'changed.gguf'; self.fixture(changed,2)
            with self.assertRaises(ValueError): load_baseline(d/'lock.json',changed)
            with open(stock,'r+b') as f: f.seek(-1,2); b=f.read(1); f.seek(-1,2); f.write(bytes([b[0]^1]))
            with self.assertRaises(ValueError): load_baseline(d/'lock.json',source)
            with self.assertRaises(ValueError): lock_baseline(source,stock,recipe,d/'bad.json',quantizer=quantizer,native_validate=validate)
            self.assertFalse((d/'bad.json').exists())

if __name__=='__main__': unittest.main()
