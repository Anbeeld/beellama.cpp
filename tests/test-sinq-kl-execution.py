"""Native execution policy contracts, without requiring a resident CUDA model."""
import sys, tempfile, unittest, subprocess
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
from tools.sinq.optimizer.kl import NativeKLEvaluator

GPU='GPU-5e87a39a-204c-4054-14a8-5b97ae4f16fd'
PCI='0000:01:00.0'
class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.exe=Path(self.tmp.name)/'perplexity.exe';self.exe.write_bytes(b'fixture')
    def native(self,args,**kwargs):
        if args[0]=='nvidia-smi':
            return subprocess.CompletedProcess(args,0,f'0, {GPU}, NVIDIA RTX 3090, 00000000:01:00.0, 550.0\n'.encode(),b'')
        if '--list-devices' in args:
            return subprocess.CompletedProcess(args,0,b'Available devices:\n  CUDA0: NVIDIA RTX 3090 (24576 MiB, 24000 MiB free)\n',b'')
        if '--no-op-offload' in args:return subprocess.CompletedProcess(args,0,b'',b'offloaded 0/33 layers to GPU\n')
        return subprocess.CompletedProcess(args,0,b'',f'llama_model_load_from_file_impl: using device CUDA0 (NVIDIA RTX 3090) ({PCI}) - 24000 MiB free\nload_tensors: offloaded 33/33 layers to GPU\n'.encode())
    def cuda(self,**kw):
        return NativeKLEvaluator(self.exe,execution=dict(backend='cuda',cuda_device=0,cache_type_k='bf16',cache_type_v='bf16',flash_attention='on',**kw))
    def test_cpu_still_hides_cuda(self):
        with patch('subprocess.run',side_effect=self.native) as run:
            e=NativeKLEvaluator(self.exe);e._run('model','corpus','ref','tokens')
        args=run.call_args.args[0];self.assertEqual(args[args.index('-ngl')+1],'0')
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(args[args.index('--device')+1],'none')
        self.assertIn('--no-op-offload',args)
        self.assertFalse(e.policy['op_offload'])
    def test_cpu_rejects_leaked_gpu_execution(self):
        e=NativeKLEvaluator(self.exe)
        for log in (b'offloaded 1/33 layers to GPU',b'using device CUDA0 (RTX)',b'CUDA0 compute buffer size = 100 MiB'):
            with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,b'',log)),self.assertRaisesRegex(RuntimeError,'CPU'):
                e._run('m','c','r','t')

    def test_cuda_command_selection_and_evidence(self):
        with patch('subprocess.run',side_effect=self.native) as run:
            e=self.cuda();e._run('model','corpus','ref','tokens')
        args=run.call_args.args[0];env=run.call_args.kwargs['env']
        for flag,value in [('-ngl','999'),('--device','CUDA0'),('--cache-type-k','bf16'),('--cache-type-v','bf16'),('--flash-attn','on')]:
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],GPU)
        self.assertNotIn('--no-op-offload',args)
        self.assertEqual(e.last_execution['offloaded_layers'],33)
        self.assertEqual(e.last_execution['gpu']['uuid'],GPU)
    def test_selected_second_gpu_and_invalid_index(self):
        gpu1='GPU-aaaaaaaa-204c-4054-14a8-5b97ae4f16fd'
        def native(args,**kwargs):
            if args[0]=='nvidia-smi':
                return subprocess.CompletedProcess(args,0,f'0, {GPU}, NVIDIA RTX 3090, 00000000:01:00.0, 550.0\n1, {gpu1}, NVIDIA RTX 3090, 00000000:03:00.0, 550.0\n'.encode(),b'')
            r=self.native(args,**kwargs)
            r.stderr=r.stderr.replace(b'0000:01:00.0',b'0000:03:00.0')
            return r
        with patch('subprocess.run',side_effect=native) as run:
            e=NativeKLEvaluator(self.exe,execution=dict(backend='cuda',cuda_device=1))
            e._run('m','c','r','t')
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],gpu1)
            self.assertEqual(e.last_execution['gpu']['pci_bus_id'],'0000:03:00.0')
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                NativeKLEvaluator(self.exe,execution=dict(backend='cuda',cuda_device=2))

    def test_reject_fallback_partial_and_wrong_physical_gpu(self):
        for log in (b'',b'offloaded 0/33 layers to GPU',b'offloaded 32/33 layers to GPU',b'using device CUDA0 (RTX) (0000:02:00.0)\noffloaded 33/33 layers to GPU'):
            with self.subTest(log=log),patch('subprocess.run',side_effect=self.native):e=self.cuda()
            with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,b'',log)),self.assertRaisesRegex(RuntimeError,'CUDA'):
                e._run('model','corpus','ref','tokens')
    def test_no_cuda_support_and_process_capacity_failure(self):
        with patch('subprocess.run',side_effect=lambda args,**kw:self.native(args,**kw) if args[0]=='nvidia-smi' else subprocess.CompletedProcess(args,0,b'Available devices:',b'')):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):self.cuda()
        with patch('subprocess.run',side_effect=self.native):e=self.cuda()
        with patch('subprocess.run',return_value=subprocess.CompletedProcess([],1,b'',b'out of memory')),self.assertRaises(RuntimeError):e._run('m','c','r','t')
    def test_invalid_policy(self):
        for p in ([],{'backend':'other'},{'backend':'cpu','cuda_device':0},{'backend':'cuda'},{'backend':'cuda','cuda_device':True},{'cache_type_k':'garbage'},{'flash_attention':'yes'},{'gpu_layers':0},{'cublas_compute_type':'garbage'}):
            with self.subTest(p=p),self.assertRaises(ValueError):NativeKLEvaluator(self.exe,execution=p)
    def test_policy_identity_and_executable_mutation(self):
        with patch('subprocess.run',side_effect=self.native):
            e=self.cuda();policies=[e.policy,NativeKLEvaluator(self.exe).policy]
            for field,value in [('cache_type_k','f16'),('flash_attention','off'),('cublas_compute_type','f32'),('mmq_precision','q8')]:
                p=dict(backend='cuda',cuda_device=0,cache_type_k='bf16',cache_type_v='bf16',flash_attention='on');p[field]=value
                policies.append(NativeKLEvaluator(self.exe,execution=p).policy)
        self.assertEqual(len({str(p) for p in policies}),len(policies))
        self.exe.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'executable|runtime'):e._run('m','c','r','t')

if __name__=='__main__':unittest.main()
