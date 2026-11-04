"""Execution policy, placement and measured-resource contract regressions."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.sinq.research.schemas import validate_execution_config
from tools.sinq.research.execution_backend import resolve_policy
from tools.sinq.research.device_placement import place_reference
from tools.sinq.research.memory_accounting import ResourceMonitor, ResourceLimitError

class ExecutionTests(unittest.TestCase):
    def test_schema(self):
        for c in ({'device':'cuda:-1'},{'cuda_devices':['cuda:0','cuda:0']},
                  {'device':'cpu','cuda_devices':['cuda:0']},{'reference_dtype':'float32'},
                  {'memory_policy':{'reserve_bytes':-1}}):
            with self.subTest(c=c),self.assertRaises(ValueError):validate_execution_config(c)
        for c in ({'reference_dtype':'fp32'},{'device':'cuda:0'}, {'memory_policy':{}}):
            with self.assertRaises(ValueError):validate_execution_config(c,local=True)
        self.assertEqual(validate_execution_config({'device':'cpu'},local=True)['device'],'cpu')
    def test_cpu_identity(self):
        p=resolve_policy({'budgets':{'workspace_bytes':1<<40}})
        self.assertEqual(p.dtype,torch.float32);self.assertEqual(str(p.input_device),'cpu')
        json.dumps(p.identity())
        q=resolve_policy({'reference_dtype':'bf16','budgets':{'workspace_bytes':1<<40}})
        self.assertNotEqual(p.identity(),q.identity())
    def test_cpu_no_cuda_and_workspace_is_not_residency(self):
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU touched CUDA')), patch.object(torch.cuda, 'is_initialized', side_effect=AssertionError('CPU touched CUDA')):
            p=resolve_policy({'budgets':{'workspace_bytes':128<<20}})
            self.assertGreater(p.memory_policy['host_limit_bytes'],128<<20)
            self.assertIn('host_limit_source',p.memory_policy)
            ResourceMonitor(p).check()
        c={'memory_policy':{'host_limit_bytes':1<<40}}
        with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':16:8'}):a=resolve_policy(c).identity()
        with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):b=resolve_policy(c).identity()
        self.assertEqual(a,b)

    def test_preserve_buffers_and_tied_storage(self):
        class Text(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed_tokens=torch.nn.Embedding(8,4,dtype=torch.bfloat16)
                self.layers=torch.nn.ModuleList([torch.nn.Linear(4,4,dtype=torch.bfloat16)])
                self.norm=torch.nn.LayerNorm(4,dtype=torch.bfloat16)
                self.register_buffer('frequency',torch.tensor([1.003],dtype=torch.float32),persistent=False)
        for device in ('cpu','cuda:0'):
            if device!='cpu' and not torch.cuda.is_available():continue
            t=Text();h=torch.nn.Linear(4,8,bias=False,dtype=torch.bfloat16);h.weight=t.embed_tokens.weight
            before=t.frequency.clone()
            p=resolve_policy({'device':device,'reference_dtype':'bf16'})
            place_reference(t,h,p)
            self.assertEqual(t.frequency.dtype,torch.float32)
            self.assertTrue(torch.equal(t.frequency.cpu(),before))
            self.assertIs(h.weight,t.embed_tokens.weight)
            self.assertEqual(h.weight.data_ptr(),t.embed_tokens.weight.data_ptr())

    def test_bounded_phase_records(self):
        p=resolve_policy({});m=ResourceMonitor(p)
        with m.phase('many'):
            for _ in range(1000):m.check()
        self.assertEqual(len(m.records),2)
        self.assertGreater(m.records[-1]['phase_elapsed_seconds'],0)
        self.assertGreater(m.records[-1]['phase_peak_current_host_bytes'],0)
        for _ in range(150):
            with m.phase('bounded'):pass
        self.assertLessEqual(len(m.records),256)

    def test_cuda_unavailable(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'CUDA'):resolve_policy({'device':'cuda:0'})
    def test_cuda_capabilities_and_indices(self):
        import os
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}), patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'device_count',return_value=1):
            with self.assertRaisesRegex(ValueError,'does not exist'):resolve_policy({'device':'cuda:2'})
            from contextlib import nullcontext
            with patch.object(torch.cuda,'device',return_value=nullcontext()), patch.object(torch.cuda,'is_bf16_supported',return_value=False):
                with self.assertRaisesRegex(ValueError,'BF16'):resolve_policy({'device':'cuda:0','reference_dtype':'bf16'})

    def test_monitor(self):
        p=resolve_policy({'memory_policy':{'host_limit_bytes':1,'reserve_bytes':0}})
        m=ResourceMonitor(p)
        with self.assertRaises(ResourceLimitError) as caught:m.check()
        self.assertEqual(caught.exception.diagnostic['code'],'RESOURCE_LIMIT')
        p=resolve_policy({'budgets':{'workspace_bytes':1<<40}});m=ResourceMonitor(p)
        with m.phase('tiny'):time.sleep(.03)
        self.assertTrue(m.records);self.assertGreater(m.records[-1]['host']['current_bytes'],0)
    def test_poll_failure(self):
        p=resolve_policy({'budgets':{'workspace_bytes':1<<40}});m=ResourceMonitor(p)
        with patch.object(m,'_sample',return_value={'host':{'current_bytes':1<<41},'cuda':{}}):
            with self.assertRaises(ResourceLimitError):
                with m.phase('over'):time.sleep(.1)
    def test_exit_poll_failure_is_not_lost(self):
        p=resolve_policy({});monitor=ResourceMonitor(p)
        class LateFailure:
            def __init__(self,**kwargs):pass
            def start(self):pass
            def join(self):
                monitor._failure=ResourceLimitError({'code':'RESOURCE_LIMIT','reason':'late poll'})
        with patch('tools.sinq.research.memory_accounting.threading.Thread',LateFailure):
            with self.assertRaisesRegex(ResourceLimitError,'late poll'):
                with monitor.phase('final'):pass

    def test_cuda_allocator_cap_restored(self):
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())]
        p=resolve_policy({'cuda_devices':devices})
        before={d:torch.cuda.get_per_process_memory_fraction(d) for d in p.devices}
        with ResourceMonitor(p).phase('allocator-cap'):
            for d in p.devices:self.assertLess(torch.cuda.get_per_process_memory_fraction(d),before[d])
        for d in p.devices:self.assertEqual(torch.cuda.get_per_process_memory_fraction(d),before[d])

    def test_placement_gradient(self):
        class Text(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed_tokens=torch.nn.Embedding(8,4)
                self.layers=torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Linear(4,4)])
                self.norm=torch.nn.LayerNorm(4)
        for devices, dtype in ((d,dt) for d in (['cpu'], ['cuda:0'], ['cuda:0','cuda:1']) for dt in ('fp32','bf16')):
            if devices[0]!='cpu' and torch.cuda.device_count()<len(devices):continue
            c={'budgets':{'workspace_bytes':1<<40},'reference_dtype':dtype}
            if devices[0]!='cpu':c['cuda_devices']=devices
            p=resolve_policy(c);t=Text().to(dtype=p.dtype);h=torch.nn.Linear(4,8).to(dtype=p.dtype)
            placement=place_reference(t,h,p);json.dumps(placement)
            x=t.embed_tokens(torch.tensor([[1,2]],device=p.input_device)).detach().requires_grad_(True)
            y=x
            for layer in t.layers:y=layer(y)
            h(t.norm(y)).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
            if len(devices)>1:self.assertEqual(set(placement.values()),set(devices))

if __name__=='__main__':
    torch.set_num_threads(32)
    unittest.main()
