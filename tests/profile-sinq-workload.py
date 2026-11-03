#!/usr/bin/env python3
"""Bounded synthetic multi-layer engineering profile, never a quality/recipe study."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
os.environ.update(OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from tools.sinq.research.resources import canonical_json,file_digest,peak_rss_bytes


def process_io():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[(name,ctypes.c_ulonglong) for name in ('read_operations','write_operations','other_operations','read_bytes','write_bytes','other_bytes')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.GetProcessIoCounters.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters)]
        value=Counters()
        if not kernel.GetProcessIoCounters(kernel.GetCurrentProcess(),ctypes.byref(value)):raise ctypes.WinError(ctypes.get_last_error())
        return {name:int(getattr(value,name)) for name,_ in value._fields_}
    path=Path('/proc/self/io')
    return {name:int(value) for line in path.read_text().splitlines() for name,value in [line.split(':')]} if path.is_file() else None


def current_rss_bytes():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),
                *[(name,ctypes.c_size_t) for name in ('peak','working','paged_peak','paged','nonpaged_peak','nonpaged','pagefile','pagefile_peak')]]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        value=Counters();value.cb=ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):raise ctypes.WinError(ctypes.get_last_error())
        return int(value.working)
    path=Path('/proc/self/statm')
    return int(path.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if path.is_file() else None


def fixture(directory,hidden,intermediate,layers):
    import torch,gguf
    from transformers import Qwen3_5TextConfig,Qwen3_5TextModel
    from conversion.qwen import Qwen3_5TextModel as Converter
    torch.manual_seed(41721);torch.set_num_threads(24)
    config=Qwen3_5TextConfig(vocab_size=512,hidden_size=hidden,intermediate_size=intermediate,
        num_hidden_layers=layers,num_attention_heads=hidden//128,num_key_value_heads=max(1,hidden//512),head_dim=128,
        linear_num_key_heads=max(2,hidden//512),linear_num_value_heads=max(4,hidden//256),linear_key_head_dim=128,
        linear_value_head_dim=128,linear_conv_kernel_dim=4,max_position_embeddings=128,
        layer_types=['linear_attention' if i%2==0 else 'full_attention' for i in range(layers)],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[11,11,10],'mrope_interleaved':True},tie_word_embeddings=True)
    config._attn_implementation='eager'
    model=Qwen3_5TextModel(config).float().eval()
    parameters=sum(p.numel() for p in model.parameters())
    model.save_pretrained(directory,safe_serialization=True)
    hparams=config.to_dict();hparams['architectures']=['Qwen3_5TextModel'];hparams['full_attention_interval']=2
    converter=Converter(directory,gguf.LlamaFileType.ALL_F32,directory/'source.gguf',eager=True,hparams=hparams)
    converter.set_gguf_parameters();writer=converter.gguf_writer
    writer.add_tokenizer_model('llama');writer.add_token_list([f'token{i}' for i in range(512)])
    writer.add_token_scores([0.]*512);writer.add_token_types([1]*512);writer.add_bos_token_id(1);writer.add_eos_token_id(2)
    converter.prepare_tensors();writer.write_header_to_file(directory/'source.gguf');writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    return parameters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--hidden-size',type=int,choices=(512,1024,2048),default=2048)
    parser.add_argument('--intermediate-size',type=int,default=5632)
    parser.add_argument('--layers',type=int,choices=(4,6),default=4)
    parser.add_argument('--tokens',type=int,choices=(16,32),default=32)
    parser.add_argument('--probes',type=int,choices=(1,2),default=2)
    args=parser.parse_args();output=Path(args.output).resolve();binaries=Path(args.bin).resolve();suffix='.exe' if os.name=='nt' else ''
    bridge=binaries/('llama-sinq-native'+suffix);loader=binaries/('llama-sinq-model'+suffix)
    if not bridge.is_file() or not loader.is_file():raise ValueError('explicit built native binaries required')
    if args.intermediate_size<512 or args.intermediate_size>5632 or args.intermediate_size%256:raise ValueError('bounded strip-aligned intermediate size required')
    if output.exists():raise FileExistsError('profile output must be new')
    if shutil.disk_usage(output.parent).free<8<<30:raise ValueError('profile requires 8 GiB free temporary disk')
    output.mkdir(parents=True)
    import numpy as np,torch,gguf
    from tools.sinq.research import native_codec
    from tools.sinq.research.cache import CandidateStore
    from tools.sinq.research.generate import source_identity
    from tools.sinq.research.cli import _advanced_reference,_validator
    from tools.sinq.research.capture import capture_block
    from tools.sinq.research.blockwise import block_delta
    from tools.sinq.research.fisher import qwen_probes
    from tools.sinq.research.hybrid import assemble
    torch.set_num_threads(24);torch.use_deterministic_algorithms(True)
    phases=[]
    @contextmanager
    def phase(name,**details):
        before=process_io();rss=current_rss_bytes();start=time.perf_counter();status='FAIL'
        try:
            yield
            status='PASS'
        finally:
            after=process_io()
            phases.append(dict(name=name,status=status,seconds=time.perf_counter()-start,parent_peak_rss_bytes=peak_rss_bytes(),
                parent_rss_before=rss,parent_rss_after=current_rss_bytes(),
                parent_io_delta={key:after[key]-before[key] for key in after} if before is not None and after is not None else None,**details))
            (output/'phases.json').write_bytes(canonical_json(phases))
    with tempfile.TemporaryDirectory(prefix='workload-',dir=output) as temporary:
        d=Path(temporary);checkpoint=d/'checkpoint';checkpoint.mkdir()
        with phase('construct_checkpoint_and_convert'):
            parameter_count=fixture(checkpoint,args.hidden_size,args.intermediate_size,args.layers)
        gc.collect();source=checkpoint/'source.gguf';source_bytes=source.stat().st_size
        with phase('hash_complete_source',input_bytes=source_bytes):source_sha=file_digest(source)
        with phase('load_verify_f32_reference',reference_parameter_bytes=parameter_count*4):
            adapter=_advanced_reference(checkpoint,source,8<<30,24)
        tokens=torch.tensor([[1]+[3+i%500 for i in range(args.tokens-1)]],dtype=torch.long)
        names=['blk.0.ffn_up.weight','blk.2.ffn_up.weight'];assignment={}
        store=CandidateStore(d/'cache',max_bytes=256<<20,max_files=100)
        with store.verified_session():
            reader=gguf.GGUFReader(source)
            try:
                for index,name in enumerate(names):
                    tensor=next(t for t in reader.tensors if t.name==name);k,n=map(int,tensor.shape)
                    raw=d/f'weights-{index}.bin';packed=d/f'packed-{index}';request_dir=d/f'request-{index}';request_dir.mkdir()
                    with phase('native_stock_encode_'+str(index),input_bytes=tensor.n_bytes):
                        native_codec.copy_tensor_bytes(tensor,raw,max_bytes=512<<20)
                        response=native_codec.request(bridge,dict(schema_version=1,operation='stock',weights=str(raw),source_type='f32',rows=n,columns=k,
                            codec='q5_0',threads=24,workspace_bytes=512<<20,max_output_bytes=64<<20,strip_width=64,iterations=16,strength=.75,output=str(packed)),request_dir)
                        provenance=dict(source_sha256=source_sha,tensor_name=name,shape=[n,k],source_qtype='F32',qtype='Q5_0',input_sha256=file_digest(raw),
                            recipe_sha256=hashlib.sha256(b'engineering-only-Q5_0').hexdigest(),imatrix_sha256=None,
                            codec=dict(implementation='ggml',chunk_rows=max(1,(16384+k-1)//k),rows_per_expert=n),
                            producer=dict(binary_sha256=file_digest(bridge),source_sha256=source_identity()),sinq=None,generator={'family':'stock'},activation=None)
                        assignment[name]=store.put_files(provenance,packed/response['payload'])
            finally:reader.data._mmap.close()
            with phase('repeated_integrity_retrieval',retrievals=40):
                for _ in range(20):
                    for key in assignment.values():store.get(key)
            for index,name in enumerate(names):
                layer=index*2
                with phase('capture_block_'+str(layer)):
                    capture=capture_block(adapter,tokens,layer,max_tokens=64,max_capture_bytes=1<<30)
                with phase('fisher_vjp_'+str(layer),requested_vjps=args.probes,output_positions=1):
                    probes=qwen_probes(adapter,capture,probes=args.probes,seed=617,positions=[(0,args.tokens-1)],max_vjps=4,max_cache_bytes=1<<30)
                phases[-1]['actual_vjps']=probes.metadata['actual_vjps']
                with phase('decode_and_fisher_pairing_'+str(layer)):
                    effective=native_codec.decode(bridge,store.get(assignment[name]),workspace_bytes=512<<20)
                    value=probes.score(block_delta(adapter,capture,{name:effective}))
                    if not np.isfinite(value) or value<0:raise ValueError('nonfinite or negative engineering Fisher smoke')
                del probes,capture,effective;gc.collect()
            for index in range(2):
                selected={name:assignment[name] for name in names[:index+1]};model=d/f'hybrid-{index}.gguf'
                with phase('assemble_and_native_validate_'+str(index),source_bytes=source_bytes,assigned_tensors=len(selected)):
                    checksum=assemble(source,model,store,selected,native_validate=_validator(loader),max_output_bytes=2<<30)
                phases[-1].update(output_bytes=model.stat().st_size,output_sha256=checksum)
                with phase('native_forward_'+str(index),tokens=args.tokens):
                    result=subprocess.run([str(loader),'--forward',str(model),*map(str,tokens[0].tolist())],capture_output=True,timeout=180)
                    if result.returncode:
                        (output/'native-forward-failure.log').write_bytes(result.stderr)
                        raise RuntimeError('profile native forward failed: '+str(result.returncode))
                    logits=json.loads(result.stdout)['logits']
                    if len(logits)!=512 or not np.isfinite(logits).all():raise ValueError('profile native forward invalid')
            cache_stats=dict(store.index_stats)
        summary=dict(schema_version=1,status='PASS',purpose='engineering profiling only; no quality A/B or recipe selection',seed=41721,
            synthetic=True,model=dict(architecture='qwen35',layers=args.layers,hidden_size=args.hidden_size,intermediate_size=args.intermediate_size,vocabulary=512,
                parameter_count=parameter_count,reference_parameter_bytes=parameter_count*4,source_bytes=source_bytes,source_sha256=source_sha),
            tokens=args.tokens,batch=1,cpu_threads=24,advanced_device='cpu',cuda_visible_devices='0',probes_per_layer=args.probes,
            phases=phases,cache_index=cache_stats,parent_peak_rss_bytes=peak_rss_bytes(),
            memory_scope='Parent cumulative high-water working set; includes construction, mapped files and reference/VJP; excludes native child peaks.',
            io_scope='OS parent logical I/O counters and explicit artifact sizes; mapped faults and child I/O are not fully accounted; not physical disk throughput.',
            limitations=['random synthetic weights and 512-token vocabulary, not a trained/9B model','no long-context or 9B throughput extrapolation','resident F32 reference, no memory-efficiency redesign','no method ranking, quality metric publication or production defaults'])
    (output/'summary.json').write_bytes(canonical_json(summary));print(json.dumps(summary,sort_keys=True))

if __name__=='__main__':main()
