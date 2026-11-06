"""Deterministic source-chunk split *before* independent reservoir sampling.

Consumes a fully source-verified Qwen35Adapter and explicit token IDs. Existing
activation-v1 manifests and ReservoirCapture are reused unchanged. No tokenizer,
checkpoint or dataset is downloaded; callers own their locked source token stream.
"""
import hashlib
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tools.sinq import export_activations as exporter
from .resources import atomic_write,canonical_json
from .schemas import validate_pairing


def capture_splits(adapter,tokens,names,output,*,context=512,rows=128,seed=0,
                   validation_chunks=1,max_tokens=4096,max_output_bytes=64<<20):
    adapter.assert_reference()
    exporter.bounds(seed,rows)
    if type(context) is not int or not 1<=context<=4096 or tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.dtype!=torch.long or tokens.device.type!='cpu' or not 1<tokens.numel()<=max_tokens:
        raise ValueError('split capture requires bounded CPU [1,tokens] integer source IDs')
    if (tokens<0).any() or (tokens>=adapter.config.vocab_size).any():
        raise ValueError('split capture token outside vocabulary')
    if not names or len(set(names))!=len(names):raise ValueError('unique selected tensor names required')
    chunks=[(start,min(start+context,tokens.numel())) for start in range(0,tokens.numel(),context)]
    if type(validation_chunks) is not int or not 0<validation_chunks<len(chunks):
        raise ValueError('split must allocate nonempty disjoint train and validation chunks')
    corpus=hashlib.sha256(tokens.numpy().astype('<u4').tobytes()).hexdigest()
    # Ranking is keyed by original chunk coordinates, not iteration order or an
    # RNG stream whose consumption could change after filtering selected sites.
    ranked=sorted(chunks,key=lambda chunk:hashlib.sha256(seed.to_bytes(4,'little')+bytes.fromhex(corpus)+
        chunk[0].to_bytes(8,'little')+chunk[1].to_bytes(8,'little')).digest())
    validation=set(ranked[:validation_chunks])
    split_chunks={'train':[c for c in chunks if c not in validation],
                  'validation':[c for c in chunks if c in validation]}
    modules={};max_bytes=0
    for name in sorted(names):
        path,site,role=exporter.module_for(name)
        module=adapter.reference.get_submodule(path)
        if type(module) is not torch.nn.Linear:raise ValueError('selected capture site must be an unfused Linear')
        modules[name]=(path,site,role,module)
        max_bytes+=rows*module.in_features*4*2+8192
    provenance_bytes=2*len(canonical_json(adapter.identity))+tokens.numel()*160+len(chunks)*512+16384
    if max_bytes+provenance_bytes>max_output_bytes:raise ValueError('activation split disk budget exceeded before capture')
    output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('immutable split output already exists')
    output.mkdir(parents=True,exist_ok=False)
    manifests={};pairing=dict(schema_version=2,source_sha256=adapter.identity['source_gguf_sha256'],split_method='split_before_reservoir')
    for split,coordinates in split_chunks.items():
        directory=output/split;directory.mkdir()
        captures=[];records=[]
        try:
            for i,(name,(path,site,role,module)) in enumerate(sorted(modules.items())):
                transform=None
                if role=='ssm_out':
                    c=adapter.config
                    transform=partial(exporter.reorder_v,axis=0,key_heads=c.linear_num_key_heads,
                        value_heads=c.linear_num_value_heads,head_dim=c.linear_value_head_dim)
                filename=f'{i:04d}.f32'
                capture=exporter.ReservoirCapture(module,name,directory/filename,rows,seed,transform)
                captures.append(capture)
                records.append(dict(name=name,site=site,reference_module=path,channel_layout='gguf-input',
                    input_dim=module.in_features,output_dim=module.out_features,dtype='<f4',order='row-major',data_file=filename))
            with torch.inference_mode():
                for begin,end in coordinates:
                    # One fresh prefill per locked original chunk: no recurrent
                    # or attention state flows between the two split streams.
                    adapter.forward(tokens[:,begin:end])
            for capture,record in zip(captures,records):
                record['count']=capture.finish();record['sha256']=exporter.sha256(capture.path)
                if capture.seen!=sum(end-begin for begin,end in coordinates):
                    raise ValueError('capture did not execute every selected source token')
        finally:
            for capture in captures:
                capture.handle.remove()
                if not capture.data._mmap.closed:capture.data.flush();capture.data._mmap.close()
        ids=[i for begin,end in coordinates for i in range(begin,end)]
        split_tokens=tokens.numpy().reshape(-1)[ids].astype('<u4')
        token_hash=hashlib.sha256(split_tokens.tobytes()).hexdigest()
        manifest=dict(schema_version=2,architecture='qwen3_5',source_sha256=adapter.identity['source_gguf_sha256'],
            source_revision=adapter.source_id,sampling=dict(contract='valid-token-prefill-reservoir-v1',seed=seed,max_rows=rows,
                corpus_sha256=corpus,token_sha256=token_hash),tensors=records,
            provenance=dict(reference_identity=adapter.identity,context=context,token_count=len(ids),
                split_contract='chunk-split-before-independent-reservoir-v1',row_id_semantics='all eligible original token coordinates; reservoirs are subsets',
                reset='fresh-sequence-cache-none'))
        manifests[split]=manifest
        atomic_write(directory/'activations.json',canonical_json(manifest),max_bytes=max_output_bytes)
        pairing[split]=dict(split_id=split,token_sha256=token_hash,
            chunks=[dict(token_source_sha256=corpus,start=begin,end=end) for begin,end in coordinates],
            row_ids=[f'{corpus}:{i}' for i in ids])
    validate_pairing(pairing,manifests['train'],manifests['validation'])
    adapter.assert_reference()
    atomic_write(output/'pairing.json',canonical_json(pairing),max_bytes=max_output_bytes)
    return pairing
