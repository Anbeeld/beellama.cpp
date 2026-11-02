#!/usr/bin/env python3
"""Tiny real Linear capture -> strict native producer -> existing sidecar loader format."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gguf-py'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--quantize', required=True)
    p.add_argument('--baseline')
    args = p.parse_args()
    if any(importlib.util.find_spec(x) is None for x in ('torch', 'numpy')):
        sys.stdout.write('SKIP: optional Torch/NumPy unavailable, no end-to-end runtime coverage\n')
        return 77
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    spec = importlib.util.spec_from_file_location('sinq_exporter_e2e', ROOT / 'tools/sinq/export_activations.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    source = Path(args.source).resolve()
    name = 'blk.0.ffn_up.weight'
    source_reader = GGUFReader(str(source))
    original = next(t for t in source_reader.tensors if t.name == name)
    w = exporter.floating_tensor(original).copy()
    source_reader.data._mmap.close()
    n, k = w.shape
    with tempfile.TemporaryDirectory(prefix='sinq-native-e2e-') as directory, ExitStack() as mappings:
        d = Path(directory)
        selected = d / 'selected.txt'
        selected.write_text(name + '\n', encoding='utf-8')
        linear = torch.nn.Linear(k, n, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(w))
        # Actual pre-inputs, deliberately correlated with nonuniform magnitudes.
        j = np.arange(k, dtype=np.float32)
        x = np.stack([np.sin(j *.13), np.cos(j *.13), np.sin(j *.13) +np.cos(j *.13), np.sin(j *.13) -np.cos(j *.13)])
        x = (x * (1 +j % 7)).astype(np.float32)
        capture = exporter.ReservoirCapture(linear, name, d / 'samples.f32', max_rows=4, seed=3)
        with torch.inference_mode():
            prediction = linear(torch.from_numpy(x)).numpy()
        assert np.allclose(prediction, x @ w.T, rtol=2e-5, atol=2e-5)
        count = capture.finish()
        token_hash = hashlib.sha256(np.arange(4, dtype='<u4').tobytes()).hexdigest()
        record = dict(name=name, site='qwen3_5.layers.0.ffn_up.input', reference_module='synthetic_linear',
                      channel_layout='gguf-input', input_dim=k, output_dim=n, count=count, dtype='<f4',
                      order='row-major', data_file='samples.f32', sha256=exporter.sha256(d /'samples.f32'))
        manifest = dict(schema_version=1, architecture='qwen3_5', source_sha256=exporter.sha256(source),
                        source_revision='synthetic-real-linear-fixture', sampling=dict(contract='valid-token-prefill-reservoir-v1',
                        seed=3, max_rows=4, corpus_sha256=token_hash, token_sha256=token_hash), tensors=[record])
        manifest_path = d / 'activations.json'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

        def run(binary, output, *, calibration=None, threads=1, width=64, strength=1, diagnostic=None, success=True, overrides=()):
            cmd = [str(binary), '--pure', '--sinq-precondition', 'sidecar', '--sinq-tensor-list', str(selected),
                   '--sinq-strip-width', str(width), '--sinq-strength', str(strength)]
            for override in overrides:
                cmd += ['--override-kv', override]
            if calibration is not None:
                cmd += ['--sinq-calibration', str(calibration)]
            if diagnostic is not None:
                cmd += ['--sinq-diagnostics-json', str(diagnostic)]
            cmd += [str(source), str(output), 'Q3_K', str(threads)]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if (result.returncode == 0) != success:
                raise AssertionError('unexpected producer status:\n' + result.stderr[-6000:])
            return result

        run(args.quantize, d /'provenance.gguf', calibration=d, overrides=(
            'quantize.imatrix.file=str:fixture', 'quantize.imatrix.dataset=str:fixture',
            'quantize.imatrix.entries_count=int:1', 'quantize.imatrix.chunks_count=int:1'))
        run(args.quantize, d /'graph-override.gguf', calibration=d, success=False,
            overrides=('qwen35.attention.layer_norm_rms_epsilon=float:0.001',))
        for threads in (1, 4):
            run(args.quantize, d /f'a{threads}.gguf', calibration=d, threads=threads, diagnostic=d /f'a{threads}.json')

        def read(path):
            reader = GGUFReader(str(path))
            mappings.callback(reader.data._mmap.close)
            return reader
        a, b = (read(d /f'a{t}.gguf') for t in (1, 4))
        ta, tb = ({t.name: t for t in reader.tensors} for reader in (a, b))
        assert ta.keys() == tb.keys()
        assert all(np.array_equal(t.data, tb[key].data) for key, t in ta.items()), 'thread-dependent weights/scales'
        assert a.fields['beellama.qk_sinq.version'].contents() == 1
        assert a.fields['beellama.qk_sinq.tensors'].contents() == [name]
        assert set(key for key in a.fields if key.startswith('beellama.qk_sinq.')) == {
            'beellama.qk_sinq.' +key for key in ('version','tensors','algorithm','strip_width','iterations','producer','numerics')}
        diagnostics = json.loads((d /'a1.json').read_text())
        item = next(t for t in diagnostics['tensors'] if t['name'] == name)
        cal = item['calibration']
        u = ta[name +'.sinq_scale2'].data.reshape(-1).astype(np.float64)
        dq = dequantize(ta[name].data, ta[name].tensor_type).reshape(n,k).astype(np.float64)
        # Independent BLAS matmuls, not the producer's compensated-dot implementation.
        actual = np.abs(x.astype(np.float64) @ (dq *u -w.astype(np.float64)).T).mean()
        assert np.isclose(actual, cal['selected_l1'], rtol=1e-10, atol=1e-12), (actual, cal)
        assert cal['selected_l1'] <= cal['uncalibrated_l1']
        assert np.ptp(cal['candidate_l1']) > 1e-10, 'activation-aware candidates did not affect output loss'
        assert json.loads((d /'a1.json').read_text()) == json.loads((d /'a4.json').read_text()), 'nondeterministic diagnostics'
        for key, value in [('input_dim', k +1), ('name','blk.0.ffn_down.weight')]:
            bad = json.loads(json.dumps(manifest))
            bad['tensors'][0][key] = value
            if key == 'name':
                bad['tensors'][0]['site'] = 'qwen3_5.layers.0.ffn_down.input'
                bad['tensors'][0]['input_dim'] = k
            manifest_path.write_text(json.dumps(bad), encoding='utf-8')
            run(args.quantize, d /'invalid.gguf', calibration=d, success=False)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        for protected in (manifest_path, d /'samples.f32', selected, source):
            before = exporter.sha256(protected)
            run(args.quantize, protected, calibration=d, success=False)
            assert exporter.sha256(protected) == before, 'input was clobbered'
        if args.baseline:
            for width in (64,128):
                for strength in (1,.5):
                    old, new = d /f'old-{width}-{strength}.gguf', d /f'ordinary-{width}-{strength}.gguf'
                    run(args.baseline,old,width=width,strength=strength)
                    run(args.quantize,new,width=width,strength=strength)
                    old_t, new_t = ({t.name:t for t in read(path).tensors} for path in (old,new))
                    assert old_t.keys()==new_t.keys()
                    assert all(np.array_equal(t.data,new_t[key].data) for key,t in old_t.items()), 'ordinary path changed bytes'
        sys.stdout.write('PASS: real Linear export/native import, independent output L1, thread parity, version1, rejection/input protection'
                         + (', frozen baseline64/128 payload parity' if args.baseline else '') + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
