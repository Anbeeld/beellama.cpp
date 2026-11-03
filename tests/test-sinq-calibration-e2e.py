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
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    from gguf.quants import dequantize
    spec = importlib.util.spec_from_file_location('sinq_exporter_e2e', ROOT / 'tools/sinq/export_activations.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    source = Path(args.source).resolve()
    name = 'blk.0.ffn_up.weight'
    source_reader = GGUFReader(str(source))
    original = next(t for t in source_reader.tensors if t.name == name)
    w = exporter.floating_tensor(original).copy()
    # Linear moments cover every matrix except the embedding lookup, whose
    # native importance-aware codec uses its explicit uniform-weight fallback.
    importance_dims = {t.name: int(t.shape[0]) for t in source_reader.tensors
                       if len(t.shape) == 2 and t.name != 'token_embd.weight'}
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

        imatrix_path = d / 'imatrix.gguf'
        imatrix_dataset = 'synthetic-real-linear-fixture'
        writer = GGUFWriter(imatrix_path, 'imatrix')
        writer.add_array('imatrix.datasets', [imatrix_dataset])
        writer.add_uint32('imatrix.chunk_count', 1)
        writer.add_uint32('imatrix.chunk_size', count)
        for tensor_name, input_dim in importance_dims.items():
            # Native GGUF sums/counts normalize to nonuniform positive moments.
            moments = (1 + np.arange(input_dim, dtype=np.float32) % 7) ** 2
            writer.add_tensor(tensor_name + '.in_sum2', moments * count)
            writer.add_tensor(tensor_name + '.counts', np.array([count], dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        def run(binary, output, *, calibration=None, threads=1, width=64, strength=1, diagnostic=None,
                success=True, overrides=(), codec='Q3_K', imatrix=None):
            cmd = [str(binary), '--pure', '--sinq-precondition', 'sidecar', '--sinq-tensor-list', str(selected),
                   '--sinq-strip-width', str(width), '--sinq-strength', str(strength)]
            for override in overrides:
                cmd += ['--override-kv', override]
            if calibration is not None:
                cmd += ['--sinq-calibration', str(calibration)]
            if diagnostic is not None:
                cmd += ['--sinq-diagnostics-json', str(diagnostic)]
            if imatrix is not None:
                cmd += ['--imatrix', str(imatrix)]
            cmd += [str(source), str(output), codec, str(threads)]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if (result.returncode == 0) != success:
                raise AssertionError('unexpected producer status:\n' + result.stderr[-6000:])
            return result

        run(args.quantize, d /'provenance.gguf', calibration=d, overrides=(
            'quantize.imatrix.file=str:fixture', 'quantize.imatrix.dataset=str:fixture',
            'quantize.imatrix.entries_count=int:1', 'quantize.imatrix.chunks_count=int:1'))
        run(args.quantize, d /'graph-override.gguf', calibration=d, success=False,
            overrides=('qwen35.attention.layer_norm_rms_epsilon=float:0.001',))

        def read(path):
            reader = GGUFReader(str(path))
            mappings.callback(reader.data._mmap.close)
            return reader
        codecs = ('Q3_K', 'IQ3_S', 'IQ4_XS', 'Q5_0', 'Q6_K')
        for codec in codecs:
            for threads in (1, 4, 32):
                run(args.quantize, d / f'{codec}-{threads}.gguf', calibration=d, threads=threads,
                    diagnostic=d / f'{codec}-{threads}.json', codec=codec, imatrix=imatrix_path)
            a, b = (read(d / f'{codec}-{t}.gguf') for t in (1, 4))
            ta, tb = ({t.name: t for t in reader.tensors} for reader in (a, b))
            assert ta.keys() == tb.keys(), codec
            assert all(np.array_equal(t.data, tb[key].data) for key, t in ta.items()), (codec, 'thread-dependent weights/scales')
            assert all(t.tensor_type == tb[key].tensor_type for key, t in ta.items()), codec
            assert ta[name].tensor_type == GGMLQuantizationType[codec], (codec, ta[name].tensor_type)
            assert a.fields['beellama.qk_sinq.version'].contents() == 1
            assert a.fields['beellama.qk_sinq.tensors'].contents() == [name]
            assert set(key for key in a.fields if key.startswith('beellama.qk_sinq.')) == {
                'beellama.qk_sinq.' + key for key in ('version', 'tensors', 'algorithm', 'strip_width', 'iterations', 'producer', 'numerics')}
            assert a.fields['quantize.imatrix.file'].contents() == str(imatrix_path)
            assert a.fields['quantize.imatrix.dataset'].contents() == imatrix_dataset
            assert a.fields['quantize.imatrix.entries_count'].contents() == len(importance_dims)
            assert a.fields['quantize.imatrix.chunks_count'].contents() == 1
            diagnostics = json.loads((d / f'{codec}-1.json').read_text())
            item = next(t for t in diagnostics['tensors'] if t['name'] == name)
            assert item['imatrix_present'] and item['imatrix_transformed'], (codec, item)
            cal = item['calibration']
            assert cal['codec'].upper() == codec, (codec, cal)
            sidecar = ta[name + '.sinq_scale2']
            assert sidecar.tensor_type == GGMLQuantizationType.F16, codec
            u = sidecar.data.reshape(-1).astype(np.float64)
            assert u.shape == (k,) and np.isfinite(u).all() and (u > 0).all(), codec
            dq = dequantize(ta[name].data, ta[name].tensor_type).reshape(n, k).astype(np.float64)
            assert np.isfinite(dq).all(), codec
            # Independent BLAS matmuls, not the producer's compensated-dot implementation.
            actual = np.abs(x.astype(np.float64) @ (dq * u - w.astype(np.float64)).T).mean()
            assert np.isfinite(actual) and np.isclose(actual, cal['selected_l1'], rtol=1e-10, atol=1e-12), (codec, actual, cal)
            assert cal['selected_l1'] <= cal['uncalibrated_l1'], (codec, cal)
            assert np.ptp(cal['candidate_l1']) > 1e-10, (codec, 'activation-aware candidates did not affect output loss')
            for threads in (4, 32):
                other = read(d / f'{codec}-{threads}.gguf')
                other_t = {t.name: t for t in other.tensors}
                assert ta.keys() == other_t.keys(), codec
                assert all(np.array_equal(t.data, other_t[key].data) and t.tensor_type == other_t[key].tensor_type
                           for key, t in ta.items()), (codec, threads, 'thread-dependent weights/scales')
                other_diagnostics = json.loads((d / f'{codec}-{threads}.json').read_text())
                # Scratch ownership changes the reported workspace, not any loss,
                # candidate, scale, encoding, or other diagnostic value.
                for reference_item, other_item in zip(diagnostics['tensors'], other_diagnostics['tensors']):
                    if 'calibration' in reference_item:
                        reference_workspace = reference_item['calibration']['workspace_bytes']
                        other_workspace = other_item['calibration'].pop('workspace_bytes')
                        assert reference_workspace <= other_workspace <= 64 * 1024 * 1024
                        other_item['calibration']['workspace_bytes'] = reference_workspace
                assert diagnostics == other_diagnostics, (codec, threads, 'nondeterministic numerical diagnostics')
            if args.baseline:
                baseline_file, baseline_diag = d / f'{codec}-baseline.gguf', d / f'{codec}-baseline.json'
                # Compare full diagnostics at the same thread count: workspace
                # accounting intentionally includes one error buffer per worker.
                run(args.baseline, baseline_file, calibration=d, threads=1, diagnostic=baseline_diag,
                    codec=codec, imatrix=imatrix_path)
                baseline_t = {t.name: t for t in read(baseline_file).tensors}
                assert ta.keys() == baseline_t.keys(), codec
                assert all(np.array_equal(t.data, baseline_t[key].data) and t.tensor_type == baseline_t[key].tensor_type
                           for key, t in ta.items()), (codec, 'frozen A-SINQ baseline payload changed')
                assert diagnostics == json.loads(baseline_diag.read_text()), (codec, 'serial baseline diagnostics changed')
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
            # Unconfigured stock CLI behavior, including its normal allocation
            # heuristics, must preserve every tensor payload and type too.
            for codec in codecs:
                paths=[]
                for label,binary in [('old',args.baseline),('new',args.quantize)]:
                    path=d / f'stock-{label}-{codec}.gguf'
                    result=subprocess.run([str(binary),str(source),str(path),codec,'24'],capture_output=True,timeout=120)
                    assert result.returncode==0, (codec,'stock baseline command failed')
                    paths.append(path)
                old,new=(read(path) for path in paths)
                assert not any(key.startswith('beellama.qk_sinq.') for key in new.fields)
                old_t,new_t=({t.name:t for t in reader.tensors} for reader in (old,new))
                assert old_t.keys()==new_t.keys()
                assert all(t.tensor_type==new_t[name].tensor_type and np.array_equal(t.data,new_t[name].data)
                           for name,t in old_t.items()), (codec,'unconfigured stock path changed bytes')
            for width in (64,128):
                for strength in (1,.5):
                    old, new = d /f'old-{width}-{strength}.gguf', d /f'ordinary-{width}-{strength}.gguf'
                    run(args.baseline,old,width=width,strength=strength)
                    run(args.quantize,new,width=width,strength=strength)
                    old_t, new_t = ({t.name:t for t in read(path).tensors} for path in (old,new))
                    assert old_t.keys()==new_t.keys()
                    assert all(np.array_equal(t.data,new_t[key].data) for key,t in old_t.items()), 'ordinary path changed bytes'
        sys.stdout.write('PASS: Q3_K/IQ3_S/IQ4_XS/Q5_0/Q6_K real Linear/native imatrix, actual qtypes, independent output L1, '
                         'thread parity, finite F16/version1, rejection/input protection'
                         + (', frozen baseline64/128 payload parity' if args.baseline else '') + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
