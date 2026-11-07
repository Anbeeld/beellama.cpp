from __future__ import annotations
import argparse, csv, ctypes, hashlib, json, math, os, re, subprocess, sys, threading, time
from ctypes import wintypes
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT))
from final_control_validation import validate_stock_baseline_control
from tools.sinq.optimizer.resources import file_digest
from tools.sinq.optimizer.trusted_reference import GuardedFile
from contextlib import ExitStack

def main():
    with ExitStack() as session:
        model_guards = {}
        def model_digest(path, expected):
            key = Path(path).absolute()
            if key not in model_guards:
                model_guards[key] = session.enter_context(GuardedFile(key))
            return model_guards[key].verified_sha256(expected)

        def stop_native(proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        parser = argparse.ArgumentParser()
        parser.add_argument('recipe', choices=['T340', 'T425'])
        parser.add_argument('gpu', type=int, choices=[0, 1])
        parser.add_argument('--models-manifest', type=Path)
        parser.add_argument('--output-directory', type=Path)
        parser.add_argument('--execution-lock', type=Path)
        args = parser.parse_args()
        if any((args.models_manifest, args.output_directory, args.execution_lock)) and not all((args.models_manifest, args.output_directory, args.execution_lock)):
            raise ValueError('external frozen models require manifest, output directory and execution lock')
        RECIPE = args.recipe
        GPU_INDEX = args.gpu
        HARDWARE = json.loads((ROOT / 'locks/hardware-build-lock.json').read_text())
        UUIDS = [item.split(',')[0] for item in HARDWARE['gpus']]
        UUID = UUIDS[GPU_INDEX]
        SOURCE = Path(r'C:\models\Qwen3.5-9B-bf16\Qwen3.5-9B-text-BF16.gguf')
        RAW = REPO / 'artifacts/sinq-v050/data/ptb.test.txt'
        LOCK_PATH = ROOT / 'locks/final-ptb-input-lock.json'
        EVAL_LOCK = ROOT / 'locks/final-wikitext-input-lock.json'
        NORMALIZED = ROOT / 'data/ptb-common-file-normalized.txt'
        TOKENS_PATH = ROOT / 'data/ptb-test-native.u32'
        EXEC = REPO / 'build-sinq-clean/bin/llama-perplexity.exe'
        EXPECTED = {
            'source': 'da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af',
            'raw': 'dd65dff31e70846b2a6030a87482edcd5d199130cdcfa1f3dccbb033728deee0',
            'normalized': '91a2fa938d84baddaa05a3d67a5698924847eca8a4e7faa1518f3fd11c3d6d73',
            'targets': 98776, 'tokens': 98777, 'context': 65536,
            'evaluator': '3698fc7d1cf6ff3d9b400a475c670ab820ada9c03f088ae5f748d44e90168ee9'
        }
        INPUT_LOCK = json.loads(LOCK_PATH.read_text())
        if INPUT_LOCK.get('status') != 'PASS_EXACT_PLANNED_PTB_INPUT_AND_NATIVE_TOKEN_STREAM_LOCKED':
            raise ValueError('exact PTB input/token stream lock is not PASS')
        if INPUT_LOCK['raw_source']['sha256'] != EXPECTED['raw'] or INPUT_LOCK['normalized_native_input']['sha256'] != EXPECTED['normalized']:
            raise ValueError('PTB input lock SHA-256 differs from the planned source')
        if INPUT_LOCK['token_stream']['token_count'] != EXPECTED['tokens'] or INPUT_LOCK['token_stream']['scored_targets'] != EXPECTED['targets']:
            raise ValueError('PTB token count differs from the locked target count')
        EVALUATOR_LOCK = json.loads(EVAL_LOCK.read_text())
        if file_digest(SOURCE) != EXPECTED['source'] or file_digest(RAW) != EXPECTED['raw'] or file_digest(NORMALIZED) != EXPECTED['normalized']:
            raise ValueError('PTB source or normalized input changed after locking')
        external_lock = json.loads(args.execution_lock.read_text()) if args.execution_lock else None
        runtime_guard = session.enter_context(GuardedFile(EXEC))
        runtime_sha256 = runtime_guard.verified_sha256(external_lock['executable_sha256'] if external_lock else EXPECTED['evaluator'])
        if external_lock:
            if (external_lock.get('status') != 'FROZEN_BEFORE_HELD_OUT_EVALUATION'
                    or external_lock.get('models_manifest_sha256') != file_digest(args.models_manifest)
                    or external_lock.get('source_sha256') != EXPECTED['source']
                    or external_lock.get('ptb_input_lock_sha256') != file_digest(LOCK_PATH)
                    or external_lock.get('executable_sha256') != runtime_sha256):
                raise ValueError('external PTB execution lock mismatch')
        elif runtime_sha256 != EXPECTED['evaluator'] or runtime_sha256 != EVALUATOR_LOCK.get('evaluator_sha256'):
            raise ValueError('native evaluator does not match the existing locked build')
        ids = np.fromfile(TOKENS_PATH, dtype='<u4')
        if len(ids) != EXPECTED['tokens'] or file_digest(TOKENS_PATH) != INPUT_LOCK['token_stream']['sha256']:
            raise ValueError('PTB native token stream changed or has the wrong count')

        FIXED_AUDIT = json.loads((ROOT / 'report/fixed-model-audit.json').read_text())
        AUTO_AUDIT = json.loads((ROOT / 'report/auto-model-audit.json').read_text())
        FIXED_RUNTIME = json.loads((ROOT / 'report/fixed-model-runtime-validation.json').read_text())
        AUTO_RUNTIME = json.loads((ROOT / 'report/auto-model-runtime-validation.json').read_text())
        if FIXED_AUDIT.get('status') != 'PASS' or len(FIXED_AUDIT.get('records', [])) != 24:
            raise ValueError('fixed model audit is incomplete')
        if AUTO_AUDIT.get('status') != 'PASS' or len(AUTO_AUDIT.get('models', [])) != 10:
            raise ValueError('automatic model audit is incomplete')
        if FIXED_RUNTIME.get('status') != 'PASS' or FIXED_RUNTIME.get('validated_arms') != 24:
            raise ValueError('fixed model runtime validation is incomplete')
        if AUTO_RUNTIME.get('status') != 'PASS' or AUTO_RUNTIME.get('validated_models') != 10:
            raise ValueError('automatic model runtime validation is incomplete')

        OUT = args.output_directory.resolve() / RECIPE if args.output_directory else ROOT / 'evaluation/ptb' / RECIPE
        OUT.mkdir(parents=True, exist_ok=True)
        PROGRESS = OUT / 'progress.json'
        models = []
        if not args.models_manifest:
            for row in FIXED_AUDIT['records']:
                if row['recipe'] == RECIPE:
                    models.append({'label': 'FIXED-' + row['arm'], 'path': ROOT / 'fixed' / RECIPE / row['arm'] / 'model.gguf',
                                   'expected_sha256': row['model_sha256'], 'kind': 'fixed', 'arm': row['arm']})
            for label in ('AUTO-LOCAL-L1', 'AUTO-LOCAL-MSE', 'AUTO-BLOCK-MSE', 'AUTO-FISHER', 'AUTO-GROUP-KLD-BUDGETED'):
                d = ROOT / 'selected' / RECIPE / label
                selection = json.loads((d / 'selection.json').read_text())
                model = d / 'best.gguf'
                if selection.get('status') != 'PASS' or selection.get('best_gguf_sha256') != model_digest(model, selection['best_gguf_sha256']):
                    raise ValueError('automatic selected model/manifest closure failed: ' + label)
                models.append({'label': label, 'path': model, 'expected_sha256': selection['best_gguf_sha256'],
                               'kind': 'auto', 'arm': label, 'selection_sha256': file_digest(d / 'selection.json')})
            if len(models) != 17 or len({m['label'] for m in models}) != 17:
                raise ValueError('expected exactly 12 fixed plus 5 automatic model outputs')
        if args.models_manifest:
            frozen = json.loads(args.models_manifest.read_text())
            if frozen.get('status') != 'FROZEN_BEFORE_HELD_OUT_EVALUATION':
                raise ValueError('external models not frozen')
            models = [dict(m, path=Path(m['path'])) for m in frozen['models'] if m['recipe'] == RECIPE]
            if not models or len({m['label'] for m in models}) != len(models):
                raise ValueError('invalid external models')
            fixed_hashes = {r['arm']: r['model_sha256'] for r in FIXED_AUDIT['records']}
            for m in models:
                if m['kind'] == 'stock-baseline':
                    validate_stock_baseline_control(m, EXPECTED['source'])
                    continue
                if m['kind'] == 'fixed':
                    if fixed_hashes.get(m['arm']) != m['expected_sha256']:
                        raise ValueError('fixed PTB control audit mismatch')
                    continue
                if m['kind'] == 'stock':
                    control = json.loads((ROOT / 'locks/historical-controls.json').read_text())['historical_controls'][RECIPE + '-stock']
                    if control['sha256'] != m['expected_sha256']:
                        raise ValueError('stock PTB control audit mismatch')
                    continue
                selection_path = Path(m['path']).parent / 'selection.json'
                selection = json.loads(selection_path.read_text())
                if (file_digest(selection_path) != m['selection_sha256'] or selection.get('status') != 'PASS'
                        or selection.get('best_gguf_sha256') != m['expected_sha256']):
                    raise ValueError('external selection/model closure mismatch')
        EXPECTED_MODELS = len(models)
        models.sort(key=lambda m: m['label'])

        config = {'source': EXPECTED['source'], 'raw': EXPECTED['raw'], 'normalized': EXPECTED['normalized'],
                  'token_stream': INPUT_LOCK['token_stream']['sha256'], 'evaluator': runtime_sha256,
                  'threads_per_process': 16, 'context': 65536, 'batch': 512, 'microbatch': 256,
                  'cache_k': 'bf16', 'cache_v': 'bf16', 'flash_attention': 'on'}
        if external_lock:
            config.update(external_lock_sha256=file_digest(args.execution_lock), models_manifest_sha256=file_digest(args.models_manifest))
        config_identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        results = {}
        failed_attempts = []
        if PROGRESS.exists():
            old = json.loads(PROGRESS.read_text())
            if old.get('recipe') != RECIPE or old.get('uuid') != UUID or old.get('config_identity') != config_identity:
                raise ValueError('existing PTB progress belongs to a different immutable run configuration')
            for row in old.get('results', []):
                record = Path(row['metrics_path'])
                tsv = Path(row['token_tsv_path'])
                if row.get('status') != 'PASS' or not record.is_file() or not tsv.is_file():
                    raise ValueError('existing PTB record is incomplete')
                if file_digest(record) != row.get('record_sha256') or file_digest(tsv) != row.get('token_tsv_sha256'):
                    raise ValueError('existing PTB output hash mismatch')
                results[row['label']] = row
            failed_attempts = list(old.get('failed_attempts', []))

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
        _KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
        _PSAPI = ctypes.WinDLL('psapi', use_last_error=True)
        _KERNEL.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        _KERNEL.OpenProcess.restype = wintypes.HANDLE
        _KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]
        _PSAPI.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        _PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL

        def peak_working_set(pid: int) -> int:
            handle = _KERNEL.OpenProcess(0x0400 | 0x0010, False, pid)
            if not handle:
                return 0
            try:
                counters = _PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                return int(counters.PeakWorkingSetSize) if _PSAPI.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else 0
            finally:
                _KERNEL.CloseHandle(handle)

        def gpu_usage() -> dict[str, int]:
            try:
                proc = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5, check=True)
                return {a.strip(): int(b.strip()) for a, b in (line.split(',', 1) for line in proc.stdout.splitlines())}
            except Exception:
                return {}

        def quantile(values: np.ndarray, p: float) -> float:
            return float(np.quantile(values, p, method='linear'))

        def summarize(tsv: Path) -> dict:
            n = EXPECTED['targets']
            nll = np.empty(n, dtype=np.float64)
            targets = np.empty(n, dtype=np.int32)
            positions = np.empty(n, dtype=np.int32)
            with tsv.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')
                expected_header = ['input_index', 'context_position', 'target_token', 'kld', 'base_nll', 'candidate_nll', 'top1_agree']
                if reader.fieldnames not in (expected_header, expected_header + ['candidate_top1']):
                    raise ValueError('PTB token TSV schema mismatch')
                count = 0
                for i, row in enumerate(reader):
                    if i >= n or int(row['input_index']) != i or int(row['context_position']) != i % EXPECTED['context']:
                        raise ValueError(f'PTB target index/context mismatch at row {i}')
                    if int(row['target_token']) != int(ids[i + 1]):
                        raise ValueError(f'PTB target-token alignment mismatch at row {i}')
                    if row['kld'] != 'NA' or row['candidate_nll'] != 'NA' or row['top1_agree'] != 'NA':
                        raise ValueError('PTB run unexpectedly reports reference-dependent KL/NLL/top-1 fields')
                    value = float(row['base_nll'])
                    if not math.isfinite(value):
                        raise ValueError(f'nonfinite PTB target NLL at row {i}')
                    nll[i] = value
                    targets[i] = int(row['target_token'])
                    positions[i] = int(row['context_position'])
                    count += 1
            if count != n:
                raise ValueError(f'PTB target count mismatch: expected {n}, got {count}')
            mean_nll = math.fsum(nll.tolist()) / n
            by_context_depth = {}
            by_corpus_index = {}
            for i, value in enumerate(nll):
                context_bin = str(int(positions[i]) // 4096)
                corpus_bin = str(i // 4096)
                by_context_depth.setdefault(context_bin, []).append(float(value))
                by_corpus_index.setdefault(corpus_bin, []).append(float(value))
            def bin_summary(groups):
                return {key: {'targets': len(values), 'mean_nll': math.fsum(values) / len(values),
                              'ppl': math.exp(math.fsum(values) / len(values))}
                        for key, values in sorted(groups.items(), key=lambda item: int(item[0]))}
            worst = np.argsort(nll)[-20:][::-1]
            return {
                'targets': n, 'input_tokens': len(ids), 'mean_nll': mean_nll, 'ppl': math.exp(mean_nll),
                'nll_distribution': {'mean': mean_nll, 'median': quantile(nll, .5), 'p90': quantile(nll, .9),
                                     'p95': quantile(nll, .95), 'p99': quantile(nll, .99),
                                     'p99_9': quantile(nll, .999), 'maximum': float(np.max(nll)),
                                     'minimum': float(np.min(nll)), 'nonfinite_count': 0,
                                     'negative_count': int(np.count_nonzero(nll < 0))},
                'context_depth_bins_4096': bin_summary(by_context_depth),
                'corpus_position_bins_4096': bin_summary(by_corpus_index),
                'worst_nll_examples': [{'input_index': int(i), 'target_token': int(targets[i]),
                                        'context_position': int(positions[i]), 'nll': float(nll[i])} for i in worst],
                'full_vocabulary_KL': None, 'top1_agreement_to_BF16': None,
                'reference_dependent_metric_note': 'No PTB BF16 distribution reference is locked; KL and top-1 agreement are intentionally not claimed.'
            }

        def save_progress(status: str) -> None:
            obj = {'schema_version': 1, 'status': status, 'recipe': RECIPE, 'uuid': UUID,
                   'config_identity': config_identity, 'input_lock_sha256': file_digest(LOCK_PATH),
                   'expected_models': EXPECTED_MODELS, 'completed_models': len(results),
                   'failed_models': sorted({r['label'] for r in failed_attempts if r.get('status') == 'FAIL'} - set(results)),
                   'failed_attempts': failed_attempts, 'results': list(results.values())}
            tmp = PROGRESS.with_suffix('.tmp')
            tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
            os.replace(tmp, PROGRESS)

        for model_info in models:
            label = model_info['label']
            model = model_info['path']
            before = model_digest(model, model_info['expected_sha256'])
            if label in results:
                continue
            if before != model_info['expected_sha256']:
                raise ValueError('model hash differs from immutable model audit: ' + label)
            arm_root = OUT / label
            arm_root.mkdir(exist_ok=True)
            occupied = lambda directory: any((directory / name).exists() for name in ('tokens.tsv', 'combined.log', 'metrics.json', 'failure.json'))
            if not occupied(arm_root):
                arm_dir = arm_root
            else:
                attempt = 1
                while (arm_root / f'attempt-{attempt:02d}').exists():
                    attempt += 1
                arm_dir = arm_root / f'attempt-{attempt:02d}'
                arm_dir.mkdir()
            tsv = arm_dir / 'tokens.tsv'
            log_path = arm_dir / 'combined.log'
            record_path = arm_dir / 'metrics.json'
            command = [str(EXEC), '-m', str(model.resolve()), '-f', str(NORMALIZED.resolve()), '--ppl-full',
                       '-c', '65536', '-b', '512', '-ub', '256', '-ctk', 'bf16', '-ctv', 'bf16', '-fa', 'on',
                       '-t', '16', '--no-warmup', '--split-mode', 'none', '--fit', 'off', '--verbose',
                       '--kld-token-output', str(tsv.resolve()), '-ngl', '999', '--device', 'CUDA0',
                       '--save-all-logits', 'NUL']
            env = {key: value for key, value in os.environ.items() if not key.startswith(('LLAMA_ARG_', 'GGML_', 'CUDA_', 'CUBLAS_', 'NVIDIA_TF32_'))}
            env.update(CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='16', MKL_NUM_THREADS='16',
                       GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto', GGML_CUDA_MMQ_PREC='auto')
            gpu_baseline = gpu_usage().get(UUID)
            gpu_peak = gpu_baseline or 0
            rss_peak = 0
            start = time.monotonic()
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            session.callback(stop_native, proc)
            output_chunks = []
            def drain_output():
                while True:
                    block = proc.stdout.read(1 << 20)
                    if not block:
                        return
                    output_chunks.append(block)
            reader = threading.Thread(target=drain_output, daemon=True)
            reader.start()
            while proc.poll() is None:
                rss_peak = max(rss_peak, peak_working_set(proc.pid))
                usage = gpu_usage().get(UUID)
                if usage is not None:
                    gpu_peak = max(gpu_peak, usage)
                time.sleep(1)
            reader.join()
            elapsed = time.monotonic() - start
            log_path.write_bytes(b''.join(output_chunks))
            if proc.returncode != 0:
                failure = {'status': 'FAIL', 'label': label, 'stage': 'native_ptb_evaluation',
                           'attempt_directory': str(arm_dir.resolve()), 'exit_code': proc.returncode,
                           'elapsed_seconds': elapsed, 'model_sha256_before': before, 'command': command,
                           'log_path': str(log_path.resolve()), 'log_sha256': file_digest(log_path),
                           'token_tsv_path': str(tsv.resolve()) if tsv.is_file() else None,
                           'token_tsv_sha256': file_digest(tsv) if tsv.is_file() else None}
                (arm_dir / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8', newline='\n')
                failed_attempts.append(failure)
                save_progress('INCOMPLETE')
                print(json.dumps({'status': 'FAIL', 'label': label, 'exit_code': proc.returncode,
                                  'attempt': str(arm_dir.resolve())}), flush=True)
                continue
            if external_lock:
                text = log_path.read_text(errors='replace')
                offload = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
                if not offload or offload[-1] != ('33', '33') or external_lock['gpus'][str(GPU_INDEX)]['pci_bus_id'] not in text:
                    raise RuntimeError('PTB CUDA execution evidence mismatch')
            try:
                metrics = summarize(tsv)
            except Exception as error:
                failure = {'status': 'FAIL', 'label': label, 'stage': 'ptb_alignment_or_metric_validation',
                           'attempt_directory': str(arm_dir.resolve()), 'error_type': type(error).__name__,
                           'error': str(error), 'elapsed_seconds': elapsed, 'model_sha256_before': before,
                           'command': command, 'log_path': str(log_path.resolve()), 'log_sha256': file_digest(log_path),
                           'token_tsv_path': str(tsv.resolve()) if tsv.is_file() else None,
                           'token_tsv_sha256': file_digest(tsv) if tsv.is_file() else None}
                (arm_dir / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8', newline='\n')
                failed_attempts.append(failure)
                save_progress('INCOMPLETE')
                print(json.dumps({'status': 'FAIL', 'label': label, 'stage': failure['stage'], 'error': str(error)}), flush=True)
                continue
            after = model_digest(model, model_info['expected_sha256'])
            if after != before:
                raise ValueError('model bytes changed during PTB evaluation: ' + label)
            if file_digest(NORMALIZED) != EXPECTED['normalized'] or file_digest(RAW) != EXPECTED['raw']:
                raise ValueError('PTB input changed during evaluation')
            record = {
                'schema_version': 1, 'status': 'PASS', 'label': label, 'recipe': RECIPE,
                'kind': model_info['kind'], 'arm': model_info['arm'], 'model_path': str(model.resolve()),
                'model_sha256_before': before, 'model_sha256_after': after,
                'model_hash_verification': 'fresh_once_under_guard_then_identity_reuse',
                'selection_manifest_sha256': model_info.get('selection_sha256'),
                'source_sha256': EXPECTED['source'], 'raw_corpus_path': str(RAW.resolve()),
                'raw_corpus_sha256': EXPECTED['raw'], 'normalized_input_path': str(NORMALIZED.resolve()),
                'normalized_input_sha256': EXPECTED['normalized'], 'token_stream_path': str(TOKENS_PATH.resolve()),
                'token_stream_sha256': INPUT_LOCK['token_stream']['sha256'],
                'target_count': EXPECTED['targets'], 'reference_path': None, 'reference_sha256': None,
                'reference_status': 'NO_PTB_BF16_DISTRIBUTION_REFERENCE; no full-vocabulary KL/top-1 claim',
                'context': 65536, 'batch': 512, 'microbatch': 256, 'cache_type_k': 'bf16',
                'cache_type_v': 'bf16', 'flash_attention': 'on', 'gpu_uuid': UUID,
                'visible_cuda_devices': 1, 'threads_argument': 16, 'command': command,
                'executable_sha256': runtime_sha256, 'config_identity': config_identity,
                'elapsed_seconds': elapsed, 'peak_process_working_set_bytes': rss_peak,
                'gpu_global_memory_baseline_mib': gpu_baseline, 'gpu_global_memory_peak_mib': gpu_peak,
                'gpu_global_memory_peak_includes_other_processes': True,
                'token_tsv_path': str(tsv.resolve()), 'token_tsv_sha256': file_digest(tsv),
                'combined_log_path': str(log_path.resolve()), 'combined_log_sha256': file_digest(log_path),
                'metrics': metrics
            }
            record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
            row = {'label': label, 'status': 'PASS', 'metrics_path': str(record_path.resolve()),
                   'record_sha256': file_digest(record_path), 'token_tsv_path': str(tsv.resolve()),
                   'token_tsv_sha256': file_digest(tsv), 'model_sha256': after,
                   'mean_nll': metrics['mean_nll'], 'ppl': metrics['ppl'],
                   'elapsed_seconds': elapsed, 'peak_process_working_set_bytes': rss_peak,
                   'gpu_global_memory_peak_mib': gpu_peak}
            results[label] = row
            save_progress('IN_PROGRESS' if len(results) < EXPECTED_MODELS else 'PASS')
            print(json.dumps({'status': 'PASS', 'completed': len(results), 'expected': EXPECTED_MODELS,
                              'label': label, 'ppl': metrics['ppl'], 'mean_nll': metrics['mean_nll'],
                              'seconds': elapsed}), flush=True)

        if len(results) == EXPECTED_MODELS and all(row['status'] == 'PASS' for row in results.values()):
            save_progress('PASS')
            print(json.dumps({'status': 'PASS', 'recipe': RECIPE, 'completed_models': len(results),
                              'progress': str(PROGRESS.resolve())}, indent=2), flush=True)
        else:
            save_progress('INCOMPLETE')
            print(json.dumps({'status': 'INCOMPLETE', 'recipe': RECIPE, 'completed_models': len(results),
                              'expected_models': EXPECTED_MODELS, 'failed_models': len(failed_attempts),
                              'progress': str(PROGRESS.resolve())}, indent=2), flush=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
