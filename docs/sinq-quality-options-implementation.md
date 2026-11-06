# SINQ quality options implementation

> Historical pre-refactor report. Its removed interfaces and v1 artifacts are
> not current instructions or v2 verification. Use [current SINQ settings](sinq-settings.md).

Audit follow-up: [complete reference identity and CPU numerical validation](sinq-audit-followup.md).
The original capture and GPU timing evidence below is historical and was not
relabelled as verification of the follow-up. No GPU tests were needed for that update.

## Source and scope

Implemented against clean local `09e81f3fa4480d9ee1628af963ed0240f6c5e122`,
not the older published reference. Kurtosis gating remains removed, with no
compatibility parser or metadata acceptance restored. Upstream mathematical
reference: `huawei-csl/SINQ` commit
`42ca83edf8fe1afc48c40fca6acc973c5d2bb59f`, especially `awq.py` and
`dual_shift.py`.

The owner's performance-test and granular-commit instructions supersede the
plan's benchmark prohibition and single-commit suggestion. No WikiText/PTB
quality evaluation, model-level KLD/PPL comparison, quality ablation, push or tag was run.
The requested conversion smoke uses only a previously disclosed TRAIN corpus.

## Delivered behavior

- Full-height widths 32/64/128/256/512, default 64, with matching producer,
  loader and inspection checks. Unsupported/misaligned/resource-exceeding
  requests fail instead of changing width. Normalizer worker buffers are
  bounded to 128 MiB in aggregate.
- Optional `--sinq-calibration` and matching nullable public parameter.
  Ordinary strength, masks, codecs and default behavior remain unchanged.
- Fixed-strength, 20-alpha AWQ candidate generation, including the separately
  scored ordinary candidate. Native final-quantizer callback and GGML
  dequantizer score actual compensated linear outputs with fixed L1 loss.
  The effective diagonal is F16-rounded only after AWQ correction. Numeric
  ties keep the earlier candidate, ordinary first.
- Strict JSON/raw-float activation import, source/digest/site/shape checks,
  contained paths and finite-data validation. Bounded sample/row buffers,
  deterministic diagnostics and logged time/workspace costs.
- Optional local Qwen3.5 exporter with real Linear pre-input hooks,
  disk-backed deterministic reservoirs, reference-weight verification and
  audited grouped-to-tiled SSM output inputs. All 12 projection roles have
  source-verified mappings; fused/missing/other-architecture sites fail.
- Existing version-1 seven-key runtime metadata and one F16 vector per weight.
  No new graph operators, kernels, runtime calibration, LoRA semantics,
  codec selection, automatic strength or median-normalization option.

Calibration currently requires one original GGUF source and an unmodified
local conditional HF checkpoint. Layer pruning/remapping and graph metadata
overrides are rejected; the four imatrix provenance overrides added by the
CLI remain supported. Missing requested samples require an explicit smaller
mask, never an imatrix-only fallback. See [settings](sinq-settings.md),
[format/mappings](sinq-activation-format.md) and
[exporter usage](../tools/sinq/README.md).

## Correctness evidence

Local evidence is under `artifacts/sinq-quality-options/`; model/activation
binaries are not committed.

- Native feasibility test covers all 15 supported codecs, actual encoder-byte
  parity, F16 compensation, original/transformed imatrix, AWQ orientation,
  covariance-sensitive output loss, uncalibrated ties and invalid factors.
- All 36 locked 64/128 normalization fixtures retain exact F16 scale bytes.
  The expanded tests cover every width, F32/F16/BF16, independent moments and
  best-iterate references, nonfinite/resource failures and thread parity.
  An obsolete out-of-bounds fixture-header read was removed to keep the
  reference assertion unconditional, not silently skipped.
- Tiny real Linear export/import verifies output loss independently with
  float64 NumPy matmuls, identical 1/4-thread payloads and deterministic
  diagnostics. Frozen-runtime ordinary 64/128 payload comparisons also pass
  at strengths 1 and 0.5.
- Eight offline-exporter CPU tests pass. Mapping was source-audited against
  Transformers 5.6.2. Repeated real capture with the final exporter reproduces both
  sample binaries and the complete manifest byte-for-byte. Strict native importer tests exercise
  malformed fields, source/data digests, duplicates, finite values, file
  lengths and path containment.
- A real imatrix conversion exposed an overly broad metadata-override
  rejection. It was fixed to admit only the four producer provenance keys,
  with a failing-then-passing regression check; graph overrides still fail.
- All-target Windows/MSVC/CUDA 13.3 Release build passed. Two complete
  registered CTest runs passed 154/154 (`full-ctest.log` and
  `full-ctest-final.log`). A strengthened forced-tail-slab native test and the
  four affected Python/fixture tests passed again after final test formatting.
- New Python files pass Flake8 7 core checks, syntax compilation and an AST
  bare-print check. Touched older Python files add no lint findings relative
  to HEAD. The existing combined lint hook cannot resolve: it pins Flake8 7,
  while `flake8-no-print` 0.1.1 requires Flake8 4.0.1. That unrelated hook
  configuration and its pre-existing file findings were left unchanged;
  evidence is in `lint-comparison.json`. Whitespace checks pass.

Useful commands from the configured Windows build environment:

```sh
cmake --build build-sinq-clean --target all
ctest --test-dir build-sinq-clean --output-on-failure
python tests/test-sinq-exporter.py
python tests/test-sinq-calibration-e2e.py --source build-sinq-clean/tests/qk-sinq-model-fixtures/source.gguf --quantize build-sinq-clean/bin/llama-quantize.exe --baseline artifacts/sinq-quality-options/baseline-bin/llama-quantize.exe
```

## Requested conversion and speed tests

Real BF16 reference capture used 8 rows per selected projection, seed 0,
fresh 128-token chunks and the existing 1,691-token TRAIN screen. The new
T425 partial A-SINQ conversion conditions only `blk.0.ffn_down.weight` and
`blk.0.ssm_out.weight`, with width 128 and fixed strength 1. Selected AWQ alphas
were 0.20 and 0.25. This is not a fully calibrated 248-projection model or an
end-to-end quality result.

Independent inspection verified all 427 base qtypes/shapes unchanged and all
425 unselected payloads byte-identical to existing T425 stock. Both new
companions are positive finite F16 vectors. Independent float64 BLAS checks
reproduce the two native calibration losses using the actual serialized
weights and scales. The imatrix is present and transformed for both.

Benchmarks compare frozen and new runtimes on existing T340/T425 stock and
new matched partial ordinary-SINQ/A-SINQ conversions. Each model uses one
UUID-isolated RTX 3090, full offload, BF16 KV, FlashAttention, 8 CPU threads,
batch 512/ubatch 256, prompts 128/512 and decode 128. ABBA process order and five
warmed repetitions per process provide 10 samples per build/model/test.
Commands, model/binary hashes, hardware observations and all native timing
samples are retained in `performance/`.

| Model | New prefill512, tokens/s | New decode128, tokens/s |
| --- | ---: | ---: |
| Existing T340 stock | 3976.39 | 118.31 |
| Existing T425 stock | 4472.38 | 138.53 |
| New T425 partial SINQ | 4462.77 | 136.62 |
| New T425 partial A-SINQ | 4451.77 | 136.48 |

Across the initial 12 matched cases, latency changes ranged from -0.68% to
+0.37%. One marginal ordinary-SINQ decode flag (+0.148%) triggered a longer,
reversed-ABBA recheck and Nsight profiling of the exact original workload.
The recheck measured -0.293% instead, with descriptive 95% bootstrap interval
[-0.591%,-0.011%]. The initial slowdown did not reproduce.

Nsight Systems 2026.1.3 old/new traces contain the same 609,519 kernel launches
and 113 kernel configurations/counts. No added kernel work was found. Profiled
times are diagnostic, not throughput claims. No reproducible runtime
regression remained, so no speculative kernel change was made. Evidence:
`recheck/summary.json`, both `.nsys-rep`/SQLite files, manifests and
`recheck/nsight-comparison.json`.

The final runtime rebuild differs from the measured executable only in the
two PE link-timestamp fields, not executable code or constants, verified by
`runtime-byte-differences.json`. Archived gate-bearing SINQ models remain
unsupported; they were not modified to make these tests pass.

Native-codec A-SINQ quality on Thireus recipes remains unmeasured. Local
calibration L1 and these speed measurements establish no KLD/PPL benefit.
