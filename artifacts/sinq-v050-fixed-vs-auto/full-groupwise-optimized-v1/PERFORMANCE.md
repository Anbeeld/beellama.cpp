# Measured SINQ pipeline performance

## Scope and attribution

`tables/exclusive-phase-attribution.csv` provides actual before/after inclusive and same-thread-exclusive measurements and the single-owner stage wall timer separately; concurrent worker phase totals are not a partition of wall time. Stage timers cover `select_config`: input verification, search and canonical finalization. The existing measured launcher also performs post-invocation model/report hash checks outside this timer; those scans are not hidden inside the stage numbers. Per-file cryptographic input bytes are reported separately from tensor-hash bytes. Neither is a physical disk-read counter.

Phase `seconds` are labeled inclusive. `exclusive_seconds` subtract same-thread child phases; concurrent worker-thread times must not be added to obtain invocation wall time. One owner supplies each top-level stage timer. The historical source snapshot is `ff7994f35` with only the current telemetry/hash-prohibition module substituted.

## Matching two-score controls on the same physical GPU

| Recipe | Legacy stage s | Prepared stage s | Stage speedup | Full-source hash passes | Stock hash passes | Canonical writes |
|---|---:|---:|---:|---:|---:|---:|
| T340 | 612.241 | 142.514 | 4.296x | 27 → 1 | 14 → 1 | 3 → 2 |
| T425 | 606.435 | 182.559 | 3.322x | 27 → 1 | 14 → 1 | 3 → 2 |

A separate complete first-group control scored stock plus all12 conditioned alternatives on the same GPU0. Legacy 2256.795s versus production-v2 112.275s (20.101x), 14 versus 2 canonical assemblies. All13 full assignment mean scores and selected model bytes match exactly. This bounded control does not claim a measured full-248 legacy speedup.

Both selected GGUF SHA-256 identities match the immutable legacy controls. The two-score pilots completed zero groups and are not full-search results. The after pilots include an extra final canonical/persistent score-parity check. Initial/final canonical writes remain; proposal evaluation does not write a GGUF.

The production v2 adapter additionally elides stock/all-one diagonals exactly like the canonical loader and rebuilds context/graphs on every score. Its all-role numerical gates and the separately recorded 13-score one-versus-two-GPU pilots precede the full studies. A single 13-score pilot completed one group, not a complete pass.

## Local scoring: bounded real controls

Each row covers the same three real tensors, all 13 candidates and original capture/settings. It is not a full 248-tensor Local stage timing. Historical 43.9/119.9-minute figures are not used as measured before values here.

| Recipe / objective | Legacy s | Batched s | Speedup | Native launches | Exact score difference |
|---|---:|---:|---:|---:|---:|
| T340 / output_l1 | 34.717 | 27.963 | 1.242x | 39 → 3 | 0.0 |
| T340 / output_mse | 110.570 | 104.391 | 1.059x | 39 → 3 | 0.0 |
| T425 / output_l1 | 33.555 | 28.358 | 1.183x | 39 → 3 | 0.0 |
| T425 / output_mse | 112.051 | 107.226 | 1.045x | 39 → 3 | 0.0 |

## Complete final evaluators

| Recipe / corpus | Models | Runner wall s | Native process s | Other orchestration s | New reference hash bytes |
|---|---:|---:|---:|---:|---:|
| T340/wikitext2 | 6 | 1132.480 | 1096.297 | 36.182 | 0 |
| T340/ptb | 6 | 339.152 | 310.924 | 28.228 | 0 |
| T425/wikitext2 | 6 | 1133.558 | 1092.702 | 40.855 | 0 |
| T425/ptb | 6 | 336.576 | 303.176 | 33.400 | 0 |

Runner minus native elapsed includes setup, model/source guards and hashes, TSV summaries and publication; it is not an isolated startup timer. The WikiText logical reference-format input is 147,601,497,864 bytes per arm. Physical disk reads, whole-run temp-disk peaks and isolated startup/resume durations were not measured. No before/after final-evaluator speedup is claimed.


Source exports and fresh source-tensor hash bytes fall by 13x for these bounded controls. MSE compute remains substantial; these measurements do not establish that historical scoring time was primarily hashing. The original local objectives and native loss routines are unchanged.

## Remaining costs and limits

Candidate registries still receive fresh content admission per guarded session. A guarded private baseline copy is required for historical hard-link aliases. Native context construction and full-vocabulary reduction remain per proposal. Checkpoint JSON publication grows with retained assignment evidence; full-study runtime is measured independently, not extrapolated from the two-score pilot.

The bounded genuine-Qwen Blockwise/Fisher profile records source/reference loading once, three native candidate decodes, zero full-GGUF assemblies and zero new source-file hash bytes during those candidate trials. It is not a production 9B throughput claim. The kernels, precision and Fisher objectives were left unchanged. Native activation-validation boundaries retain their independent full checks; individual advanced candidate decodes still launch a bounded native process.

Full WikiText probability-body reads are required for KLD and are not hash passes. Normal full-reference content hashes and content bytes hashed are zero. The production metadata/guard probe and mocked success/resume/error tripwires are recorded separately. Final evaluator and full-study figures are added only after completed runs; missing metrics remain explicitly unmeasured.

Large GGUFs, probability references and temporary score TSVs remain local. Immutable small manifests, code, commands, hashes and compact coverage tables are the publication surface.
