# SINQ end-to-end refactor and exhaustive first-pass A/B

Both T340 and T425 completed all 248 eligible groups in their first coordinate-descent pass. Every group accounted for stock plus all 12 pre-generated conditioned candidates. The two canonical GGUFs were independently audited and frozen before the new held-out runs.

## Measured quality

| Recipe | Model | Completed groups | Selection KLD (564) | WikiText KLD (297,193) | WikiText PPL | PTB PPL (98,776) | GGUF SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| T340 | LOCKED-STOCK-BASELINE-CONTROL | N/A | 0.064806199 | 0.104081454 | 7.209603 | 12.113470 | `3287225bf587cd3cbf48b6f3891327e8892002fac33ccda6afdf23aaae8a8c7f` |
| T340 | FIXED-ANCHOR | N/A | 0.063441264 | 0.096561983 | 7.112571 | 12.106593 | `9c23848879505d1699a039459f72c7899b9f5fe75c4c8fda7cf28fba69aeba70` |
| T340 | CURRENT-GROUP-KLD-CUDA27 | 3 attempted; incomplete | 0.062489739 | 0.096052488 | 7.101471 | 12.082488 | `3364bb34a9b58dbff0653317fb7ece1a55f0093701c9f8ec5d6710fc4f612bb0` |
| T340 | NEW-FULL-248-GROUPWISE | 248/248 | 0.040979841 | 0.096565834 | 7.118288 | 12.130691 | `912de0e6e4c375a07382f08ccd91662b59cfbad3c2ef2629fad153c8ea35cf20` |
| T340 | BEST-HISTORICAL-AUTOMATIC | N/A | 0.062489739 | 0.096052488 | 7.101471 | 12.082488 | `3364bb34a9b58dbff0653317fb7ece1a55f0093701c9f8ec5d6710fc4f612bb0` |
| T340 | BEST-HISTORICAL-FIXED-SELECTION-KL | N/A | 0.053299104 | 0.102258598 | 7.254296 | 11.827471 | `cf06e0109c7ee6217df5c6be60da9ae67800ceb7809025d5307fd6fd4ae3a942` |
| T425 | LOCKED-STOCK-BASELINE-CONTROL | N/A | 0.020701105 | 0.044583063 | 6.601768 | 10.934203 | `cee08bcd9087742509b0ee74c327ed4741a88414da4a9a3f3ba12dc81b80c998` |
| T425 | FIXED-ANCHOR | N/A | 0.018371355 | 0.038698239 | 6.693302 | 11.040010 | `ca88481fdc5e9b009af3d69593af2a9bdcc881075408f4aca33f93416a8d5153` |
| T425 | CURRENT-GROUP-KLD-CUDA27 | 3 attempted; incomplete | 0.017441887 | 0.038755844 | 6.688022 | 11.033516 | `9192cdd67b1a1e0bffd35637d6e08d0c1de1721e155236a7f032cea698cad5d7` |
| T425 | NEW-FULL-248-GROUPWISE | 248/248 | 0.013192414 | 0.038295344 | 6.716062 | 11.067749 | `8586a243a7d75ae38e1793f6da99e07fb55f1baa7dd4e3ac5a16551d8cbff854` |
| T425 | BEST-HISTORICAL-AUTOMATIC | N/A | 0.018833309 | 0.038488083 | 6.683044 | 11.127261 | `83e523cc2199bcd87c18b04ad9dab8854b8c0e79928f3300ed9d9ed0a295e046` |
| T425 | BEST-HISTORICAL-FIXED-SELECTION-KL | N/A | 0.016789615 | 0.038888090 | 6.692046 | 11.070902 | `f874a9cff314415bf5672c78017e11c7b28295134e5be2389034b755e7b7f598` |

### Interpretation

T340: selection KLD changed -35.405% versus the fixed anchor. Full WikiText KLD changed +0.000003851 nats, WikiText PPL +0.080%, and PTB PPL +0.199%. Lower is better for all four metrics. Improved selection KLD did not produce a general held-out PPL improvement; no statistical significance or reserved-validation stability estimate is available.
T425: selection KLD changed -28.190% versus the fixed anchor. Full WikiText KLD changed -0.000402894 nats, WikiText PPL +0.340%, and PTB PPL +0.251%. Lower is better for all four metrics. Improved selection KLD did not produce a general held-out PPL improvement; no statistical significance or reserved-validation stability estimate is available.

Selection KLD optimizes the original 564-target stream. Full WikiText KLD and WikiText PPL use all 297,193 locked targets; PTB PPL uses all 98,776 targets. PTB has no locked full-vocabulary BF16 reference, so PTB KLD is deliberately not reported. Median/tail WikiText KLD, top-1 agreement, exact commands and runtime evidence are preserved in the CSV/JSON records.

## Limits and controls

The external historical stock GGUFs were missing after user disk cleanup. With explicit user approval, preserved locked stock baselines were audited and used as separately labeled replacement controls. Their hashes differ; no byte identity with the missing historical files is claimed. Other historical controls were retained. A completed first pass is not a global optimum or necessarily a coordinate-wise local optimum: later accepted changes can make earlier groups worth revisiting. Thousands of alternatives were tested against only 564 optimization targets, creating overfitting risk. No new full-test observation was used to change selection or choose a different new winner.

Historical CUDA-27 controls entered three groups but did not exhaust the third; they are incomplete, not three fully completed groups. The best historical fixed control is the already-recorded CPU selection-KL winner. The best historical automatic control is explicitly retrospective, chosen from already-observed historical WikiText results. Prior full-test arms were already examined, limiting claims of blind holdout validation.

## Integrity and performance

Normal full-reference SHA-256 passes and bytes hashed are zero. The historical reference digest is explicitly non-fresh evidence under a live mandatory file guard; it does not prove unchanged bytes during unguarded historical intervals. Probability-body reads needed for full KLD remain legitimate inference I/O. See `INTEGRITY.md` and `reference-trust-policy.json`.

The persistent native evaluator retains full-model CUDA inference, private F32 scale buffers, canonical stock/identity elision and a fresh context/graph per score. Production qualification passed 19 exact per-token cases per recipe and five 16-thread cases; the final build reported no work to do and all49 relevant CTest programs passed in 224.51 seconds. Final selected canonical models also matched persistent predictions exactly. Source/candidate identities, allocation and numerical policy were preserved; no historical pools were regenerated.

See `PERFORMANCE.md`, `performance-before-after.json` and `hash-io-accounting.json` for scoped timings and byte counts. Local controls are bounded real-tensor benchmarks, not timings of the full historical Local stages. Full-study telemetry after interruption is not an uninterrupted whole-study baseline comparison. Large GGUFs, logits references and per-token TSVs remain local; compact coverage, assignments, hashes, commands and reports are published.

The relevant build/test commands and initial failures are retained in `WORK_LOG.md`, `build-v2.log` and `tests-v2.log`. Exact runtime digests and source-content fingerprints are in `implementation-provenance.json`; publication commit and remote verification are reported with the final delivery.
