# SINQ session integrity and numerical evidence

## Evidence status

The immutable-session, assembly and native evaluator changes passed the 48 relevant CTest programs, including production v2 qualification. Both recipes then completed 248/248 groups and 2,977 committed evaluations. Independent deterministic replay, all-score receipt/token checks, final tensor inventories, CPU/CUDA forward checks and the pre-held-out freeze passed. All six frozen arms per recipe completed full WikiText and PTB evaluation. The final build succeeded and all49 relevant CTest programs passed. Detailed measured results are in `REPORT.md` and `final-results.json`.

## What a reused digest means

`VerifiedInputs` owns live Windows `GuardedFile` handles. Each newly admitted small input receives a fresh SHA-256 under a deny-write/delete handle. Later calls check the file object and reader path while that handle remains open; they reuse the authenticated digest, not a size/mtime cache. Nested preparation can borrow an already-guarded source from its parent invocation. Closing a child does not close the parent's source guard.

The file guard checks a local fixed NTFS/ReFS volume, canonical path, file identity, length, reparse-point constraints and single-link status. Ancestor handles prevent path rebinding. Same-size writes and replacement attempts are denied by the OS. Candidate payloads continue to use `CandidateStore.verified_session()` and its existing guarded receipts; this refactor does not replace those protections with metadata-only receipts.

A historical stock GGUF can have publication hard links. `PreparedAssembly` does not relax alias rejection. It makes a private, single-link byte copy, hashes that copy under a live guard against the original baseline lock, validates its complete inventory, and uses only that snapshot thereafter. It does not modify the historical file or rewrite the old lock. Maps close before input handles, and handles close before temporary storage is removed.

On platforms without mandatory guards, the prepared selection path fails closed. Ordinary generation retains its existing full-verification path when initial guard acquisition is unsupported. A guard or integrity failure after execution starts does not silently downgrade an active session.

## The 147.6 GB reference is historical evidence, not a fresh hash

Normal evaluation uses `TrustedLargeReference`, with:

- `trust_mode=previously_verified_immutable_local_file`;
- `fresh_content_hash=false` and `hash_performed_in_this_session=false`;
- the previously recorded digest, size, native header, token identity, original evidence-lock digest and current guarded file identity;
- zero full-reference cryptographic hash passes.

`reference-trust-policy.json` records an actual production metadata/guard probe. It read 1,188,800 header/token-alignment bytes, not the probability body, with a full-reference hash tripwire active. The probe's handle was then released; final evaluation must acquire its own guard. The normal hash helper rejects the pinned production filename, and the generic selection evaluator rejects the held-out large reference rather than hashing it. `audit-reference` is a standalone opt-in command, tested only on a small synthetic reference. It has not been invoked on the production reference.

This policy does **not** cryptographically prove unchanged content during historical intervals when no guard was held. Matching file identity, size and header cannot establish that fact. The historical SHA-256 remains a previously verified claim. Full WikiText KLD must still read the probability body for inference; those required reads are separate from hash reads. PTB has no locked BF16 distribution reference and supplies PPL only.

## Prepared assembly versus publication

Preparation performs source/stock content and tensor-inventory verification once per guarded session. Repeated canonical assembly reuses those authenticated input facts. Every published output still receives a complete payload readback, native-loader check and output SHA-256. An independent final audit must verify the selected candidate payload/scale pairs, allocation and SINQ metadata against the published output.

`tests/test-sinq-prepared-assembly.py` checks identical canonical output bytes, one inventory scan across repeated assemblies, parent-session source-hash reuse, hard-link snapshot isolation, denied mutation/replacement and expired-session rejection. Existing assembly, stock-baseline and corruption tests remain registered.

## Persistent evaluator semantics

`llama-sinq-groupwise` loads one canonical starting model on the explicitly selected CUDA device. It assigns each mutable tensor a private F32 runtime diagonal, avoiding the loader's deduplicated scale buffers. Stock and all-one conditioned diagonals are elided exactly as in the canonical loader. Context reconstruction handles those graph-structure transitions. Payload qtype/byte-length changes are rejected. The Python wrapper checks source, tensor, allocation, recipe, imatrix and candidate provenance before sending a delta.

Each score uses a fresh native context. Graph bindings, KV/recurrent state and logits therefore cannot survive from the preceding proposal. Unchanged model weights stay resident. Native full-vocabulary scoring calls the same `perplexity_full` implementation as the ordinary evaluator. Both physical GPU identity and 33/33 full offload are attested; no CPU fallback is accepted.

The original assembled-GGUF evaluator remains independent. The production v2 real-model parity runs cover 19 cases per recipe: initial assignments, stock transitions for all 12 eligible roles, combined stock-role changes, two late-layer changes, conditioned transitions and restoration. Per-token KL, reference NLL, candidate NLL, top-1 agreement and candidate token IDs matched exactly for all 19 T340/GPU0 and 19 T425/GPU1 cases. The v2 T340/16-thread run independently matched in five cases. `cross-device-thread-parity.json` supplies additional earlier same-oracle physical-device and thread comparisons; its prototype provenance is distinct from the v2 all-role runs. These results concern the 564-target selection stream, not held-out quality.

Before a prepared search publishes `best.gguf`, it re-scores the selected state and the just-assembled canonical model and requires exact per-token equality and equality with the checkpoint score.

## Search state and recovery

Checkpoint keys bind full assignments, implementation/runtime identities, reference evidence, scorer policy and effective GPU. Scores are committed only after successful finite, aligned evaluation. Resume rebuilds native state from the canonical initial assignment and replays deterministic decisions from the score cache. Failed scores do not become completed alternatives.

Progress distinguishes an attempted group from a group whose admissible alternatives are exhausted. `COMPLETE_FIRST_PASS`, `COMPLETE_CONVERGED`, `INCOMPLETE_BUDGET` and `FAILED` are separate search statuses. The stage's historical `PASS` publication status is not evidence of full group coverage. Final freeze independently required and verified 248/248 completed groups for each recipe. Two external interruptions were recovered without reducing alternatives or changing policy. The final resumed invocation wall times are not presented as uninterrupted whole-study timing.

## Preserved boundaries

The original floating source, two candidate manifests, stock locks, candidate IDs/payloads, numerical objectives and selection stream remain inputs, not regenerated substitutes. During execution, external historical stock controls at `C:/models/Qwen3.5-9B-thireus-v050/{T340,T425}-stock.gguf` became unavailable after user disk cleanup. The user explicitly approved preserved locked stock baselines as replacement controls. They are labeled `LOCKED-STOCK-BASELINE-CONTROL`, carry their own baseline-manifest/source/hash closure, and are not claimed byte-identical to the missing files. Other historical controls remain unchanged. No full WikiText or PTB result may change selection after freeze. A completed first pass is not a claim of global optimality or, when later changes were accepted, a coordinate-wise local optimum. The 564-target objective has substantial selection-overfitting risk; earlier historical held-out arms were already observed.
