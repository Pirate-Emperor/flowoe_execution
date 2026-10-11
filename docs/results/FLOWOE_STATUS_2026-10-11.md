# FlowOE Resume-Target Status — 11 October 2026

## Executive answer

**Neither resume performance target is established.** CFM/PF-ODE training, historical execution replay, CUDA timing, and an INT8 TensorRT build have been implemented and tested, but the real-data improvement is not statistically supported and the strict TensorRT-plus-fused sub-2 ms target has not passed an accuracy-gated benchmark.

## Current repository / test state

- FlowOE `main`: `d8db41368d27a20e4959b2058b126cd55ee671a2`.
- A100 runner repository `main`: `f7ce6a5ad1736e434bbb2e579c1742f4c2e61c80`.
- FlowOE CI #256: passed; 95 passed, 16 skipped, 0 failures, 93% line coverage. It ran on the code revision immediately before the latest documentation-only caveat update.
- A100 fused-extension run #5: passed its core CUDA/PyTorch checks (103 tests passed, 0 failures, 92% coverage), data-preparation checks, and both latency sweeps. Ninja is now checked explicitly before PyTorch compiles the extension.
- Data unit CI #14: 36 tests passed in 1.44 s; all five workflow YAML files parsed successfully.
- New A100 research run #33: failed before publishing artifacts, during/after the alternate market-data fallback. The available run record has inconsistent step state and no retrievable log/artifact, so the terminal download error cannot be confirmed from available evidence.
- New real-execution sweep #8 and data-validation run #42 remain queued in the latest Actions view. No newer current-source GPU result is available yet.

## Target 1 — 2.4 bps lower slippage against trade VWAP

**Status: not achieved; no unbiased result from the current leakage-safe code is available yet.**

| Experiment | Measured result | Interpretation |
|---|---|---|
| Earlier real-data run #27, 24 evaluation episodes | TWAP -0.55897 bps; FlowOE -0.58896 bps; mean improvement +0.0300 bps; paired 95% CI [-0.03336, +0.09304] bps | CI includes zero; far below 2.4 bps. Diagnostic only after a later feature-normalization leakage issue was found. |
| A100 sweep #5, 12-hour BTCUSDT USD-M futures window, 3 training seeds × buy/sell, 144 held-out episodes per seed/side | Across tested policies, mean improvement was about +0.004 to +0.012 bps; all 10 pooled paired 95% intervals crossed zero. Example RK4: buy +0.00530 bps, CI [-0.01566, +0.02887]; sell +0.00457 bps, CI [-0.01858, +0.02935]. | Larger than 24 episodes, but still no statistically reliable gain. This run predates trailing-only OFI normalization and did not record the exact FlowOE checkout SHA; it is explicitly diagnostic-only. |

The 12-hour market files were independently validated: 8,639 reconstructed L2 snapshots, 895,566 trades inside the window, 410,004 sequence transitions checked, no sequence gaps, and recorded SHA-256 hashes. FI-2010 DecPre (not ZScore) was validated and 50,000 rows were loaded for training. The held-out market window is still only one historical period, and nearby windows are dependent; three seeds do not create three independent market samples.

**Required next proof:** complete a fresh run using the current trailing-only feature normalization; retain exact FlowOE/runner source SHAs and raw input hashes; evaluate multiple non-overlapping dates; pool paired execution windows with a time-block bootstrap; compare CFM + FI auxiliary training against TWAP and simpler schedule baselines. Until then, do not put 2.4 bps in the resume as an achieved result.

## Target 2 — INT8 TensorRT + fused CUDA, p99 below 2 ms

**Status: not achieved as a combined, accuracy-passing inference path.**

| Measurement | Result | Decision |
|---|---|---|
| Fixed-step Euler, synthetic checkpoint, batch 1 | 4 steps: wall p99 1.741 ms, mean absolute schedule error 0.00894, mean KL vs 64-step RK4 0.00393. 6 steps: 1.942 ms, error 0.00502, KL 0.00123. | Some fixed-step PyTorch configurations are below 2 ms on this synthetic benchmark, with a speed/quality trade-off. This does not establish the resume target. |
| RK4, synthetic checkpoint, batch 1 | Wall p99 about 11.86 ms at 8 steps and 40.47 ms at 32 steps. | Full higher-step RK4 is much slower. |
| INT8 TensorRT, 50 warmups / 300 calls, synthetic checkpoint | Batch 1: wall p99 0.452 ms, max abs error 0.000257. Batch 4: wall p99 0.426 ms, max abs error 0.084224. | Engine built and executed, but the overall accuracy gate failed (batch 4 exceeded even the prior 0.05 threshold). Raw speed ratios are not accepted. |
| Fused CUDA update | Max observed output error about 2.98e-8; measured speedup varied about 0.943×–1.202×. | Correctness passed, but performance was configuration-dependent. The extension fuses the Euler update only, not the vector-field network. |

The follow-up benchmark code now checks 32 distinct context/noise inputs at batch sizes 1 and 4, records max/mean output error and KL divergence, and defaults to a 0.02 max-error threshold. It has **not yet produced a completed A100 result**. Therefore the sub-2 ms p99 goal using TensorRT INT8 plus fused CUDA is not proven.

## Milestones completed

- CFM policy and probability-flow ODE comparison path, fixed-step sampler, schedule generation, and full-depth L2 replay.
- Trade-VWAP benchmark, TWAP comparison, completion/slippage metrics, paired moving-block bootstrap, chronological split, and feature-prefix invariance tests.
- FI-2010 and public futures L2/trade ingestion, strict update-ID reconstruction checks, provenance/hash reports, and real/synthetic result separation.
- CUDA Euler-update extension; current-stream and invalid-input tests; warmup/repeated latency grids; p50/p95/p99 and schedule-error/KL comparisons.
- TensorRT conversion path, reproducible shape-tensor dtype patch reporting, engine execution checks, and tighter multi-input accuracy validation.
- Execution simulator schedule-copy fix and regression tests for inclusive VWAP time window and buy/sell slippage sign.
- Experiment provenance fields for UTC time and source/workflow commits. Workflow data downloads now have timeouts and FI archive-first mode; YAML syntax checks run in data CI.

## Blockers / acceptance criteria

1. **Runner execution:** current-source real-data sweep #8 is queued; research run #33 ended in failure during the alternate data stage without a retrievable artifact. A fresh A100 result must finish before accepting the new code's performance claims.
2. **Slippage evidence:** use leakage-safe features, more independent dates, and matched paired baselines. No currently valid evidence supports a 2.4 bps reduction.
3. **Inference evidence:** run the stricter multi-context INT8 benchmark on the current trained model; require engine build/load/execute, finite normalized outputs, stated error gate, CUDA-event and wall p50/p95/p99, and schedule-quality measures. Combine with the fused path only if profiling shows a repeatable total-path win.
4. Keep real-data, synthetic smoke, and inference-only measurements separate. Do not state either resume target as achieved without new passing evidence.
