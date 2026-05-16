# Qwen3-235B + SWE-bench — fully-async agentic RL with slime

Twelve-node H200 cluster, Qwen3-235B-A22B-Thinking-2507, Modal-backed
Docker sandboxes from Epoch AI's per-instance SWE-bench images, six-tool
coding agent, fully-async Megatron + 8× SGLang TP=8 engines.

This example is structured as a **linear log of experiments** — the
journey from initial bring-up through context scaling, PD attempts,
partial-rollout, metric instrumentation, and the final
partial-rollout-vs-base comparison.

## Layout

```
.
├── README.md             ← you are here (high-level journey)
├── EXPERIMENTS.md        ← status table + index of all experiments
├── lib/                  ← shared library code — single source of truth
│   ├── coding_sandbox.py           ← Modal pool + per-sample sandbox class
│   ├── generate_with_codingagent.py ← agent loop (gated resume branch)
│   ├── rollout_metrics.py        ← metric wrapper for slime hooks
│   └── gen_prompt_data.py          ← dataset prep (--source verified|lite)
├── tests/                ← tests for lib/ — all hermetic
│   ├── test_coding_sandbox.py      ← 18 tests, stubs `modal`
│   ├── test_metric_fix.py          ← 10 tests, stubs slime
│   ├── test_codingagent.py         ← parser tests (no live sandbox needed)
│   ├── test_rollout_metrics.py   ← older metric tests
│   └── smoke_test_sandbox.py       ← end-to-end live Modal smoke test
├── 01-initial-bringup-29747/
├── 02-context-scaling-29922/
├── 03-slime-metrics-30045/
├── 04-pd-disaggregation-abandoned/
├── 05-resumable-smoke-30707/
├── 06-update-weights-interval-30922/
├── 07-switch-to-lite-40/
├── 08-metadata-metric-fix-30939/
├── 09-resumable-full-run-31011/
├── 10-base-control-31079/
├── 11-sglang-deepep-16k-response/
├── 12-deepep-normal-16k-response/
└── 13-ep8-no-deepep-16k-response/
```

## The journey, summarized

**Bring-up → context scaling → instrumentation → PD disaster → resumable
smoke → tuning → bug-hunting → headline runs.**

1. **01 — Initial bring-up.** Modal + 6-tool agent + fully-async Megatron
   on 12 nodes. Verified the whole stack lit up on SWE-bench Verified-10.
   Trajectories hit 32K context — needed more headroom.

2. **02 — Context scaling sweep.** Tested 16K / 32K / 64K / 128K trainer
   context caps. **64K is the knee:** smallest cap with 0 truncations on
   Qwen3-Thinking's natural trajectory distribution.

3. **03 — Slime metric instrumentation.** Added `rollout_metrics.py`
   wrapping the agent's `generate`, `reward_func`, and a custom rollout-log
   hook. Per-tool sandbox metrics, per-SWE-bench outcomes, and a
   policy-staleness metric. Validated on a 20-step run (job 30045).

4. **04 — PD disaggregation.** Tried prefill/decode separation on the
   rollout pool to potentially speed up rollouts. Three config-level fixes
   all failed with the same `KVTransferError: Failed to get kvcache from
   prefill instance`. Traced to a Mooncake C++ binding bug. **Abandoned.**

5. **05 — Resumable smoke.** Added park/resume to the sandbox pool and a
   resume-detection branch to the agent loop, both gated behind
   `--partial-rollout`. Validated PARKED/RESUMED end-to-end. Ran 20 steps,
   8h19m. *But:* `policy_staleness` reported 0 throughout — suspicious.

6. **06 — `--update-weights-interval=5`.** Cut weight-push overhead by
   pushing every 5 steps. **52% step-time reduction** (1078s → 511s mean).
   The staleness silence got more inexplicable — now staleness should
   definitely be >0.

7. **07 — Switch to SWE-bench Lite.** Lite-40 (4 instances × 10 repos)
   has denser per-group reward variance than Verified-10. Used for all
   subsequent experiments.

8. **08 — Metric bug found.** `policy_staleness` and `resume_count` were
   stored as plain attributes on the `Sample` object. Ray's cloudpickle
   strips arbitrary attributes when round-tripping through the partial-
   rollout buffer; only declared dataclass fields survive. **Fix:** move
   both into `sample.metadata` (a real field). Also made staleness
   interval-aware (`weight_version = rollout_id // update_weights_interval`).
   Added 10 hermetic tests covering sticky stamping, pickle round-trip,
   interval-aware math, etc.

9. **09 — Resumable full run with fixed metrics.** Same setup as 05 but
   on Lite-40 with interval=5 and the metric fix. **FAILED at 7h25m**
   with a gloo recv timeout. Forensics: the 37 samples parked at the
   first weight push got stuck in an abort-cycling pathology, each
   re-dispatched 400-700 times producing ~70 tokens per cycle. By step
   14, rollout_time was 4969s — exceeded gloo's 30-minute collective
   timeout. **Trainer crashed.** The metric pipeline is verified working
   (PARKED=63=RESUMED=63), but the training dynamics under naive
   partial-rollout are pathological for long-context agentic tasks.

10. **10 — Base control, no partial-rollout.** Identical setup minus
    `--partial-rollout`. **TIMEOUT at 8h, 10 training steps.** Different
    pathology: monolithic slow tails. Step 8 alone took 4h 5m — one or
    two trajectories produced ~40-50K tokens of agentic + thinking
    output and gated the entire rollout.

11. **11 — SGLang DeepEP `auto`/`low_latency` MoE backend, EP=8, 16K cap.**
    First test of inference-side DeepEP. Faster weight broadcast (35s vs
    10's 43s), but the post-push regime saw the **straggler-tail collapse**:
    once one slow sample is alone on an engine, DeepEP `low_latency`'s
    fixed RDMA-buffer overhead is paid for a 1-token dispatch instead of
    amortized across 8 → ~8× per-token slowdown on the straggler. Tail
    steps `p7-p8 ≈ 3500s/step`. **Cancelled at 11 perfs** with no reward
    movement.

12. **12 — DeepEP `normal` mode, same config otherwise. DOA.** `deepep_mode=normal`
    is incompatible with SGLang's cuda graph capture pipeline: 0% cuda
    graph hit rate across 306 decode batches. Pure eager decode at
    5-11 tok/s (vs 22-35 in 11's eager fallback). **Cancelled at 59 min,
    0 perfs.** Confirms there's no usable DeepEP mode for our scaffold.

13. **13 — Plain `alltoall` + EP=8 + 16K cap.** Final isolating test: same
    EP and response cap as 11 but with the default SGLang MoE backend.
    Result: **per-step times nearly identical to 11** in the tail
    (p7-p8 sum = 6749s vs 11's 6709s, <1% difference). So the straggler
    problem is **workload-fundamental, not DeepEP-specific**. The
    inference backend doesn't matter for the long-context decode tail.
    TIMEOUT at 8h, 9 perfs, raw_reward still flat at ~5%.

## What this told us about agentic RL on Qwen3-Thinking

- The plumbing is solid. Slime, SGLang, Megatron, Modal, the metric
  wrapper, partial-rollout, weight intervals, JSONL metric capture —
  all verified.
- The bottleneck is per-sample wall-clock cost. Mean trajectory does 20-22
  tool calls (max_turns=30 doesn't bind), but per-turn cost varies wildly
  because Qwen3-Thinking emits 8-15K chain-of-thought tokens per hard turn.
- **`--rollout-max-response-len 16K` is the single most impactful flag.**
  10 (32K) had a 4-hour catastrophic step. 11/13 (16K) cap the worst step
  at ~3.5K seconds, yielding ~2× cumulative wallclock improvement.
- **No inference-side config fixes the straggler tail.** DeepEP modes
  (auto, normal) and plain alltoall all converge to the same post-push
  per-step time once the rollout is in the long-tail regime. The driver
  is single-sample-on-engine decode inefficiency at long context, not
  the all-to-all backend.
- The effective best config is **alltoall + EP=8 + 16K + interval=5**
  (= experiment 13's setup). DeepEP isn't worth the complexity.
- Reward signal is **flat at ~5% across all of 09-13**. The slime
  plumbing isn't the blocker for learning — the workload is. Real
  progress would need reward shaping, a smaller/faster model, or
  more curated easy instances.

## Running tests

All hermetic — no Modal, no slime, no GPUs required:

```bash
python3 tests/test_coding_sandbox.py    # 18 tests
python3 tests/test_metric_fix.py        # 10 tests
```

`smoke_test_sandbox.py` uses real Modal — needs `MODAL_CONFIG_PATH` and
network access.
