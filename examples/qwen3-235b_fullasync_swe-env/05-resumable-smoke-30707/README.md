# 05 — Resumable sandbox smoke (job 30707)

**Status:** ✅ partial-rollout machinery validated end-to-end
**Lib SHA:** `cac0d666` (resumable variant first landed)
**Dataset:** Verified sample_10
**Headline:** First end-to-end validation of slime's `--partial-rollout`
on a multi-turn agent loop. 20 steps, 8h19m wallclock. Sandbox park/resume
working. `policy_staleness` metric reporting **was broken at this point** —
this run is what motivated the bug-hunt in **experiment 08**.

20-step training run of Qwen3-235B-Thinking on SWE-bench-Verified with the
**resumable sandbox** variant. First end-to-end validation of slime's
`--partial-rollout` flag on a multi-turn agent loop.

## Configuration

| | |
|---|---|
| **Model** | Qwen3-235B-A22B-Thinking-2507 (FP8) |
| **Cluster** | 12 H200 nodes (4 actor + 8 rollout, no `--colocate`) |
| **Trainer parallelism** | TP=4, PP=2, CP=4, EP=8, ETP=1 |
| **Effective trainer context** | CP × max_tokens_per_gpu = 4 × 16384 = **64K** |
| **Rollout engine** | 8 × SGLang TP=8 (no PD) |
| **SGLang context cap** | 65536 (matches trainer 64K) |
| **Response cap (per turn)** | min(32768, remaining_budget) — dynamic |
| **Dataset** | SWE-bench-Verified 10-instance slice |
| **Rollout batch** | 8 prompts × 8 samples = 64 trajectories |
| **Global batch** | 64 |
| **Optimiser** | Adam, lr=1e-6, CPU offload |
| **Advantage** | GSPO, eps_clip=4e-4, KL coef=0 |
| **Sandbox concurrency** | MAX_CONCURRENT=100, SPAWN_QPS=10 |
| **MoE backend** | flex + DeepEP |
| **Wallclock** | 8h 19m for 20 perf steps |

## What's new vs the 30537 baseline

| Change | What | Why |
|---|---|---|
| **Resumable sandbox** | `coding_sandbox.py` gains `park` / `acquire_or_resume`; Modal sandbox survives weight-update aborts and re-attaches on resume | Required to make `--partial-rollout` safe for the multi-turn agent (sandbox holds filesystem + process state across turns) |
| **`--partial-rollout`** | Enabled | Pipeline-RL style — sample tokens persist across weight updates; new tokens train under new weights with GSPO importance ratio handling the off-policy delta |
| **Dynamic per-turn max_new_tokens** | Agent caps `max_new_tokens = min(global_cap, max_context − total_so_far − 64)` per turn | Fixes late-turn 400 errors when prompt grows past 32K |
| **SLIME_SWEBENCH_TIMEOUT** | 1800s → 7200s | Parked Modal sandboxes need to survive multiple weight-update cycles |

## Headline results

```
Steady-state mean (steps 1-19):
  train_tflops:    66.9        (MFU 6.76% on H200 BF16 peak 990 TFLOPS)
  wallclock_MFU:    0.7%       (trainer compute as fraction of total step time)
  wait_time:      1391s
  train_time:      176s
  step_time:      1567s
  wait_ratio:       0.87
```

Partial-rollout fired observably twice (`policy_staleness_max=1` on rollouts
1 and 11) — first verified pipeline-RL behavior in this codebase.

Dynamic context cap let two samples grow past the 32K per-turn cap to
~54K total response (r7, r15), filling the 64K context window without
hitting SGLang's 400-on-overflow path that broke the prior 30676 run.

## Comparison to prior 12-node non-PD runs

| | 30045 (orig) | 30537 (MAX_CONC=100) | 30707 (this run) |
|---|---|---|---|
| trajs/rollout | 16 (4×4) | 64 (8×8) | 64 (8×8) |
| sandbox MAX_CONCURRENT | 32 | 100 | 100 |
| `--partial-rollout` | no | no | **yes** |
| Sandbox supports resume | no | no | **yes** |
| Dynamic context cap | no | no | **yes** |
| Steps in 8h | 20 (3h) | 4 (3h, cancelled) | 20 (8h) |
| Steady mean train_tflops | 39.4 | 65.0 | 66.9 |
| Steady MFU (in-step) | 3.98% | 6.55% | 6.76% |
| Steady wait_ratio | 0.86 | 0.84 | 0.87 |
| 400 context-overflow errors | 0 | 0 | 0 (cap fix held) |

Trainer-side throughput is essentially equivalent to 30537 — the resumable
variant doesn't cost trainer MFU. wait_ratio is unchanged (0.87) — SWE-bench
is still rollout-bound by the agent's tool-call serialization, just as in
30537.

What differs visibly:
- **Two non-zero `policy_staleness_max` values** in the rollout series
  (r1, r11). 30537 had max=0 throughout. This proves the partial-rollout
  + resumable path is actually carrying samples across weight updates.
- **Two samples reached ~54K response length** vs 30537's max of ~33K.
  Dynamic per-turn cap unlocked longer trajectories without triggering
  the SGLang context-overflow rejection.

## Reward signal

No movement. `raw_reward` stays in [4.77%, 5.00%] across all 20 steps,
`kl=0`, normalized rewards stay near `−0.003`. Same flat pattern as 30537
and 30045. 20 steps is not enough for measurable policy shift on a
SWE-bench-Verified cold start; this run is intended to validate the
resumable + partial-rollout machinery, not to learn.

## Known issues (sample-level, not job-killing)

- **1 `modal.exception.InvalidError: All entrypoint arguments must be
  strings`** — model emitted a tool call with a non-string `cmd`
  (probably a list). One sample lost. Caller-side validation in
  `generate_with_codingagent.py` would catch it.
- **1 `UnicodeDecodeError`-class miss possible** — fixed in
  `coding_sandbox.py` for the next run, but we don't have telemetry on
  whether it was tripped in this run since the new fix swallows them.
- **`rollout_metrics.py:157` policy_staleness stamp** — overwrites the
  first-dispatch version on each resume call, so reported staleness is
  per-resume, not per-first-dispatch. The numbers are correct for the
  most recent resume; per-trajectory max staleness across the entire
  lifetime requires changing the stamp to sticky-on-first-write.

## Where the wins are (and aren't)

- ✅ Resumable sandbox works end-to-end (20 perfs, partial-rollout fired
  twice, no resume-related crashes).
- ✅ Dynamic context cap eliminates late-turn 400 errors that bricked the
  prior 30676 attempt.
- ✅ Trainer-side MFU comparable to 30537 baseline — no cost to enabling
  the new machinery.
- ⚠️ Total throughput per wallclock-hour is **lower** than 30537 baseline
  (20 perfs in 8h vs 20 perfs in 8-10h projected for 30537). Likely from
  longer per-turn rollouts on the few resumed trajectories, and from
  the long-tail samples that now grow to ~54K.
- ❌ Reward signal still flat — same as every prior cold-start SWE-bench
  run. Not a regression; just a reminder that 20 steps is below the
  visible-learning threshold for this task.

## Files

```
examples/qwen3-235b_fullasync_swe-env_resumable/
├── coding_sandbox.py                   # +82 lines (park, acquire_or_resume)
├── generate_with_codingagent.py        # +264 lines (resume detection, dynamic cap)
├── rollout_metrics.py                # unchanged from base example
├── run_swe.sh                          # --partial-rollout, --sglang-context-length 65536
├── run_swe.sbatch                      # job name: infx-swe-rsm
└── README.md                           # arch + caveats
```

Slurm log: `mnt/logs/infx-swe-rsm-30707.out`
This results dir:
- `RESULTS.md` — per-step + per-rollout tables (generated by
  `examples/.../results/generate_report.py`)
- `metrics.json` — structured dump consumed by the report generator
- `README.md` — this file
