# 07 — Switch from SWE-bench Verified-10 to Lite-40

**Status:** ✅ adopted as default
**Lib SHA:** `a5878973` (current; added `--source lite|verified` to `gen_prompt_data.py`)
**Dataset:** Lite sample_40 — 4 instances × 10 repos = 40 problems
**Headline:** Lite has denser per-group reward variance than Verified
(simpler patches → more groups with non-zero std → more learning signal
for GRPO). Switch was a one-line `--source` flag in the prompt-data
generator.

## Hypothesis

With `--n-samples-per-prompt=8` and ~5% pass rate, most groups have 0/8
successes (zero in-group variance → zero advantage → no gradient signal).
Switching to Lite (simpler patches) should raise the pass rate, putting
more groups in the 1-7/8 regime where GRPO has meaningful signal.

## What changed

`lib/gen_prompt_data.py` gained `--source {verified, lite}`:
```bash
# Verified (default in old setup)
./gen_prompt_data.py --source verified --n-instances 10 --per-repo 2 --out swebench_verified/sample_10.jsonl

# Lite (new default)
./gen_prompt_data.py --source lite --n-instances 40 --per-repo 4 --out swebench_lite/sample_40.jsonl
```

Then `PROMPT_DATA` in `run_swe.sh` flipped to the Lite jsonl.

## Result

`rollout/zero_std/count_0.1` (samples in zero-variance groups) dropped from
~24/64 on Verified → ~7/64 on Lite-40. So ~7/8 of Lite groups had usable
variance, vs ~5/8 of Verified groups.

Raw_reward at baseline was similar (~5%) but the *gradient signal density*
roughly doubled. This is the dataset used in **experiments 08, 09, 10**.

## Caveats

- Lite is "simpler" only in the sense of having smaller gold patches. SWE-bench
  Lite leaderboard top entries (with elaborate SWE-agent scaffolds) hit ~60%;
  our 6-tool agent on Qwen3-Thinking sits at ~5%. The dataset doesn't fix the
  agent/scaffold limitations.
- Lite-40's denser variance doesn't translate to faster learning in our
  experiments because **rollout cost dominates** — see 09 and 10.
