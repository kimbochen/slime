# Reproducing the Lite-40 dataset

```bash
cd examples/qwen3-235b_fullasync_swe-env/lib
./gen_prompt_data.py \
  --source lite \
  --n-instances 40 \
  --per-repo 4 \
  --out ../../../mnt/data/swebench_lite/sample_40.jsonl
```

Then run any experiment that has `PROMPT_DATA=...swebench_lite/sample_40.jsonl`
in its `run_swe.sh` — experiments 09, 10, etc. all default to this.
