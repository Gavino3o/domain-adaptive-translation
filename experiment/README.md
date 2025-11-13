## Instruction

1. Install all requirements given in `requirements.txt`
2. Download all three models and WMT 2022 test set. The model are given in these link:
  - https://huggingface.co/AIDC-AI/Marco-MT-Algharb
  - https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B
  - https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1
3. Replace all `path/to/model` in all python file to respective model.
4. Copy WMT 2022 test set as `wmttest2022.zh` and `wmttest2022.AnnA.en`.
5. Run all python file. You may use our `sbatch` file to run in slurm. The output is in `model.en`.
6. To score the result, use `score.py` in root folder.
