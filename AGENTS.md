# Basis Sharing – Project Map

## What this repo does
- Implements the “Basis Sharing” cross-layer parameter sharing method for LLM compression (paper: Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression).
- Supports GPT-2, LLaMA-2, OPT, and Mistral; replaces per-layer weight matrices with shared basis modules plus layer-specific coefficients to reduce parameters.
- Uses calibration activations to derive bases; optionally builds update datasets to refine coefficients after compression; integrates LoRA finetuning and evaluation utilities (perplexity, reasoning benchmarks, throughput).

## Layout
- `config.py` – CLI arg parsing (`--cf/--yaml_config_file`, dataset overrides) and `ShareConfig` that loads YAML configs, attaches attributes, maps model names to short names, and stores reference weight shapes for basis math.
- `prepare_data.py` – Dataset loaders/tokenizers for WikiText-2, PTB, C4 (JSON cache), and Alpaca; handles tokenization, chunking to fixed context length, and returns collators.
- `calib.py` – Hooks to collect activation covariance (`X^T X`) per layer; saves/loads `.pkl` calibration shards; builds calibration datasets (initial compression) and update datasets (post-compression coefficient refresh); computes scaling matrices (`S`, `S^-1`) via Cholesky.
- `group.py` – Forms groups of layers that share a basis. `Group` builds bases/coefficients from calibration stats, writes weights into shared basis modules and per-layer coefficient modules; `change_model` applies basis sharing once; `update_model` recomputes coefficients later with new calibration data.
- `model_factory.py` – Orchestrates model creation and compression:
  - Loads tokenizer/std model, optionally builds calibration data.
  - Computes layer groups and number of bases from config/compression ratio.
  - Instantiates shared model variants (`ShareGPT2...`, `ShareLlama...`, `ShareOPT...`, `ShareMistral...`), loads matching weights, and applies `change_model`.
  - Can reload previously saved “untrained” compressed models or apply `update_model`; saves untrained/updated checkpoints.
- `models/` – Shared-model implementations:
  - `model_utils.py` supplies `Basis`/`Coefficient` linear layers and `build_basis_collection`.
  - `gpt2.py`, `llama.py`, `opt.py`, `mistral.py` wrap HF models, swapping projections/MLPs for basis+coefficient modules, with grouping-aware sharing.
- `utils.py` – `match_state_dict` for compatible parameter loading; `compute_num_basis` computes basis count given compression ratio/group size.
- Experiments/scripts:
  - `test.py` – Compute perplexity over a tokenized test set (sliding window).
  - `lora.py` – LoRA finetuning on compressed model; reuses `create_model`, trains via HF `Trainer`, optionally tests PPL.
  - `train.py` – Full finetuning of compressed model with HF `Trainer` (baseline setup).
  - `test_adapter.py` – Runs `lm_eval` tasks with a HF model (currently loads vanilla Mistral).
  - `test_latency_throughput.py` – Torch-compile + batches to measure throughput (tokens/sec) after a warmup.
- `tasks/configs/.../*.yaml` – Experiment configs organized by dataset/model. Key sections:
  - `model_args`: model type/name, target submodule names, grouping size, compression ratio, context/stride, which parts are shared vs private.
  - `calibration_args`: dataset name/cache, calibration toggle/paths, calib sample size/batch.
  - `after_calibration_update_args`: update calib paths/toggles.
  - `model_saving`: paths/toggles for untrained (compressed) and updated models.
  - `lora_args`: LoRA hyperparams and output path.
- `requirements.txt` – Full dependency pin set (Transformers, Accelerate, PEFT, lm_eval, datasets, etc.).

## Typical flows
- **Compress & eval PPL**: `python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml` (set `build_calib=true` first run). Creates calibration, builds shared model, saves untrained model if enabled, then reports perplexity.
- **Reuse compressed checkpoint**: Set `save_untrained_model=true` once; later runs load from `untrained_model_path` to skip calibration/compression.
-. **Coefficient update**: With `build_update_calib=true`, builds update stats and runs `update_model` to refresh coefficients; can save to `updated_model_path`.
- **LoRA finetune**: `python lora.py --cf <config>`; loads compressed model, wraps with PEFT LoRA, trains/evals, optionally saves adapters.
- **Throughput check**: `python test_latency_throughput.py --cf <config>`; uses subset of val data, torch.compile, measures tokens/sec.

## Key concepts/notes
- Grouping: Layers are grouped (size `group_size`) so basis weights are shared; private parts default to group-of-1.
- Basis size: `compute_num_basis` converts compression ratio into basis rank given weight shape and group size.
- Calibration files: Stored under `calib_path/<submodule>/<layer_id>.pkl`; reused across runs. Missing files will raise during compression/update.
- HF compatibility: `match_state_dict` only loads matching-shaped params from the reference model; remaining weights keep module defaults.
- Tokenizer: Pads with literal `"[PAD]"` to support collation; RoPE models use HF rotary embedding functions unchanged.

## Worklog (Dynamic Rank idea)
- 2024-xx-xx: Reviewed `modification.md` chat guidance. Plan: add config hooks for dynamic rank (threshold, max basis), capture scaling matrices, and prototype dynamic coefficient layer using S-scaled projections; start with non-invasive scaffolding.
- 2024-xx-xx: Added CLI options `--dynamic_rank_threshold` and `--max_basis_rank` (config.py) for future dynamic rank control. Introduced masking-aware `Coefficient` that supports optional scaling/thresholding/max-rank; currently defaults to prior behavior unless these fields are set.
- 2024-xx-xx: Persisted scaling diag during calibration (calib.py) and propagate Cholesky diag per group. `Coefficient` now registers optional scaling diag buffers, applies scaling + dynamic masking in forward, tracks last active ratio, and respects max-rank truncation. All model wrappers pass dynamic config; change_model sets scaling diag per coefficient when available.
- 2024-xx-xx: Hardened scaling broadcast in `Coefficient._apply_scaling`; ensured dynamic config injected before `from_pretrained` loads so thresholds are honored. Added active-ratio monitor in `test.py` that averages `Coefficient.last_active_ratio` across batches and prints during PPL eval.
- 2024-xx-xx: Switched dynamic pruning to cumulative energy pruning. Added `--dynamic_energy_threshold` to config, propagated into all model wrappers, and reworked `Coefficient.forward` to sort energy, compute cumsum cutoff, scatter mask back, and report active ratio under energy-based sparsity.
- 2024-xx-xx: Added mask capture support in `Coefficient` (store_mask/last_mask_tensor) and a visualization script `analyze_sparsity_pattern.py` to run a sample forward, grab a layer mask, plot a heatmap, and compute block sparsity stats.
- 2024-xx-xx: Prototype decode gather path in `Coefficient` (optional `decode_gather` flag) for seq_len=1 using energy-based index selection, plus offline channel importance tool `reorder_channels.py` (saves permutations; application optional).
- 2024-xx-xx: `reorder_channels.py` now applies permutations to Coefficient/Basis (and scaling_diag) and saves reordered checkpoints; `benchmark_decode.py` benchmarks dense vs decode-gather latency on reordered models; decode gather includes prefix fast-path when active indices are contiguous.
- 2024-xx-xx: Added `assume_sorted` no-sort decode path; decode gather can skip sort/gather and slice by cumulative energy; benchmark now forces no-sort path; profiling support remains for scaling/decision/gather/compute.
- 2024-xx-xx: Added `static_k` support via config; all model wrappers propagate it to Coefficient layers. `test.py` can now run PPL with static slicing (disables dynamic). Benchmark gathers k stats and supports static slice benchmarking.
- 2024-xx-xx: Refactored `Coefficient` to explicit strategies (baseline/global_static/prompt_adaptive) driven by target_ratio; removed legacy flags/static_k/energy thresholds. Added `verify_strategies.py` to compare strategies under same budget (PPL, avg k, energy kept); configs/wrappers use `strategy` and `target_ratio`.
