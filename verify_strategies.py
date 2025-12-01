import argparse
import copy

import torch
from tqdm import tqdm

from config import ShareConfig, add_args
from model_factory import create_model
from prepare_data import prepare_data
from models.model_utils import Coefficient, reset_k_stats, collect_avg_k, collect_energy_ratio
from transformers import LlamaTokenizer, AutoTokenizer


def compute_ppl(max_length, stride, data, model, device):
    model.to(device)
    model = model.eval()
    seq_len = data.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            output = model(input_ids, labels=target_ids)
            nlls.append(output.loss)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def get_tokenizer(cfg):
    if cfg.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(cfg.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = "[PAD]"
    return tokenizer


def eval_strategy(base_cfg, dataset_name, strategy, target_ratio):
    cfg = copy.deepcopy(base_cfg)
    cfg.dataset_name = dataset_name
    cfg.strategy = strategy
    cfg.target_ratio = target_ratio
    tokenizer = get_tokenizer(cfg)
    train_dataset, val_dataset, test_dataset, data_collator = prepare_data(
        cfg.dataset_name, tokenizer, cfg.context_length, cfg.dataset_cache_dir
    )
    model = create_model(cfg)
    reset_k_stats(model)
    ppl = compute_ppl(cfg.context_length, cfg.stride, test_dataset, model, device="cuda")
    avg_k = collect_avg_k(model)
    energy_ratio = collect_energy_ratio(model)
    return ppl, avg_k, energy_ratio


def main():
    parser = argparse.ArgumentParser(description="Compare strategies under same target_ratio")
    parser.add_argument("--cf", "--yaml_config_file",
                        default="tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml", type=str)
    parser.add_argument("--reordered_model_dir", type=str, default=None,
                        help="Path to reordered model to load as untrained_model_path")
    parser.add_argument("--target_ratio", type=float, default=0.5, help="Retention ratio for basis channels")
    parser.add_argument("--datasets", nargs="+", default=["wikitext"],
                        help="Datasets to evaluate (default: wikitext). Add c4 only if locally available.")
    args = parser.parse_args()

    base_args = add_args()
    base_args.yaml_config_file = args.cf
    cfg = ShareConfig(base_args)
    if args.reordered_model_dir is not None:
        cfg.untrained_model_path = args.reordered_model_dir

    datasets = args.datasets
    strategies = ["global_static", "prompt_adaptive"]

    results = []
    for ds in datasets:
        for strat in strategies:
            ppl, avg_k, energy = eval_strategy(cfg, ds, strategy=strat, target_ratio=args.target_ratio)
            results.append((ds, strat, ppl, avg_k, energy))

    print("\nIso-Compute Comparison (target_ratio={})".format(args.target_ratio))
    print("dataset   | strategy         | ppl     | avg_k   | energy_kept")
    for ds, strat, ppl, avg_k, energy in results:
        print(f"{ds:8s} | {strat:15s} | {ppl:7.3f} | {avg_k} | {energy}")


if __name__ == "__main__":
    main()
