import argparse
import os
import torch
from collections import defaultdict

from config import ShareConfig, add_args
from model_factory import create_model
from models.model_utils import Coefficient, Basis
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Reorder basis channels by global importance")
    parser.add_argument("--cf", "--yaml_config_file",
                        default="tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml",
                        type=str)
    parser.add_argument("--save_path", type=str, default="channel_permutation.pt",
                        help="Where to save the permutation dictionary")
    parser.add_argument("--output_dir", type=str, default="untrained_model_reordered",
                        help="Directory to save reordered checkpoint")
    return parser.parse_args()


def collect_importance(model):
    importance = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, Coefficient):
            score = module.weight.abs().mean(dim=0)
            importance[name].append(score)
    merged = {}
    for name, scores in importance.items():
        merged_score = torch.stack(scores).mean(dim=0)
        perm = torch.argsort(merged_score, descending=True)
        merged[name] = {"score": merged_score, "perm": perm}
    return merged


def apply_reorder(model, perm_map):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Coefficient) and name in perm_map:
                perm = perm_map[name]["perm"]
                module.weight.copy_(module.weight[:, perm])
                if hasattr(module, "scaling_diag") and module.scaling_diag is not None:
                    if module.scaling_diag.shape[0] >= perm.shape[0]:
                        module.scaling_diag.copy_(module.scaling_diag[perm])
            if isinstance(module, Basis) and name in perm_map:
                perm = perm_map[name]["perm"]
                module.weight.copy_(module.weight[perm, :])


def main():
    args = parse_args()
    base_args = add_args()
    base_args.yaml_config_file = args.cf
    cfg = ShareConfig(base_args)
    model = create_model(cfg)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    perm_map = collect_importance(model)
    torch.save(perm_map, args.save_path)
    print(f"Saved permutation map to {os.path.abspath(args.save_path)}")

    apply_reorder(model, perm_map)
    print("Applied in-place reorder on model weights.")
    model.save_pretrained(args.output_dir, safe_serialization=False)
    print(f"Saved reordered model to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
