import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import Calib
from config import ShareConfig
from prepare_data import prepare_data
from utils import compute_num_basis


SHARED_PARTS = {
    "v": "self_attn.v_proj",
    "k": "self_attn.k_proj",
    "q": "self_attn.q_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
}


class FactorizedLinear(nn.Module):
    def __init__(self, basis, coefficient):
        super().__init__()
        self.basis = basis
        self.coefficient = coefficient

    def forward(self, x):
        return self.coefficient(self.basis(x))


def make_layer_groups(num_layers, group_size):
    full_groups = num_layers // group_size
    rest = num_layers % group_size
    groups = [[group_size * i + j for j in range(group_size)] for i in range(full_groups)]
    if rest:
        groups.append([full_groups * group_size + i for i in range(rest)])
    return groups


def split_module_name(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def get_group_weight(model, group, module_name):
    weights = []
    for layer_idx in group:
        weight = model.model.layers[layer_idx].get_submodule(module_name).weight.detach().T
        weights.append(weight)
    return torch.cat(weights, dim=-1).double()


def factorize_group(model, group, module_name, num_basis, calib_path):
    weight = get_group_weight(model, group, module_name)
    device = weight.device
    s, inv_s = Calib.get_s_inv_s(group, module_name, "llama2", calib_path)
    s = s.to(device=device, dtype=torch.float64)
    inv_s = inv_s.to(device=device, dtype=torch.float64)

    weighted = s @ weight
    gram = weighted @ weighted.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:num_basis]
    sigma = torch.clamp(eigenvalues[order], min=0).sqrt()
    u = eigenvectors[:, order]

    basis = inv_s @ (u * sigma)
    coeff = (u.T @ weighted) / torch.clamp(sigma[:, None], min=1e-12)

    del weight, s, inv_s, weighted, gram, eigenvalues, eigenvectors, u, sigma
    return basis.float(), coeff.float()


def parse_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def target_rank(nx, basis_rank, num_basis_tensors, compression):
    keep_ratio = 1.0 - compression / 100.0
    original = nx * basis_rank * num_basis_tensors
    denom = nx + basis_rank * num_basis_tensors
    return max(1, int((original * keep_ratio) // denom))


def compute_ppl(model, data, max_length, stride, device, max_eval_steps):
    model.eval()
    seq_len = data.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    steps = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="ppl", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            output = model(input_ids, labels=target_ids)
        nlls.append(output.loss.detach().float().cpu())
        prev_end_loc = end_loc
        steps += 1
        if end_loc == seq_len or (max_eval_steps and steps >= max_eval_steps):
            break
    return torch.exp(torch.stack(nlls).mean()).item()


def install_factorized_shared_parts(model, model_name, calib_path, group_size, compression_ratio, device, selected_parts):
    short_name = ShareConfig.name_map[model_name]
    groups = make_layer_groups(model.config.num_hidden_layers, group_size)
    basis_modules = {}
    baseline_basis = {}
    logical_params = {}

    for part in selected_parts:
        module_name = SHARED_PARTS[part]
        nx, nf = ShareConfig.weight_info[short_name][module_name]
        num_basis = compute_num_basis(nx, nf, group_size, compression_ratio)
        basis_modules[part] = []
        baseline_basis[part] = []
        logical_params[part] = {
            "nx": nx,
            "nf": nf,
            "num_basis": num_basis,
            "num_basis_tensors": len(groups),
            "original_basis_params": nx * num_basis * len(groups),
        }
        print(f"first-level {part}: {module_name}, num_basis={num_basis}, groups={len(groups)}", flush=True)

        for group in tqdm(groups, desc=f"build {part}"):
            basis_weight, coeff_weight = factorize_group(model, group, module_name, num_basis, calib_path)

            basis = nn.Linear(nx, num_basis, bias=False, device=device, dtype=torch.float16)
            basis.weight.data.copy_(basis_weight.T.to(device=device, dtype=torch.float16))
            basis_modules[part].append(basis)
            baseline_basis[part].append(basis_weight.contiguous())

            for group_offset, layer_idx in enumerate(group):
                coefficient = nn.Linear(num_basis, nf, bias=False, device=device, dtype=torch.float16)
                start = group_offset * nf
                end = start + nf
                coefficient.weight.data.copy_(
                    coeff_weight[:, start:end].T.to(device=device, dtype=torch.float16)
                )
                parent, attr = split_module_name(model.model.layers[layer_idx], module_name)
                setattr(parent, attr, FactorizedLinear(basis, coefficient))

            del basis_weight, coeff_weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return basis_modules, baseline_basis, logical_params


def apply_basis_of_basis(basis_modules, baseline_basis, bob_group_size, compression):
    summary = {}
    for part, tensors in baseline_basis.items():
        nx, basis_rank = tensors[0].shape
        ranks = []
        original_params = 0
        hierarchical_params = 0
        weighted_error_num = 0.0
        weighted_error_den = 0.0

        for start in range(0, len(tensors), bob_group_size):
            chunk = tensors[start:start + bob_group_size]
            chunk_size = len(chunk)
            rank = target_rank(nx, basis_rank, chunk_size, compression)
            matrix = torch.cat(chunk, dim=1)
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            effective_rank = min(rank, u.shape[1])
            rec = (u[:, :effective_rank] * s[:effective_rank]) @ vh[:effective_rank, :]

            energy = s.square()
            total = energy.sum().item()
            kept = energy[:effective_rank].sum().item()
            weighted_error_num += max(total - kept, 0.0)
            weighted_error_den += total
            ranks.append(effective_rank)
            original_params += nx * basis_rank * chunk_size
            hierarchical_params += nx * effective_rank + effective_rank * basis_rank * chunk_size

            for offset, tensor in enumerate(chunk):
                block = rec[:, offset * basis_rank:(offset + 1) * basis_rank]
                module = basis_modules[part][start + offset]
                module.weight.data.copy_(block.T.to(device=module.weight.device, dtype=module.weight.dtype))

        summary[part] = {
            "ranks": ranks,
            "mean_rank": sum(ranks) / len(ranks),
            "original_basis_params": original_params,
            "hierarchical_basis_params": hierarchical_params,
            "basis_saving_pct": (1 - hierarchical_params / original_params) * 100,
            "relative_error": math.sqrt(weighted_error_num / weighted_error_den) if weighted_error_den else 0.0,
        }
    return summary


def restore_baseline_basis(basis_modules, baseline_basis):
    for part, tensors in baseline_basis.items():
        for module, tensor in zip(basis_modules[part], tensors):
            module.weight.data.copy_(tensor.T.to(device=module.weight.device, dtype=module.weight.dtype))


def aggregate_summary(summary):
    original = sum(item["original_basis_params"] for item in summary.values())
    hierarchical = sum(item["hierarchical_basis_params"] for item in summary.values())
    rel_num = sum((item["relative_error"] ** 2) * item["original_basis_params"] for item in summary.values())
    rel_den = original
    return {
        "basis_params": hierarchical,
        "basis_saving_pct": (1 - hierarchical / original) * 100,
        "relative_error": math.sqrt(rel_num / rel_den) if rel_den else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run LLaMA shared-basis basis-of-basis pilot with PPL.")
    parser.add_argument("--model-name", default="jeffwan/llama-7b-hf")
    parser.add_argument("--calib-path", required=True)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--first-group-size", type=int, default=2)
    parser.add_argument("--compression-ratio", type=float, default=20)
    parser.add_argument("--bob-group-sizes", default="2,4,8,16")
    parser.add_argument("--bob-basis-compression", type=float, default=20)
    parser.add_argument("--shared-parts", default="v,k,q,up,gate")
    parser.add_argument("--max-eval-steps", type=int, default=32)
    parser.add_argument("--output-csv", default="experiments/llama7b_shared_bob_group_size_pilot.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = "[PAD]"
    _, _, test_dataset, _ = prepare_data(
        args.dataset_name,
        tokenizer,
        args.context_length,
        args.dataset_cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    model.config.use_cache = False
    model.to(device)

    selected_parts = [part.strip() for part in args.shared_parts.split(",") if part.strip()]
    unknown_parts = sorted(set(selected_parts) - set(SHARED_PARTS))
    if unknown_parts:
        raise ValueError(f"Unknown shared parts: {unknown_parts}")

    basis_modules, baseline_basis, _ = install_factorized_shared_parts(
        model,
        args.model_name,
        args.calib_path,
        args.first_group_size,
        args.compression_ratio,
        device,
        selected_parts,
    )

    rows = []
    print("Evaluate first-level shared-basis baseline", flush=True)
    restore_baseline_basis(basis_modules, baseline_basis)
    baseline_ppl = compute_ppl(model, test_dataset, args.context_length, args.stride, device, args.max_eval_steps)
    rows.append(
        {
            "bob_group_size": "baseline",
            "basis_params": sum(t.numel() for tensors in baseline_basis.values() for t in tensors),
            "basis_saving_pct": 0.0,
            "relative_error": 0.0,
            "ppl": baseline_ppl,
        }
    )

    for bob_group_size in parse_ints(args.bob_group_sizes):
        print(f"Evaluate basis-of-basis group size {bob_group_size}", flush=True)
        restore_baseline_basis(basis_modules, baseline_basis)
        summary = apply_basis_of_basis(
            basis_modules,
            baseline_basis,
            bob_group_size,
            args.bob_basis_compression,
        )
        agg = aggregate_summary(summary)
        ppl = compute_ppl(model, test_dataset, args.context_length, args.stride, device, args.max_eval_steps)
        rows.append(
            {
                "bob_group_size": bob_group_size,
                "basis_params": agg["basis_params"],
                "basis_saving_pct": agg["basis_saving_pct"],
                "relative_error": agg["relative_error"],
                "ppl": ppl,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bob_group_size", "basis_params", "basis_saving_pct", "relative_error", "ppl"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nResults")
    print("bob_group_size,basis_params,basis_saving_pct,relative_error,ppl")
    for row in rows:
        print(
            f"{row['bob_group_size']},{row['basis_params']},"
            f"{row['basis_saving_pct']:.4f},{row['relative_error']:.6f},{row['ppl']:.6f}"
        )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
