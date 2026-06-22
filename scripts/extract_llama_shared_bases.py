import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import Calib
from config import ShareConfig
from utils import compute_num_basis


SHARED_PARTS = {
    "v": "self_attn.v_proj",
    "k": "self_attn.k_proj",
    "q": "self_attn.q_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
}


def make_layer_groups(num_layers, group_size):
    full_groups = num_layers // group_size
    rest = num_layers % group_size
    groups = [[group_size * i + j for j in range(group_size)] for i in range(full_groups)]
    if rest:
        groups.append([full_groups * group_size + i for i in range(rest)])
    return groups


def get_llama_group_weight(std_model, group, module_name):
    weights = []
    for layer_idx in group:
        weight = std_model.model.layers[layer_idx].get_submodule(module_name).weight.detach().T
        weights.append(weight)
    return torch.cat(weights, dim=-1).double()


def compute_basis_only(std_model, group, module_name, num_basis, calib_path):
    weight = get_llama_group_weight(std_model, group, module_name)
    device = weight.device
    s, inv_s = Calib.get_s_inv_s(group, module_name, "llama2", calib_path)
    s = s.to(device=device, dtype=torch.float64)
    inv_s = inv_s.to(device=device, dtype=torch.float64)

    weighted = s @ weight
    gram = weighted @ weighted.T
    del weighted, weight, s
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    order = order[:num_basis]
    eigenvalues = torch.clamp(eigenvalues[order], min=0)
    eigenvectors = eigenvectors[:, order]
    basis = inv_s @ (eigenvectors * eigenvalues.sqrt())
    return basis.float()


def main():
    parser = argparse.ArgumentParser(description="Extract first-level shared LLaMA bases without building the full model.")
    parser.add_argument("--model-name", default="jeffwan/llama-7b-hf")
    parser.add_argument("--calib-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--compression-ratio", type=float, default=20)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model_config = AutoConfig.from_pretrained(args.model_name)
    std_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=dtype)
    std_model.config.use_cache = False

    short_model_name = ShareConfig.name_map[args.model_name]
    groups = make_layer_groups(model_config.num_hidden_layers, args.group_size)

    pseudo_config = model_config.to_dict()
    pseudo_config["model_type"] = "llama2"
    pseudo_config["basis_source_model"] = args.model_name
    pseudo_config["basis_group_size"] = args.group_size
    pseudo_config["basis_compression_ratio"] = args.compression_ratio

    state_dict = {}
    for part, module_name in SHARED_PARTS.items():
        nx, nf = ShareConfig.weight_info[short_model_name][module_name]
        num_basis = compute_num_basis(nx, nf, args.group_size, args.compression_ratio)
        pseudo_config[f"{part}_groups"] = groups
        pseudo_config[f"num_basis_{part}"] = num_basis

        print(f"Extract {part}: module={module_name} num_basis={num_basis} groups={len(groups)}")
        for group in groups:
            basis = compute_basis_only(std_model, group, module_name, num_basis, args.calib_path)
            weight = basis.T.detach().cpu().contiguous()
            state_dict[f"model.{part}_basis.{group[0]}.weight"] = weight
            del basis, weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(pseudo_config, f, indent=2)
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    print(f"Wrote shared basis pseudo checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
