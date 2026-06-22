import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch


BASIS_RE = re.compile(r"(?:^|\.)(?P<name>[A-Za-z_]+_basis)\.(?P<layer>\d+)\.weight$")


def load_state_dict(model_dir):
    model_dir = Path(model_dir)
    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No pytorch_model*.bin found in {model_dir}")
    if len(bin_files) > 1:
        raise NotImplementedError("Sharded pytorch_model*.bin files are not supported yet")
    return torch.load(bin_files[0], map_location="cpu")


def load_config(model_dir):
    config_path = Path(model_dir) / "config.json"
    with config_path.open() as f:
        return json.load(f)


def parse_ranks(rank_text):
    if not rank_text:
        return [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    ranks = []
    for item in rank_text.split(","):
        item = item.strip()
        if item:
            ranks.append(int(item))
    if not ranks:
        raise ValueError("--ranks did not contain any integer rank")
    return sorted(set(ranks))


def rank_for_target_basis_compression(nx, basis_rank, num_groups, target_compression):
    keep_ratio = 1.0 - target_compression / 100.0
    original_params = nx * basis_rank * num_groups
    denominator = nx + basis_rank * num_groups
    rank = int((original_params * keep_ratio) // denominator)
    return max(rank, 1)


def chunk_items(tensors, groups, chunk_size):
    if chunk_size <= 0:
        return [(tensors, groups)]
    chunks = []
    for start in range(0, len(tensors), chunk_size):
        chunks.append((tensors[start:start + chunk_size], groups[start:start + chunk_size]))
    return chunks


def index_basis_keys(state_dict):
    basis_keys = {}
    for key in state_dict.keys():
        match = BASIS_RE.search(key)
        if not match:
            continue
        basis_name = match.group("name")
        layer_idx = int(match.group("layer"))
        basis_keys.setdefault(basis_name, {})[layer_idx] = key
    return basis_keys


def groups_for_basis(config, basis_name, layer_keys):
    groups_key = basis_name[: -len("_basis")] + "_groups"
    groups = config.get(groups_key)
    if groups is None:
        return [[layer_idx] for layer_idx in sorted(layer_keys)]
    return groups


def collect_unique_basis(state_dict, config, basis_name, layer_to_key):
    groups = groups_for_basis(config, basis_name, layer_to_key.keys())
    tensors = []
    source_layers = []
    collected_groups = []

    for group in groups:
        group_key = None
        source_layer = None
        for layer_idx in group:
            if layer_idx in layer_to_key:
                source_layer = layer_idx
                group_key = layer_to_key[layer_idx]
                break
        if group_key is None:
            continue

        weight = state_dict[group_key].detach().float().cpu()
        if weight.ndim != 2:
            raise ValueError(f"Expected 2D basis weight for {group_key}, got {tuple(weight.shape)}")
        tensors.append(weight.T.contiguous())
        source_layers.append(source_layer)
        collected_groups.append(group)

    if not tensors:
        raise ValueError(f"No basis tensors collected for {basis_name}")

    shape = tensors[0].shape
    for tensor in tensors:
        if tensor.shape != shape:
            raise ValueError(f"{basis_name} has mixed basis shapes: {shape} and {tensor.shape}")

    return tensors, source_layers, collected_groups


def coefficient_key(config, basis_name, layer_idx):
    stem = basis_name[: -len("_basis")]
    model_type = config.get("model_type")
    module_by_type = {
        "gpt2": {
            "attn": "attn.c_attn",
            "o": "attn.c_proj",
            "up": "mlp.c_fc",
            "down": "mlp.c_proj",
        },
        "llama": {
            "k": "self_attn.k_proj",
            "q": "self_attn.q_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
            "up": "mlp.up_proj",
            "gate": "mlp.gate_proj",
            "down": "mlp.down_proj",
        },
        "llama2": {
            "k": "self_attn.k_proj",
            "q": "self_attn.q_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
            "up": "mlp.up_proj",
            "gate": "mlp.gate_proj",
            "down": "mlp.down_proj",
        },
        "mistral": {
            "k": "self_attn.k_proj",
            "q": "self_attn.q_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
            "up": "mlp.up_proj",
            "gate": "mlp.gate_proj",
            "down": "mlp.down_proj",
        },
        "opt": {
            "k": "self_attn.k_proj",
            "q": "self_attn.q_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.out_proj",
            "up": "fc1",
            "down": "fc2",
        },
    }
    prefix_by_type = {
        "gpt2": f"transformer.h.{layer_idx}",
        "llama": f"model.layers.{layer_idx}",
        "llama2": f"model.layers.{layer_idx}",
        "mistral": f"model.layers.{layer_idx}",
        "opt": f"model.decoder.layers.{layer_idx}",
    }
    if model_type not in module_by_type or stem not in module_by_type[model_type]:
        raise ValueError(f"No coefficient mapping for model_type={model_type}, basis={basis_name}")
    return f"{prefix_by_type[model_type]}.{module_by_type[model_type][stem]}.weight"


def weighted_basis_matrix(tensors, groups, state_dict, config, basis_name):
    weighted = []
    for basis_tensor, group in zip(tensors, groups):
        basis_rank = basis_tensor.shape[1]
        covariance = torch.zeros((basis_rank, basis_rank), dtype=torch.float32)
        for layer_idx in group:
            key = coefficient_key(config, basis_name, layer_idx)
            if key not in state_dict:
                raise KeyError(f"Coefficient weight not found for {basis_name}: {key}")
            coefficient_weight = state_dict[key].detach().float().cpu()
            if coefficient_weight.shape[1] != basis_rank:
                raise ValueError(
                    f"{key} has basis rank {coefficient_weight.shape[1]}, expected {basis_rank}"
                )
            covariance += coefficient_weight.T @ coefficient_weight
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        sqrt_covariance = eigenvectors @ torch.diag(torch.clamp(eigenvalues, min=0).sqrt())
        weighted.append(basis_tensor @ sqrt_covariance)
    return torch.cat(weighted, dim=1)


def analysis_matrix(tensors, groups, state_dict, config, basis_name, mode):
    if mode == "basis":
        return torch.cat(tensors, dim=1)
    if mode == "coefficient-weighted":
        return weighted_basis_matrix(tensors, groups, state_dict, config, basis_name)
    raise ValueError(f"Unknown analysis mode: {mode}")


def analyze_chunk(tensors, groups, state_dict, config, basis_name, mode, ranks):
    nx, basis_rank = tensors[0].shape
    num_groups = len(tensors)
    matrix = analysis_matrix(tensors, groups, state_dict, config, basis_name, mode)
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    total_energy = energy.sum().item()
    cumulative = torch.cumsum(energy, dim=0)

    original_params = nx * basis_rank * num_groups
    max_rank = min(matrix.shape)
    rows = []
    for rank in ranks:
        effective_rank = min(rank, max_rank)
        kept = cumulative[effective_rank - 1].item()
        omitted = max(total_energy - kept, 0.0)
        rel_error = math.sqrt(omitted / total_energy) if total_energy > 0 else 0.0
        hierarchical_params = nx * effective_rank + effective_rank * basis_rank * num_groups
        rows.append(
            {
                "rank": rank,
                "effective_rank": effective_rank,
                "relative_error": rel_error,
                "original_params": original_params,
                "hierarchical_params": hierarchical_params,
                "basis_param_saving": 1.0 - hierarchical_params / original_params,
                "omitted_energy": omitted,
                "total_energy": total_energy,
            }
        )

    return {
        "nx": nx,
        "basis_rank": basis_rank,
        "num_groups": num_groups,
        "max_rank": max_rank,
        "original_params": original_params,
        "singular_values": singular_values,
        "rows": rows,
    }


def analyze_one_basis(tensors, groups, state_dict, config, basis_name, mode, ranks, bob_group_size):
    chunks = chunk_items(tensors, groups, bob_group_size)
    chunk_results = [
        analyze_chunk(chunk_tensors, chunk_groups, state_dict, config, basis_name, mode, ranks)
        for chunk_tensors, chunk_groups in chunks
    ]
    nx, basis_rank = tensors[0].shape
    num_groups = len(tensors)
    original_params = sum(result["original_params"] for result in chunk_results)
    rows = []
    for rank in ranks:
        total_energy = 0.0
        omitted_energy = 0.0
        hierarchical_params = 0
        effective_ranks = []
        for result in chunk_results:
            row = next(item for item in result["rows"] if item["rank"] == rank)
            total_energy += row["total_energy"]
            omitted_energy += row["omitted_energy"]
            hierarchical_params += row["hierarchical_params"]
            effective_ranks.append(row["effective_rank"])
        rows.append(
            {
                "rank": rank,
                "effective_rank": max(effective_ranks),
                "effective_ranks": effective_ranks,
                "relative_error": math.sqrt(omitted_energy / total_energy) if total_energy > 0 else 0.0,
                "original_params": original_params,
                "hierarchical_params": hierarchical_params,
                "basis_param_saving": 1.0 - hierarchical_params / original_params,
                "omitted_energy": omitted_energy,
                "total_energy": total_energy,
            }
        )

    max_chunk_size = max(len(chunk_tensors) for chunk_tensors, _ in chunks)
    break_even = original_params / sum(nx + basis_rank * len(chunk_tensors) for chunk_tensors, _ in chunks)
    return {
        "nx": nx,
        "basis_rank": basis_rank,
        "num_groups": num_groups,
        "num_bob_groups": len(chunks),
        "bob_group_size": bob_group_size if bob_group_size > 0 else num_groups,
        "max_bob_group_size": max_chunk_size,
        "max_rank": max(result["max_rank"] for result in chunk_results),
        "break_even_rank": break_even,
        "original_params": original_params,
        "chunk_results": chunk_results,
        "rows": rows,
    }


def reconstruct_chunk(tensors, groups, state_dict, config, basis_name, rank, mode):
    matrix = analysis_matrix(tensors, groups, state_dict, config, basis_name, mode)
    max_rank = min(matrix.shape)
    effective_rank = min(rank, max_rank)

    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if mode == "basis":
        reconstructed = (u[:, :effective_rank] * singular_values[:effective_rank]) @ vh[:effective_rank, :]
    else:
        left_basis = u[:, :effective_rank]
        reconstructed = torch.cat([left_basis @ (left_basis.T @ tensor) for tensor in tensors], dim=1)

    _, basis_rank = tensors[0].shape
    for group_index, group in enumerate(groups):
        start = group_index * basis_rank
        end = start + basis_rank
        rec_weight = reconstructed[:, start:end].T.contiguous()
        yield group, rec_weight


def reconstruct_basis(state_dict, config, basis_name, layer_to_key, rank, mode, bob_group_size, target_compression):
    tensors, _, groups = collect_unique_basis(state_dict, config, basis_name, layer_to_key)
    chunks = chunk_items(tensors, groups, bob_group_size)
    ranks_used = []
    for chunk_tensors, chunk_groups in chunks:
        chunk_rank = rank
        if chunk_rank <= 0:
            nx, basis_rank = chunk_tensors[0].shape
            chunk_rank = rank_for_target_basis_compression(
                nx, basis_rank, len(chunk_tensors), target_compression
            )
        ranks_used.append(chunk_rank)
        for group, rec_weight in reconstruct_chunk(chunk_tensors, chunk_groups, state_dict, config, basis_name, chunk_rank, mode):
            for layer_idx in group:
                key = layer_to_key.get(layer_idx)
                if key is not None:
                    state_dict[key] = rec_weight.to(dtype=state_dict[key].dtype)
    return ranks_used


def format_int(value):
    return f"{int(value):,}"


def print_report(results, ranks, mode):
    print(f"\nBasis-of-basis rank sweep ({mode})")
    print("=" * 80)
    for basis_name, result in results.items():
        print(
            f"\n{basis_name}: groups={result['num_groups']} "
            f"bob_groups={result['num_bob_groups']} "
            f"bob_group_size={result['bob_group_size']} "
            f"basis_shape=({result['nx']}, {result['basis_rank']}) "
            f"original_basis_params={format_int(result['original_params'])} "
            f"break_even_rank={result['break_even_rank']:.1f}"
        )
        print(f"{'rank':>8} {'eff':>8} {'rel_err':>12} {'hier_params':>16} {'saving':>10}")
        for row in result["rows"]:
            if row["rank"] not in ranks:
                continue
            print(
                f"{row['rank']:>8} {row['effective_rank']:>8} "
                f"{row['relative_error']:>12.6f} "
                f"{format_int(row['hierarchical_params']):>16} "
                f"{row['basis_param_saving'] * 100:>9.2f}%"
            )

    print("\nAggregate across basis tensors")
    print("-" * 80)
    print(f"{'rank':>8} {'rel_err':>12} {'orig_params':>16} {'hier_params':>16} {'saving':>10}")
    for rank in ranks:
        total_energy = 0.0
        omitted_energy = 0.0
        original_params = 0
        hierarchical_params = 0
        for result in results.values():
            row = next(item for item in result["rows"] if item["rank"] == rank)
            total_energy += row["total_energy"]
            omitted_energy += row["omitted_energy"]
            original_params += row["original_params"]
            hierarchical_params += row["hierarchical_params"]
        rel_error = math.sqrt(omitted_energy / total_energy) if total_energy > 0 else 0.0
        saving = 1.0 - hierarchical_params / original_params
        print(
            f"{rank:>8} {rel_error:>12.6f} "
            f"{format_int(original_params):>16} {format_int(hierarchical_params):>16} "
            f"{saving * 100:>9.2f}%"
        )


def write_json(output_path, results, ranks):
    serializable = {"ranks": ranks, "basis": {}, "aggregate": []}
    for basis_name, result in results.items():
        serializable["basis"][basis_name] = {
            key: value
            for key, value in result.items()
            if key not in {"singular_values", "rows", "chunk_results"}
        }
        serializable["basis"][basis_name]["rows"] = result["rows"]

    for rank in ranks:
        total_energy = 0.0
        omitted_energy = 0.0
        original_params = 0
        hierarchical_params = 0
        for result in results.values():
            row = next(item for item in result["rows"] if item["rank"] == rank)
            total_energy += row["total_energy"]
            omitted_energy += row["omitted_energy"]
            original_params += row["original_params"]
            hierarchical_params += row["hierarchical_params"]
        serializable["aggregate"].append(
            {
                "rank": rank,
                "relative_error": math.sqrt(omitted_energy / total_energy) if total_energy > 0 else 0.0,
                "original_params": original_params,
                "hierarchical_params": hierarchical_params,
                "basis_param_saving": 1.0 - hierarchical_params / original_params,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Analyze whether learned basis tensors can share a second-level basis.")
    parser.add_argument("--model-dir", required=True, help="Compressed model directory created by Basis Sharing")
    parser.add_argument("--basis-names", default="", help="Comma-separated basis names, e.g. attn_basis,o_basis")
    parser.add_argument(
        "--mode",
        default="basis",
        choices=["basis", "coefficient-weighted"],
        help="Use raw basis SVD or coefficient-weighted basis SVD",
    )
    parser.add_argument("--ranks", default="", help="Comma-separated second-level ranks to sweep")
    parser.add_argument(
        "--bob-group-size",
        type=int,
        default=0,
        help="Number of unique first-level basis tensors per second-level SVD. 0 means all tensors together.",
    )
    parser.add_argument("--output-json", default="", help="Optional path for JSON results")
    parser.add_argument("--save-reconstructed-dir", default="", help="Optional model copy with low-rank reconstructed basis")
    parser.add_argument("--save-rank", type=int, default=0, help="Rank used with --save-reconstructed-dir")
    parser.add_argument(
        "--target-basis-compression",
        type=float,
        default=0.0,
        help="When saving, choose one second-level rank per basis to compress basis parameters by this percent",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --save-reconstructed-dir if it exists")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = load_config(model_dir)
    state_dict = load_state_dict(model_dir)
    basis_keys = index_basis_keys(state_dict)
    if not basis_keys:
        raise ValueError(f"No *_basis.<layer>.weight tensors found in {model_dir}")

    requested_basis = [item.strip() for item in args.basis_names.split(",") if item.strip()]
    basis_names = requested_basis or sorted(basis_keys)
    ranks = parse_ranks(args.ranks)

    results = {}
    for basis_name in basis_names:
        if basis_name not in basis_keys:
            raise ValueError(f"{basis_name} not found. Available: {sorted(basis_keys)}")
        tensors, _, groups = collect_unique_basis(state_dict, config, basis_name, basis_keys[basis_name])
        results[basis_name] = analyze_one_basis(
            tensors, groups, state_dict, config, basis_name, args.mode, ranks, args.bob_group_size
        )

    print_report(results, ranks, args.mode)

    if args.output_json:
        write_json(args.output_json, results, ranks)
        print(f"\nWrote JSON results to {args.output_json}")

    if args.save_reconstructed_dir:
        if args.save_rank <= 0:
            if args.target_basis_compression <= 0:
                raise ValueError(
                    "Use either --save-rank or --target-basis-compression with --save-reconstructed-dir"
                )
        output_dir = Path(args.save_reconstructed_dir)
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        shutil.copytree(model_dir, output_dir)

        reconstructed_state = {key: value.clone() for key, value in state_dict.items()}
        for basis_name in basis_names:
            ranks_used = reconstruct_basis(
                reconstructed_state,
                config,
                basis_name,
                basis_keys[basis_name],
                args.save_rank,
                args.mode,
                args.bob_group_size,
                args.target_basis_compression,
            )
            print(f"{basis_name}: saved with second-level ranks {ranks_used}")
        torch.save(reconstructed_state, output_dir / "pytorch_model.bin")
        print(f"Wrote reconstructed model to {output_dir}")


if __name__ == "__main__":
    main()
