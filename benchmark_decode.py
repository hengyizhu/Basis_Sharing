import argparse
import time
import torch

from config import ShareConfig, add_args
from model_factory import create_model
from models.model_utils import Coefficient


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark decode gather latency")
    parser.add_argument("--cf", "--yaml_config_file",
                        default="tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml",
                        type=str)
    parser.add_argument("--reordered_model_dir", type=str,
                        default="untrained_model_reordered",
                        help="Path to reordered model checkpoint")
    parser.add_argument("--dynamic_energy_threshold", type=float, default=0.99,
                        help="Energy threshold for decode gather")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--static_k", type=int, default=None, help="Use static slice size instead of dynamic")
    return parser.parse_args()


def toggle_decode_gather(model, enabled: bool, energy_thresh: float):
    for module in model.modules():
        if isinstance(module, Coefficient):
            module.decode_gather = enabled
            module.dynamic_energy_threshold = energy_thresh if enabled else None
            module.assume_sorted = True  # enable no-sort fast path for benchmark
            module.static_k = None


def run_once(model, input_ids, attention_mask):
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)


def benchmark(model, input_ids, attention_mask, warmup, iters):
    # warmup
    for _ in range(warmup):
        run_once(model, input_ids, attention_mask)
    torch.cuda.synchronize()
    start = time.time()
    keeps = []
    for _ in range(iters):
        run_once(model, input_ids, attention_mask)
        # collect keep_len if available
        if Coefficient.profile_stats.get("last_keep", None) is not None:
            keeps.append(Coefficient.profile_stats["last_keep"])
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000.0 / iters, keeps  # ms/iter


def main():
    args = parse_args()
    base_args = add_args()
    base_args.yaml_config_file = args.cf
    cfg = ShareConfig(base_args)
    cfg.untrained_model_path = args.reordered_model_dir
    model = create_model(cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = torch.randint(low=0, high=model.config.vocab_size, size=(1, 1), device=device)
    attention_mask = torch.ones_like(input_ids)

    # Case A: dense decode
    toggle_decode_gather(model, enabled=False, energy_thresh=None)
    dense_ms, _ = benchmark(model, input_ids, attention_mask, args.warmup, args.iters)

    # Case B: decode gather
    toggle_decode_gather(model, enabled=True, energy_thresh=args.dynamic_energy_threshold)
    Coefficient.reset_profile()
    Coefficient.profile_enabled = True
    gather_ms, keeps_dynamic = benchmark(model, input_ids, attention_mask, args.warmup, args.iters)
    Coefficient.profile_enabled = False
    prof = Coefficient.get_profile()

    # Case C: static slice if provided
    static_ms = None
    if args.static_k is not None:
        toggle_decode_gather(model, enabled=True, energy_thresh=args.dynamic_energy_threshold)
        for module in model.modules():
            if isinstance(module, Coefficient):
                module.static_k = args.static_k
        static_ms, keeps_static = benchmark(model, input_ids, attention_mask, args.warmup, args.iters)
        # reset static_k
        for module in model.modules():
            if isinstance(module, Coefficient):
                module.static_k = None

    speedup = dense_ms / gather_ms if gather_ms > 0 else 0
    print(f"Dense decode:  {dense_ms:.3f} ms")
    print(f"Gather decode: {gather_ms:.3f} ms (threshold={args.dynamic_energy_threshold})")
    print(f"Speedup: {speedup:.2f}x")
    if static_ms is not None:
        print(f"Static slice (k={args.static_k}): {static_ms:.3f} ms (speedup {dense_ms/static_ms:.2f}x)")
    if prof.get("count", 0) > 0:
        c = prof["count"]
        print("[Profile Breakdown] (avg per iteration)")
        print(f"Scaling:  {prof['scaling']/c:.4f} ms")
        print(f"Decision: {prof['decision']/c:.4f} ms")
        print(f"Gather:   {prof['gather']/c:.4f} ms")
        print(f"Compute:  {prof['compute']/c:.4f} ms")
    if keeps_dynamic:
        k_tensor = torch.tensor(keeps_dynamic, dtype=torch.float32)
        print(f"[k stats dynamic] mean={k_tensor.mean():.2f}, std={k_tensor.std():.2f}, "
              f"min={k_tensor.min():.0f}, max={k_tensor.max():.0f}")
    if args.static_k is None and keeps_dynamic:
        suggested_k = int(torch.tensor(keeps_dynamic, dtype=torch.float32).mean().item())
        print(f"Suggested static_k based on mean k: {suggested_k}")


if __name__ == "__main__":
    main()
