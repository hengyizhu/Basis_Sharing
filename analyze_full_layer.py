import argparse
import math
import os
import sys
import matplotlib.pyplot as plt
import torch
from config import ShareConfig, add_args
from model_factory import create_model
from prepare_data import prepare_data
from models.model_utils import Coefficient
from transformers import LlamaTokenizer, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sparsity mask patterns for all modules in a layer")
    # 脚本专用参数
    parser.add_argument("--cf", "--yaml_config_file",
                        default="tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml",
                        type=str)
    parser.add_argument("--layer_idx", type=int, default=15, help="Decoder layer index to inspect")
    # 默认扫描该层所有关键线性层
    parser.add_argument("--modules", nargs="+", 
                        default=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", 
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
                        help="List of submodules to inspect")
    parser.add_argument("--dynamic_energy_threshold", type=float, default=0.95,
                        help="Energy fraction to retain for dynamic pruning")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Number of tokens from the sample to run through the model")
    parser.add_argument("--output_dir", type=str, default="sparsity_analysis_result",
                        help="Directory to save heatmap images")
    
    # 使用 parse_known_args，只解析上面定义的参数，剩下的留给 add_args
    return parser.parse_known_args()

def get_tokenizer(config):
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = "[PAD]"
    return tokenizer

def set_store_mask(model, layer_idx, module_name):
    try:
        layer = model.model.layers[layer_idx]
        target = layer.get_submodule(module_name)
    except AttributeError:
        print(f"Warning: Could not find layer {layer_idx} or module {module_name}")
        return None
        
    if isinstance(target, Coefficient):
        target.store_mask = True
        return target
    else:
        print(f"Warning: Target {module_name} at layer {layer_idx} is not a Coefficient layer (type: {type(target)})")
        return None

def compute_block_stats(mask_2d, block_size=32):
    seq_len, dim = mask_2d.shape
    empty_blocks = 0
    partial_blocks = 0
    total_blocks = 0
    for t in range(seq_len):
        for start in range(0, dim, block_size):
            block = mask_2d[t, start:start + block_size]
            total_blocks += 1
            if block.sum() == 0:
                empty_blocks += 1
            else:
                partial_blocks += 1
    sparsity_rate = empty_blocks / total_blocks if total_blocks > 0 else 0.0
    return {
        "total_blocks": total_blocks,
        "empty_blocks": empty_blocks,
        "partial_blocks": partial_blocks,
        "block_sparsity_rate": sparsity_rate,
    }

def main():
    # 1. 解析脚本参数
    args, remaining_argv = parse_args()
    
    # 2. Hack sys.argv：临时替换 argv，骗过 add_args 只看剩下的参数
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    base_args = add_args()
    sys.argv = original_argv # 还原（虽然这里用不到了）
    base_args.yaml_config_file = args.cf

    # 3. 强制覆盖配置
    if base_args.dynamic_energy_threshold is None:
        base_args.dynamic_energy_threshold = args.dynamic_energy_threshold
    
    config = ShareConfig(base_args)
    # 双重保险：确保 config 对象里的值也是对的
    config.dynamic_energy_threshold = args.dynamic_energy_threshold
    
    print(f"=== Analysis Config ===")
    print(f"Layer Index: {args.layer_idx}")
    print(f"Energy Threshold: {config.dynamic_energy_threshold}")
    print(f"Modules: {args.modules}")
    print(f"=======================")

    # 准备数据
    tokenizer = get_tokenizer(config)
    _, _, tokenized_test, _ = prepare_data(config.dataset_name, tokenizer, config.context_length,
                                           config.dataset_cache_dir)
    
    # 兼容不同的 dataset 返回格式
    if hasattr(tokenized_test, "input_ids"):
        input_ids = tokenized_test.input_ids[:, :args.max_tokens]
    else:
        # 假设是 list 或 dict 结构
        input_ids = tokenized_test[0]["input_ids"][:args.max_tokens].unsqueeze(0)
        
    attention_mask = torch.ones_like(input_ids)

    # 加载模型
    print("Loading model (this may take a while)...")
    model = create_model(config)
    model.eval()

    # 注册 Hook (开启 store_mask)
    target_modules = {}
    for mod_name in args.modules:
        mod = set_store_mask(model, args.layer_idx, mod_name)
        if mod is not None:
            target_modules[mod_name] = mod

    if not target_modules:
        print("Error: No valid modules found to inspect.")
        return

    # 推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    print("Running forward pass...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 分析与绘图
    print("\n=== Sparsity Analysis Results ===")
    for mod_name, module in target_modules.items():
        if module.last_mask_tensor is None:
            print(f"[{mod_name}] No mask captured. Did the forward pass run?")
            continue

        mask = module.last_mask_tensor.squeeze(0)  # (seq, rank)
        if mask.dim() != 2:
            mask = mask.view(mask.shape[0], -1)
        mask_2d = mask.float()
        
        mask_mean = mask_2d.mean().item()
        stats = compute_block_stats(mask_2d, block_size=32)
        
        print(f"\nModule: {mod_name}")
        print(f"  > Element Sparsity (Keep Ratio): {mask_mean:.4f}")
        print(f"  > Block Sparsity (Dead Blocks):  {stats['block_sparsity_rate']:.2%} ({stats['empty_blocks']}/{stats['total_blocks']})")

        # 绘图
        plt.figure(figsize=(12, 6))
        # cmap='gray': 0(黑) -> Dead, 1(白) -> Active
        # 也可以试 cmap='binary_r' (0白 1黑) 看你喜好
        plt.imshow(mask_2d.cpu().numpy(), aspect='auto', cmap='gray', interpolation='nearest')
        plt.xlabel("Basis Index")
        plt.ylabel("Token Sequence")
        plt.title(f"Layer {args.layer_idx} - {mod_name}\nActive: {mask_mean:.1%} | Block Sparse: {stats['block_sparsity_rate']:.1%}")
        
        # 保存
        safe_name = mod_name.replace(".", "_")
        out_path = os.path.join(args.output_dir, f"L{args.layer_idx}_{safe_name}.png")
        plt.savefig(out_path)
        print(f"  > Heatmap saved to: {out_path}")
        plt.close()

if __name__ == "__main__":
    main()