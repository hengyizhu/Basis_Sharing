import argparse
import os
import torch
from collections import defaultdict

from config import ShareConfig, add_args
from model_factory import create_model
# 确保导入了正确的类
from models.model_utils import Coefficient, Basis 

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
    """
    计算重要性并生成置换索引。
    注意：这里假设了简单的 1-to-1 或局部共享关系。
    如果有跨层共享 (Group>1)，需要确保同一组共享同样的 Permutation。
    """
    perm_map = {}
    
    # 1. 遍历所有模块，找到包含 Coefficient 和 Basis 的父容器（通常是 LinearShared）
    # 或者直接通过 name 匹配。
    
    # 为了处理 Basis Sharing 的 Group 逻辑，我们需要更聪明的聚合。
    # 这里我们采用一个简化的策略：
    # 计算每个 Coefficient 的重要性，然后生成 Permutation。
    # **关键：** 我们需要把这个 Permutation 同时注册给 Coefficient 和对应的 Basis。
    
    # 由于反向查找 Basis 比较困难，我们这里先只生成 Coefficient 的 Map。
    # 在 apply 阶段，我们会尝试去“猜测”或者“同步” Basis 的 Permutation。
    
    # 修正维度计算错误
    for name, module in model.named_modules():
        if isinstance(module, Coefficient):
            # [Fix] dim=1 (沿着输入特征求均值)，得到 [Rank] 长度的向量
            score = module.weight.abs().mean(dim=1) 
            perm = torch.argsort(score, descending=True)
            perm_map[name] = perm
            
            # [Hack] 尝试找到对应的 Basis 并共享 Permutation
            # 假设结构是 ...mlp.down_proj.coefficient / ...mlp.down_proj.basis
            # 我们把 'coefficient' 替换为 'basis' 尝试注册
            if 'coefficient' in name:
                basis_name = name.replace('coefficient', 'basis')
                # 检查模型里是否有这个模块（可选，这里直接存进去也没事，apply时会check）
                perm_map[basis_name] = perm

    return perm_map

def apply_reorder(model, perm_map):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in perm_map:
                continue
                
            perm = perm_map[name]
            
            # 确保 perm 在正确的设备上
            perm = perm.to(module.weight.device)

            if isinstance(module, Coefficient):
                # Coefficient: 重排 行 (Dim 0)
                # 形状: [Rank, In] -> [Rank, In]
                module.weight.copy_(module.weight[perm, :])
                if module.bias is not None:
                    module.bias.copy_(module.bias[perm])
                print(f"Reordered Coefficient: {name}")

            elif isinstance(module, Basis):
                # Basis: 重排 列 (Dim 1)
                # 形状: [Out, Rank] -> [Out, Rank]
                module.weight.copy_(module.weight[:, perm])
                print(f"Reordered Basis: {name}")

def main():
    args = parse_args()
    base_args = add_args()
    base_args.yaml_config_file = args.cf
    cfg = ShareConfig(base_args)
    
    print("Loading model...")
    model = create_model(cfg)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Collecting importance...")
    perm_map = collect_importance(model)
    torch.save(perm_map, args.save_path)
    print(f"Saved permutation map to {os.path.abspath(args.save_path)}")

    print("Applying reorder...")
    apply_reorder(model, perm_map)
    
    print("Saving reordered model...")
    model.save_pretrained(args.output_dir, safe_serialization=False)
    print(f"Saved reordered model to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()