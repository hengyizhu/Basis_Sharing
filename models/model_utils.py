import torch
import torch.nn as nn
import torch.nn.functional as F


def build_basis_collection(groups, num_basis, nx):
    model_dict = torch.nn.ModuleDict()
    for group in groups:
        basis = Basis(num_basis, nx)
        for item in group:
            model_dict[str(item)] = basis
    return model_dict


class Basis(nn.Linear):
    def __init__(self, num_basis, nx):
        super().__init__(nx, num_basis, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())


class Coefficient(nn.Linear):
    """
    Inference strategies:
    - baseline: dense compute
    - global_static: always slice first k channels (static_k)
    - prompt_adaptive: prefill determines k by energy, decode reuses that k
    """

    def __init__(
            self,
            nf,
            num_basis,
            bias=False,
            scaling_diag=None,
            strategy="baseline",
            target_ratio=1.0,
            strategy_debug=False,
            store_mask=False,
    ):
        super().__init__(num_basis, nf, bias=bias)
        self.nf = nf
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.static_k = None
        self.energy_threshold = None
        self.strategy_debug = strategy_debug
        self.store_mask = store_mask
        self.last_mask_tensor = None
        self.cached_k = None
        self.last_keep = None
        self.k_sum = 0.0
        self.k_count = 0
        self.energy_kept = 0.0
        self.energy_total = 0.0
        if scaling_diag is not None:
            self.register_buffer("scaling_diag", scaling_diag)
        else:
            self.scaling_diag = None

    @property
    def last_active_ratio(self):
        if self.last_keep is None:
            return None
        return self.last_keep / self.in_features

    def reset_cached_k(self):
        self.cached_k = None

    def reset_stats(self):
        self.k_sum = 0.0
        self.k_count = 0
        self.last_keep = None
        self.energy_kept = 0.0
        self.energy_total = 0.0

    def _record_k(self, k):
        self.last_keep = int(k)
        self.k_sum += float(k)
        self.k_count += 1

    def _record_energy(self, kept, total):
        self.energy_kept += float(kept)
        self.energy_total += float(total)

    def set_weight(self, weight, bias=None):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())

    def set_scaling_diag(self, scaling_diag):
        if scaling_diag is None:
            return
        if scaling_diag.dim() == 2:
            scaling_diag = torch.diag(scaling_diag)
        scaling_diag = scaling_diag.flatten()
        if scaling_diag.shape[0] >= self.in_features:
            scaling_diag = scaling_diag[: self.in_features]
        else:
            return
        if hasattr(self, "scaling_diag") and isinstance(self.scaling_diag, torch.Tensor):
            with torch.no_grad():
                self.scaling_diag.copy_(scaling_diag.to(self.scaling_diag.device))
        else:
            if hasattr(self, "scaling_diag"):
                del self.scaling_diag
            self.register_buffer("scaling_diag", scaling_diag)

    def _apply_scaling(self, x):
        scaling = self.scaling_diag
        if scaling is None:
            return x
        feat_dim = x.shape[-1]
        if scaling.shape[0] < feat_dim:
            return x
        scale_vec = scaling[:feat_dim].to(x.device)
        shape = [1] * (x.dim() - 1) + [feat_dim]
        return x * scale_vec.view(*shape)

    def _select_k_from_energy(self, x_scaled):
        # x_scaled shape: [batch, seq, hidden]
        dims = tuple(range(x_scaled.dim() - 1))
        energy = x_scaled.pow(2).sum(dim=dims)  # [hidden]
        cumulative = torch.cumsum(energy, dim=-1)
        total = cumulative[-1]
        thr = self.energy_threshold if self.energy_threshold is not None else 1.0
        target = total * thr
        cutoff = torch.nonzero(cumulative >= target, as_tuple=True)[0]
        if cutoff.numel() == 0:
            k = energy.shape[-1]
        else:
            k = int(cutoff[0].item()) + 1
        return k

    def forward(self, x):
        x_proc = self._apply_scaling(x)
        size_out = x.size()[:-1] + (self.nf,)
        seq_len = x_proc.shape[-2] if x_proc.dim() >= 2 else 1
        feat_dim = x_proc.shape[-1]

        basis_dim = self.weight.shape[0]
        budget_k = max(1, int(round(basis_dim * self.target_ratio)))
        if self.static_k is None:
            self.static_k = budget_k

        if self.strategy == "global_static" and self.static_k is not None:
            k = min(self.static_k, self.weight.shape[0])
            weight_active = self.weight[:k, :]
            bias_active = self.bias[:k] if self.bias is not None else None
            z_full = F.linear(x_proc, self.weight, self.bias)
            z_small = F.linear(x_proc, weight_active, bias_active)
            z = torch.zeros_like(z_full)
            z[..., :k] = z_small
            energy_total = z_full.pow(2).sum().item()
            energy_keep = z.pow(2).sum().item()
            self._record_k(k)
            self._record_energy(energy_keep, energy_total)
            if self.store_mask:
                mask = torch.zeros_like(z_full, dtype=torch.bool)
                mask[..., :k] = True
                self.last_mask_tensor = mask.detach().cpu()
            if self.strategy_debug and not self.training:
                print(f"[GlobalStatic] x={tuple(x_proc.shape)}, weight={tuple(self.weight.shape)}, k={k}, "
                      f"weight_active={tuple(weight_active.shape)}")
            return z.view(size_out)

        if self.strategy == "prompt_adaptive":
            if seq_len > 1 or self.cached_k is None:
                k = budget_k
                self.cached_k = k
            k = min(self.cached_k, self.weight.shape[0])
            z_full = F.linear(x_proc, self.weight, self.bias)
            energy_full = z_full.pow(2)
            topk = min(k, energy_full.shape[-1])
            vals, idx = torch.topk(energy_full, k=topk, dim=-1, largest=True, sorted=False)
            mask = torch.zeros_like(z_full, dtype=z_full.dtype)
            mask.scatter_(-1, idx, 1.0)
            z = z_full * mask
            energy_total = energy_full.sum().item()
            energy_keep = (energy_full * (mask > 0)).sum().item()
            self._record_k(k)
            self._record_energy(energy_keep, energy_total)
            if self.store_mask:
                self.last_mask_tensor = (mask > 0).detach().cpu()
            if self.strategy_debug and not self.training:
                energy_vec = energy_full.mean(dim=tuple(range(energy_full.dim() - 1)))
                print(f"[PromptAdaptive] seq_len={seq_len} k={k}, weight={tuple(self.weight.shape)}, "
                      f"energy_head={energy_vec[:5].tolist()}")
            return z.view(size_out)

        # baseline dense
        energy_total = x_proc.pow(2).sum().item()
        self._record_energy(energy_total, energy_total)
        z = F.linear(x_proc, self.weight, self.bias)
        self._record_k(feat_dim)
        return z.view(size_out)


def reset_k_stats(model):
    for module in model.modules():
        if isinstance(module, Coefficient):
            module.reset_stats()
            module.reset_cached_k()


def collect_avg_k(model):
    total_k = 0.0
    total_count = 0
    for module in model.modules():
        if isinstance(module, Coefficient):
            total_k += module.k_sum
            total_count += module.k_count
    if total_count == 0:
        return None
    return total_k / total_count


def collect_energy_ratio(model):
    kept = 0.0
    total = 0.0
    for module in model.modules():
        if isinstance(module, Coefficient):
            kept += module.energy_kept
            total += module.energy_total
    if total == 0:
        return None
    return kept / total
