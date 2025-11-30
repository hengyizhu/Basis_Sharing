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
    def __init__(self, nf, num_basis, bias=False, dynamic_threshold=None, max_rank=None, scaling_diag=None,
                 log_active=False, dynamic_energy_threshold=None, store_mask=False, decode_gather=False,
                 assume_sorted=False, static_k=None):
        super().__init__(num_basis, nf, bias=bias)
        self.nf = nf
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.max_rank = max_rank
        self.log_active = log_active
        self._last_active_ratio = None
        self.store_mask = store_mask
        self.last_mask_tensor = None
        self.decode_gather = decode_gather
        self.assume_sorted = assume_sorted
        self.static_k = static_k
        if scaling_diag is not None:
            self.register_buffer("scaling_diag", scaling_diag)
        else:
            self.scaling_diag = None

    # Profiling support
    profile_enabled = False
    profile_stats = {
        "scaling": 0.0,
        "decision": 0.0,
        "gather": 0.0,
        "compute": 0.0,
        "count": 0,
    }

    @classmethod
    def reset_profile(cls):
        cls.profile_stats = {
            "scaling": 0.0,
            "decision": 0.0,
            "gather": 0.0,
            "compute": 0.0,
            "count": 0,
        }

    @classmethod
    def get_profile(cls):
        return dict(cls.profile_stats)

    @property
    def last_active_ratio(self):
        return self._last_active_ratio

    def set_weight(self, weight, bias=None):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())

    def set_scaling_diag(self, scaling_diag):
        if scaling_diag is None:
            return
        if scaling_diag.dim() == 2:
            scaling_diag = torch.diag(scaling_diag)
        scaling_diag = scaling_diag.flatten()
        # align size with input features; ignore if mismatched
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

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        x_proc = x

        if self.max_rank is not None:
            x_proc = x_proc[..., : self.max_rank]
            weight = weight[:, : self.max_rank]

        decode_path = (
            self.decode_gather
            and x_proc.dim() >= 2
            and x_proc.shape[-2] == 1
            and self.dynamic_energy_threshold is not None
        )
        if self.static_k is not None and decode_path:
            keep_len = min(self.static_k, x_proc.shape[-1])
            x_active = x_proc[..., :keep_len]
            weight_active = weight[:, :keep_len]
            z = F.linear(x_active, weight_active, bias)
            z = z.view(x.shape[:-1] + (self.nf,))
            self._last_active_ratio = keep_len / x_proc.shape[-1]
            return z

        if self.dynamic_threshold is not None or self.dynamic_energy_threshold is not None:
            if not decode_path:
                x_proc = self._apply_scaling(x_proc)

        mask = None
        # Decode path gather (seq_len == 1) to avoid full matmul
        if decode_path and x_proc.shape[0] == 1:
            prof = self.profile_enabled and x_proc.is_cuda

            def _event():
                return torch.cuda.Event(enable_timing=True) if prof else None

            scaling_start = _event(); scaling_end = _event()
            decision_start = _event(); decision_end = _event()
            gather_start = _event(); gather_end = _event()
            compute_start = _event(); compute_end = _event()

            x_scaled = x_proc
            if self.dynamic_threshold is not None or self.dynamic_energy_threshold is not None:
                if prof:
                    scaling_start.record()
                x_scaled = self._apply_scaling(x_scaled)
                if prof:
                    scaling_end.record()

            if prof:
                decision_start.record()
            energy_in = x_scaled.view(-1).pow(2)

            if self.assume_sorted:
                cumulative = torch.cumsum(energy_in, dim=-1)
                target = cumulative[..., -1:] * self.dynamic_energy_threshold
                # find smallest k where cumulative >= target
                cutoff = torch.nonzero(cumulative >= target, as_tuple=True)[0]
                if cutoff.numel() == 0:
                    keep_len = energy_in.numel()
                else:
                    keep_len = int(cutoff[0].item()) + 1
                active_mask = None
            else:
                sorted_energy, indices = torch.sort(energy_in, dim=-1, descending=True)
                cumulative = torch.cumsum(sorted_energy, dim=-1)
                target = cumulative[..., -1:] * self.dynamic_energy_threshold
                sorted_mask = cumulative <= target
                sorted_mask[..., 0] = True
                keep_len = int(sorted_mask.sum().item())
                if keep_len <= 0:
                    keep_len = 1
                active_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
                active_mask.scatter_(0, indices, sorted_mask)
                flat_mask = active_mask.view(-1)
                prefix_indices = torch.nonzero(flat_mask, as_tuple=True)[0]
                if prefix_indices.numel() > keep_len:
                    prefix_indices = prefix_indices[:keep_len]
            if prof:
                decision_end.record()

            # track keep_len for stats
            self.profile_stats["last_keep"] = keep_len

            if prof:
                gather_start.record()
            if self.assume_sorted:
                x_active = x_scaled[..., :keep_len]
                weight_active = weight[:, :keep_len]
            else:
                if prefix_indices.numel() == keep_len and torch.all(
                        prefix_indices == torch.arange(keep_len, device=prefix_indices.device)):
                    x_active = x_scaled[..., :keep_len]
                    weight_active = weight[:, :keep_len]
                else:
                    active_indices = prefix_indices if prefix_indices.numel() > 0 else indices[:keep_len]
                    x_active = x_scaled[..., active_indices]
                    weight_active = weight[:, active_indices]
            if prof:
                gather_end.record()

            if prof:
                compute_start.record()
            z = F.linear(x_active, weight_active, bias)
            if prof:
                compute_end.record()
            z = z.view(x.shape[:-1] + (self.nf,))
            if active_mask is not None:
                mask_vec = active_mask
                mask = mask_vec.view(1, 1, -1)
                self._last_active_ratio = mask_vec.float().mean().detach().cpu().item()
                if self.store_mask:
                    self.last_mask_tensor = mask.detach().cpu()
            else:
                self._last_active_ratio = keep_len / energy_in.numel()
                if self.store_mask:
                    mask = torch.zeros(1, 1, energy_in.numel(), device=x.device, dtype=torch.bool)
                    mask[..., :keep_len] = True
                    self.last_mask_tensor = mask.detach().cpu()
            if self.log_active and not self.training:
                print(f"[Coefficient decode-gather] active fraction: {self._last_active_ratio:.4f}")
            if prof:
                torch.cuda.synchronize()

                def _acc(phase, start_ev, end_ev):
                    if start_ev is not None and end_ev is not None:
                        self.profile_stats[phase] += start_ev.elapsed_time(end_ev)

                _acc("scaling", scaling_start, scaling_end)
                _acc("decision", decision_start, decision_end)
                _acc("gather", gather_start, gather_end)
                _acc("compute", compute_start, compute_end)
                self.profile_stats["count"] += 1
            return z

        z = F.linear(x_proc, weight, bias)

        if self.dynamic_energy_threshold is not None:
            energy = z.pow(2)
            sorted_energy, indices = torch.sort(energy, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_energy, dim=-1)
            total_energy = cumulative[..., -1:]
            target = total_energy * self.dynamic_energy_threshold
            sorted_mask = cumulative <= target
            # ensure at least one element kept
            sorted_mask[..., 0] = True
            recovered_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
            recovered_mask.scatter_(-1, indices, sorted_mask)
            mask = recovered_mask
            z = z * recovered_mask
        elif self.dynamic_threshold is not None:
            mask = (z.abs() >= self.dynamic_threshold)
            self._last_active_ratio = mask.float().mean().detach().cpu().item()
            if self.log_active and not self.training:
                print(f"[Coefficient] active fraction: {self._last_active_ratio:.4f}")
            z = z * mask
        if mask is not None:
            self._last_active_ratio = mask.float().mean().detach().cpu().item()
            if self.log_active and not self.training and self.dynamic_energy_threshold is not None:
                print(f"[Coefficient] active fraction: {self._last_active_ratio:.4f}")
            if self.store_mask:
                mask_to_store = mask.view(*x.size()[:-1], mask.shape[-1]).detach().cpu()
                self.last_mask_tensor = mask_to_store

        size_out = x.size()[:-1] + (self.nf,)
        z = z.view(size_out)
        return z
