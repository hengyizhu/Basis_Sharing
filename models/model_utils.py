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
                 log_active=False, dynamic_energy_threshold=None):
        super().__init__(num_basis, nf, bias=bias)
        self.nf = nf
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.max_rank = max_rank
        self.log_active = log_active
        self._last_active_ratio = None
        if scaling_diag is not None:
            self.register_buffer("scaling_diag", scaling_diag)
        else:
            self.scaling_diag = None

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

        if self.dynamic_threshold is not None or self.dynamic_energy_threshold is not None:
            x_proc = self._apply_scaling(x_proc)

        z = F.linear(x_proc, weight, bias)

        mask = None
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

        size_out = x.size()[:-1] + (self.nf,)
        z = z.view(size_out)
        return z
