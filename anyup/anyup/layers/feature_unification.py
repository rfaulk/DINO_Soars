import torch
from torch import nn
import torch.nn.functional as F

compute_basis_size = {"gauss_deriv": lambda order, mirror: ((order + 1) * (order + 2)) // (1 if mirror else 2)}


def herme_vander_torch(z, m):
    He0 = z.new_ones(z.shape)
    if m == 0: return He0[:, None]
    H = [He0, z]
    for n in range(1, m):
        H.append(z * H[-1] - n * H[-2])
    return torch.stack(H, 1)


def gauss_deriv(max_order, device, dtype, kernel_size, sigma=None, include_negations=False, scale_magnitude=True):
    sigma = (kernel_size // 2) / 1.645 if sigma is None else sigma
    if kernel_size % 2 == 0: raise ValueError("ksize must be odd")
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    z = x / sigma
    g = torch.exp(-0.5 * z ** 2) / (sigma * (2.0 * torch.pi) ** 0.5)
    He = herme_vander_torch(z, max_order)
    derivs_1d = [(((-1) ** n) / (sigma ** n) if scale_magnitude else (-1) ** n) * He[:, n] * g for n in
                 range(max_order + 1)]
    bank = []
    for o in range(max_order + 1):
        for i in range(o + 1):
            K = torch.outer(derivs_1d[o - i], derivs_1d[i])
            bank.append(K)
            if include_negations: bank.append(-K)
    return torch.stack(bank, 0)


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if init_gaussian_derivatives:
            # find smallest order that gives at least out_channels basis functions
            order = 0
            while compute_basis_size["gauss_deriv"](order, False) < out_channels:
                order += 1
            print(f"FeatureUnification: initializing with Gaussian derivative basis of order {order}")
            self.basis = nn.Parameter(
                gauss_deriv(
                    order, device='cpu', dtype=torch.float32, kernel_size=kernel_size, scale_magnitude=False
                )[:out_channels, None]
            )
        else:
            self.basis = nn.Parameter(
                torch.randn(out_channels, 1, kernel_size, kernel_size)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(b, self.out_channels, c, h, w)
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p), value=0)
        x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(F.pad(mask, (p, p, p, p), value=0), torch.ones(1, 1, k, k, device=x.device))
        return x / denom  # (B, out_channels*C, H, W)
