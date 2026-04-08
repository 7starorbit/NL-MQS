import torch
import torch.nn.functional as F
from model import froehlich_B


def dxx_center(H, dx):
    # H: (B,T,X)
    return (H[:, :, :-2] - 2 * H[:, :, 1:-1] + H[:, :, 2:]) / (dx * dx)


def dt_forward(B, dt):
    # B: (B,T,X), dt: (B,1,1) broadcast
    return (B[:, 1:, :] - B[:, :-1, :]) / dt


def pde_residual_physical(Hn_hat, sample):
    """
    Compute residual in physical units:
      r = (1/sigma) H_xx - d/dt B(H)
    where H = H_scale * Hn_hat.
    """
    sigma = sample["sigma"]  # (B,)
    a = sample["a"]
    b = sample["b"]
    eps = sample["eps"]      # (B,) or scalar

    x = sample["x"]          # (B,X)
    t = sample["t"]          # (B,T)
    H_scale = sample["H_scale"]  # (B,)

    # dx, dt per sample
    dx = (x[:, 1] - x[:, 0]).view(-1, 1, 1)              # (B,1,1)
    dt = (t[:, 1] - t[:, 0]).view(-1, 1, 1)              # (B,1,1)

    # physical H
    H_phys = Hn_hat * H_scale[:, None, None]             # (B,T,X)

    # broadcast params
    sigma_b = sigma[:, None, None]
    a_b = a[:, None, None]
    b_b = b[:, None, None]
    eps_b = eps[:, None, None]

    B_phys = froehlich_B(H_phys, a_b, b_b, eps_b)        # (B,T,X)

    dxx = dxx_center(H_phys, dx)[:, :-1, :]              # (B,T-1,X-2)
    dBt = dt_forward(B_phys, dt)[:, :, 1:-1]             # (B,T-1,X-2)

    r = (1.0 / sigma_b) * dxx - dBt                      # (B,T-1,X-2)

    # nondimensionalize to reduce scale variation
    r_nd = r * (a_b[:, :1, :1] * dt) / (H_scale[:, None, None] + 1e-12)
    return r_nd


def compute_losses(sample, Hn_hat, weights):
    Hn_true = sample["H"]  # (B,T,X) normalized
    L_data = F.mse_loss(Hn_hat, Hn_true)

    r_nd = pde_residual_physical(Hn_hat, sample)
    L_pde = torch.mean(r_nd * r_nd)

    # IC (already hard-enforced, keep tiny penalty optional)
    L_ic = torch.mean(Hn_hat[:, 0, :] ** 2)

    total = (weights["data"] * L_data +
             weights["pde"] * L_pde +
             weights["ic"] * L_ic)
    return total, {"L_data": L_data.detach(), "L_pde": L_pde.detach(), "L_ic": L_ic.detach()}