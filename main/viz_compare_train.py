from __future__ import annotations

import argparse
import os
import numpy as np

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from dataset import MQSDataset
from model import MQSOperator, froehlich_B


def make_input(sample):
    # channels: H_left_map, x01_map, t01_map
    H_left = sample["H_left"]  # (B,T) normalized
    x01 = sample["x01"]        # (B,X)
    t01 = sample["t01"]        # (B,T)

    Bsz, T = H_left.shape
    Xn = x01.shape[1]

    H_left_map = H_left[:, :, None].repeat(1, 1, Xn)
    x_map = x01[:, None, :].repeat(1, T, 1)
    t_map = t01[:, :, None].repeat(1, 1, Xn)

    X_in = torch.stack([H_left_map, x_map, t_map], dim=1)  # (B,3,T,X)
    return X_in


def nearest_idx(arr, x0):
    arr = np.asarray(arr)
    return int(np.argmin(np.abs(arr - x0)))


def plot_contour(ax, x, t, Z, title, use_symlog=False):
    X, T = np.meshgrid(x, t)
    if use_symlog:
        vmax = float(np.max(np.abs(Z))) + 1e-12
        norm = SymLogNorm(linthresh=1e-3 * vmax, vmin=-vmax, vmax=vmax, base=10)
        cf = ax.contourf(X, T, Z, levels=60, cmap="viridis", norm=norm)
    else:
        cf = ax.contourf(X, T, Z, levels=60, cmap="viridis")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title(title)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="e.g. data_mqs50/train")
    ap.add_argument("--ckpt", required=True, help="e.g. ckpts/mqsop_epoch050.pt")
    ap.add_argument("--idx", type=int, default=0, help="which sample index in dataset")
    ap.add_argument("--outdir", default="figs_pred_train")
    ap.add_argument("--xs", default="0.0,0.01", help="physical x positions for curve plot, comma-separated (m)")
    ap.add_argument("--use_symlog", action="store_true", help="symlog colormap for H plots (enhance small values)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load one sample from training set
    ds = MQSDataset(args.train_dir, normalize=True)
    sample = ds[args.idx]

    # batchify
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)   # 0D/1D/2D 全部变成带 batch 维
        else:
            batch[k] = v

    # tensors to device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    T = int(batch["t"].shape[1])  # (B,T)
    model = MQSOperator(T=T, c_in=3, base=32, wave_emb=32).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    with torch.no_grad():
        X_in = make_input(batch).to(device)
        Hn_hat = model(
            X_in,
            batch["x01"],
            batch["t01"],
            batch["H_left"],
            batch["sigma"],
            batch["a"],
            batch["b"],
            batch["f"],
            batch["H_scale"],
        )  # (B,T,X) normalized

    # to numpy (single sample)
    x = batch["x"][0].detach().cpu().numpy()       # (X,)
    t = batch["t"][0].detach().cpu().numpy()       # (T,)
    Hn_true = batch["H"][0].detach().cpu().numpy() # (T,X) normalized
    Hn_pred = Hn_hat[0].detach().cpu().numpy()     # (T,X) normalized
    H_scale = float(batch["H_scale"][0].detach().cpu().item())

    # physical H
    H_true = Hn_true * H_scale
    H_pred = Hn_pred * H_scale

    # physical B (use same constitutive as training loss)
    a = float(batch["a"][0].detach().cpu().item())
    b = float(batch["b"][0].detach().cpu().item())
    eps = float(batch["eps"][0].detach().cpu().item())
    # compute via torch for consistency
    with torch.no_grad():
        H_pred_t = torch.tensor(H_pred, dtype=torch.float32, device=device)[None, :, :]
        H_true_t = torch.tensor(H_true, dtype=torch.float32, device=device)[None, :, :]
        a_b = torch.tensor([[a]], device=device).view(1, 1, 1)
        b_b = torch.tensor([[b]], device=device).view(1, 1, 1)
        e_b = torch.tensor([[eps]], device=device).view(1, 1, 1)
        B_pred = froehlich_B(H_pred_t, a_b, b_b, e_b)[0].detach().cpu().numpy()
        B_true = froehlich_B(H_true_t, a_b, b_b, e_b)[0].detach().cpu().numpy()

    # error (normalized, like your reference figure)
    En = Hn_pred - Hn_true

    # --- 4-panel figure ---
    fig = plt.figure(figsize=(12.5, 8.5), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = plot_contour(ax1, x, t, H_true, "H (A/m) by FEM", use_symlog=args.use_symlog)
    fig.colorbar(cf1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = plot_contour(ax2, x, t, H_pred, "H (A/m) by Model", use_symlog=args.use_symlog)
    fig.colorbar(cf2, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    Xg, Tg = np.meshgrid(x, t)
    cf3 = ax3.contourf(Xg, Tg, En, levels=60, cmap="coolwarm")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("t (s)")
    ax3.set_title("Error of normalized H  (H_pred/H_scale - H_true/H_scale)")
    fig.colorbar(cf3, ax=ax3)

    # curves: B(t) at selected x
    ax4 = fig.add_subplot(gs[1, 1])
    xs = [float(s) for s in args.xs.split(",") if s.strip()]
    for x0 in xs:
        ix = nearest_idx(x, x0)
        ax4.plot(t, B_true[:, ix], "--", lw=2, label=f"FEM, x={x[ix]:.3f}")
        ax4.plot(t, B_pred[:, ix], "-", lw=1.5, label=f"Model, x={x[ix]:.3f}")
    ax4.set_xlabel("t (s)")
    ax4.set_ylabel("B (T)")
    ax4.grid(True, alpha=0.25)
    ax4.set_title("B(t) at selected x")
    ax4.legend(fontsize=8, ncol=2)

    # meta title
    sigma = float(batch["sigma"][0].detach().cpu().item())
    f = float(batch["f"][0].detach().cpu().item())
    kappa = float(batch["kappa"][0].detach().cpu().item())
    L = float(batch["L"][0].detach().cpu().item())
    fig.suptitle(
        f"train idx={args.idx} | sigma={sigma:.3e}, a={a:g}, b={b:g}, f={f:g}Hz, kappa={kappa:g}, L={L:g}, eps={eps:g}\n"
        f"H_scale={H_scale:.3g} | ckpt={os.path.basename(args.ckpt)}",
        fontsize=11,
        y=0.98,
    )

    outpath = os.path.join(args.outdir, f"train_idx{args.idx:06d}.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", outpath)


if __name__ == "__main__":
    main()