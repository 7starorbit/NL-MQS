import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import MQSDataset
from model import MQSOperator
from loss import compute_losses


def make_input(sample):
    """
    Build X_in: (B,C,T,X)
    channels: H_left(t) map, x01 map, t01 map
    """
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


def to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="e.g. data_mqs50/train")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--wave_emb", type=int, default=32)

    ap.add_argument("--pde_weight", type=float, default=0.1)
    ap.add_argument("--pde_warmup_epochs", type=int, default=10)
    ap.add_argument("--ic_weight", type=float, default=0.01)

    ap.add_argument("--save_dir", type=str, default="ckpts")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    ds = MQSDataset(args.train_dir, normalize=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # infer T from one sample
    s0 = ds[0]
    T = int(s0["t"].shape[0])

    model = MQSOperator(T=T, c_in=3, base=args.base, wave_emb=args.wave_emb).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        # warmup PDE weight
        if args.pde_warmup_epochs > 0:
            alpha = min(1.0, epoch / args.pde_warmup_epochs)
        else:
            alpha = 1.0

        weights = {"data": 1.0, "pde": args.pde_weight * alpha, "ic": args.ic_weight}

        meters = {"L_data": 0.0, "L_pde": 0.0, "L_ic": 0.0}
        nsteps = 0

        for batch in loader:
            batch = to_device(batch, device)

            X_in = make_input(batch)
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
            )

            loss, comps = compute_losses(batch, Hn_hat, weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            for k in meters:
                meters[k] += float(comps[k])
            nsteps += 1

        for k in meters:
            meters[k] /= max(1, nsteps)

        dt = time.time() - t0
        print(
            f"epoch {epoch:03d} | "
            f"w_pde={weights['pde']:.3g} | "
            f"data {meters['L_data']:.3e} pde {meters['L_pde']:.3e} ic {meters['L_ic']:.3e} | "
            f"time {dt:.1f}s"
        )

        # save checkpoint
        if (epoch % 10 == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(args.save_dir, f"mqsop_epoch{epoch:03d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, ckpt_path)
            print("[save]", ckpt_path)


if __name__ == "__main__":
    main()