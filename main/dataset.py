import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_meta(npz):
    if "meta" not in npz:
        return {}
    try:
        m = npz["meta"].item()
    except Exception:
        m = npz["meta"]
    return m if isinstance(m, dict) else {}


class MQSDataset(Dataset):
    """
    Expects each .npz contains:
      x: (X,) float
      t: (T,) float
      H: (T,X) float  (physical)
      H_left: (T,) float (physical)  (optional; fallback H[:,0])
      meta: dict with sigma,a,b,f,kappa,H_peak,eps,dt_out,...
    """
    def __init__(self, root_split_dir: str, normalize: bool = True):
        self.files = sorted(glob.glob(os.path.join(root_split_dir, "case_*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No npz files found under: {root_split_dir}/case_*.npz")
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        meta = _safe_meta(d)

        x = d["x"].astype(np.float32)          # (X,) physical m
        t = d["t"].astype(np.float32)          # (T,) physical s
        H = d["H"].astype(np.float32)          # (T,X) physical A/m

        if "H_left" in d:
            H_left = d["H_left"].astype(np.float32)  # (T,) physical
        else:
            H_left = H[:, 0].copy()

        # parameters from meta
        sigma = float(meta["sigma"])
        a = float(meta["a"])
        b = float(meta["b"])
        f = float(meta["f"])
        kappa = float(meta.get("kappa", np.nan))
        H_peak = float(meta.get("H_peak", np.max(np.abs(H_left))))
        eps = float(meta.get("eps", 1e-3))
        L = float(meta.get("L", float(x.max() - x.min())))

        # scales (per-sample)
        H_scale = float(np.max(np.abs(H_left)) + 1e-6)  # physical amplitude scale

        if self.normalize:
            Hn = H / H_scale
            H_left_n = H_left / H_scale
        else:
            Hn = H
            H_left_n = H_left

        # coords normalized to [0,1] for NN input
        x01 = (x - x.min()) / (x.max() - x.min() + 1e-12)
        t01 = (t - t.min()) / (t.max() - t.min() + 1e-12)

        sample = {
            # grids
            "x": torch.from_numpy(x),          # (X,) physical
            "t": torch.from_numpy(t),          # (T,) physical
            "x01": torch.from_numpy(x01),      # (X,)
            "t01": torch.from_numpy(t01),      # (T,)

            # fields (normalized if normalize=True)
            "H": torch.from_numpy(Hn),              # (T,X)
            "H_left": torch.from_numpy(H_left_n),   # (T,)

            # params (physical)
            "sigma": torch.tensor(sigma, dtype=torch.float32),
            "a": torch.tensor(a, dtype=torch.float32),
            "b": torch.tensor(b, dtype=torch.float32),
            "f": torch.tensor(f, dtype=torch.float32),
            "kappa": torch.tensor(kappa, dtype=torch.float32),
            "H_peak": torch.tensor(H_peak, dtype=torch.float32),
            "eps": torch.tensor(eps, dtype=torch.float32),
            "L": torch.tensor(L, dtype=torch.float32),

            # scale
            "H_scale": torch.tensor(H_scale, dtype=torch.float32),
        }
        return sample