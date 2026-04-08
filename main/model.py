import torch
import torch.nn as nn


def froehlich_B(H, a, b, eps):
    # H: (B,T,X) physical
    Habs = torch.sqrt(H * H + eps * eps)
    return H / (a + b * Habs)


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 2 * feat_ch),
            nn.SiLU(),
            nn.Linear(2 * feat_ch, 2 * feat_ch),
        )
        self.feat_ch = feat_ch

    def forward(self, feat, cond):
        # feat: (B,C,T,X)
        gb = self.mlp(cond)  # (B,2C)
        gamma, beta = gb[:, : self.feat_ch], gb[:, self.feat_ch :]
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feat * (1.0 + gamma) + beta


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(cond_dim, out_ch)

    def forward(self, x, cond):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.film(x, cond)
        x = self.act(self.norm2(self.conv2(x)))
        return x


class CondUNet2D(nn.Module):
    """
    Input: (B,C_in,T,X)
    Output: (B,1,T,X)
    """
    def __init__(self, c_in, cond_dim, base=32):
        super().__init__()
        self.enc1 = ConvBlock(c_in, base, cond_dim)
        self.enc2 = ConvBlock(base, base * 2, cond_dim)
        self.enc3 = ConvBlock(base * 2, base * 4, cond_dim)

        self.pool = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 4, base * 4, cond_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, cond_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, cond_dim)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x, cond):
        e1 = self.enc1(x, cond)
        e2 = self.enc2(self.pool(e1), cond)
        e3 = self.enc3(self.pool(e2), cond)

        m = self.mid(e3, cond)

        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, cond)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, cond)

        return self.out(d1)


class WaveformEncoder(nn.Module):
    def __init__(self, T, emb=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T, 128),
            nn.SiLU(),
            nn.Linear(128, emb),
        )

    def forward(self, h_left):
        # h_left: (B,T) (normalized)
        return self.net(h_left)


class MQSOperator(nn.Module):
    """
    Predict normalized H_n(t,x), with hard BC and hard IC.

    Condition includes:
      log(sigma), log(a), log(b), log(f), log(H_scale), waveform_embedding
    """
    def __init__(self, T, c_in=3, base=32, wave_emb=32):
        super().__init__()
        self.wave_enc = WaveformEncoder(T, emb=wave_emb)
        self.cond_dim = 5 + wave_emb
        self.unet = CondUNet2D(c_in=c_in, cond_dim=self.cond_dim, base=base)

    def forward(self, X_in, x01, t01, h_left, sigma, a, b, f, H_scale):
        """
        X_in: (B,C_in,T,X)
        x01: (B,X)
        t01: (B,T)
        h_left: (B,T) normalized
        params: (B,)
        Return: Hn_hat (B,T,X)
        """
        w = self.wave_enc(h_left)  # (B,wave_emb)

        cond = torch.cat([
            torch.log(sigma[:, None] + 1e-12),
            torch.log(a[:, None] + 1e-12),
            torch.log(b[:, None] + 1e-12),
            torch.log(f[:, None] + 1e-12),
            torch.log(H_scale[:, None] + 1e-12),
            w
        ], dim=1)

        delta = self.unet(X_in, cond)[:, 0, :, :]  # (B,T,X)

        # Hard BC: Hn(0,t)=h_left, Hn(1,t)=0
        x = x01[:, None, :]               # (B,1,X)
        H_left_map = h_left[:, :, None]   # (B,T,1)
        H_base = (1.0 - x) * H_left_map

        # Hard IC: multiply correction by t01 to enforce H(x,0)=0
        tt = t01[:, :, None]              # (B,T,1) in [0,1]
        Hn_hat = H_base + (x * (1.0 - x)) * tt * delta
        return Hn_hat