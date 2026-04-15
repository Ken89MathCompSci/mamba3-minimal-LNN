"""
train_ssd_lnn.py
================
Integrates Liquid Neural Network (LNN) dynamics into the Mamba-3 SSD algorithm.

LNN Key Idea (Hasani et al., 2021 — Liquid Time-Constant Networks):
    Standard SSM:  h_t = exp(A·Δt) · h_{t-1} + B·x_t             (fixed per-head decay)
    LNN-SSM:       h_t = exp((A·Δt + ΔA(x_t))) · h_{t-1} + B·x_t (data-dependent decay)

    where ΔA(x_t) = -Δt / τ(x_t),   τ(x_t) = softplus(W·x_t) + τ_min > 0

The LiquidGate module maps the SSM input to a per-head, per-timestep modulation
of the log-decay, making time constants data-dependent. This is injected into
ssd_lnn() which otherwise runs the standard Mamba-2/3 chunked SSD unchanged.

Training Task — Damped-Oscillation Regression:
    Sequences of the form  y(t) = sin(ω·t) · exp(-γ·t),  γ ~ Uniform(0.01, 0.5)
    The model sees a window of past values and must predict the next one.
    Since γ varies per sequence, a model with data-dependent time constants (LNN)
    can infer γ from context and adapt its forgetting rate accordingly —
    directly demonstrating the advantage of LNN over fixed-decay SSMs.
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from einops import rearrange, repeat


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SSDLNNConfig:
    """Configuration for the SSD-LNN regression model.

    Geometry follows Mamba-3Config conventions (d_inner = expand * d_model,
    nheads = d_inner // headdim). d_state must be even for RoPE pairing.

    LNN-specific fields:
        use_liquid   — enable the LiquidGate (False → standard Mamba-3 SISO baseline)
        tau_min      — minimum time constant (prevents division-by-zero in τ(x))
        liquid_scale — multiplier on the ΔA modulation (warm-start near zero effect)
    """
    d_model: int   = 64
    n_layer: int   = 2
    d_state: int   = 16     # must be even (RoPE pairing)
    expand:  int   = 2
    headdim: int   = 16
    chunk_size: int = 32

    # LNN-specific
    use_liquid:   bool  = True
    tau_min:      float = 0.1
    liquid_scale: float = 1.0

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        assert self.d_state % 2 == 0, "d_state must be even for RoPE pairing"


# ──────────────────────────────────────────────────────────────────────────────
# Utility: segsum  (identical to mamba3.py)
# ──────────────────────────────────────────────────────────────────────────────

def segsum(x: Tensor, device=None) -> Tensor:
    """Stable segment sum — produces the lower-triangular decay mask L.

    exp(segsum(A))[..., i, j] = exp(Σ_{k=j+1}^{i} A[k])  for i ≥ j, else 0.
    Source: Mamba-2 (Dao & Gu, 2024).
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


# ──────────────────────────────────────────────────────────────────────────────
# Utility: apply_rope  (identical to mamba3.py)
# ──────────────────────────────────────────────────────────────────────────────

def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Apply data-dependent rotary position embedding (Proposition 3, Mamba-3).

    Arguments
        x:      (..., d_state)
        angles: (..., d_state // 2) — cumulative rotation angles
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    return torch.stack([cos_a * x1 - sin_a * x2,
                        sin_a * x1 + cos_a * x2], dim=-1).flatten(-2)


# ──────────────────────────────────────────────────────────────────────────────
# SSD-LNN: Core Algorithm
# ──────────────────────────────────────────────────────────────────────────────

def ssd_lnn(
    x, A, B, C, chunk_size,
    liquid_A_mod=None,
    initial_states=None,
    device=None,
):
    """Structured State Space Duality with Liquid decay modulation.

    Extends standard SSD (Mamba-2/3) by injecting a data-dependent perturbation
    into the log-decay before chunking:

        Standard SSD:  effective decay = A                      (fixed per-head scalar)
        SSD-LNN:       effective decay = A + liquid_A_mod(x_t)  (data-dependent)

    The perturbation is computed by LiquidGate upstream and passed in as
    liquid_A_mod: (batch, seqlen, nheads).  All four chunked-SSD steps
    (intra-chunk, per-chunk states, inter-chunk recurrence, state-to-output)
    are otherwise identical to the Mamba-2 algorithm.

    Arguments
        x:             (batch, seqlen, nheads, headdim)
        A:             (batch, seqlen, nheads)  — log-decay (Δ·A, already ≤ 0)
        B:             (batch, seqlen, nheads, d_state)
        C:             (batch, seqlen, nheads, d_state)
        chunk_size:    int  — seqlen must be divisible by chunk_size
        liquid_A_mod:  (batch, seqlen, nheads) or None — LNN modulation ΔA(x)

    Returns
        y:             (batch, seqlen, nheads, headdim)
        final_state:   (batch, nheads, headdim, d_state)
    """
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    # ── LNN: inject data-dependent modulation into log-decay ──
    # liquid_A_mod ≤ 0 (computed as -dt/τ), so this increases the decay magnitude
    # for inputs that signal "short memory needed" (large damping).
    if liquid_A_mod is not None:
        A = A + liquid_A_mod  # (batch, seqlen, nheads) — now data-dependent

    # ── Rearrange into chunks: (batch, n_chunks, chunk_size, ...) ──
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
        for m in (x, A, B, C)
    ]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # ── Step 1: Intra-chunk output (diagonal blocks) ──
    # Quadratic attention-like computation within each chunk of size Q.
    L = torch.exp(segsum(A, device=device))  # (batch, nheads, n_chunks, Q, Q)
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # ── Step 2: Per-chunk states (B-terms for low-rank factorization) ──
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # ── Step 3: Inter-chunk SSM recurrence (A-terms) ──
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # ── Step 4: State-to-output per chunk (C-terms) ──
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


# ──────────────────────────────────────────────────────────────────────────────
# LiquidGate — Data-Dependent Time Constants (the LNN component)
# ──────────────────────────────────────────────────────────────────────────────

class LiquidGate(nn.Module):
    """Maps SSM input to a per-head log-decay modulation — the LNN mechanism.

    Implements the Liquid Time-Constant (LTC) principle in discrete time:

        τ(x_t) = softplus(W_τ · x_t + b_τ) + τ_min   [positive time constant]
        ΔA(x_t) = −scale · Δt / τ(x_t)               [negative: adds extra decay]

    Semantics:
        • Small τ(x_t) → fast forgetting (useful for high-frequency / transient content)
        • Large τ(x_t) → slow forgetting (useful for slowly-varying / persistent content)

    The module is initialized so that W_τ ≈ 0 and b_τ = 1, meaning at init
    τ(x) ≈ softplus(1) + τ_min ≈ 1.3 + τ_min.  This ensures a near-zero
    perturbation at the start of training — the model can grow liquid dynamics
    gradually via gradient descent.
    """

    def __init__(self, d_inner: int, nheads: int, tau_min: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.tau_proj = nn.Linear(d_inner, nheads, bias=True)
        self.tau_min = tau_min
        self.scale = scale
        # Init: near-zero weights so liquid effect starts small
        nn.init.zeros_(self.tau_proj.weight)
        nn.init.ones_(self.tau_proj.bias)

    def forward(self, x: Tensor, dt: Tensor) -> Tensor:
        """
        Args:
            x:  (batch, seqlen, d_inner) — SSM input (pre-reshape)
            dt: (batch, seqlen, nheads)  — discretized step size (after softplus)
        Returns:
            liquid_A_mod: (batch, seqlen, nheads) — ΔA(x_t), always ≤ 0
        """
        tau = F.softplus(self.tau_proj(x)) + self.tau_min  # (b, l, nheads)
        return -self.scale * dt / tau  # (b, l, nheads), negative → more decay


# ──────────────────────────────────────────────────────────────────────────────
# Utility Modules (RMSNorm, SwiGLU — mirrors mamba3.py)
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU(x) = W_down(SiLU(W_gate(x)) ⊙ W_up(x))  — Llama-style MLP."""

    def __init__(self, d_model: int, device=None):
        super().__init__()
        d_inner = 256 * ((int(2 * (4 * d_model) / 3) + 255) // 256)
        self.w_gate = nn.Linear(d_model, d_inner, bias=False, device=device)
        self.w_up   = nn.Linear(d_model, d_inner, bias=False, device=device)
        self.w_down = nn.Linear(d_inner, d_model, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ──────────────────────────────────────────────────────────────────────────────
# Mamba3LNNBlock — Mamba-3 SISO SSM mixer with optional LiquidGate
# ──────────────────────────────────────────────────────────────────────────────

class Mamba3LNNBlock(nn.Module):
    """Mamba-3 SISO SSM block augmented with a LiquidGate on the SSD decay.

    Includes all key Mamba-3 innovations:
      • Trapezoidal discretization (Proposition 1, Eq. 4) via two-SSD calls
      • Data-dependent RoPE on B, C (Proposition 3)
      • QK-normalization on B, C (Section 3.4)
      • Learnable BC bias (Section 3.4, Appendix G)

    LNN addition:
      • LiquidGate computes ΔA(x_t) = −Δt / τ(x_t) from the SSM input
      • Added to dA before both trapezoidal SSD calls (same decay for γ and β)
      • Disabled when cfg.use_liquid = False → exact Mamba-3 SISO baseline
    """

    def __init__(self, cfg: SSDLNNConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # ── Input projection: d_model → z + x + B + C + dt + λ + θ ──
        d_in = (
            2 * cfg.d_inner     # z + x
            + 2 * cfg.d_state   # B + C
            + 2 * cfg.nheads    # dt + λ (one scalar per head each)
            + cfg.d_state // 2  # θ (RoPE angles, d_state/2 pairs)
        )
        self.in_proj = nn.Linear(cfg.d_model, d_in, bias=False, device=device)

        # ── SSM parameters ──
        self.A_log   = nn.Parameter(torch.empty(cfg.nheads, device=device))
        self.D       = nn.Parameter(torch.empty(cfg.nheads, device=device))
        self.dt_bias = nn.Parameter(torch.empty(cfg.nheads, device=device))

        # ── QK-norm + BC bias ──
        self.B_norm = RMSNorm(cfg.d_state, device=device)
        self.C_norm = RMSNorm(cfg.d_state, device=device)
        self.B_bias = nn.Parameter(torch.ones(cfg.nheads, cfg.d_state, device=device))
        self.C_bias = nn.Parameter(torch.ones(cfg.nheads, cfg.d_state, device=device))

        # ── LNN component (optional) ──
        self.liquid_gate = (
            LiquidGate(cfg.d_inner, cfg.nheads, cfg.tau_min, cfg.liquid_scale)
            if cfg.use_liquid else None
        )

        # ── Output projection ──
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False, device=device)

        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.A_log, -1, 0)        # A = -exp(A_log) < 0
        nn.init.ones_(self.D)
        nn.init.constant_(self.dt_bias, math.log(0.1))

    def forward(self, u: Tensor) -> Tensor:
        """
        Args:
            u: (batch, seqlen, d_model) — pre-normed input
        Returns:
            y: (batch, seqlen, d_model)
        """
        cfg = self.cfg
        batch, seqlen, _ = u.shape

        # seqlen must be divisible by chunk_size
        assert seqlen % cfg.chunk_size == 0, (
            f"seqlen ({seqlen}) must be divisible by chunk_size ({cfg.chunk_size})"
        )

        # ── Negative diagonal of A ──
        A = -torch.exp(self.A_log)  # (nheads,)

        # ── Project input ──
        proj = self.in_proj(u)
        z, x, B, C, dt, lam, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.nheads, cfg.d_state // 2],
            dim=-1,
        )

        # ── Discretization (Eq. 4, Proposition 1) ──
        dt  = F.softplus(dt + self.dt_bias)   # (batch, seqlen, nheads), > 0
        lam = torch.sigmoid(lam)              # (batch, seqlen, nheads), ∈ (0,1)

        # ── QK-norm on B, C ──
        B = self.B_norm(B)
        C = self.C_norm(C)

        # ── Data-dependent RoPE angles ──
        # raw_angles[t, h, j] = Δ_t[h] * θ_t[j]
        raw_angles = dt.unsqueeze(-1) * rearrange(theta, "b l n -> b l 1 n")
        cum_angles = -torch.cumsum(raw_angles, dim=1)  # (batch, seqlen, nheads, d_state//2)

        # ── Trapezoidal coefficients (Eq. 4) ──
        dA    = dt * rearrange(A, "h -> 1 1 h")   # (batch, seqlen, nheads)
        beta  = (1 - lam) * dt * torch.exp(dA)    # β_t: left-endpoint weight
        gamma = lam * dt                           # γ_t: right-endpoint weight

        # ── Add head bias + RoPE to B and C ──
        B = rearrange(B, "b l n -> b l 1 n") + self.B_bias   # (b, l, nheads, d_state)
        C = rearrange(C, "b l n -> b l 1 n") + self.C_bias
        B = apply_rope(B, cum_angles)
        C = apply_rope(C, cum_angles)

        # ── LNN: compute data-dependent decay modulation ──
        # liquid_A_mod has the same shape as dA: (batch, seqlen, nheads)
        liquid_A_mod = None
        if self.liquid_gate is not None:
            liquid_A_mod = self.liquid_gate(x, dt)

        # ── Reshape x for SSD ──
        x_ssd = rearrange(x, "b l (h p) -> b l h p", p=cfg.headdim)

        # ── Trapezoidal SSD: two-SSD decomposition (see mamba3.py Mamba3.forward) ──
        # γ term: current-timestep contribution
        y_gamma, _ = ssd_lnn(
            x_ssd * gamma.unsqueeze(-1),
            dA, B, C, cfg.chunk_size,
            liquid_A_mod=liquid_A_mod,
            device=self.device,
        )

        # β term: previous-timestep contribution (B and x shifted right by 1)
        B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))          # (b, l, nheads, d_state)
        x_prev = F.pad(x_ssd[:, :-1], (0, 0, 0, 0, 1, 0))      # (b, l, nheads, headdim)
        y_beta, _ = ssd_lnn(
            x_prev * beta.unsqueeze(-1),
            dA, B_prev, C, cfg.chunk_size,
            liquid_A_mod=liquid_A_mod,
            device=self.device,
        )

        y = y_gamma + y_beta  # (batch, seqlen, nheads, headdim)

        # ── Skip connection + output gate ──
        y = y + rearrange(self.D, "h -> 1 1 h 1") * x_ssd
        y = rearrange(y, "b l h p -> b l (h p)")
        y = y * F.silu(z)
        return self.out_proj(y)


# ──────────────────────────────────────────────────────────────────────────────
# Mamba3LNNRegressor — Full model for sequence regression
# ──────────────────────────────────────────────────────────────────────────────

class Mamba3LNNRegressor(nn.Module):
    """Sequence regressor built from Mamba3LNN blocks.

    Architecture:
        Input (batch, seqlen, input_dim)
        → Linear input embedding
        → N × [RMSNorm → Mamba3LNNBlock → Residual, RMSNorm → SwiGLU → Residual]
        → RMSNorm
        → take last-timestep output
        → Linear regression head → (batch, output_dim)
    """

    def __init__(self, cfg: SSDLNNConfig, input_dim: int = 1, output_dim: int = 1, device=None):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(input_dim, cfg.d_model, bias=True, device=device)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                mixer_norm = RMSNorm(cfg.d_model, device=device),
                mixer      = Mamba3LNNBlock(cfg, device=device),
                mlp_norm   = RMSNorm(cfg.d_model, device=device),
                mlp        = SwiGLU(cfg.d_model, device=device),
            ))
            for _ in range(cfg.n_layer)
        ])

        self.norm_f = RMSNorm(cfg.d_model, device=device)
        self.head   = nn.Linear(cfg.d_model, output_dim, bias=True, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seqlen, input_dim)
        Returns:
            pred: (batch, output_dim) — prediction from last timestep
        """
        h = self.input_proj(x)  # (batch, seqlen, d_model)

        for layer in self.layers:
            h = layer["mixer"](layer["mixer_norm"](h)) + h
            h = layer["mlp"](layer["mlp_norm"](h)) + h

        h = self.norm_f(h)
        return self.head(h[:, -1])  # last timestep → (batch, output_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset — Damped Oscillations with Variable Damping
# ──────────────────────────────────────────────────────────────────────────────

def generate_damped_oscillation_dataset(
    n_sequences: int = 4000,
    seq_len: int = 128,
    window_size: int = 64,
    gamma_range: tuple = (0.01, 0.5),
    omega_range: tuple = (0.5, 3.0),
    noise_std: float = 0.02,
    val_fraction: float = 0.2,
    seed: int = 42,
    batch_size: int = 64,
    chunk_size: int = 32,
):
    """Generate sliding-window dataset of damped oscillations.

    Each sequence:  y(t) = sin(ω·t + φ) · exp(-γ·t)
    where γ ~ Uniform(gamma_range) and ω ~ Uniform(omega_range).

    Task: given a window of `window_size` timesteps, predict the next value.

    The window_size is padded to the nearest multiple of chunk_size so it
    can be processed by the chunked SSD algorithm without remainder.

    Returns a dict with train_loader, val_loader, and metadata.
    """
    rng = np.random.default_rng(seed)

    # Round window_size up to nearest multiple of chunk_size
    if window_size % chunk_size != 0:
        window_size = ((window_size // chunk_size) + 1) * chunk_size
        print(f"[dataset] window_size padded to {window_size} (multiple of chunk_size={chunk_size})")

    t = np.linspace(0, 6 * math.pi, seq_len + 1)

    X_list, y_list = [], []
    for _ in range(n_sequences):
        gamma = rng.uniform(*gamma_range)
        omega = rng.uniform(*omega_range)
        phi   = rng.uniform(0, 2 * math.pi)

        signal = np.sin(omega * t + phi) * np.exp(-gamma * t)
        signal += rng.normal(0, noise_std, size=signal.shape)

        # Slide over valid positions
        for start in range(seq_len - window_size):
            window = signal[start : start + window_size]
            target = signal[start + window_size]
            X_list.append(window)
            y_list.append(target)

    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)  # (N, window, 1)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    n_val = int(len(X) * val_fraction)
    idx   = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[dataset] {len(train_ds)} train / {len(val_ds)} val samples  "
          f"| window={window_size}  chunk_size={chunk_size}")
    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "window_size":  window_size,
        "input_dim":    1,
        "output_dim":   1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_regression_metrics(targets: np.ndarray, preds: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for regression evaluation."""
    mae  = float(np.mean(np.abs(targets - preds)))
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop  (mirrors train_tcn_lnn.py structure)
# ──────────────────────────────────────────────────────────────────────────────

def train_ssd_lnn_model(data_dict, model_params, train_params, save_dir="models"):
    """Train a Mamba3LNN (or baseline Mamba3 SISO) regression model.

    Args:
        data_dict:    dict from generate_damped_oscillation_dataset()
        model_params: dict with SSDLNNConfig fields + optionally use_liquid flag
        train_params: dict with lr, epochs, patience
        save_dir:     directory to save checkpoints and plots

    Returns:
        (model, history, best_model_path)
    """
    os.makedirs(save_dir, exist_ok=True)

    train_loader = data_dict["train_loader"]
    val_loader   = data_dict["val_loader"]

    # ── Build config and model ──
    cfg = SSDLNNConfig(
        d_model     = model_params.get("d_model",      64),
        n_layer     = model_params.get("n_layer",       2),
        d_state     = model_params.get("d_state",      16),
        expand      = model_params.get("expand",        2),
        headdim     = model_params.get("headdim",      16),
        chunk_size  = model_params.get("chunk_size",   32),
        use_liquid  = model_params.get("use_liquid",  True),
        tau_min     = model_params.get("tau_min",     0.1),
        liquid_scale= model_params.get("liquid_scale",1.0),
    )

    device = get_device()
    model  = Mamba3LNNRegressor(
        cfg,
        input_dim  = data_dict["input_dim"],
        output_dim = data_dict["output_dim"],
        device     = device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = "SSD-LNN" if cfg.use_liquid else "SSD-Baseline"
    print(f"\nStarting {tag} training on {device} | params: {n_params:,}")

    # ── Optimizer + loss ──
    lr      = train_params.get("lr",       1e-3)
    epochs  = train_params.get("epochs",    50)
    patience= train_params.get("patience",  10)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_val_loss   = float("inf")
    best_model_path = None
    counter         = 0

    for epoch in range(epochs):
        # ── Training phase ──
        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{tag}]")
        for inputs, targets in bar:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ── Validation phase ──
        model.eval()
        val_loss    = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                val_loss += loss.item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        metrics = calculate_regression_metrics(all_targets, all_outputs)
        history["val_metrics"].append(metrics)

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  "
            f"MAE: {metrics['mae']:.4f}  RMSE: {metrics['rmse']:.4f}  R²: {metrics['r2']:.4f}"
        )

        # ── Early stopping + checkpoint ──
        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            counter         = 0
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                "model_state": model.state_dict(),
                "model_params": model_params,
                "train_params": train_params,
                "metrics": metrics,
            }, best_model_path)
            print(f"  ✓ Saved best model → {best_model_path}")
        else:
            counter += 1
            print(f"  EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("  Early stopping triggered.")
                break

    print(f"{tag} training complete.\n")

    # ── Save final checkpoint ──
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save({"model_state": model.state_dict(), "model_params": model_params}, final_path)

    # ── Loss curves ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{tag} — Training History", fontsize=13, fontweight="bold")

    epochs_x = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_x, history["train_loss"], label="Train")
    axes[0].plot(epochs_x, history["val_loss"],   label="Val")
    axes[0].set_title("MSE Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    mae_vals  = [m["mae"]  for m in history["val_metrics"]]
    rmse_vals = [m["rmse"] for m in history["val_metrics"]]
    r2_vals   = [m["r2"]   for m in history["val_metrics"]]

    axes[1].plot(epochs_x, mae_vals,  color="#29B5C8"); axes[1].set_title("Val MAE")
    axes[1].set_xlabel("Epoch"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_x, r2_vals, color="#E8851B"); axes[2].set_title("Val R²")
    axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save history JSON ──
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(
            {
                "train_loss":  [float(v) for v in history["train_loss"]],
                "val_loss":    [float(v) for v in history["val_loss"]],
                "val_metrics": [{k: float(v) for k, v in m.items()} for m in history["val_metrics"]],
            },
            f, indent=4,
        )

    return model, history, best_model_path


# ──────────────────────────────────────────────────────────────────────────────
# Compare Baseline vs SSD-LNN
# ──────────────────────────────────────────────────────────────────────────────

def train_ssd_lnn_all_configs(save_dir="models/ssd_lnn"):
    """Train both the Mamba-3 SISO baseline and the SSD-LNN model, then compare.

    Results are saved to:
        save_dir/baseline/   — standard Mamba-3 SISO (use_liquid=False)
        save_dir/lnn/        — SSD-LNN (use_liquid=True)
        save_dir/comparison.png  — side-by-side metric curves
        save_dir/summary.json    — final metrics for both models
    """
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save    = os.path.join(save_dir, timestamp)
    os.makedirs(base_save, exist_ok=True)

    # ── Shared dataset ──
    data_dict = generate_damped_oscillation_dataset(
        n_sequences  = 4000,
        seq_len      = 128,
        window_size  = 64,
        gamma_range  = (0.01, 0.5),
        omega_range  = (0.5, 3.0),
        noise_std    = 0.02,
        val_fraction = 0.2,
        batch_size   = 64,
        chunk_size   = 32,
    )

    # ── Shared model/training params ──
    model_params_base = dict(
        d_model     = 64,
        n_layer     = 2,
        d_state     = 16,
        expand      = 2,
        headdim     = 16,
        chunk_size  = 32,
    )
    train_params = dict(lr=1e-3, epochs=60, patience=15)

    results = {}

    for use_liquid, tag in [(False, "baseline"), (True, "lnn")]:
        mp = {**model_params_base, "use_liquid": use_liquid, "tau_min": 0.1, "liquid_scale": 1.0}
        model, history, best_path = train_ssd_lnn_model(
            data_dict, mp, train_params,
            save_dir=os.path.join(base_save, tag),
        )
        results[tag] = {"history": history, "best_model_path": best_path}

    # ── Comparison plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SSD-LNN vs Mamba-3 SISO Baseline — Damped Oscillation Regression",
                 fontsize=13, fontweight="bold")

    colors = {"baseline": "#888888", "lnn": "#29B5C8"}
    labels = {"baseline": "Mamba-3 SISO (baseline)", "lnn": "SSD-LNN (ours)"}

    for tag, info in results.items():
        vm = info["history"]["val_metrics"]
        ex = range(1, len(vm) + 1)
        c  = colors[tag]
        l  = labels[tag]
        axes[0].plot(ex, info["history"]["val_loss"], color=c, label=l, linewidth=1.5)
        axes[1].plot(ex, [m["mae"]  for m in vm],    color=c, label=l, linewidth=1.5)
        axes[2].plot(ex, [m["r2"]   for m in vm],    color=c, label=l, linewidth=1.5)

    for ax, title, ylabel in zip(
        axes,
        ["Validation MSE Loss", "Validation MAE", "Validation R²"],
        ["MSE",  "MAE",  "R²"],
    ):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_save, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved → {base_save}/comparison.png")

    # ── Summary JSON ──
    summary = {}
    for tag, info in results.items():
        final_metrics = info["history"]["val_metrics"][-1]
        summary[tag] = {
            "best_model_path": info["best_model_path"],
            "final_val_loss":  float(info["history"]["val_loss"][-1]),
            "final_metrics":   {k: float(v) for k, v in final_metrics.items()},
        }
    with open(os.path.join(base_save, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # ── Print comparison table ──
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 55)
    for tag, info in summary.items():
        m = info["final_metrics"]
        print(f"{labels[tag]:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
    print("=" * 55)
    print(f"\nAll results saved to {base_save}")

    return results, base_save


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, save_dir = train_ssd_lnn_all_configs(save_dir="models/ssd_lnn")
