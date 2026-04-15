"""
train_ssd_lnn.py
================
Integrates Liquid Neural Network (LNN) dynamics into the Mamba-3 SSD algorithm,
trained on the UK-DALE dataset for Non-Intrusive Load Monitoring (NILM).

LNN Key Idea (Hasani et al., 2021 — Liquid Time-Constant Networks):
    Standard SSM:  h_t = exp(A·Δt) · h_{t-1} + B·x_t             (fixed per-head decay)
    LNN-SSM:       h_t = exp((A·Δt + ΔA(x_t))) · h_{t-1} + B·x_t (data-dependent decay)

    where ΔA(x_t) = -Δt / τ(x_t),   τ(x_t) = softplus(W·x_t) + τ_min > 0

The LiquidGate module maps the SSM input to a per-head, per-timestep modulation
of the log-decay, making time constants data-dependent. This is injected into
ssd_lnn() which otherwise runs the standard Mamba-2/3 chunked SSD unchanged.

Training Task — NILM on UK-DALE:
    Given a window of aggregate mains power, predict the power consumption of a
    target appliance (sequence-to-point). Metrics follow calculate_nilm_metrics:
    MAE (regression), SAE (aggregate energy error), F1 (on/off classification).
"""

import os
import sys
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from einops import rearrange, repeat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics, save_model

# ---------------------------------------------------------------------------
# Constants  (matches test_multiscale_lnn_ukdale_specific_splits.py)
# ---------------------------------------------------------------------------

STRIDE = 5
APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def load_data():
    """Load raw UK-DALE DataFrames from data/ukdale/*.pkl"""
    print("Loading UKDALE data...")
    with open('data/ukdale/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/ukdale/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/ukdale/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]
    print(f"Train: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val:   {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test:  {test_data.index.min()} to {test_data.index.max()}")
    print(f"Columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=100):
    mains = data['main'].values
    X = []
    for i in range(0, len(mains) - window_size + 1, STRIDE):
        X.append(mains[i:i + window_size])
    return np.array(X).reshape(-1, window_size, 1)


def prepare_appliance_data(raw_data, appliance_name, window_size=100):
    """Create scaled DataLoaders for one appliance from raw DataFrames."""
    X_tr = create_sequences(raw_data['train'], window_size)
    X_va = create_sequences(raw_data['val'],   window_size)
    X_te = create_sequences(raw_data['test'],  window_size)

    y_tr = raw_data['train'][appliance_name].iloc[::STRIDE].values.reshape(-1, 1)[:len(X_tr)]
    y_va = raw_data['val'][appliance_name].iloc[::STRIDE].values.reshape(-1, 1)[:len(X_va)]
    y_te = raw_data['test'][appliance_name].iloc[::STRIDE].values.reshape(-1, 1)[:len(X_te)]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_tr = y_scaler.fit_transform(y_tr)
    y_va = y_scaler.transform(y_va)
    y_te = y_scaler.transform(y_te)

    print(f"Training sequences:   {X_tr.shape}")
    print(f"Validation sequences: {X_va.shape}")
    print(f"Test sequences:       {X_te.shape}")

    return {
        'train_loader': DataLoader(SimpleDataset(X_tr, y_tr), batch_size=32, shuffle=True),
        'val_loader':   DataLoader(SimpleDataset(X_va, y_va), batch_size=32, shuffle=False),
        'test_loader':  DataLoader(SimpleDataset(X_te, y_te), batch_size=32, shuffle=False),
        'y_scaler':       y_scaler,
        'appliance_name': appliance_name,
        'input_size':     1,
        'output_size':    1,
    }


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
# Training Loop  (mirrors train_tcn_lnn.py structure)
# ──────────────────────────────────────────────────────────────────────────────

def train_ssd_lnn_model(data_dict, model_params, train_params, save_dir="models"):
    """Train a Mamba3LNN regression model on UK-DALE NILM data.

    Args:
        data_dict:    dict from load_and_preprocess_ukdale()
        model_params: dict with SSDLNNConfig fields
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
        input_dim  = data_dict["input_size"],
        output_dim = data_dict["output_size"],
        device     = device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStarting SSD-LNN training on {device} | params: {n_params:,}")

    # ── Optimizer + loss ──
    lr      = train_params.get("lr",       1e-3)
    epochs  = train_params.get("epochs",   80)
    patience= train_params.get("patience", 20)

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
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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
        y_scaler = data_dict.get("y_scaler")
        if y_scaler is not None:
            all_targets = y_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
            all_outputs = y_scaler.inverse_transform(all_outputs.reshape(-1, 1)).flatten()
        threshold = data_dict.get("threshold", 10.0)
        metrics = calculate_nilm_metrics(all_targets, all_outputs, threshold=threshold)
        history["val_metrics"].append(metrics)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
            f"Val MAE: {metrics['mae']:.2f}, Val SAE: {metrics['sae']:.4f}, "
            f"Val F1: {metrics['f1']:.4f}"
        )

        # ── Early stopping + checkpoint ──
        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            counter         = 0
            best_model_path = os.path.join(save_dir, "ssd_lnn_model_best.pth")
            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    # ── Save final checkpoint ──
    final_path = os.path.join(save_dir, "ssd_lnn_model_final.pth")
    save_model(model, model_params, train_params, metrics, final_path)

    # ── Training history plot (mirrors train_tcn_lnn.py) ──
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    val_mae = [m["mae"] for m in history["val_metrics"]]
    plt.plot(val_mae, label="Validation MAE")
    plt.title("Validation MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ssd_lnn_training_history.png"))
    plt.close()

    # ── Save history JSON ──
    with open(os.path.join(save_dir, "ssd_lnn_history.json"), "w") as f:
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
# Train on All Appliances  (mirrors train_tcn_lnn_all_appliances)
# ──────────────────────────────────────────────────────────────────────────────

def train_ssd_lnn_all_appliances(house_number=1, window_size=100, save_dir="models/ssd_lnn"):
    """Train SSD-LNN on all appliances in the specified UK-DALE house.

    Uses the same window_size=100 as train_tcn_lnn.py. chunk_size is set to 25
    (a divisor of 100) so the chunked SSD can process it evenly (4 chunks of 25).

    Args:
        house_number: House number in the UK-DALE dataset
        window_size:  Input sequence length (must be divisible by chunk_size=25)
        save_dir:     Directory to save the models

    Returns:
        (results dict, base_save_dir)
    """
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save    = os.path.join(save_dir, f"house{house_number}_{timestamp}")
    os.makedirs(base_save, exist_ok=True)

    # Load data once, then loop over appliances
    raw_data = load_data()
    appliances = {i: name for i, name in enumerate(APPLIANCES)}

    print(f"Training SSD-LNN models for {len(appliances)} appliances:")
    for idx, name in appliances.items():
        print(f"  Index {idx}: {name}")

    config = {
        "window_size": window_size,
        "timestamp":   timestamp,
        "appliances":  appliances,
    }
    with open(os.path.join(base_save, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    model_params = {
        "input_size":    1,
        "output_size":   1,
        "d_model":       64,
        "n_layer":       2,
        "d_state":       16,
        "expand":        2,
        "headdim":       16,
        "chunk_size":    25,
        "use_liquid":    True,
        "tau_min":       0.1,
        "liquid_scale":  1.0,
    }
    train_params = {"lr": 0.001, "epochs": 80, "patience": 20}

    results = {}

    for appliance_idx, appliance_name in appliances.items():
        print(f"\n{'-'*50}")
        print(f"Training SSD-LNN model for {appliance_name} (index {appliance_idx})")
        print(f"{'-'*50}\n")

        appliance_dir = os.path.join(base_save, appliance_name)
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            data_dict = prepare_appliance_data(raw_data, appliance_name, window_size=window_size)
            data_dict["threshold"] = THRESHOLDS[appliance_name]

            model, history, best_model_path = train_ssd_lnn_model(
                data_dict, model_params, train_params,
                save_dir=appliance_dir,
            )

            results[appliance_name] = {
                "model_path":    best_model_path,
                "appliance_index": appliance_idx,
                "final_metrics": history["val_metrics"][-1] if history["val_metrics"] else None,
                "history":       history,
            }
            print(f"Successfully trained SSD-LNN model for {appliance_name}")

        except Exception as e:
            print(f"Error training SSD-LNN model for {appliance_name}: {str(e)}")

    # ── Summary JSON ──
    summary = {
        "timestamp":    timestamp,
        "house_number": house_number,
        "results": {
            name: {
                "model_path":    info["model_path"],
                "final_metrics": {k: float(v) for k, v in info["final_metrics"].items()}
                                  if info["final_metrics"] else None,
            }
            for name, info in results.items()
        },
    }
    with open(os.path.join(base_save, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # ── Combined val metrics plot (appliances × metrics grid) ──
    trained = {
        name: info for name, info in results.items()
        if info.get("history") and info["history"]["val_metrics"]
    }
    if trained:
        n_apps = len(trained)
        fig, axes = plt.subplots(n_apps, 3, figsize=(15, 4 * n_apps))
        if n_apps == 1:
            axes = [axes]
        fig.suptitle(
            f"Val Metrics per Epoch — SSD-LNN (80 epochs)",
            fontsize=13, fontweight="bold",
        )

        for row, (app_name, info) in enumerate(trained.items()):
            vm       = info["history"]["val_metrics"]
            epochs_x = range(1, len(vm) + 1)
            mae_vals = [m["mae"] for m in vm]
            sae_vals = [m["sae"] for m in vm]
            f1_vals  = [m["f1"]  for m in vm]

            for col, (vals, ylabel, title) in enumerate([
                (mae_vals, "MAE (W)",  f"{app_name} — MAE (W)"),
                (sae_vals, "SAE",      f"{app_name} — SAE"),
                (f1_vals,  "F1",       f"{app_name} — F1"),
            ]):
                ax = axes[row][col]
                ax.plot(epochs_x, vals, color="#29B5C8", linewidth=1.5)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Epoch", fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(base_save, "ssd_lnn_val_metrics.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        print(f"Val metrics plot saved to {base_save}/ssd_lnn_val_metrics.png")

    print("\nSSD-LNN training completed for all appliances!")
    print(f"Results saved to {base_save}")

    return results, base_save


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, save_dir = train_ssd_lnn_all_appliances(house_number=5)
