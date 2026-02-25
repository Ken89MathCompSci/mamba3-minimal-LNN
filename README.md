
# mamba3-minimal

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white)
![Mac](https://img.shields.io/badge/Apple_Silicon-MPS_Ready-black?style=for-the-badge&amp;logo=apple)
![CPU](https://img.shields.io/badge/CPU-0071C5?style=for-the-badge&amp;logo=intel&amp;logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)

A minimal, single-file implementation of **Mamba-3** in pure PyTorch. Built as a readable "Rosetta Stone" to bridge the gap between the paper's dense mathematics and working code, without relying on custom C++ or Triton kernels.

> **MAMBA-3: IMPROVED SEQUENCE MODELING USING STATE SPACE PRINCIPLES**\
> *Under review at ICLR 2026*

Mamba-3 is the latest evolution of the Mamba family of State Space Models (SSMs). Unlike Transformers, Mamba models map sequences through a hidden state, enabling **linear** scaling in computation/memory during training and **constant time** per step during inference. 

Mamba-3 introduces several major innovations over Mamba-2:

| Innovation | Paper Ref | What it does |
|---|---|---|
| **Trapezoidal Discretization** | Eq. 4, Prop. 1 | Second-order accurate state update (vs. first-order Euler) |
| **Complex SSM / RoPE** | Eq. 9, Prop. 3–4 | Enables state-tracking via data-dependent rotary embeddings |
| **QK-Normalization on B, C** | § 3.4 | Repositioned norm for training stability |
| **Learnable BC Bias** | § 3.4, App. G | Head-specific, channel-wise bias initialized to ones |
| **No Short Convolution** | § 3.4 | Trapezoidal rule + bias makes Conv1d unnecessary |
| **Llama-Style Architecture** | § 3.4 | Alternating Mamba-3 SSM + SwiGLU MLP blocks |
| **MIMO (Rank-R)** | App. D | Rank-R state update for higher expressivity at same state size |

This implementation is **hardware-agnostic** (CUDA, Apple Silicon MPS, CPU) and emphasizes **readability** with 1-to-1 equation references from the paper.

## Architecture

```text
Input Token IDs
      │
      ▼
  Embedding
      │
      ▼
┌─────────────────────────────────────────┐
│  × n_layer                              │
│                                         │
│  ┌─ RMSNorm → Mamba3 SSM → Residual     │
│  │   • in_proj → (z, x, B, C, Δ, λ, θ)  │
│  │   • B, C: QK-Norm → +Bias → RoPE     │
│  │   • Trapezoidal SSD (γ + β terms)    │
│  │   • y + D·x → y · SiLU(z) → out      │
│  │                                      │
│  └─ RMSNorm → SwiGLU MLP → Residual     │
│       • SiLU(W_gate·x) ⊙ W_up·x         │
│       • → W_down                        │
└─────────────────────────────────────────┘
      │
      ▼
  RMSNorm → LM Head (weight-tied)
      │
      ▼
   Logits

```

## Two-SSD Decomposition

Because Mamba-3 uses a Trapezoidal rule, the recurrence (Eq. 9) introduces a strict cross-boundary dependency ($\beta_t$):

> $h_t = \alpha_t h_{t-1} + \beta_t (B_{t-1} x_{t-1}) + \gamma_t (B_t x_t)$

Standard PyTorch chunking breaks here, which is why official implementations rely on custom CUDA. To solve this in pure PyTorch while keeping memory bounded, `mamba3-minimal` decomposes the equation into **two standard SSD calls**:

1. **γ SSD** (Current-timestep): `ssd(x * γ, dA, B, C)`
2. **β SSD** (Previous-timestep): `ssd(x_prev * β, dA, B_prev, C)`

By pre-shifting `B` and `x` at the **global sequence level** before chunking, cross-chunk boundaries are handled naturally without explicit boundary correction logic.

## Usage

Install dependencies:

```bash
pip install torch einops

```

### Quick Test

Run the built-in consistency check. This proves that the highly-parallel chunked training pass produces the exact same logits as the sequential autoregressive inference step (`max_diff < 1e-2`).

```bash
python mamba3.py 

```

### As a Library

```python
import torch
from mamba3 import Mamba3Config, Mamba3LMHeadModel, InferenceCache

# 1. Create model
args = Mamba3Config(d_model=768, n_layer=24, vocab_size=50277)
model = Mamba3LMHeadModel(args, device="cuda")

# 2. Forward pass (Training)
input_ids = torch.randint(0, args.vocab_size, (1, 128), device="cuda")
logits, h = model(input_ids)  # logits: (1, 128, vocab_size)

# 3. Autoregressive Generation (O(1) inference)
prompt = torch.tensor([1, 2, 3], device="cuda")  # 1D tensor (no batch dim)
for token, h in model.generate(prompt, max_new_length=50, temperature=0.8):
    print(token, end=" ")

```

## Tests & Verification

This repo includes specialized tests to verify the core architectural claims of Mamba-3:

```bash
# 1. State-Tracking Parity Test (Table 4b)
# Proves Complex SSM / RoPE enables 100% accuracy on parity tasks (where Mamba-2 fails)
python tests/test_parity.py

# 2. MIMO Forward/Step Consistency
# Verifies the rank-R matrix implementation matches sequential inference
python tests/test_mimo.py

# 3. Real Vocabulary Generation
# Tests tokenization (tiktoken) and generation loop with 50k+ vocab size
python tests/test_text.py
```

## File Structure

| File | Description |
| --- | --- |
| `mamba3.py` | Complete Mamba-3 + MIMO implementation (~800 lines, single file) |
| `demo.py` | Demo script: architecture, training loop, generation, MIMO pass |
| `tests/` | Verification scripts: Parity, MIMO, Text Generation |

## Credits & Resources

* [Albert Gu](https://github.com/albertfgu) and [Tri Dao](https://github.com/tridao) — Authors of the Mamba architecture family.
* [Tommy Ip](https://github.com/tommyip) — Author of [tommyip/mamba2-minimal](https://github.com/tommyip/mamba2-minimal), which provided the foundational SSD chunking algorithm this repo builds upon.
* [John Ma](https://github.com/johnma2006) — Author of [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal), which inspired the minimal, educational approach.
