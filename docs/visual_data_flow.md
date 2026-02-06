# Visual Data Flow Diagram

This document provides visual representations of data flow through the hBehaveMAE model.

## Configuration

```
input_size: (400, 1, 72)
patch_stride: (2, 1, 3)
q_strides: [(2,1,4), (2,1,6)]
stages: [2, 3, 4]
```

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         hBehaveMAE Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘

Input: [B, 400, 72]
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Patch Embedding (Conv3d)                                                │
│  kernel=(2,1,3), stride=(2,1,3)                                          │
│  Output: [B, 4800, 96]                                                   │
│  Tokens: 4,800 | Dim: 96 | RF: 2×1×3                                    │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 0: Fine-grained Features (Blocks 0-1)                             │
│  • Mask Unit Attention (local)                                           │
│  • q_stride = 1 (no pooling)                                             │
│  • Tokens: 4,800 | Dim: 96 | RF: 2×1×3                                  │
│  • Learns: Local motion patterns                                         │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼ (Q-pooling: stride=(2,1,4), reduce by 8×)
    │
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Mid-level Features (Blocks 2-4)                                │
│  • Global Attention                                                      │
│  • Pooling at Block 2: q_stride = (2,1,4)                               │
│  • Tokens: 600 | Dim: 192 | RF: 4×1×12                                  │
│  • Learns: Action sequences                                              │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼ (Q-pooling: stride=(2,1,6), reduce by 12×)
    │
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: High-level Features (Blocks 5-8)                               │
│  • Global Attention                                                      │
│  • Pooling at Block 5: q_stride = (2,1,6)                               │
│  • Tokens: 50 | Dim: 384 | RF: 8×1×72                                   │
│  • Learns: Complex activities                                            │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼ (Project to 256)
    │
┌─────────────────────────────────────────────────────────────────────────┐
│  Encoder Output                                                          │
│  [B, 50, 256]                                                            │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Decoder                                                                 │
│  • Embed: 256 → 128                                                      │
│  • Add mask tokens for masked positions                                  │
│  • 1 Transformer block                                                   │
│  • Predict: 128 → 288 (reconstruction)                                   │
│  Output: [B, 50, 288]                                                    │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Loss (MSE on masked tokens)
```

## Token Evolution Through Stages

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Token Count & Receptive Field                     │
└────────────────────────────────────────────────────────────────────────┘

Stage 0:  ████████████████████████████████████████████ 4,800 tokens
          RF: 2×1×3 = 6 input elements
          [Fine-grained: many small receptive fields]

          ▼ Q-pooling (÷8)

Stage 1:  █████████████████████ 600 tokens
          RF: 4×1×12 = 48 input elements
          [Mid-level: fewer, larger receptive fields]

          ▼ Q-pooling (÷12)

Stage 2:  ███ 50 tokens
          RF: 8×1×72 = 576 input elements
          [High-level: very few, very large receptive fields]
```

## How Q-Stride Pooling Works

### Stage 0 → Stage 1: q_stride = (2, 1, 4)

```
Before pooling (Stage 0):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ T │ T │ T │ T │ T │ T │ T │ T │  Each cell = 1 token
└───┴───┴───┴───┴───┴───┴───┴───┘
 200 tokens in temporal dim, 24 in width dim
 = 4,800 total tokens

After pooling with q_stride=(2,1,4):
┌───────┬───────┐
│   T   │   T   │  Each cell = pooled from 8 tokens
└───────┴───────┘   (2 temporal × 4 width = 8)
 100 tokens in temporal dim, 6 in width dim
 = 600 total tokens

Pooling operation: Max pooling over 8 tokens
  • Temporal: 200 → 100 (stride 2)
  • Width: 24 → 6 (stride 4)
  • Total: 4,800 → 600 (÷8)
```

### Stage 1 → Stage 2: q_stride = (2, 1, 6)

```
Before pooling (Stage 1):
┌───┬───┬───┬───┬───┬───┐
│ T │ T │ T │ T │ T │ T │  600 tokens
└───┴───┴───┴───┴───┴───┘

After pooling with q_stride=(2,1,6):
┌─────────────┐
│      T      │  50 tokens (pooled from 12 each)
└─────────────┘

Pooling operation: Max pooling over 12 tokens
  • Temporal: 100 → 50 (stride 2)
  • Width: 6 → 1 (stride 6)
  • Total: 600 → 50 (÷12)
```

## Receptive Field Growth

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Receptive Field Growth                          │
└────────────────────────────────────────────────────────────────────────┘

Input sequence:
│0──10──20──30──40──50──60──70│ (72 width units, 400 time steps)

After Patch Embedding (each token):
│█│█│█│  (covers 3 width units, 2 time steps)

After Stage 0 (each token):
│████│████│  (covers 3 width units, 2 time steps) - no change

After Stage 1 (each token):
│█████████████│  (covers 12 width units, 4 time steps)
Pooling accumulated adjacent tokens' receptive fields

After Stage 2 (each token):
│████████████████████████████████████████████████████████████████████████│
(covers 72 width units, 8 time steps)
Final tokens have receptive field spanning entire width!
```

## Attention Patterns

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Attention Patterns                            │
└────────────────────────────────────────────────────────────────────────┘

Stage 0: Mask Unit Attention (Local)
┌────────┬────────┬────────┬────────┐
│ Window │ Window │ Window │ Window │  Tokens attend within windows
└────────┴────────┴────────┴────────┘
  Local spatial attention within mask units


Stage 1: Global Attention
┌────────────────────────────────────┐
│    All tokens attend to all        │  Full attention across all tokens
└────────────────────────────────────┘
  But fewer tokens (600 vs 4,800)


Stage 2: Global Attention
┌──────────────┐
│   50 tokens  │  Full attention, but very efficient
└──────────────┘   (only 50 tokens)
```

## Masking Strategy

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Mask Units                                   │
└────────────────────────────────────────────────────────────────────────┘

Mask unit size: (4, 1, 24) in original token space
                [temporal, height, width]

Original token layout (200×1×24 = 4,800 tokens):
┌─────────┬─────────┬─────────┬─────────┐
│ MU: 4×24│ MU: 4×24│ MU: 4×24│ MU: 4×24│  50 mask units total
└─────────┴─────────┴─────────┴─────────┘
Each MU contains 96 tokens (4 temporal × 1 height × 24 width)

Masking at mask unit level:
┌─────────┬─────────┬─────────┬─────────┐
│  Keep   │  MASK   │  MASK   │  MASK   │  Example: 75% mask ratio
└─────────┴─────────┴─────────┴─────────┘
   ✓         ✗         ✗         ✗

Benefits:
• Prevents information leakage between nearby tokens
• More challenging reconstruction task
• Encourages learning hierarchical patterns
```

## Decoder Reconstruction

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Decoder Reconstruction                           │
└────────────────────────────────────────────────────────────────────────┘

Encoder output: 50 tokens (some visible, some masked)
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ V │ M │ M │ M │ V │ M │ M │ V │  V=Visible, M=Masked
└───┴───┴───┴───┴───┴───┴───┴───┘

Decoder input: Fill masked positions with learnable mask token
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ V │[M]│[M]│[M]│ V │[M]│[M]│ V │  [M]=Mask token
└───┴───┴───┴───┴───┴───┴───┴───┘

Decoder processing:
• Add positional embeddings
• Self-attention across all 50 tokens
• MLP transformation
• Predict reconstruction for each token

Prediction head output: 50 × 288 values
Each token predicts 288 values:
  • These reconstruct the original patch (after striding)
  • pred_stride = 288 elements per token
  • Covers: 3×1×96 = 288 input elements

Loss computed only on masked tokens:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │ ✓ │ ✓ │ ✓ │   │ ✓ │ ✓ │   │  MSE loss here
└───┴───┴───┴───┴───┴───┴───┴───┘
```

## Summary Visualization

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Key Transformations Summary                         │
└────────────────────────────────────────────────────────────────────────┘

Dimension            Stage 0    →    Stage 1    →    Stage 2
────────────────────────────────────────────────────────────────────────
Token Count          4,800           600             50
Feature Dim          96              192             384
Receptive Field      2×1×3           4×1×12          8×1×72
Attention Type       Local           Global          Global
Computation          Heavy           Medium          Light
Representation       Fine            Mid             Coarse

Overall Transform: 4,800 tokens (fine) → 50 tokens (coarse)
                   96× reduction in tokens
                   4× increase in features
                   96× increase in receptive field
```
