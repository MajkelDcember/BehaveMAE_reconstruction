# Detailed Tensor Shape Transformations

This document provides a detailed walkthrough of tensor shape transformations through the hBehaveMAE model with the specific configuration:

```
input_size: (400, 1, 72)
patch_kernel: (2, 1, 3)
q_strides: [(2,1,4), (2,1,6)]
stages: [2, 3, 4]  (9 blocks total)
init_embed_dim: 96
out_embed_dims: [78, 128, 256]
```

## Complete Tensor Shape Evolution

### Input Layer
```
Input Tensor:
  Shape: [B, 400, 72]
  Description: Raw behavioral sequence
  
After adding channel dimension:
  Shape: [B, 1, 400, 1, 72]
  Description: (Batch, Channels, Time, Height, Width)
```

### Patch Embedding Layer
```
Operation: Conv3d(in=1, out=96, kernel=(2,1,3), stride=(2,1,3))

Input:  [B, 1, 400, 1, 72]
Output: [B, 96, 200, 1, 24]
Flatten & Transpose: [B, 4800, 96]

Spatial shape: (200, 1, 24)
Total tokens: 4,800
Feature dim: 96
```

### After Positional Embedding
```
Positional embedding shape: [1, 4800, 96]
Output: [B, 4800, 96]  (position embedding added)
```

### Unroll Operation
```
Purpose: Organize tokens for hierarchical processing
Input:  [B, 4800, 96]
Output: [B, 4800, 96]  (internal organization changed for efficient pooling)

Internal structure after unrolling:
- Tokens organized into mask units
- Mask unit size: (4, 1, 24) at final resolution
- Enables efficient window attention and pooling
```

### Stage 0: Blocks 0-1 (Initial Encoding)

**Block 0:**
```
Input:  [B, 4800, 96]

MaskUnitAttention:
  - q_stride: 1 (no pooling)
  - use_mask_unit_attn: True (local attention)
  - window_size: 4800 (flattened mask unit size)
  - heads: 2
  - dim: 96 → 96

Output: [B, 4800, 96]

MLP:
  - dim: 96 → 384 → 96 (mlp_ratio=4.0)

Final Output: [B, 4800, 96]
```

**Block 1:**
```
Input:  [B, 4800, 96]
Output: [B, 4800, 96]
(Same structure as Block 0)

End of Stage 0:
  Tokens: [B, 4800, 96]
  Token count: 4,800
  Receptive field: 2×1×3 input elements per token
```

### Stage 1: Blocks 2-4 (First Pooling)

**Block 2 (First pooling block):**
```
Input:  [B, 4800, 96]

MaskUnitAttention:
  - q_stride: 8 (flattened from (2,1,4))
  - use_mask_unit_attn: False (global attention)
  - heads: 4 (doubled via head_mul=2.0)
  - dim: 96 → 192 (doubled via dim_mul=2.0)
  
  Q pooling: [B, 4800, 96] → [B, 600, 192]
  (maxpool over every 8 tokens in query)
  
Output: [B, 600, 192]

Residual connection via projection:
  - Input pooled: [B, 4800, 96] → [B, 600, 96] (maxpool)
  - Projected: [B, 600, 96] → [B, 600, 192]

MLP:
  - dim: 192 → 768 → 192

Final Output: [B, 600, 192]
```

**Blocks 3-4:**
```
Input:  [B, 600, 192]
Output: [B, 600, 192]
(No pooling, q_stride=1)

End of Stage 1:
  Tokens: [B, 600, 192]
  Token count: 600
  Spatial shape: (100, 1, 6)
  Receptive field: 4×1×12 input elements per token
```

### Stage 2: Blocks 5-8 (Second Pooling)

**Block 5 (Second pooling block):**
```
Input:  [B, 600, 192]

MaskUnitAttention:
  - q_stride: 12 (flattened from (2,1,6))
  - use_mask_unit_attn: False (global attention)
  - heads: 8 (doubled again)
  - dim: 192 → 384 (doubled again)
  
  Q pooling: [B, 600, 192] → [B, 50, 384]
  
Output: [B, 50, 384]

MLP:
  - dim: 384 → 1536 → 384

Final Output: [B, 50, 384]
```

**Blocks 6-8:**
```
Input:  [B, 50, 384]
Output: [B, 50, 384]
(No pooling, q_stride=1)

End of Stage 2:
  Tokens: [B, 50, 384]
  Token count: 50
  Spatial shape: (50, 1, 1)
  Receptive field: 8×1×72 input elements per token
```

### Multi-Scale Fusion (only for decoding_strategy="multi")

For `decoding_strategy="single"`, skip to encoder norm.

### Encoder Output Processing

**Reroll (if using multi-scale):**
```
Converts from flattened token sequence back to spatial layout
[B, 50, 384] → [B, #MUs, MUy, MUx, 384]
where #MUs × MUy × MUx = 50
```

**Projection:**
```
Linear projection to final encoder dimension:
[B, 50, 384] → [B, 50, 256]  (via out_embed_dims[-1])
```

**Encoder Norm:**
```
LayerNorm: [B, 50, 256]
```

### Decoder

**Decoder Embedding:**
```
Input:  [B, 50, 256]
Linear: [B, 50, 256] → [B, 50, 128]  (decoder_embed_dim)
```

**Mask Token Insertion:**
```
Mask token shape: [1, 1, 128]

For masked positions (e.g., 75% with mask_ratio=0.75):
- Visible tokens: [B, 12.5, 128]  (25% of 50 mask units)
- Add mask tokens: → [B, 50, 128]  (fill in 37.5 masked positions)
```

**Undo Windowing:**
```
Restore spatial layout:
[B, 50, 128] (organized in mask units)
  → [B, 50, 128] (spatial order restored)
```

**Decoder Positional Embedding:**
```
Pos embed shape: [1, 50, 128]
Output: [B, 50, 128]
```

**Decoder Blocks (1 block):**
```
Input:  [B, 50, 128]

MaskUnitAttention:
  - heads: 1 (decoder_num_heads)
  - dim: 128 → 128
  
MLP:
  - dim: 128 → 512 → 128

Output: [B, 50, 128]
```

**Decoder Norm:**
```
LayerNorm: [B, 50, 128]
```

**Prediction Head:**
```
pred_stride = patch_stride × cumulative q_strides
            = (2, 1, 3) × (2×2, 1×1, 4×6)
            = (4, 1, 72)
            = 4 × 1 × 72 × 1 channel = 288

Wait, let me recalculate based on code:
pred_stride = patch_stride[-1] * patch_stride[-2] * prod(overall_q_strides)
            = 3 × 1 × (2*2 × 1*1 × 4*6)
            = 3 × 1 × 96 = 288

Linear projection:
[B, 50, 128] → [B, 50, 288]

Each of 50 tokens predicts 288 values:
- These 288 values represent the reconstruction of the corresponding 
  spatiotemporal patch in the original input
```

### Loss Computation

**Target Preparation:**
```
Original input: [B, 1, 400, 1, 72]

Strided sampling for loss:
- Sample every patch_stride[0] frames: [B, 1, 400//2, 1, 72] = [B, 1, 200, 1, 72]

Partition into blocks:
- Temporal blocks: 200 → 50 (divide by cumulative temporal stride)
- Spatial blocks: (1, 72) → (1, 1) (divide by cumulative spatial strides)
- Block size: (4, 1, 72)

Target shape: [B, 50, 288]
```

**Loss Calculation:**
```
Predictions: [B, 50, 288]
Targets:     [B, 50, 288]

Mask: [B, 50] boolean (True for masked, False for visible)

Masked predictions:  [B, ~37, 288]  (75% masked)
Masked targets:      [B, ~37, 288]

MSE Loss: mean((predictions - targets) ** 2)
```

## Summary Table

| Stage | Block Range | Input Shape | Output Shape | q_stride | Token Count | Receptive Field |
|-------|-------------|-------------|--------------|----------|-------------|-----------------|
| Input | - | [B, 400, 72] | [B, 1, 400, 1, 72] | - | - | - |
| Patch Embed | - | [B, 1, 400, 1, 72] | [B, 4800, 96] | - | 4,800 | 2×1×3 |
| Stage 0 | 0-1 | [B, 4800, 96] | [B, 4800, 96] | 1 | 4,800 | 2×1×3 |
| Stage 1 | 2-4 | [B, 4800, 96] | [B, 600, 192] | 8 | 600 | 4×1×12 |
| Stage 2 | 5-8 | [B, 600, 192] | [B, 50, 384] | 12 | 50 | 8×1×72 |
| Encoder Proj | - | [B, 50, 384] | [B, 50, 256] | - | 50 | 8×1×72 |
| Decoder | - | [B, 50, 256] | [B, 50, 288] | - | 50 | - |

## Key Observations

1. **Token Count Reduction**: 4,800 → 600 → 50 (96× reduction overall)
2. **Feature Dimension Growth**: 96 → 192 → 384 (4× increase)
3. **Receptive Field Expansion**: 2×1×3 → 8×1×72 (96× expansion)
4. **Efficient Attention**: Token reduction makes later stages computationally efficient
5. **Hierarchical Learning**: Early stages: local patterns, Late stages: global context
