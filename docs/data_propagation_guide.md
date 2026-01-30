# Data Propagation Through hBehaveMAE Model

This document provides a comprehensive explanation of how data propagates through the hBehaveMAE (Hierarchical Behavior Masked Autoencoder) model, with specific focus on the configuration parameters provided.

## Additional Resources

- **[Visual Data Flow Diagram](visual_data_flow.md)**: ASCII art diagrams showing data flow and transformations
- **[Detailed Tensor Shapes](tensor_shapes_detailed.md)**: Block-by-block tensor shape transformations
- **[Demo Script](demo_data_propagation.py)**: Executable Python script that calculates and displays data flow for any configuration

## Configuration Overview

The following configuration is analyzed in this document:

```bash
--model hbehavemae \
--input_size 400 1 72 \
--stages 2 3 4 \
--q_strides 2,1,4;2,1,6 \
--mask_unit_attn True False False \
--patch_kernel 2 1 3 \
--init_embed_dim 96 \
--init_num_heads 2 \
--out_embed_dims 78 128 256 \
--epochs 200 \
--num_frames 400 \
--decoding_strategy single \
--decoder_embed_dim 128 \
--decoder_depth 1 \
--decoder_num_heads 1
```

## What is a Token?

In the hBehaveMAE model, a **token** is a learned representation of a small spatiotemporal patch of the input data. Here's how tokens are created:

### Token Creation Process

1. **Input Shape**: The model receives input of shape `[B, C, T, H, W]` where:
   - `B` = batch size
   - `C` = channels (1 for the given config)
   - `T` = temporal dimension (400 frames in config)
   - `H` = height dimension (1 in config)
   - `W` = width dimension (72 in config)

2. **Patch Embedding**: The `PatchEmbed` layer applies a 3D convolution with:
   - Kernel size: `patch_kernel = (2, 1, 3)` → divides input into patches
   - Stride: `patch_stride = (2, 1, 3)` → determines patch sampling rate
   - This creates initial tokens from overlapping or non-overlapping patches

3. **Initial Token Spatial Shape**: After patch embedding:
   ```python
   tokens_spatial_shape = [input_size[i] // patch_stride[i] for i in range(3)]
   # = [400//2, 1//1, 72//3] = [200, 1, 24]
   ```
   
   So initially, we have **200 × 1 × 24 = 4,800 tokens**, each with dimension `init_embed_dim = 96`.

### Token Representation

Each token represents:
- **Temporal extent**: 2 frames (from patch_kernel[0]=2, patch_stride[0]=2)
- **Height extent**: 1 unit (from patch_kernel[1]=1, patch_stride[1]=1)
- **Width extent**: 3 units (from patch_kernel[2]=3, patch_stride[2]=3)

A token is essentially a vector that encodes information from this local spatiotemporal region of the input.

## Understanding q_stride and Its Effects

The `q_stride` (query stride) is a crucial parameter in the hierarchical architecture that controls **pooling** operations within the attention mechanism. It directly affects:

1. **Token reduction** (downsampling)
2. **Receptive field expansion**
3. **Hierarchical feature learning**

### q_stride Configuration

From the config: `q_strides = [(2,1,4), (2,1,6)]`

This means:
- **First pooling stage**: stride of `(2, 1, 4)` in (T, H, W) dimensions
- **Second pooling stage**: stride of `(2, 1, 6)` in (T, H, W) dimensions

### How q_stride Affects Pooling

The `q_stride` is applied in the `MaskUnitAttention` module during the attention computation:

```python
# From MaskUnitAttention.forward()
if self.q_stride > 1:
    # Performs a maxpool operation on queries
    q = q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
         .max(dim=3)
         .values
```

**What happens:**
- The queries (Q) in the attention mechanism are pooled by taking the maximum over every `q_stride` tokens
- Keys (K) and values (V) remain at the original resolution
- This effectively reduces the number of output tokens by a factor of `q_stride`

**Flattened q_strides:**
```python
flat_q_stride_1 = 2 × 1 × 4 = 8
flat_q_stride_2 = 2 × 1 × 6 = 12
```

### Receptive Field Evolution

The receptive field of a token grows as it passes through stages with q_stride > 1:

#### Stage 0 (Blocks 0-1): Initial Encoding
- **q_stride**: 1 (no pooling)
- **Token shape**: [200, 1, 24]
- **Total tokens**: 4,800
- **Receptive field per token**: 2 frames × 1 height × 3 width (from initial patch)

#### Stage 1 (Blocks 2-4): First Pooling Stage
- **q_stride**: (2, 1, 4) → flattened = 8
- **Token shape**: [200/2, 1/1, 24/4] = [100, 1, 6]
- **Total tokens**: 600
- **Receptive field per token**: 
  - Temporal: 2 × 2 = 4 frames (initial 2 frames × stride 2)
  - Height: 1 × 1 = 1 unit
  - Width: 3 × 4 = 12 units (initial 3 units × stride 4)

#### Stage 2 (Blocks 5-8): Second Pooling Stage
- **q_stride**: (2, 1, 6) → flattened = 12
- **Token shape**: [100/2, 1/1, 6/6] = [50, 1, 1]
- **Total tokens**: 50
- **Receptive field per token**:
  - Temporal: 4 × 2 = 8 frames (previous 4 frames × stride 2)
  - Height: 1 × 1 = 1 unit
  - Width: 12 × 6 = 72 units (previous 12 units × stride 6)

### Mask Units and Attention Windows

The `mask_unit_attn` parameter controls whether to use **local** (mask unit) attention or **global** attention:

- `mask_unit_attn = [True, False, False]` means:
  - **Stage 0**: Uses Mask Unit Attention (local attention within windows)
  - **Stage 1**: Uses Global Attention (attention across all tokens)
  - **Stage 2**: Uses Global Attention

**Mask unit size** is computed as the cumulative product of q_strides:
```python
mask_unit_size = (2×2, 1×1, 4×6) = (4, 1, 24)
```

This means tokens are organized into windows of size 4×1×24 for masked attention in Stage 0.

## Complete Data Flow Through the Model

### 1. Input Processing

```
Input: [B, 400, 72]  (behavior sequences)
  ↓
Add channel dimension: [B, 1, 400, 1, 72]
```

### 2. Patch Embedding

```
PatchEmbed (Conv3d with kernel=(2,1,3), stride=(2,1,3))
  ↓
Tokens: [B, 4800, 96]  (4800 = 200×1×24 tokens, each with 96 dims)
  ↓
Add positional embedding
```

### 3. Unrolling (Windowing)

```
Unroll operation organizes tokens for efficient hierarchical processing
  ↓
Tokens organized into mask units for efficient attention
```

### 4. Encoder Stages

#### Stage 0: Fine-grained Features (Blocks 0-1)
- **Attention**: Mask Unit Attention (local windows)
- **q_stride**: 1 (no pooling)
- **Dimensions**: 96 → 96
- **Tokens**: [B, 4800, 96]
- **Learns**: Fine-grained, local motion patterns

#### Stage 1: Mid-level Features (Blocks 2-4)
- **Attention**: Global Attention
- **q_stride**: (2, 1, 4) pooling applied at block 2
- **Dimensions**: 96 → 192 (dim_mul=2.0)
- **Tokens**: [B, 600, 192] after pooling
- **Learns**: Intermediate motion patterns, short action sequences

#### Stage 2: High-level Features (Blocks 5-8)
- **Attention**: Global Attention
- **q_stride**: (2, 1, 6) pooling applied at block 5
- **Dimensions**: 192 → 384
- **Tokens**: [B, 50, 384] after pooling
- **Learns**: Complex actions, long-term dependencies

### 5. Multi-scale Fusion (if decoding_strategy="multi")

For the given config with `decoding_strategy="single"`, only the final stage output is used.

```
Stage 2 output: [B, #mask_units, features, 256]  (projected to out_embed_dims[-1])
  ↓
Encoder norm
```

### 6. Decoder

```
Decoder embed: 256 → 128
  ↓
Combine with mask tokens for masked positions
  ↓
Undo windowing to restore spatial structure: [B, 50, 128]
  ↓
Add decoder positional embedding
  ↓
Decoder block (1 layer, 1 head)
  ↓
Decoder norm
  ↓
Prediction head: 128 → pred_stride × channels
```

Where `pred_stride` is computed as:
```python
pred_stride = patch_stride[0] × patch_stride[1] × product(q_strides)
            = 2 × 1 × (2×2) × (4×6)
            = 2 × 4 × 24 = 192
```

### 7. Loss Computation

The model predicts reconstructions for masked tokens at the final resolution:
- Predictions: [B, 50, 192]
- Each prediction reconstructs a patch of 192 consecutive frames (due to striding)
- Loss is computed only on masked tokens using MSE

## Key Insights

### 1. Hierarchical Token Pooling

The q_stride mechanism implements a form of learned hierarchical pooling:
- Lower stages maintain high resolution with many tokens (fine-grained)
- Higher stages aggressively pool to fewer tokens (coarse-grained)
- Each stage's receptive field grows multiplicatively

### 2. Token Receptive Field Growth

The receptive field of tokens grows exponentially through the stages:
- **Stage 0**: 2 × 1 × 3 = 6 input elements per token
- **Stage 1**: 4 × 1 × 12 = 48 input elements per token (8× growth)
- **Stage 2**: 8 × 1 × 72 = 576 input elements per token (96× growth from start!)

This means the final 50 tokens in Stage 2 each have a receptive field covering most or all of the spatial width and a significant temporal extent.

### 3. Efficient Computation via Unroll/Reroll

The `Unroll` and `Reroll` operations:
- Organize tokens into contiguous memory blocks
- Enable efficient max pooling as simple tensor operations
- Allow flexible support for arbitrary dimensional inputs
- Support mask unit attention by grouping tokens into windows

### 4. Masking Strategy

The model uses mask units for masking:
- Mask unit size: (4, 1, 24) in final resolution
- Entire mask units are either kept or masked together
- This prevents information leakage and makes the task more challenging
- With mask_ratio=0.75, only 25% of mask units are kept during training

## Summary

The hBehaveMAE model processes behavioral sequences through a hierarchical encoder where:

1. **Tokens** are spatiotemporal patch embeddings that represent local regions of the input
2. **q_stride** controls token pooling, reducing token count while expanding receptive fields
3. Data flows from fine-grained (many small-receptive-field tokens) to coarse-grained (few large-receptive-field tokens)
4. The hierarchical structure enables learning representations at multiple temporal and spatial scales
5. The decoder reconstructs masked regions using the hierarchical representations

This architecture is particularly suited for behavioral data where patterns exist at multiple scales (e.g., individual keypoint movements → limb movements → full-body actions → activity sequences).
