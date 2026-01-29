# Documentation Index

This directory contains detailed documentation about the hBehaveMAE model architecture and data flow.

## Available Documentation

### 📘 [Data Propagation Guide](data_propagation_guide.md)
**Start here!** Comprehensive explanation of how data propagates through the hBehaveMAE model, including:
- What tokens are and how they're created
- How q_stride affects pooling and receptive fields
- Complete data flow from input to output
- Key insights about hierarchical token processing

### 🎨 [Visual Data Flow Diagram](visual_data_flow.md)
ASCII art diagrams and visualizations showing:
- High-level architecture overview
- Token evolution through stages
- How Q-stride pooling works
- Receptive field growth
- Attention patterns
- Masking strategy
- Decoder reconstruction process

### 📊 [Detailed Tensor Shapes](tensor_shapes_detailed.md)
Block-by-block tensor shape transformations including:
- Complete tensor shape evolution from input to output
- Shape at each encoder block
- Decoder processing steps
- Summary table of all transformations

### 🔧 [Demo Script](demo_data_propagation.py)
Executable Python script that:
- Calculates and displays data flow for any configuration
- Shows token counts, dimensions, and receptive fields at each stage
- Can be customized for different model configurations
- Run with: `python docs/demo_data_propagation.py`

## Quick Reference

### Configuration Used in Examples

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
--decoder_embed_dim 128 \
--decoder_depth 1 \
--decoder_num_heads 1
```

### Key Numbers for This Configuration

| Metric | Value |
|--------|-------|
| Initial tokens | 4,800 |
| Final tokens | 50 |
| Token reduction | 96× |
| Initial feature dim | 96 |
| Final feature dim | 384 |
| Feature expansion | 4× |
| Initial receptive field | 2×1×3 |
| Final receptive field | 8×1×72 |
| Receptive field growth | 96× |

## Understanding the Architecture

The hBehaveMAE model is a hierarchical masked autoencoder designed for behavioral data. Key concepts:

1. **Tokens**: Learned representations of spatiotemporal patches
2. **Stages**: Model has 3 stages with progressively fewer tokens
3. **Q-stride**: Controls pooling between stages, reducing token count
4. **Receptive Field**: Grows as tokens are pooled, capturing larger context
5. **Hierarchical Learning**: Early stages learn fine details, later stages learn global patterns

## How to Use This Documentation

1. **New to the model?** Start with [Data Propagation Guide](data_propagation_guide.md)
2. **Want visual intuition?** Check [Visual Data Flow Diagram](visual_data_flow.md)
3. **Need exact shapes?** See [Detailed Tensor Shapes](tensor_shapes_detailed.md)
4. **Working with different configs?** Run [Demo Script](demo_data_propagation.py)

## Questions?

If you have questions or find issues with the documentation, please open an issue in the repository.
