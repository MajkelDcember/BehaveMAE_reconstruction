# Quick Reference: hBehaveMAE Data Propagation

## One-Page Summary for Config: `--input_size 400 1 72 --q_strides 2,1,4;2,1,6 --stages 2 3 4`

### The Journey of Data Through hBehaveMAE

```
INPUT [B, 400, 72] → Behavioral sequence
    ↓ Patch Embedding (Conv3d)
    
TOKENS [B, 4800, 96] → 4,800 tokens, each covering 2×1×3 input elements
    ↓ Stage 0: Blocks 0-1 (Local Attention)
    
TOKENS [B, 4800, 96] → Fine-grained features, local patterns
    ↓ Q-POOLING: stride=(2,1,4) → Reduce by 8×
    
TOKENS [B, 600, 192] → 600 tokens, each covering 4×1×12 input elements
    ↓ Stage 1: Blocks 2-4 (Global Attention)
    
TOKENS [B, 600, 192] → Mid-level features, action sequences
    ↓ Q-POOLING: stride=(2,1,6) → Reduce by 12×
    
TOKENS [B, 50, 384] → 50 tokens, each covering 8×1×72 input elements
    ↓ Stage 2: Blocks 5-8 (Global Attention)
    
ENCODER OUTPUT [B, 50, 256] → High-level features, complex activities
    ↓ Decoder (with mask tokens)
    
PREDICTIONS [B, 50, 288] → Reconstruct masked regions
    ↓ Loss on masked tokens only
    
TRAINED MODEL → Learns hierarchical behavioral representations
```

### Key Concepts Explained

#### 🎯 What is a Token?
A **token** is a learned vector representation of a small spatiotemporal patch from your input data. 
- Initially: 4,800 tokens from patch embedding
- Each token starts covering 2 frames × 3 width units
- Tokens become more abstract as they flow through stages

#### 🔄 What is Q-Stride?
**Q-stride** controls token pooling in the attention mechanism:
- **Mechanism**: Max-pools queries (Q) while keeping keys (K) and values (V) at original resolution
- **Effect**: Reduces number of tokens, increases receptive field
- **Example**: q_stride=(2,1,4) reduces 4,800 tokens → 600 tokens (8× reduction)

#### 📡 Receptive Field Growth
As tokens are pooled, their receptive field grows:
```
Stage 0: 2×1×3    = 6 input elements per token     (fine-grained)
Stage 1: 4×1×12   = 48 input elements per token    (mid-level)
Stage 2: 8×1×72   = 576 input elements per token   (high-level)
```
Final tokens see 96× more input area than initial tokens!

### The Magic of Hierarchical Learning

```
┌──────────────────────────────────────────────────────────────┐
│  Lower Stages (Stage 0)                                      │
│  • Many tokens (4,800)                                       │
│  • Small receptive fields (2×1×3)                            │
│  • Local attention                                           │
│  → Learns fine-grained patterns (individual movements)       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Middle Stages (Stage 1)                                     │
│  • Fewer tokens (600)                                        │
│  • Medium receptive fields (4×1×12)                          │
│  • Global attention                                          │
│  → Learns mid-level patterns (limb movements, short actions) │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Higher Stages (Stage 2)                                     │
│  • Very few tokens (50)                                      │
│  • Large receptive fields (8×1×72)                           │
│  • Global attention                                          │
│  → Learns high-level patterns (complex actions, activities)  │
└──────────────────────────────────────────────────────────────┘
```

### Why This Architecture Works

1. **Computational Efficiency**: Fewer tokens in later stages = faster attention
2. **Multi-scale Learning**: Different stages capture different temporal/spatial scales
3. **Hierarchical Abstraction**: Natural progression from details to concepts
4. **Information Bottleneck**: Forces model to learn meaningful representations

### Common Questions

**Q: Why reduce tokens so aggressively (4,800 → 50)?**
A: This forces the model to compress information efficiently and learn hierarchical abstractions. The 50 final tokens must encode everything needed to reconstruct the input!

**Q: How does pooling differ from downsampling?**
A: Q-pooling is special: it only pools the queries in attention, not the keys/values. This allows attending to fine-grained information while producing coarser outputs.

**Q: What's the mask unit size (4, 1, 24)?**
A: Mask units are groups of tokens that are masked together. Size (4,1,24) means 4 temporal × 1 height × 24 width tokens = 96 tokens per mask unit. This prevents information leakage.

**Q: Why use different attention types (local vs global)?**
A: Stage 0 uses local attention for efficiency with many tokens. Later stages use global attention because there are fewer tokens (600, then 50), making global attention feasible.

### Performance Impact

```
Configuration Trade-offs:

Larger q_strides:
  ✓ Faster computation (fewer tokens)
  ✓ Larger receptive fields
  ✗ Less fine-grained control
  ✗ May lose local details

Smaller q_strides:
  ✓ More fine-grained features
  ✓ Better detail preservation
  ✗ More tokens = slower
  ✗ More memory usage

Your config (2,1,4;2,1,6) strikes a balance:
  • Good reduction (96×) for efficiency
  • Maintains spatial width information (aggressive width pooling only at end)
  • Gradual temporal pooling (2× at each stage)
```

### See Also

- 📘 [Full Data Propagation Guide](data_propagation_guide.md) - Comprehensive explanation
- 🎨 [Visual Diagrams](visual_data_flow.md) - ASCII art visualizations
- 📊 [Tensor Shapes](tensor_shapes_detailed.md) - Detailed shape transformations
- 🔧 [Demo Script](demo_data_propagation.py) - Calculate for any config
- 🏠 [Documentation Index](README.md) - All documentation

### Quick Command to Run Demo

```bash
cd docs
python demo_data_propagation.py
```

This will show you the exact data flow with your configuration!
