#!/usr/bin/env python3
"""
Demonstration script showing data propagation through hBehaveMAE model.

This script calculates and displays the tensor shapes at each stage of the model
given specific configuration parameters.

Usage:
    python docs/demo_data_propagation.py
"""

import math
from functools import reduce
from operator import mul


def calculate_data_flow(
    input_size=(400, 1, 72),
    patch_kernel=(2, 1, 3),
    patch_stride=(2, 1, 3),
    q_strides=[(2, 1, 4), (2, 1, 6)],
    stages=[2, 3, 4],
    init_embed_dim=96,
    init_num_heads=2,
    dim_mul=2.0,
    head_mul=2.0,
    out_embed_dims=[78, 128, 256],
    decoder_embed_dim=128,
):
    """
    Calculate and print the data flow through hBehaveMAE model.
    
    Args:
        input_size: Tuple of (T, H, W) for input dimensions
        patch_kernel: Kernel size for initial patch embedding
        patch_stride: Stride for initial patch embedding
        q_strides: List of strides for each pooling stage (T, H, W) tuples
        stages: Number of blocks in each stage
        init_embed_dim: Initial embedding dimension
        init_num_heads: Initial number of attention heads
        dim_mul: Dimension multiplier between stages
        head_mul: Head multiplier between stages
        out_embed_dims: Output embedding dimensions for each stage
        decoder_embed_dim: Decoder embedding dimension
    """
    
    print("="*80)
    print("hBehaveMAE Data Propagation Analysis")
    print("="*80)
    print()
    
    # Input
    print(f"INPUT:")
    print(f"  Original shape: [B, {input_size[0]}, {input_size[2]}]")
    print(f"  With channels: [B, 1, {input_size[0]}, {input_size[1]}, {input_size[2]}]")
    print()
    
    # Patch Embedding
    tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)]
    num_tokens = math.prod(tokens_spatial_shape)
    
    print(f"PATCH EMBEDDING:")
    print(f"  Kernel: {patch_kernel}, Stride: {patch_stride}")
    print(f"  Token spatial shape: {tokens_spatial_shape}")
    print(f"  Total tokens: {num_tokens:,}")
    print(f"  Feature dimension: {init_embed_dim}")
    print(f"  Output shape: [B, {num_tokens}, {init_embed_dim}]")
    print(f"  Receptive field per token: {patch_kernel[0]}×{patch_kernel[1]}×{patch_kernel[2]}")
    print()
    
    # Calculate cumulative strides
    cumulative_strides = []
    for i in range(len(q_strides)):
        stride_so_far = tuple(
            reduce(mul, [q_strides[j][k] for j in range(i + 1)])
            for k in range(len(q_strides[0]))
        )
        cumulative_strides.append(stride_so_far)
    
    # Track evolution through stages
    current_tokens = num_tokens
    current_shape = tokens_spatial_shape.copy()
    current_dim = init_embed_dim
    current_heads = init_num_heads
    current_receptive_field = list(patch_kernel)
    
    stage_idx = 0
    block_idx = 0
    q_pool_idx = 0
    stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
    q_pool_blocks = [x + 1 for x in stage_ends[:len(q_strides)]]
    
    print(f"ENCODER STAGES:")
    print()
    
    for stage_num, num_blocks in enumerate(stages):
        print(f"Stage {stage_num}: Blocks {block_idx} to {block_idx + num_blocks - 1}")
        print(f"  Number of blocks: {num_blocks}")
        
        # Check if pooling happens at start of this stage
        pools_at_start = block_idx in q_pool_blocks
        
        for local_block in range(num_blocks):
            # Dimension change happens after stage boundary (except first stage)
            if local_block == 0 and stage_num > 0:
                current_dim = int(current_dim * dim_mul)
                current_heads = int(current_heads * head_mul)
            
            # Pooling at first block of stage (if applicable)
            if local_block == 0 and pools_at_start:
                flat_q_stride = math.prod(q_strides[q_pool_idx])
                current_tokens = current_tokens // flat_q_stride
                current_shape = [
                    s // q_strides[q_pool_idx][i]
                    for i, s in enumerate(current_shape)
                ]
                current_receptive_field = [
                    r * q_strides[q_pool_idx][i]
                    for i, r in enumerate(current_receptive_field)
                ]
                print(f"    Block {block_idx}: POOLING with q_stride={q_strides[q_pool_idx]} (flat={flat_q_stride})")
                print(f"      Tokens: {current_tokens:,}")
                print(f"      Shape: {current_shape}")
                print(f"      Dim: {current_dim}")
                print(f"      Heads: {current_heads}")
                print(f"      Receptive field: {current_receptive_field[0]}×{current_receptive_field[1]}×{current_receptive_field[2]}")
                q_pool_idx += 1
            
            block_idx += 1
        
        # Project to output dimension
        print(f"  Stage output projection: {current_dim} → {out_embed_dims[stage_num]}")
        print(f"  Final output: [B, {current_tokens}, {out_embed_dims[stage_num]}]")
        print()
    
    # Encoder output
    encoder_dim = out_embed_dims[-1]
    print(f"ENCODER OUTPUT:")
    print(f"  Shape: [B, {current_tokens}, {encoder_dim}]")
    print(f"  Each token covers: {current_receptive_field[0]}×{current_receptive_field[1]}×{current_receptive_field[2]} input elements")
    print()
    
    # Decoder
    overall_q_strides = list(
        map(lambda elements: reduce(mul, elements), zip(*q_strides))
    )
    pred_stride = patch_stride[-1] * patch_stride[-2] * math.prod(overall_q_strides)
    
    print(f"DECODER:")
    print(f"  Embed: {encoder_dim} → {decoder_embed_dim}")
    print(f"  Tokens with mask tokens: [B, {current_tokens}, {decoder_embed_dim}]")
    print(f"  Prediction stride: {pred_stride}")
    print(f"  Prediction output: [B, {current_tokens}, {pred_stride}]")
    print()
    
    # Summary
    print(f"SUMMARY:")
    print(f"  Token reduction: {num_tokens:,} → {current_tokens} ({num_tokens/current_tokens:.1f}× reduction)")
    print(f"  Feature expansion: {init_embed_dim} → {current_dim} ({current_dim/init_embed_dim:.1f}× increase)")
    print(f"  Receptive field expansion: {patch_kernel} → {current_receptive_field}")
    rf_expansion = math.prod(current_receptive_field) / math.prod(patch_kernel)
    print(f"  Receptive field growth: {rf_expansion:.1f}×")
    print()
    
    # Mask unit information
    mask_unit_size = tuple(math.prod([q_strides[j][i] for j in range(len(q_strides))]) for i in range(3))
    print(f"MASKING:")
    print(f"  Mask unit size (cumulative q_strides): {mask_unit_size}")
    print(f"  Mask spatial shape: {[t // m for t, m in zip(tokens_spatial_shape, mask_unit_size)]}")
    print()


def main():
    """Run the demonstration with default configuration."""
    
    print("\n")
    print("Configuration 1: Given configuration from issue")
    print("-" * 80)
    
    calculate_data_flow(
        input_size=(400, 1, 72),
        patch_kernel=(2, 1, 3),
        patch_stride=(2, 1, 3),
        q_strides=[(2, 1, 4), (2, 1, 6)],
        stages=[2, 3, 4],
        init_embed_dim=96,
        init_num_heads=2,
        dim_mul=2.0,
        head_mul=2.0,
        out_embed_dims=[78, 128, 256],
        decoder_embed_dim=128,
    )
    
    print("\n\n")
    print("Configuration 2: Example with different parameters")
    print("-" * 80)
    
    # Another example configuration
    calculate_data_flow(
        input_size=(600, 3, 24),
        patch_kernel=(4, 1, 2),
        patch_stride=(4, 1, 2),
        q_strides=[(1, 1, 3), (1, 1, 4), (1, 3, 1)],
        stages=[2, 3, 4],
        init_embed_dim=48,
        init_num_heads=2,
        dim_mul=2.0,
        head_mul=2.0,
        out_embed_dims=[32, 64, 96],
        decoder_embed_dim=128,
    )


if __name__ == "__main__":
    main()
