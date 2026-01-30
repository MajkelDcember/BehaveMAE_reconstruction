#!/usr/bin/env python3
"""
Simple debug script to run hBehaveMAE training without DataJoint.
Uses static functions from hbehavemae_nodj.py.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from argparse import Namespace

from hbehavemae_nodj import HBehaveMAEModelSettings, HBehaveMAEModelTraining


def main():
    """
    Run a simple training process for hBehaveMAE.
    """
    
    # Define paths
    project_data_path = Path("/scratch/michal/projects/dvc_ofd_2025/data/interim/hbmae_training_data")
    train_data_path = project_data_path / "hbmaeproject-0_shuffle-1/hbmaeproj-0_shuffle-1_train.npy"
    test_data_path = project_data_path / "hbmaeproject-0_shuffle-1/hbmaeproj-0_shuffle-1_test.npy"
    
    # Create output directories
    output_base = Path(f"/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/outputs/hbmae_debug_{datetime.now()}")
    output_dir = output_base / "outputs"
    log_dir = output_base / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("hBehaveMAE Training Debug Script")
    print("=" * 80)
    print(f"Train data: {train_data_path}")
    print(f"Test data: {test_data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Log dir: {log_dir}")
    print()
    
    # Get default settings (optimized for OFD mice experiments)
    print("\nLoading default model settings...")
    settings_dict = HBehaveMAEModelSettings.get_default_settings()
    
    # Optionally modify settings for quick debug run
    settings_dict['epochs'] = 5  # Quick test with just 5 epochs
    settings_dict['batch_size'] = 32  # Smaller batch for debugging
    settings_dict['checkpoint_period'] = 2  # Save checkpoints more frequently
    settings_dict['num_workers'] = 1  # Fewer workers for debugging
    settings_dict['blr'] = 1e-5  # Much lower learning rate for stability
    settings_dict['warmup_epochs'] = 2  # Shorter warmup for debug
    settings_dict['clip_grad'] = 0.01  # More aggressive gradient clipping
    
    print(f"\nModel settings:")
    print(f"  Epochs: {settings_dict['epochs']}")
    print(f"  Batch size: {settings_dict['batch_size']}")
    print(f"  Input size: {settings_dict['input_size']}")
    print(f"  Mask ratio: {settings_dict['mask_ratio']}")
    print(f"  Learning rate (blr): {settings_dict['blr']}")
    print()
    
    # Build training arguments
    print("Building training arguments...")
    trainer = HBehaveMAEModelTraining()
    args = trainer._build_training_args(
        settings_dict=settings_dict,
        train_data_path=str(train_data_path),
        test_data_path=str(test_data_path),
        output_dir=str(output_dir),
        log_dir=str(log_dir)
    )
    
    print("Training arguments prepared.")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device: {args.device}")
    print(f"  Num frames: {args.num_frames}")
    print()
    
    # Run training
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    try:
        final_checkpoint = trainer._run_training(args, output_base)
        
        if final_checkpoint:
            print("\n" + "=" * 80)
            print("Training completed successfully!")
            print("=" * 80)
            print(f"Final checkpoint: {final_checkpoint}")
            print(f"Output directory: {output_dir}")
            print(f"Log directory: {log_dir}")
            return 0
        else:
            print("\n" + "=" * 80)
            print("Training failed - no checkpoint produced")
            print("=" * 80)
            return 1
            
    except Exception as e:
        print("\n" + "=" * 80)
        print("Training failed with error:")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
