import os
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import torch
import subprocess
import sys
from typing import Dict, Any

class HBehaveMAEModelSettings():
    """
    hBehaveMAE model hyperparameter settings.
    Stores all configuration parameters as a serialized dict.
    """
    definition = """
    hbmae_settings_id: INT
    ---
    hbmae_settings: LONGBLOB  # Complete serialized settings dict
    hbmae_settings_description = NULL: VARCHAR(512)
    """

    @classmethod
    def add_settings(cls, settings_id, settings_dict, description=None):
        """
        Add model settings.
        
        Args:
            settings_id: Unique settings ID
            settings_dict: Dictionary containing all training settings
            description: Optional description
        """
        entry = {
            'hbmae_settings_id': settings_id,
            'hbmae_settings': settings_dict,
            'hbmae_settings_description': description
        }
        
        cls.insert1(entry)
        print(f"Added hBehaveMAE settings with ID {settings_id}")
        return entry
    
    @staticmethod
    def get_default_settings():
        """
        Get default settings for OFD mice experiments.
        Based on mabe22_mice training script, adapted for single-mouse OFD data.
        
        OFD mice: 1 individual, 8 keypoints
        After centeralign: 2 (center) + 2 (rotation) + 8*2 (keypoints) = 20 features
        """
        return {
            'batch_size': 768,
            'model_name': 'hbehavemae',
            'input_size': (900, 1, 58),  # OFD: 1 individual, 20 features after centeralign
            'stages': (3, 4, 5),
            'q_strides': [(5, 1, 1), (1, 1, 1)],  # Don't stride in individuals dim (it's only 1)
            'mask_unit_attn': [True, False, False],
            'patch_kernel': (3, 1, 58),  # Match last dimension to input_size
            'init_embed_dim': 128,
            'init_num_heads': 2,
            'out_embed_dims': (128, 192, 256),
            'epochs': 200,
            'num_frames': 900,
            'decoding_strategy': 'multi',
            'decoder_embed_dim': 128,
            'decoder_depth': 1,
            'decoder_num_heads': 1,
            'pin_mem': True,
            'num_workers': 8,
            'sliding_window': 17,
            'blr': 1.6e-4,
            'warmup_epochs': 40,
            'masking_strategy': 'random',
            'mask_ratio': 0.875,
            'clip_grad': 0.02,
            'checkpoint_period': 20,
            'fill_holes': True,
            'data_augment': False,
            'norm_loss': False,
            'seed': 0,
            'non_hierarchical': False,
            'weight_decay': 0.05,
            'accum_iter': 1,
            'sampling_rate': 1,
            'centeralign': True,
            'include_test_data': False,
            'num_checkpoint_del': 3,
        }




# @schema
class HBehaveMAEModelTraining():
    """
    Trains the hBehaveMAE model with automatic resumption from checkpoints.
    Training progress is tracked via HBehaveMAETrainingProgress (insert-only).
    """
    def _build_training_args(self, settings_dict, train_data_path, test_data_path, output_dir, log_dir):
        """
        Build arguments for hBehaveMAE training.
        Returns a Namespace object compatible with main_pretrain.py
        """
        from argparse import Namespace
        
        # Get input_size - stored as tuple
        input_size = settings_dict['input_size']
        
        args = Namespace(
            # Dataset parameters
            dataset='OFD_mouse',
            path_to_data_dir=train_data_path,
            test_data_path=test_data_path,

            # Model architecture
            model=settings_dict['model_name'],
            non_hierarchical=settings_dict['non_hierarchical'],
            input_size=input_size,
            stages=settings_dict['stages'],
            q_strides=settings_dict['q_strides'],
            mask_unit_attn=settings_dict['mask_unit_attn'],
            patch_kernel=settings_dict['patch_kernel'],
            init_embed_dim=settings_dict['init_embed_dim'],
            init_num_heads=settings_dict['init_num_heads'],
            out_embed_dims=settings_dict['out_embed_dims'],
            
            # Decoder
            decoder_embed_dim=settings_dict['decoder_embed_dim'],
            decoder_depth=settings_dict['decoder_depth'],
            decoder_num_heads=settings_dict['decoder_num_heads'],
            decoding_strategy=settings_dict['decoding_strategy'],
            
            # Training parameters
            batch_size=settings_dict['batch_size'],
            epochs=settings_dict['epochs'],
            num_frames=settings_dict['num_frames'],
            sliding_window=settings_dict['sliding_window'],
            
            # Masking
            mask_ratio=settings_dict['mask_ratio'],
            masking_strategy=settings_dict['masking_strategy'],
            
            # Optimizer
            lr=None,  # Will be computed from blr
            blr=settings_dict['blr'],
            min_lr=0.0,
            warmup_epochs=settings_dict['warmup_epochs'],
            weight_decay=settings_dict['weight_decay'],
            clip_grad=settings_dict['clip_grad'],
            
            # Data augmentation
            fill_holes=settings_dict.get('fill_holes', True),
            data_augment=settings_dict.get('data_augment', True),
            centeralign=settings_dict.get('centeralign', False),
            include_test_data=settings_dict.get('include_test_data', False),
            norm_loss=settings_dict.get('norm_loss', True),
            
            # Other
            checkpoint_period=settings_dict.get('checkpoint_period', 20),
            accum_iter=settings_dict.get('accum_iter', 1),
            num_workers=settings_dict.get('num_workers', 8),
            sampling_rate=settings_dict.get('sampling_rate', 1),
            
            # Paths
            output_dir=output_dir,
            log_dir=log_dir,
            
            # System
            device='cuda',
            seed=settings_dict.get('seed', 0),
            resume='',  # Will be set if checkpoint exists
            start_epoch=0,
            num_checkpoint_del=settings_dict.get('num_checkpoint_del', 3),
            pin_mem=settings_dict.get('pin_mem', True),
            distributed=False,
            world_size=1,
            local_rank=-1,
            dist_on_itp=False,
            dist_url='env://',
            no_env=False,
            
            # Model specific flags
            no_qkv_bias=False,
            bias_wd=False,
            sep_pos_embed=True,
            trunc_init=False,
            fp32=True,
            beta=None,
            
            # Wandb (disabled by default)
            use_wandb=False,
            wandb_project='hbehavemae',
            wandb_entity=None,
            
            # AMASS/hBABEL (not used for mice)
            joints3d_procrustes=settings_dict.get('joints3d_procrustes', True)
        )
        
        return args
    
    def _run_training(self, args, project_dir):
        """
        Run the hBehaveMAE training.
        
        Args:
            args: Training arguments namespace
            project_dir: Project directory path
            
        Returns:
            str: Path to final checkpoint, or None if failed
        """
        try:
            # Import main_pretrain (support running as script or module)
            try:
                from .main_pretrain import main
            except ImportError:
                from main_pretrain import main
            
            # Check for existing checkpoint to resume from in output_dir
            output_dir = Path(args.output_dir)
            checkpoints = list(output_dir.glob('checkpoint-*.pth'))
            if checkpoints:
                # Sort by epoch number and get latest
                checkpoints.sort(key=lambda x: int(x.stem.split('-')[1]))
                latest_checkpoint = checkpoints[-1]
                args.resume = str(latest_checkpoint)
                print(f"Resuming from checkpoint: {latest_checkpoint}")
            
            # Run training
            main(args)
            
            # Find final checkpoint
            checkpoints = list(output_dir.glob('checkpoint-*.pth'))
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.stem.split('-')[1]))
                final_checkpoint = checkpoints[-1]
                return str(final_checkpoint)
            
            return None
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @classmethod
    def cleanup_orphaned_files(cls, project_id=None, dry_run=True):
        """Clean up orphaned model files."""
        pass