#!/usr/bin/env python3

import subprocess
from datetime import datetime

num_gpus = 2
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction"
data_root = "/scratch/michal/projects/dvc_ofd_2025/data/interim/hbmae_training_data"
run_name = f"ofd_hbmae_{run_id}"
train_args = [
    "--dataset", "OFD_mouse",
    "--path_to_data_dir", f"{data_root}/hbmaeproject-0_shuffle-1/hbmaeproj-0_shuffle-1_train.npy",
    "--test_data_path", f"{data_root}/hbmaeproject-0_shuffle-1/hbmaeproj-0_shuffle-1_test.npy",
    "--wandb_run_name", run_name,
    "--model", "hbehavemae",
    "--non_hierarchical", "True",
    "--input_size", "900", "1", "56",
    "--stages", "2", "2", "2",
    "--q_strides", "5,1,1;1,1,1",
    "--mask_unit_attn", "True", "False", "False",
    "--patch_kernel", "1", "1", "56",
    "--init_embed_dim", "128",
    "--init_num_heads", "2",
    "--out_embed_dims", "128", "192", "256",
    "--decoding_strategy", "single",
    "--decoder_embed_dim", "128",
    "--decoder_depth", "1",
    "--decoder_num_heads", "1",
    "--batch_size", "64",
    "--epochs", "200",
    "--num_frames", "900",
    "--blr", "1.6e-4",
    "--warmup_epochs", "40",
    "--weight_decay", "0.05",
    "--clip_grad", "0.02",
    "--sliding_window", "17",
    "--mask_ratio", "0.75",
    "--norm_loss", "False",
    "--data_augment", "True",
    "--return_likelihoods", "True",
    "--centeralign",
    "--fill_holes", "True",
    "--augmentations", "True",
    "--num_workers", "8",
    "--pin_mem",
    "--checkpoint_period", "20",
    "--output_dir", f"/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/outputs/ofd_run_{run_id}/outputs",
    "--log_dir", f"/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/outputs/ofd_run_{run_id}/logs",
    "--use_wandb",
    "--wandb_project", "hbehavemae-ofd",
    "--wandb_entity", "majkel_d_cember",
]

if num_gpus > 1:
    train_args.insert(0, "--distributed")
    training_cmd = (
        f"cd {project_root} && "
        f"OMP_NUM_THREADS=1 uv run torchrun "
        f"--nproc_per_node={num_gpus} "
        f"--node_rank=0 "
        f"--master_addr=127.0.0.1 "
        f"--master_port=2999 "
        f"run_pretrain.py {' '.join(train_args)}"
    )
else:
    training_cmd = f"cd {project_root} && OMP_NUM_THREADS=1 uv run python run_pretrain.py {' '.join(train_args)}"

subprocess.run([
    "runai", "submit", f"hbmae-ofd_{run_id}",
    "--node-pool", "default",
    "--gpu", str(num_gpus),
    "--cpu", "10",
    "--memory", "40G",
    "--large-shm",
    "--backoff-limit", "0",
    "--image", "registry.rcp.epfl.ch/upamathis-grudzien/mlruntime:1.0",
    "--run-as-user", "grudzien",
    "--run-as-uid", "298315",
    "--run-as-gid", "79678",
    "--existing-pvc", "claimname=upamathis-scratch,path=/scratch",
    "--existing-pvc", "claimname=home,path=/home/grudzien",
    "--environment", "HOME=/home/grudzien",
    "--environment", f"UV_PROJECT_ENVIRONMENT={project_root}/.venv",
    "--environment", f"VIRTUAL_ENV={project_root}/.venv",
    "--environment", "WANDB_API_KEY=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",
    "--command", "--", "/bin/bash", "-lc", training_cmd,
], check=True)
