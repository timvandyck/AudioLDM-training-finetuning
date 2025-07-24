# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import sys

sys.path.append("src")
import shutil
import os

# Set TOKENIZERS_PARALLELISM to false to avoid potential deadlocks with multiprocessing
# on macOS, especially when using MPS.
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import argparse
import yaml
import torch

from tqdm import tqdm
# DDPStrategy is for distributed training (multiple GPUs), which is not typical for a single Mac ARM machine.
# We'll remove it or make it conditional.
# from pytorch_lightning.strategies.ddp import DDPStrategy 
from audioldm_train.utilities.data.dataset import AudioDataset

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from audioldm_train.utilities.tools import (
    get_restore_step,
    copy_test_subset_data,
)
from audioldm_train.utilities.model_util import instantiate_from_config
import logging

logging.basicConfig(level=logging.WARNING)


def print_on_rank0(msg):
    # This function is primarily for distributed training.
    # On a single Mac ARM machine, rank will always be 0.
    # We can simplify it or keep it as is, it won't hurt.
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(msg)
    else:
        print(msg)


def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    # MPS does not currently support `torch.set_float32_matmul_precision`.
    # This line should be skipped or conditionally executed.
    # if "precision" in configs.keys():
    #     torch.set_float32_matmul_precision(
    #         configs["precision"]
    #     )  # highest, high, medium

    # Determine the device
    if torch.backends.mps.is_available():
        #device = "mps"
        device = "cpu"
        print("Using Apple Silicon (MPS) for acceleration.")
    else:
        # Fallback to CPU if MPS is not available (e.g., older macOS, no M-chip)
        device = "cpu"
        print("MPS not available, falling back to CPU.")
    
    # In PyTorch Lightning, the 'accelerator' argument handles device placement.
    # We'll set 'accelerator="mps"' or "cpu" directly in the Trainer.

    log_path = configs["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    dataset = AudioDataset(configs, split="train", add_ons=dataloader_add_ons)

    # For MPS, num_workers is often best set to 0 or 1.
    # Higher values can sometimes lead to overhead or issues.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0, # Changed from 16 to 0 for MPS/CPU efficiency on Mac
        pin_memory=True if device == "mps" else False, # Pin memory can be beneficial for MPS
        shuffle=True,
    )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = AudioDataset(configs, split="test", add_ons=dataloader_add_ons)

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=0, # Changed from default to 0 for MPS/CPU
        pin_memory=True if device == "mps" else False,
    )

    # Copy test data
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs["log_directory"]),
        "testset_data",
        val_dataset.dataset_name,
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset.data, test_data_subset_folder)

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    try:
        limit_val_batches = configs["step"]["limit_val_batches"]
    except:
        limit_val_batches = None

    validation_every_n_epochs = configs["step"]["validation_every_n_epochs"]
    save_checkpoint_every_n_steps = configs["step"]["save_checkpoint_every_n_steps"]
    max_steps = configs["step"]["max_steps"]
    save_top_k = configs["step"]["save_top_k"]

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step", # Monitor a metric that exists during training, e.g., 'global_step'
        mode="max", # Or 'min' depending on what 'global_step' indicates. For global step, 'max' is fine.
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=False,
    )

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    is_external_checkpoints = False
    if len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        is_external_checkpoints = True
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    # devices = torch.cuda.device_count() # This will be 0 on Mac ARM without CUDA
    # For MPS, we typically use 1 device. PyTorch Lightning handles this with `accelerator="mps"`.
    devices = 1 # We'll let Trainer figure out the actual device count for MPS

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=configs["project"],
        config=configs,
        name="%s/%s" % (exp_group_name, exp_name),
    )

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epochs" % validation_every_n_epochs)

    # Instantiate Trainer
    # Change accelerator to "mps" if available, else "cpu"
    # Remove DDPStrategy for single-device training.
    trainer = Trainer(
        accelerator=device, # Changed from "gpu" to dynamic 'device' ('mps' or 'cpu')
        devices=1, # Always 1 for single-device MPS or CPU training
        logger=wandb_logger,
        max_steps=max_steps,
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        precision="32-true", # Use "32-true" for MPS/CPU
        # Remove DDPStrategy since we're targeting single Mac ARM.
        # strategy=DDPStrategy(find_unused_parameters=True), 
        callbacks=[checkpoint_callback],
    )

    if is_external_checkpoints:
        if resume_from_checkpoint is not None:
            ckpt = torch.load(resume_from_checkpoint, map_location=device)["state_dict"] # Map to device

            key_not_in_model_state_dict = []
            size_mismatch_keys = []
            state_dict = latent_diffusion.state_dict()
            print("Filtering key for reloading:", resume_from_checkpoint)
            print(
                "State dict key size:",
                len(list(state_dict.keys())),
                len(list(ckpt.keys())),
            )
            for key in tqdm(list(ckpt.keys())):
                if key not in state_dict.keys():
                    key_not_in_model_state_dict.append(key)
                    del ckpt[key]
                    continue
                if state_dict[key].size() != ckpt[key].size():
                    del ckpt[key]
                    size_mismatch_keys.append(key)

            latent_diffusion.load_state_dict(ckpt, strict=False)

        trainer.fit(latent_diffusion, loader, val_loader)
    else:
        trainer.fit(
            latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )

    parser.add_argument("--val", action="store_true")

    args = parser.parse_args()

    perform_validation = args.val

    # Check for MPS availability first, then CUDA (though CUDA won't be on Mac ARM)
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("Neither MPS nor CUDA is available. Running on CPU.")
    elif torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available. Using Apple Silicon GPU.")
    elif torch.cuda.is_available():
        print("CUDA is available. Using NVIDIA GPU.")


    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if perform_validation:
        # These specific config changes related to `cond_stage_config` might need to be verified
        # against the actual structure of your `audioldm_train` library and the model's design.
        # Assuming they are general settings that don't directly conflict with device change.
        if "model" in config_yaml and "params" in config_yaml["model"] and \
           "cond_stage_config" in config_yaml["model"]["params"] and \
           "crossattn_audiomae_generated" in config_yaml["model"]["params"]["cond_stage_config"] and \
           "params" in config_yaml["model"]["params"]["cond_stage_config"]["crossattn_audiomae_generated"]:
            config_yaml["model"]["params"]["cond_stage_config"]["crossattn_audiomae_generated"]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation)