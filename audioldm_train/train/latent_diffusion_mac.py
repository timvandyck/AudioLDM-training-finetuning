# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import sys

sys.path.append("src")
import shutil
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import yaml
import torch

from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
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
    if torch.distributed.get_rank() == 0:
        print(msg)

"""
configs: das Konfigurations dictionary! 
exp_group_name: ist der name des überordners wo die configurations yaml datei ist. z.B:2025_22_07_audio_ldm_2
exp_name: Name der konfigurations .yaml -> z.B: audioldm2_full
"""
def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation):
    if "seed" in configs.keys():
        #setze pytorch random seed auf value von seed
        seed_everything(configs["seed"])
    else:
        #ansonst -> Seed = 0 für random num generation
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(
            configs["precision"]
        )  # highest, high, medium

    log_path = configs["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    # --- Datasets und DataLoader für train und val laden --- 
    dataset = AudioDataset(configs, split="train", add_ons=dataloader_add_ons)#TODO: AudioDataset klasse checken, um herauszufinden wie die mit den configs arbeitet

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
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
    )

    # Copy test data
    #TODO: AudioDataset klasse checken -> Ich nehme an der split erfolgt hier via random?

    test_data_subset_folder = os.path.join(
        os.path.dirname(configs["log_directory"]),
        "testset_data",
        val_dataset.dataset_name,
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset.data, test_data_subset_folder)#TODO: diese funktion checken
    #setze lokale variable auf den checkpoint path
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
    """
    ModelCheckpoint macht folgendes:
    - Speichert Checkpoints im angegebenen Verzeichnis
    - Überwacht den globalen Schritt und speichert den besten Checkpoint basierend auf
        dem Frechet Inception Distance (FID) Wert
    - Speichert Checkpoints alle `save_checkpoint_every_n_steps` Schritte
    - Speichert die besten `save_top_k` Checkpoints
    - Fügt den Metriknamen automatisch hinzu
    - Speichert den letzten Checkpoint nicht automatisch

    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=False,
    )

    os.makedirs(checkpoint_path, exist_ok=True)# erstellt den Ordner falls er nicht existiert
    shutil.copy(config_yaml_path, wandb_path)# kopiert die config.yaml nach wandb_path

    is_external_checkpoints = False 
    """Falls weitere checkpoints im checkpoint_path vorhanden sind,
    lade den letzten checkpoint, 
    
    ansonst, lade den in der config.yaml angegebenen checkpoint,
    ansonst, trainiere von scratch"""
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

    devices = torch.cuda.device_count()#TODO: Für mac anpassen?
    #devices = torch.mps.device_count()
    latent_diffusion = instantiate_from_config(configs["model"])#TODO: instantiate_from_config checken, um zu verstehen was es macht
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)
   
    wandb_logger = WandbLogger(
        save_dir=wandb_path,#log_path + "/" + exp_group_name (name des überordner von yaml ) + "/" + exp_name (name des .yaml),
        project=configs["project"],
        config=configs,
        name="%s/%s" % (exp_group_name, exp_name),
    )#TODO: wandb_logger checken, um zu verstehen was es macht

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epochs" % validation_every_n_epochs)

    trainer = Trainer(
        accelerator="mps",#MPS für macOS
        devices=devices, # Anzahl der GPUs, die verwendet werden sollen
        logger=wandb_logger, # Logger für WandB
        max_steps=max_steps, # Maximale Anzahl an Trainingsschritten aus config[step][max_steps]
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,# Wie oft die Validierung durchgeführt wird aus config[step][validation_every_n_epochs]
        strategy=DDPStrategy(find_unused_parameters=True),#TODO: DDPStrategy checken, um zu verstehen was es macht
        callbacks=[checkpoint_callback],#logik um checkpoints zu speichern
    )
    # Dieser 'if'-Block wird ausgeführt, wenn in der .yaml-Datei ein 'reload_from_ckpt'
    # Pfad angegeben wurde. Er ist speziell dafür ausgelegt, die Gewichte aus einem
    # bereits trainierten Modell (wie unserem audioldm2-full.pth) in die neu
    # initialisierte Modellarchitektur zu laden. 
    # Es werden nur jene Gewichte geladen, die in der neuen Modellarchitektur
    # vorhanden sind.
    # =================================================================================
    # So funktioniert es im Detail:
    # 1. torch.load(...): Lädt die angegebene .pth-Datei. Das Skript geht davon aus,
    #    dass dies ein Dictionary ist, und versucht, auf den Schlüssel ["state_dict"]
    #    zuzugreifen, wo die eigentlichen Gewichte liegen.
    #
    # 2. Filter-Schleife (for key in ...): Dies ist ein "intelligenter Filter". Er vergleicht
    #    jede Ebene (jeden "key") aus der geladenen .pth-Datei mit den Ebenen des
    #    aktuellen Modells ('latent_diffusion').
    #    - Es werden alle Ebenen aus der .pth-Datei entfernt, die im neuen Modell nicht existieren.
    #    - Es werden auch alle Ebenen entfernt, deren Dimensionen (Größe/Shape) nicht übereinstimmen.
    #    Das macht den Prozess robust gegenüber kleinen Architekturänderungen.
    #
    # 3. latent_diffusion.load_state_dict(ckpt, strict=False): Dies ist der letzte
    #    Schritt. Er lädt die GEFILTERTEN Gewichte in das neue Modell.
    #    - 'strict=False' ist hier entscheidend: Es erlaubt PyTorch, die Gewichte auch dann
    #      zu laden, wenn nicht alle Ebenen des neuen Modells in der Checkpoint-Datei
    #      vorhanden sind. Das ist perfekt für Fine-Tuning, da z.B. der Optimierer-Status
    #      fehlt und neu initialisiert werden muss.
    # =================================================================================
    if is_external_checkpoints:
        if resume_from_checkpoint is not None:
            """Lade externen Model Checkpoint"""
            ckpt = torch.load(resume_from_checkpoint)["state_dict"]#TODO eventuell was einstellen damit am mac geht! 

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

            # if(len(key_not_in_model_state_dict) != 0 or len(size_mismatch_keys) != 0):
            # print("⛳", end=" ")

            # print("==> Warning: The following key in the checkpoint is not presented in the model:", key_not_in_model_state_dict)
            # print("==> Warning: These keys have different size between checkpoint and current model: ", size_mismatch_keys)

            latent_diffusion.load_state_dict(ckpt, strict=False)

        # if(perform_validation):
        #     trainer.validate(latent_diffusion, val_loader)

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

    #assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)

    #config_yaml ist ein dict welches die Konfiguration enthält
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        #setze den pfad zum checkpoint des gesamt-models welcher geladen werden soll
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    #TODO: Herausfinden was das tut?
    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation)
