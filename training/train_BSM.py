from pathlib import Path
from datetime import datetime
from itertools import count
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import AdamW
import segmentation_models_pytorch as smp

import repackage
repackage.up()
from utils.helpers import unnormalize
from utils.models import VGGContentLossMultiLayer
from utils.ball_flag_dataset_BSM import BallFlagDatasetBSM

if __name__ == '__main__':
    torch.manual_seed(1337)

    lr = 1e-4
    wd = 1e-4
    betas = (.5, .999)
    bs = 8
    test_percentage = 0.2
    test_interval = 10  # Every n:th batch
    save_interval = 100 # Every n:th batch
    note = "lr=1e-4"    # Tag the log files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VGG_loss = VGGContentLossMultiLayer([27]).to(device)
    L1_loss = nn.L1Loss().to(device)
    L2_loss = nn.MSELoss().to(device)
    def critereon(input, output, target):
        return VGG_loss(output, target) + L2_loss(output, target) + L1_loss(output, input)

    black_flags = open("./data/black_flags.txt", encoding="utf-8").read().splitlines()
    dataset = BallFlagDatasetBSM("./data", exclude_countries=black_flags, use_augmentation=True)
    test_len = int(len(dataset)*test_percentage)
    train_len = len(dataset)-test_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_dataloader = DataLoader(
        train_dataset, bs, shuffle=True, pin_memory=True, drop_last=True, num_workers=1)
    test_dataloader = DataLoader(
        test_dataset, bs, shuffle=False, pin_memory=True, drop_last=True, num_workers=0)
    assert len(test_dataloader) > 0, "too few samples"
    # Weird code but this is how it has to be done
    # in order to set augmentation on/off for test/train.
    test_dataset.dataset = deepcopy(test_dataset.dataset)
    test_dataset.dataset.use_augmentation = False

    aux_params = dict(
        activation="softmax",
        classes=dataset.n_classes
    )
    G = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", decoder_use_batchnorm=True,
                 decoder_attention_type="scse", in_channels=4, classes=3, activation=nn.Tanh)
    G = G.to(device)

    scaler = torch.cuda.amp.GradScaler()

    optimizer_G = AdamW(G.parameters(), lr=lr,
                        betas=betas, weight_decay=wd)

    run_id = '_'.join(["BSM", note, datetime.today().strftime('%Y-%m-%d-%H.%M.%S')])

    saved_e = 0
    saved_i = 0


    first_run = True

    # Loading previous progress
    # saved_state = torch.load(
    #     r"checkpoints\S2_2020-12-09-13.36.28_V14_INIT_2\e2_gs4000.pth")
    # G.load_state_dict(saved_state["G"])
    # optimizer_G.load_state_dict(saved_state["optimizer_G"])
    # # run_id = saved_state["run_id"]
    # saved_e = saved_state["e"]
    # saved_i = saved_state["i"]
    
    logger = SummaryWriter(f"./training/logs/{run_id}")

    for e in count(saved_e):
        running_loss = {
            "loss_G": []
        }
        for i, inputs in enumerate(tqdm(train_dataloader, desc=f"Training epoch {e}")):
            global_step = e*len(train_dataloader) + i + saved_i

            ball = inputs["ball"].to(device)
            ball_mask = inputs["ball_mask"].to(device)
            outline = inputs["outline"].to(device)
            GMM_morph = inputs["GMM_morph"].to(device)

            interest_mask = ball_mask - outline

            target = ball * interest_mask
            input = torch.cat([GMM_morph, outline], dim=1)

            #######################################
            #              Train G                #
            #######################################

            G.train()
            G.zero_grad()

            with torch.cuda.amp.autocast():
                output = G(input)
                output = output * interest_mask

                loss_G = critereon(GMM_morph*interest_mask, output, target)
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

            scaler.update()

            #######################################
            #       Export loss for logging       #
            #######################################

            running_loss["loss_G"].append(loss_G.item())

            #######################################
            #            Log and test             #
            #######################################

            if global_step % test_interval == 0:
                for key, value in running_loss.items():
                    logger.add_scalar(
                        f"train/{key}", np.mean(value), global_step)
                running_loss = {k: [] for k in running_loss}

                G.eval()
                for inputs in tqdm(test_dataloader, desc=f"Testing global step {global_step}"):
                    ball = inputs["ball"].to(device)
                    ball_mask = inputs["ball_mask"].to(device)
                    outline = inputs["outline"].to(device)
                    GMM_morph = inputs["GMM_morph"].to(device)

                    interest_mask = ball_mask - outline

                    target = ball * interest_mask
                    input = torch.cat([GMM_morph, outline], dim=1)

                    output = G(input)
                    output = output * interest_mask

                    loss_G = critereon(GMM_morph*interest_mask, output, target)

                    running_loss["loss_G"].append(loss_G.item())

                for key, value in running_loss.items():
                    logger.add_scalar(
                        f"test/{key}", np.mean(value), global_step)
                running_loss = {k: [] for k in running_loss}

                logger.add_images(
                    "test/0_output", unnormalize(output), global_step)
                logger.add_images("test/2_output_outline",
                                  unnormalize(output)*(1-outline), global_step)
                # Log these only once
                if first_run:
                    logger.add_images(
                        "test/1_target", unnormalize(target), global_step)
                    logger.add_images("test/3_target_outline",
                                    unnormalize(target)*(1-outline), global_step)
                    logger.add_images("test/4_input_morphed",
                                    unnormalize(GMM_morph), global_step)
                    logger.add_images("test/5_input_outline", outline, global_step)
                    first_run = False

            if global_step % save_interval == 0:
                output_path = Path(f"./training/checkpoints/{run_id}/E{e}_L{loss_G.item()}.pth")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "G": G.state_dict(),
                    "e": e,
                    "i": i,
                    "run_id": run_id,
                    "optimizer_G": optimizer_G.state_dict()
                }, output_path)
                print(f"Saved {output_path.stem}.")