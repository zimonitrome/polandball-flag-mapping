from pathlib import Path
from typing import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
from itertools import count

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import repackage
repackage.up()
from utils.models import GMM
from utils.ball_flag_dataset_GMM import BallFlagDatasetGMM
from utils.helpers import unnormalize


if __name__ == '__main__':
    torch.manual_seed(1337)

    lr = 1e-4
    betas = (0.9, 0.999)
    wd = 1e-4
    bs = 16
    test_interval = 100 # Every n:th batch
    save_interval = 100 # Every n:th batch
    note = "lr=1e-4"    # Tag the log files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    excluded_countries = open("./data/black_flags.txt", "r", encoding="utf-8").read().splitlines()
    dataset = BallFlagDatasetGMM("./data", exclude_countries=excluded_countries)
    train_dataset, test_dataset = random_split(dataset, [len(dataset)-bs, bs])
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs, drop_last=True, num_workers=16)

    test_samples = next(iter(DataLoader(test_dataset, shuffle=False, batch_size=bs)))
    test_flag = test_samples["flag"].to(device)
    test_outline = test_samples["outline"].to(device)
    test_ball_flag = test_samples["ball_flag"].to(device)
    test_ball_flag_mask = test_samples["ball_flag_mask"].to(device)
    test_ball_flag = test_ball_flag*test_ball_flag_mask


    test_grid = dataset.transform([Image.open("./data/grid.png")])[
        0].repeat(bs, 1, 1, 1).to(device)

    dataloader.dataset.use_augmentation = True


    model = GMM(256, 256)
    # Load trained flag fromg GMM phase 1
    model.load_state_dict(torch.load(
        r"training\checkpoints\GMM_P1_lr=1e-4_2021-02-13-16.43.07\E00040_L0.05678.pth"))
    model.to(device)
    model.train()


    # criterion
    criterion = nn.L1Loss().to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

    run_id = '_'.join(["GMM_P2", note, datetime.today().strftime('%Y-%m-%d-%H.%M.%S')])
    logger = SummaryWriter(f"./training/logs/{run_id}")

    logger.add_images("test/4_outline", test_outline)
    logger.add_images("test/3_flags", unnormalize(test_flag))
    logger.add_images("test/2_target", unnormalize(test_ball_flag))


    for e in count():
        running_loss = []
        for i, inputs in enumerate(tqdm(dataloader, desc=f"Epoch {e}")):
            global_step = e*len(dataloader) + i

            flag = inputs["flag"].to(device)
            outline = inputs["outline"].to(device)
            target = inputs["ball_flag"].to(device)
            target_mask = inputs["ball_flag_mask"].to(device)
            target = target*target_mask

            grid, theta = model(flag, outline)

            warped_flag = F.grid_sample(
                flag, grid, padding_mode="border", align_corners=False)
            warped_flag = warped_flag*target_mask

            loss = criterion(warped_flag, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            if global_step % test_interval == 0:
                logger.add_scalar("train/loss", np.mean(running_loss), global_step)

                model.eval()
                with torch.no_grad():
                    grid, theta = model(test_flag, test_outline)

                    warped_flag = F.grid_sample(
                        test_flag, grid, padding_mode="border", align_corners=False)
                    warped_flag = warped_flag*test_ball_flag_mask
                    
                    warped_grid = F.grid_sample(
                        test_grid, grid, padding_mode="border", align_corners=False)

                    loss = criterion(warped_flag, test_ball_flag)

                logger.add_scalar("test/loss", loss, global_step)
                
                logger.add_images(
                    "test/1_warped", unnormalize(warped_flag), global_step)
                logger.add_images(
                    "test/0_grid", unnormalize(warped_grid), global_step)

                running_lossL2 = []
                running_loss = []
                model.train()

            if global_step % save_interval == 0:
                output_path = Path(f"./training/checkpoints/{run_id}/E{str(e).rjust(5, '0')}_L{round(loss.item(), 5)}.pth")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_path)