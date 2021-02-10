from pathlib import Path
from typing import Counter
from PIL import Image
import numpy as np
import torch
from torch import nn
from utils.models import GMM, VGGContentLossMultiLayer
from utils.ball_flag_dataset_GMM import BallFlagDatasetGMM
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from itertools import count
from utils import unnormalize

torch.manual_seed(1337)

lr = 1e-4
betas = (0.9, 0.999)
wd = 1e-2
bs = 50
test_interval = 50
save_interval = 972
note = "adam_lr=1e-4_"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

excluded_countries = open("black_flags.txt", "r", encoding="utf-8").read().splitlines()
dataset = BallFlagDatasetGMM("./data", exclude_countries=excluded_countries)
dataloader = DataLoader(dataset, shuffle=True, batch_size=bs, drop_last=True)

test_samples = next(iter(dataloader))
test_flag = test_samples["flag"].to(device)
test_outline = test_samples["outline"].to(device)
test_ball_flag = test_samples["ball_flag"].to(device)
test_grid = dataset.transform([Image.open("grid2.png")])[
    0].repeat(bs, 1, 1, 1).to(device)

dataloader.dataset.use_augmentation = True


model = GMM(256, 256)
# model.load_state_dict(torch.load(
#     r"checkpoints\r2020-11-11-21.41.07_s30666_l0.054862797260284424.pth"))
model.to(device)
model.train()


# criterion
criterionL1 = nn.L1Loss().to(device)
criterionVGG = VGGContentLossMultiLayer([26]).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

run_id = "S0_" + note + datetime.today().strftime('%Y-%m-%d-%H.%M.%S')
logger = SummaryWriter(f"./logs/{run_id}")

logger.add_images("test/4_outline", test_outline)
logger.add_images("test/3_flags", unnormalize(test_flag))
logger.add_images("test/2_target", unnormalize(test_ball_flag))


for e in count():
    running_lossL1 = []
    running_lossVGG = []
    running_loss = []
    for i, inputs in enumerate(tqdm(dataloader, desc=f"Epoch {e}")):
        global_step = e*len(dataloader) + i

        flag = inputs["flag"].to(device)
        outline = inputs["outline"].to(device)
        target = inputs["ball_flag"].to(device)

        TF.to_pil_image(unnormalize(target[0])).show()

        grid, theta = model(flag, outline)

        warped_flag = F.grid_sample(
            flag, grid, padding_mode="border", align_corners=False)

        lossL1 = criterionL1(warped_flag, target)
        lossVGG = criterionVGG(warped_flag, target)
        
        loss = lossL1 + lossVGG

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_lossL1.append(lossL1.item())
        running_lossVGG.append(lossVGG.item())
        running_loss.append(loss.item())

        if global_step % test_interval == 0:
            logger.add_scalar("train/loss L1", np.mean(running_lossL1), global_step)
            logger.add_scalar("train/loss VGG", np.mean(running_lossVGG), global_step)
            logger.add_scalar("train/loss", np.mean(running_loss), global_step)

            model.eval()
            with torch.no_grad():
                grid, theta = model(test_flag, test_outline)

                warped_flag = F.grid_sample(
                    test_flag, grid, padding_mode="border", align_corners=False)
                warped_grid = F.grid_sample(
                    test_grid, grid, padding_mode="border", align_corners=False)

                lossL1 = criterionL1(warped_flag, test_ball_flag)
                lossVGG = criterionVGG(warped_flag, test_ball_flag)
                
                loss = lossL1 + lossVGG

            logger.add_scalar("test/loss L1", lossL1, global_step)
            logger.add_scalar("test/loss VGG", lossVGG, global_step)
            logger.add_scalar("test/loss", loss, global_step)
            
            logger.add_images(
                "test/1_warped", unnormalize(warped_flag), global_step)
            logger.add_images(
                "test/0_grid", unnormalize(warped_grid), global_step)

            running_lossL1 = []
            running_lossVGG = []
            running_loss = []
            model.train()

        # if global_step % save_interval == 0:
    # Save every epoch
    output_path = Path(f"./checkpoints/{run_id}/e{e}_l{round(loss.item(), 5)}.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
