from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import DataLoader

import repackage
repackage.up()
from utils.models import GMM
from utils.ball_flag_dataset_GMM import BallFlagDatasetGMM


if __name__ == '__main__':
    bs = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GMM(256, 256, use_cuda=(str(device) == "cuda"))
    model.load_state_dict(torch.load(r"../main_weights/GMM.pth"))
    model.to(device)
    model.eval()

    dataset = BallFlagDatasetGMM("./data", use_augmentation=False)
    dataloader = DataLoader(dataset, batch_size=bs, drop_last=False, num_workers=8)

    output_path = Path("data/morphed_flags_GMM")

    def unnormalize(t):
        return t*0.5+0.5

    for inputs in tqdm(dataloader):
        flag = inputs["flag"].to(device)
        outline = inputs["outline"].to(device)
        target_mask = inputs["ball_flag_mask"].to(device)
        countries = inputs["country_name"]
        file_names = inputs["file_name"]

        grid, theta = model(flag, outline)
        warped_flag = F.grid_sample(flag, grid, padding_mode="border", align_corners=False)
        warped_flag = unnormalize(warped_flag*target_mask).cpu()

        for warped, country, file in zip(warped_flag, countries, file_names):
            country_path = output_path / country
            country_path.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(warped).save(country_path / file)