from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from utils.helpers import quantize_pil_image, unnormalize
from utils.models import GMM
from utils.ball_flag_dataset_BSM import BallFlagDatasetBSM

use_cuda = True
input_folder = Path(r"prediction_FID")
img_size = (256, 256)
bs = 8
quantize_thresh = 0.01
use_first_model = True
use_second_model = True
use_quantization = True
note = "mediumquant"

device = torch.device("cuda" if torch.cuda.is_available()
                      and use_cuda else "cpu")

# Set up models
GMM_model = GMM(*img_size, use_cuda=use_cuda)
GMM_model.load_state_dict(torch.load(r"//MODEL.pth"))
M2_model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", decoder_use_batchnorm=True,
                    decoder_attention_type="scse", in_channels=4, classes=3, activation=torch.nn.Tanh)
M2_model.load_state_dict(torch.load(r"//MODEL.pth")["G"])
GMM_model = GMM_model.to(device).eval()
M2_model = M2_model.to(device).eval()

# Set up data
dataset = BallFlagDatasetBSM(
    input_folder, use_augmentation=False, img_size=img_size)
dataloader = DataLoader(dataset, bs, shuffle=False, pin_memory=True)

# Predict
countries_colored = {}

for inputs in tqdm(dataloader):
    flag = inputs["flag"].to(device)
    ball_mask = inputs["ball_flag_mask"].to(device)
    outline = inputs["outline"].to(device)
    country_name = inputs["country_name"]
    file_name = inputs["file_name"]

    interest_mask = ball_mask - outline

    # First model
    if use_first_model:
        grid, _ = GMM_model(flag, outline)
        morphed_flag = F.grid_sample(flag, grid, padding_mode="border", align_corners=False)
        input_m2 = torch.cat([morphed_flag*interest_mask, outline], dim=1)
    else:
        input_m2 = torch.cat([flag*interest_mask, outline], dim=1)
    
    # Second model
    if use_second_model:
        output = M2_model(input_m2)
        output = output*interest_mask
    else:
        output = input_m2[:,:-1]

    # Add to data structure
    for cnam, outp, outl, mask, fnam in zip(country_name, output, outline, ball_mask, file_name):
        image = TF.to_pil_image(unnormalize(outp))
        flag = Image.open(input_folder / "flags" / f"{cnam}.png").convert("RGB")

        # Quantize
        if use_quantization:
            image = quantize_pil_image(image, flag, quantize_thresh)

        # Add outline
        image = Image.composite(
            Image.new("RGB", img_size, (0, 0, 0)), 
            image, 
            TF.to_pil_image(outl)
        )
        # Add transparent background
        image = Image.composite(
            Image.new("RGBA", img_size, (0, 0, 0, 0)),
            image.convert("RGBA"),
            TF.to_pil_image(1-(mask+outl))
        )

        if cnam not in countries_colored:
            countries_colored[cnam] = []
        countries_colored[cnam].append((fnam, image))

# Save images
output_folder = input_folder.parent / (f"{input_folder.stem}_output" + note)

for country, items in countries_colored.items():
    country_output_folder = output_folder / country
    country_output_folder.mkdir(parents=True, exist_ok=True)
    for (file_name, image) in items:
        image.save(country_output_folder / file_name)