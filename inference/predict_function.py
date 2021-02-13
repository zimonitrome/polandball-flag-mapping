from PIL import Image, ImageDraw
from copy import copy

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from preprocessing.process_traning_data import fit_resize, get_black_mask, resize_and_pad_ball
from utils.helpers import quantize_pil_image, unnormalize
from utils.models import GMM

def predict(
        outlines, 
        flags, 
        img_size=256,
        use_M1 = True,
        use_M2 = True,
        use_quantization = True,
        quantize_threshold = 0.01,
        use_cuda=True
    ):
    outlines = copy(outlines)
    flags = copy(flags)
    
    # Convert to uniform types
    if type(img_size) == int:
        img_size = (img_size, img_size)

    if type(outlines) != list:
        outlines = [outlines]

    if type(flags) != list:
        flags = [flags]


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available()
                        and use_cuda else "cpu")

    # Set up models
    GMM_model = GMM(*img_size, use_cuda=use_cuda)
    GMM_model.load_state_dict(torch.load(r"../main_weights/GMM.pth"))
    M2_model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", decoder_use_batchnorm=True,
                        decoder_attention_type="scse", in_channels=4, classes=3, activation=torch.nn.Tanh)
    M2_model.load_state_dict(torch.load(r"../main_weights/BSM.pth")["G"])
    GMM_model = GMM_model.to(device).eval()
    M2_model = M2_model.to(device).eval()

    masks = []
    uncrop_coords = []

    # Pre-process data
    for i, outline in enumerate(outlines):
        outline = outline.convert("RGBA")

        # For uncropping later
        coords = (img_size[0]//2-outline.width//2, img_size[1]//2-outline.height//2, outline.width, outline.height)
        uncrop_coords.append(coords)

        mask = outline
        ImageDraw.floodfill(mask, (0,0), (0,0,0,0), border=None, thresh=0)
        mask = resize_and_pad_ball(mask, img_size)
        masks.append(TF.to_tensor(mask.split()[-1]))

        outline = resize_and_pad_ball(outline, img_size)
        outline = get_black_mask(outline).astype(int)

        outlines[i] = TF.to_tensor(outline)

    color_flags = []

    # Pre-process data
    for i, flag in enumerate(flags):
        flag = flag.convert("RGBA")
        color_flags.append(flag.convert("RGB"))
        flag = fit_resize(flag, (img_size[0]-15, img_size[1]-15) )
        flag = TF.center_crop(flag, img_size[::-1])
        flag = Image.composite(flag, Image.new("RGB", img_size, (255, 255, 255)), flag)
        flags[i] = TF.to_tensor(flag)

    flags = torch.stack(flags).float().to(device)
    flags = TF.normalize(flags, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    outlines = torch.stack(outlines).float().to(device)
    masks = torch.stack(masks).float().to(device)

    interest_mask = masks - outlines

    # First model
    if use_M1:
        grid, _ = GMM_model(flags, outlines)
        morphed_flag = F.grid_sample(flags, grid, padding_mode="border", align_corners=False)
        input_m2 = torch.cat([morphed_flag*interest_mask, outlines], dim=1)
    else:
        input_m2 = torch.cat([flags*interest_mask, outlines], dim=1)
    
    # Second model
    if use_M2:
        output = M2_model(input_m2)
        output = output*interest_mask
    else:
        output = input_m2[:,:-1]

    finished_images = []

    # Add to data structure
    for outp, outl, mask, color_flag, uc in zip(output, outlines, masks, color_flags, uncrop_coords):
        image = TF.to_pil_image(unnormalize(outp))

        # Quantize
        if use_quantization:
            image = quantize_pil_image(
                image, 
                color_flag,
                quantize_threshold
            )

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
        # Uncrop
        background = Image.new("RGBA", uc[-2:], (0, 0, 0, 0))
        background.paste(image, (-uc[0], -uc[1]))
        image = background

        finished_images.append(image)

    return finished_images
