from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF


class BallFlagDatasetBSM(Dataset):
    def __init__(self, data_dir, use_augmentation=False, img_size=(256, 256), exclude_countries=[]):
        data_folder = Path(data_dir)
        self.flags_rgb_folder = data_folder / "processed_flags_rgb"
        self.flags_mask_folder = data_folder / "processed_flags_mask"
        self.ball_folder = data_folder / "processed_balls_resized"
        self.outlines_folder = data_folder / "processed_balls_outlines"
        self.balls_flags_folder = data_folder / "processed_balls_flags"
        self.balls_flags_masks_folder = data_folder / "processed_balls_flags_masks"
        self.morphed_flags_folder = data_folder / "morphed_flags_GMM"

        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.files = []
        self.n_classes = len([*self.balls_flags_folder.iterdir()])

        for class_idx, country in enumerate(self.balls_flags_folder.iterdir()):
            if country.name.startswith('_'):
                continue
            if not (self.flags_rgb_folder / f"{country.name}.png").exists():
                continue
            if country.name in exclude_countries:
                continue
            for file in country.iterdir():
                self.files.append((class_idx, country.name, file.name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        class_idx, country, file = self.files[i]

        ball_org = Image.open(self.ball_folder / country / file)
        ball = ball_org.convert("RGB")
        ball_mask = ball_org.split()[-1]
        outline = Image.open(self.outlines_folder / country / file)
        morphed_flag = Image.open(
            self.morphed_flags_folder / country / file)

        ball, ball_mask, outline, morphed_flag = self.transform(
            [ball, ball_mask, outline, morphed_flag])

        return {
            "country_name": country,
            "ball": ball,
            "ball_mask": ball_mask,
            "outline": outline,
            "GMM_morph": morphed_flag,
            "label": class_idx
        }

    def transform(self, images):
        if self.use_augmentation:
            # Flip all
            if np.random.rand() > 0.5:
                images = [TF.hflip(i) for i in images]

            # Change hue of RGB images
            if np.random.rand() > 0.5:
                hue_val = np.random.rand() - 0.5  # random val [-0.5, 0.5]
                images = [TF.adjust_hue(
                    i, hue_val) if i.mode == "RGB" else i for i in images]

            # Change saturation of RGB images
            if np.random.rand() > 0.5:
                sat_val = np.random.rand() + 0.5  # random val [0.5, 1.5]
                images = [TF.adjust_saturation(
                    i, sat_val) if i.mode == "RGB" else i for i in images]

        # Convert to tensor
        images = [TF.to_tensor(i) for i in images]

        # Normalize RGB images
        images = [TF.normalize(i, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  if i.shape[0] == 3 else i for i in images]

        return images
