from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from scipy.signal.signaltools import convolve2d
from scipy.spatial.kdtree import KDTree
from tqdm import tqdm

img_size = (256, 256)
data_folder = Path(r"/data")
balls_folder = data_folder / "balls"
flags_folder = data_folder / "flags"

##########################################################
#################### Helper functions ####################
##########################################################

def resize_and_pad_ball(ball, img_size):
    # Resize ball if needed
    if ball.width > img_size[0] or ball.height > img_size[1]:
        ball = fit_resize(ball, img_size)

    # Pad ball to img_size
    ball_padded = Image.new("RGBA", img_size, (0, 0, 0, 0))
    ball_padded.paste(
        ball, (img_size[0]//2-ball.width//2, img_size[1]//2-ball.height//2), ball)
    return ball_padded 

def fit_resize(img, size):
    if(img.size[0] > img.size[1]):
        return TF.resize(img, (round((size[0]/img.width)*img.height), size[0]), 0)
    else:
        return TF.resize(img, (size[1], round((size[1]/img.height)*img.width)), 0)

def get_black_mask(ball):
    ball_white_bg = Image.new("RGB", ball.size, (255, 255, 255))
    ball_white_bg.paste(ball, (0, 0), ball)
    return np.asarray(ball_white_bg.convert("L")) == 0

def get_eye_mask(ball):
    # Get segments of white pixels
    white_pixels = (np.asarray(ball.convert("L")) == 255)
    white_segments, n_labels = ndimage.label(white_pixels)

    # Get border colors of each white segment
    eye_labels = []
    for label in range(1, n_labels+1):
        segment = white_segments == label
        kernel = np.asarray([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
        edges = convolve2d(segment.astype(int), kernel.astype(
            int), mode='same').astype(bool)
        diff = edges ^ segment
        border_colors = np.unique(np.asarray(ball)[diff == 1], axis=0)

        # Check if only color white segment borders is black
        if len(border_colors) == 1 and (border_colors[0] == (0, 0, 0, 255)).all():
            # If so, then it is an eye
            eye_labels.append(label)

    return np.isin(white_segments, eye_labels)

def nearest_neighbor_fill(ball, mask):
    # Fill flag holes with nearest neighbor
    ff = np.asarray(ball).copy()
    x, y = np.mgrid[0:ff.shape[0], 0:ff.shape[1]]
    xygood = np.array((x[~mask], y[~mask])).T
    xybad = np.array((x[mask], y[mask])).T
    ff[mask] = ff[~mask][KDTree(xygood).query(xybad)[1]]
    return Image.fromarray(ff)

################################################################
################################################################
################################################################

print("OBS: This script can take some time to run, but can be run multiple instances at the same time.")

countries = [n.name for n in balls_folder.glob("*")]

flag_rgb_path = data_folder / "processed_flags_rgb"
flag_mask_path = data_folder / "processed_flags_mask"

flag_rgb_path.mkdir(parents=True, exist_ok=True)
flag_mask_path.mkdir(parents=True, exist_ok=True)

for c in tqdm(countries):
    ball_country_folder = balls_folder / c

    resized_ball_folder = data_folder / "processed_balls_resized" / c
    ball_outline_folder = data_folder / "processed_balls_outlines" / c
    ball_eyes_folder = data_folder / "processed_balls_eyes" / c
    ball_flag_folder = data_folder / "processed_balls_flags" / c
    ball_flag_mask_folder = data_folder / "processed_balls_flags_masks" / c

    # To use multiple scripts at the same time
    if resized_ball_folder.exists():
        continue
    # Ignore some files
    if c.startswith('_'):
        continue

    resized_ball_folder.mkdir(parents=True, exist_ok=True)
    ball_outline_folder.mkdir(parents=True, exist_ok=True)
    ball_eyes_folder.mkdir(parents=True, exist_ok=True)
    ball_flag_folder.mkdir(parents=True, exist_ok=True)
    ball_flag_mask_folder.mkdir(parents=True, exist_ok=True)


    flag_path = flags_folder / f"{c}.png"
    if flag_path.exists():
        flag = Image.open(flag_path).convert("RGBA")
        flag = fit_resize(flag, (img_size[0]-15, img_size[1]-15) )
        flag = TF.center_crop(flag, img_size[::-1])
        flag_rgb = Image.composite(flag, Image.new("RGB", img_size, (255, 255, 255)), flag)
        flag_mask = np.array(flag.split()[-1]) > 128
        flag_mask = Image.fromarray(flag_mask).convert('1')

        flag_rgb.save(flag_rgb_path / f"{c}.png")
        flag_mask.save(flag_mask_path / f"{c}.png")

    ball_country_folder = balls_folder / c
    balls = [n.name for n in ball_country_folder.glob("*")]
    for b in tqdm(balls, desc=c):
        ball = Image.open(ball_country_folder / b).convert("RGBA")

        ball_padded = resize_and_pad_ball(ball, img_size)

        outline = get_black_mask(ball_padded)
        eye_mask = get_eye_mask(ball_padded)
        fill_mask = (outline ^ eye_mask)    # combine outline and eyes
        filled_ball = nearest_neighbor_fill(ball_padded, fill_mask)

        filled_ball_rgb = Image.composite(filled_ball, Image.new("RGB", img_size, (255, 255, 255)), filled_ball)
        filled_ball_mask = np.array(filled_ball.split()[-1]) > 128
        filled_ball_mask = Image.fromarray(filled_ball_mask).convert('1')

        outline = Image.fromarray(outline).convert('1')
        eyes = Image.fromarray(eye_mask).convert('1')

        ball_padded.save(resized_ball_folder / b)
        outline.save(ball_outline_folder / b)
        eyes.save(ball_eyes_folder / b)
        filled_ball_rgb.save(ball_flag_folder / b)
        filled_ball_mask.save(ball_flag_mask_folder / b)
