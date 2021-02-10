from PIL import Image
import numpy as np


def unnormalize(t):
    return t*0.5+0.5


def custom_norm(t):
    t[t >= 0] = 0.5*(t[t >= 0] / t.max())
    t[t < 0] = -0.5*(t[t < 0] / t.min())
    t = t + 0.5
    return t


def quantize_pil_image(ball, flag, color_thresh=0.01):
    flag_pixels = flag.width * flag.height

    flag_colors = flag.getcolors(148279)
    flag_colors = sorted(flag_colors, reverse=True)
    flag_colors = {c for n, c in flag_colors if n/flag_pixels > color_thresh}
    # Always add white
    flag_colors = flag_colors.union({(255, 255, 255)})
    flag_colors = np.array([[*flag_colors]], np.uint8)

    palette_image = Image.fromarray(flag_colors, "RGB").convert(
        'P', palette=Image.ADAPTIVE, colors=flag_colors.size, dither=None)
    
    return ball.quantize(palette=palette_image, dither=Image.NONE).convert("RGB")
