import random
import numpy as np
from PIL import Image


def prepare_sprite(sprite_img, canvas_size, cfg):
    base_w = int(canvas_size[0] * cfg["sprite_scale_start"])
    s_w, s_h = sprite_img.size
    new_h = int(s_h * (base_w / s_w))
    return sprite_img.resize((base_w, new_h), Image.LANCZOS)


def rotate_image(img, angle):
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)


def compute_targets(outline_canvas, sprites_count):
    indices = np.linspace(
        0, len(outline_canvas), sprites_count, endpoint=False, dtype=int
    )
    return outline_canvas[indices % len(outline_canvas)]


def apply_shake(cfg):
    mag = cfg.get("shake_magnitude", 0)
    return (random.uniform(-mag, mag), random.uniform(-mag, mag))


def compose_frame(
    background_pil,
    center,
    targets,
    t_norm,
    sprite_base,
    rotation_angle,
    cfg,
    shake_offset=(0, 0),
):
    canvas = background_pil.copy().convert("RGBA")
    # interpolate sprite size
    scale_factor = (
        cfg["sprite_scale_start"]
        + (cfg["sprite_scale_end"] - cfg["sprite_scale_start"]) * t_norm
    )
    base_w = int(canvas.size[0] * scale_factor)
    s_w, s_h = sprite_base.size
    new_h = int(s_h * (base_w / s_w))
    sprite_scaled = sprite_base.resize((base_w, new_h), Image.LANCZOS)

    rot = rotate_image(sprite_scaled, rotation_angle)

    for tgt in targets:
        x = center[0] + (tgt[0] - center[0]) * t_norm + shake_offset[0]
        y = center[1] + (tgt[1] - center[1]) * t_norm + shake_offset[1]
        w, h = rot.size
        px, py = int(x - w / 2), int(y - h / 2)
        canvas.alpha_composite(rot, (px, py))

    return canvas
