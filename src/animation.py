import os
import numpy as np
from pathlib import Path
from PIL import Image
import imageio

from glyphs import extract_outline
from sprites import (
    prepare_sprite,
    compose_frame,
    compute_targets,
    apply_shake,
)


def generate_animation(cfg: dict) -> None:
    sprite_img = Image.open(cfg["sprite_path"]).convert("RGBA")
    bg_img = Image.open(cfg["background_path"]).convert("RGBA")
    canvas_size = tuple(cfg["canvas_size"])

    # upscale background nicely (using OpenCV for better quality if upscaling needed)
    bg_arr = np.array(bg_img)
    import cv2
    bg_resized = cv2.resize(
        bg_arr, canvas_size, interpolation=cv2.INTER_LANCZOS4
    )
    bg_img = Image.fromarray(bg_resized)

    sprite_img = prepare_sprite(sprite_img, canvas_size, cfg)

    all_frames = []
    global_frame_idx = 0
    os.makedirs(cfg["frames_dir"], exist_ok=True)

    for char_index, ch in enumerate(cfg["text"]):
        outline_canvas = extract_outline(
            ch,
            cfg["font_path"],
            cfg["glyph_render_size"],
            canvas_size,
            cfg["outline_points"],
        )
        if outline_canvas is None:
            print(f"Skipping char '{ch}' (no outline).")
            continue

        targets = compute_targets(outline_canvas, cfg["sprites_count"])
        center = (canvas_size[0] / 2.0, canvas_size[1] / 2.0)

        for f in range(cfg["frames_per_char"] + cfg["shake_duration"]):
            if f < cfg["frames_per_char"]:
                # Interpolation phase
                t_norm = (f + 1) / cfg["frames_per_char"]
                shake_offset = (0, 0)
            else:
                # Shake phase
                t_norm = 1.0
                shake_offset = apply_shake(cfg)

            rotation = f * cfg["rotation_speed_deg_per_frame"]

            frame = compose_frame(
                bg_img,
                center,
                targets,
                t_norm,
                sprite_img,
                rotation,
                cfg,
                shake_offset,
            )

            all_frames.append(np.array(frame.convert("RGBA")))

            if cfg["save_frames"]:
                p = Path(cfg["frames_dir"]) / f"frame_{global_frame_idx:05d}.png"
                frame.save(p)
            global_frame_idx += 1

    imageio.mimsave(cfg["output_gif"], all_frames, duration=cfg["frame_duration"], loop=cfg["loop"])
    print(f"Saved GIF with {len(all_frames)} frames to {cfg['output_gif']}")
