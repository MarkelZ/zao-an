import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def render_glyph_mask(char, font_path, render_size):
    img = Image.new("L", (render_size, render_size), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(render_size * 0.75))
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (render_size - w) // 2 - bbox[0]
    y = (render_size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return img


def find_largest_contour(mask_pil):
    arr = np.array(mask_pil)
    _, bw = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).squeeze(axis=1).astype(float)


def resample_contour(points, n_samples):
    if len(points) < 2:
        return points.copy()
    pts = np.vstack([points, points[0]])  # close
    segs = np.diff(pts, axis=0)
    seg_lens = np.hypot(segs[:, 0], segs[:, 1])
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    ts = np.linspace(0, total, n_samples, endpoint=False)
    res = []
    idx = 0
    for t in ts:
        while idx < len(cum) - 2 and not (cum[idx] <= t < cum[idx + 1]):
            idx += 1
        seg_t = (
            (t - cum[idx]) / (cum[idx + 1] - cum[idx]) if cum[idx + 1] > cum[idx] else 0
        )
        p = (1 - seg_t) * pts[idx] + seg_t * pts[idx + 1]
        res.append(p)
    return np.array(res)


def transform_points(points, render_size, canvas_size):
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    w, h = xmax - xmin, ymax - ymin
    w, h = (w if w > 0 else 1, h if h > 0 else 1)
    cw, ch = canvas_size
    margin = int(min(cw, ch) * 0.08)
    scale = min((cw - 2 * margin) / w, (ch - 2 * margin) / h)
    scaled = (points - [xmin, ymin]) * scale
    offset = [(cw - w * scale) / 2, (ch - h * scale) / 2]
    return scaled + offset

def find_all_contours(mask_pil):
    """
    Return all contours (outer shapes) from mask image.
    Each contour is an (N,2) numpy array.
    """
    arr = np.array(mask_pil)
    _, bw = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.squeeze(axis=1).astype(float) for c in contours if len(c) >= 3]


def extract_outline(char, font_path, render_size, canvas_size, n_points):
    mask = render_glyph_mask(char, font_path, render_size)
    contours = find_all_contours(mask)
    if not contours:
        return None

    # distribute points across all contours proportional to length
    lengths = [cv2.arcLength(c.astype(np.int32), True) for c in contours]
    total_len = sum(lengths)
    outlines = []
    for contour, L in zip(contours, lengths):
        pts = resample_contour(contour, max(1, int(n_points * L / total_len)))
        outlines.append(pts)

    # concatenate all contours into one array
    merged = np.vstack(outlines)
    return transform_points(merged, render_size, canvas_size)