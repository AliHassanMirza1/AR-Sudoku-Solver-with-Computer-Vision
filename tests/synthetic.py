"""Synthetic camera frames for tests: a printed puzzle or a blank sheet on a textured table."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
SIZE = 450
CELL = SIZE // 9


def puzzle_image(grid=PUZZLE, lines=True):
    """A flat 450x450 white puzzle, black lines and printed digits."""
    img = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT, 34)
    except OSError:
        font = ImageFont.load_default()
    if lines:
        for i in range(10):
            width = 4 if i % 3 == 0 else 1
            p = min(i * CELL, SIZE - 2)
            draw.line([(p, 0), (p, SIZE)], fill=0, width=width)
            draw.line([(0, p), (SIZE, p)], fill=0, width=width)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                draw.text((c * CELL + CELL / 2, r * CELL + CELL / 2), str(grid[r][c]),
                          fill=0, font=font, anchor="mm")
    return np.array(img)


def table(width=640, height=480, seed=0):
    """A wood-like background with enough texture for feature tracking."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (height // 8, width // 8)).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    grain = np.sin(np.linspace(0, 40, height))[:, None] * 0.5
    base = 110 + 25 * noise + 20 * grain
    bgr = np.dstack([base * 0.55, base * 0.8, base]).clip(0, 255).astype(np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, width), rng.integers(0, height)
        cv2.circle(bgr, (int(x), int(y)), int(rng.integers(2, 6)), (40, 60, 80), -1)
    return bgr


def scene(corners, flat=None, sheet_corners=None, width=640, height=480, seed=0):
    """Paste a flat 450x450 image (and optionally a blank sheet under it) onto the table."""
    frame = table(width, height, seed)
    square = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    if sheet_corners is not None:
        cv2.fillConvexPoly(frame, np.int32(sheet_corners), (235, 238, 240))
    if flat is not None:
        h = cv2.getPerspectiveTransform(square, np.float32(corners))
        warped = cv2.warpPerspective(flat, h, (width, height), borderValue=0)
        mask = cv2.warpPerspective(np.full_like(flat, 255), h, (width, height)) > 128
        frame[mask] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask]
    return frame


TILTED = np.float32([[150, 70], [480, 95], [500, 420], [130, 400]])


def printed_scene(**kwargs):
    return scene(TILTED, puzzle_image(), **kwargs)
