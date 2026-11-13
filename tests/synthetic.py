"""Synthetic camera frames for tests: a printed puzzle or a blank sheet on a textured table."""

import os

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


DRAWN_CORNERS = np.float32([[60, 120], [580, 110], [590, 400], [50, 410]])  # a wide rectangle


def handdrawn_scene(digits, width=640, height=480, seed=0):
    """A grid drawn by hand on ruled notebook paper: uneven, wavy lines on a wide rectangle, with
    blue ruled lines across the page and handwritten `digits` {(row, col): digit}. Returns the
    frame and the true crossings of the drawn lines (10 x 10 x 2, frame pixels)."""
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), (215, 222, 228), np.uint8)  # warm white paper (BGR)
    for x in range(8, width, 18):  # ruled lines, light blue, across the page
        cv2.line(frame, (x, 0), (x + 6, height), (205, 170, 140), 1, cv2.LINE_AA)

    def even(n):  # uneven but ordered positions of 10 lines between 0 and 1
        steps = rng.uniform(0.7, 1.3, n - 1)
        return np.concatenate([[0], np.cumsum(steps) / steps.sum()])

    cols, rows = even(10), even(10)
    square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    to_frame = cv2.getPerspectiveTransform(square, DRAWN_CORNERS)
    t = np.linspace(0, 1, 2000)

    def wobble():
        return 0.012 * np.sin(2 * np.pi * (rng.uniform(0.5, 1.5) * t + rng.uniform()))

    verticals = [np.stack([c + wobble() + 0.02 * (t - 0.5) * rng.uniform(-1, 1), t], 1) for c in cols]
    horizontals = [np.stack([t, r + wobble()], 1) for r in rows]
    for line in verticals + horizontals:
        points = cv2.perspectiveTransform(np.float32(line).reshape(-1, 1, 2), to_frame)
        cv2.polylines(frame, [np.int32(points)], False, (60, 60, 60), 2, cv2.LINE_AA)

    crossings = np.zeros((10, 10, 2), np.float32)
    for i, h in enumerate(horizontals):
        for j, v in enumerate(verticals):
            x = np.interp(h[:, 1], v[:, 1], v[:, 0])  # the vertical line's x at each y of h ...
            k = np.argmin(np.abs(x - h[:, 0]))  # ... is closest to h's own x where they cross
            crossings[i, j] = h[k]
    crossings = cv2.perspectiveTransform(crossings.reshape(-1, 1, 2), to_frame).reshape(10, 10, 2)

    data = np.load(os.path.join(os.path.dirname(__file__), "..", "sudoku_cv", "data", "handwritten.npz"))
    for (r, c), digit in digits.items():
        sample = data["digits"][np.flatnonzero(data["labels"] == digit)[0]]
        cell = np.float32([crossings[r, c], crossings[r, c + 1], crossings[r + 1, c + 1], crossings[r + 1, c]])
        centre, size = cell.mean(axis=0), 0.7 * min(np.ptp(cell[:, 0]), np.ptp(cell[:, 1]))
        ink = cv2.dilate(cv2.resize(sample, (int(size), int(size))), np.ones((2, 2), np.uint8)) > 100  # a pen stroke
        y0, x0 = int(centre[1] - size / 2), int(centre[0] - size / 2)
        frame[y0:y0 + ink.shape[0], x0:x0 + ink.shape[1]][ink] = (50, 50, 50)
    return frame, crossings
