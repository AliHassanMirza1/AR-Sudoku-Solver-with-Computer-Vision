"""Hand-drawn grids: wobbly, slanted, unevenly spaced lines, often on ruled notebook paper.

detect.py expects 10 straight, evenly spaced lines each way, so it misses these. Here:
  1. the printed lines of ruled paper are painted over with paper (straight and coloured),
  2. the board's outline is found with contours (any 4-sided shape, not only a square),
  3. inside the flattened board, each drawn line is followed strip by strip, and the 10 x 10
     crossings of these curves make a mesh,
  4. the mesh is remapped to even cells, so the usual digit reader works unchanged.

The mesh is in board coordinates (0..SQUARE), so it holds at any resolution: find the board in
a small frame, then read it from a sharper one with read_board.
"""

import cv2
import numpy as np

from .detect import SQUARE, get_homography, grid_score, quad_from_hull, threshold
from .digits import read_digits

FRAME_WIDTH = 640  # pixel sizes below are for frames this wide; larger frames are scaled

RULED_VOTES = 0.6  # a ruled line is straight for at least this share of the frame's short side
RULED_WIDTH = 2
COLOUR_RATIO = 0.1  # blue-minus-red shift per unit of darkness above which a pixel is coloured
MIN_AREA_RATIO = 0.05
MAX_CANDIDATES = 5
STRIPS = 18  # strips across the lines when following them
REACH = 4  # a line may wander this many pixels within one strip
GAP = 6  # gaps up to twice this long along a line are bridged (where a ruled line crossed it)
LINE_COVER = 0.75  # share of a strip's length a line must cross
LINK_PIXELS = 12  # a line moves at most this much from one strip to the next
MIN_SPAN = 0.4  # share of the strips a line must be found in (dashed or short lines)
MIN_CELL, MAX_CELL = 0.4, 2.0  # cell size limits, in even cells
MIN_GRID_SCORE = 0.3  # lower than sudoku_cv: drawn lines are dashed where ruled lines crossed
OUTLINE_MARGIN = 0.04  # the outline is rough: look a little outside it for the border lines


# cleaning

def clean_gray(bgr):
    """Grey image without the printed lines of ruled paper.

    Two cues, each too weak alone: ruled lines are perfectly straight and cross the whole page,
    so each collects many votes in one bin of a Hough transform (hand-drawn lines wobble and
    spread their votes); and they are coloured, so they are bluer or redder than their darkness
    explains (pencil and black pen are grey). Pixels with both get the paper's brightness; pen
    crossing a ruled line is grey and stays.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, max(gray.shape) / FRAME_WIDTH)
    small = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    lines = cv2.HoughLines(threshold(small), 1, np.pi / 360, int(RULED_VOTES * min(small.shape)))
    if lines is None:
        return gray
    straight = np.zeros_like(gray)
    reach = 2 * max(gray.shape)
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x, y = a * rho * scale, b * rho * scale
        cv2.line(straight, (int(x - reach * b), int(y + reach * a)), (int(x + reach * b), int(y - reach * a)),
                 255, round(RULED_WIDTH * scale))

    paper = cv2.dilate(gray, np.ones((odd(5 * scale),) * 2, np.uint8))
    tint = bgr[..., 0].astype(np.float32) - bgr[..., 2].astype(np.float32)  # blue minus red
    paper_tint = cv2.medianBlur((tint + 128).clip(0, 255).astype(np.uint8), min(odd(21 * scale), 255))
    coloured = np.abs(tint - paper_tint.astype(np.float32) + 128) > COLOUR_RATIO * (paper.astype(np.float32) - gray)
    return np.where((straight > 0) & coloured, paper, gray)


def odd(size):
    return 2 * round(size / 2) + 1


# outline

def outlines(thresh):
    """Rough 4-sided outlines of the largest shapes, largest first."""
    height, width = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CANDIDATES]:
        if cv2.contourArea(contour) < MIN_AREA_RATIO * height * width:
            return
        corners = quad_from_hull(contour)
        if corners is not None:
            yield corners


# drawn lines

def line_centres(mask):
    """Centres of the runs of True in a 1-D mask."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return (edges[0::2] + edges[1::2] - 1) / 2


def follow_lines(ink):
    """The roughly vertical lines in a square ink image, each as a curve x = f(y), left to right.

    The image is cut into horizontal strips; in each strip a line is a column band that most
    rows have ink in (within REACH pixels, so slanted lines count, and with gaps up to 2 GAP
    bridged, so dashed lines count). Bands are linked from strip to strip, and lines found in
    enough strips become curves.
    """
    size = ink.shape[0]
    strip = size // STRIPS
    near = cv2.dilate(ink, np.ones((1, 2 * REACH + 1), np.uint8))
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((2 * GAP + 1, 1), np.uint8)) > 0
    chains = []  # each a list of (y, x)
    for k in range(STRIPS):
        y = (k + 0.5) * strip
        cover = near[k * strip:(k + 1) * strip].mean(axis=0)
        for x in line_centres(cover >= LINE_COVER):
            open_chains = [c for c in chains if c[-1][0] < y and abs(c[-1][1] - x) < LINK_PIXELS]
            if open_chains:
                min(open_chains, key=lambda c: abs(c[-1][1] - x)).append((y, x))
            else:
                chains.append([(y, x)])
    chains = [c for c in chains if len(c) >= MIN_SPAN * STRIPS]
    chains = sorted(chains, key=len, reverse=True)[:10]
    return sorted((Curve(*np.array(c).T) for c in chains), key=lambda f: f(size / 2))


class Curve:
    """A drawn line through samples (t, x): x = f(t). Follows the samples (smoothed a little)
    between the first and last, and goes on straight beyond them."""

    def __init__(self, ts, xs):
        if len(xs) >= 3:  # median of 3 removes single samples caught on a digit
            xs = np.concatenate([xs[:1], np.median([xs[:-2], xs[1:-1], xs[2:]], axis=0), xs[-1:]])
        self.ts, self.xs = ts, xs
        self.start = np.polyfit(ts[:4], xs[:4], 1)
        self.end = np.polyfit(ts[-4:], xs[-4:], 1)

    def __call__(self, t):
        if t < self.ts[0]:
            return np.polyval(self.start, t)
        if t > self.ts[-1]:
            return np.polyval(self.end, t)
        return np.interp(t, self.ts, self.xs)


def find_mesh(ink):
    """The 10 x 10 crossings of the drawn lines in a square ink image, or None."""
    verticals = follow_lines(ink)  # x = f(y)
    horizontals = follow_lines(ink.T)  # y = g(x)
    if len(verticals) != 10 or len(horizontals) != 10:
        return None
    mesh = np.zeros((10, 10, 2), np.float32)
    middle = ink.shape[0] / 2
    for i, g in enumerate(horizontals):
        for j, f in enumerate(verticals):
            y = g(middle)
            for _ in range(5):  # where x = f(y) and y = g(x) meet
                x = f(y)
                y = g(x)
            mesh[i, j] = x, y
    widths = np.diff(mesh[..., 0], axis=1)
    heights = np.diff(mesh[..., 1], axis=0)
    cell = (mesh[-1, -1] - mesh[0, 0]).mean() / 9
    sizes = np.concatenate([widths.ravel(), heights.ravel()]) / cell
    return mesh if MIN_CELL < sizes.min() and sizes.max() < MAX_CELL else None


# the board

def grown(corners, margin):
    corners = np.float32(corners)
    centre = corners.mean(axis=0)
    return centre + (corners - centre) * (1 + 2 * margin)


def transform(points, homography):
    shape = np.shape(points)
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), homography).reshape(shape)


def even_board(gray, corners, mesh, true_shape=False):
    """The board remapped so its cells are even: SQUARE x SQUARE, ready for the digit reader.
    `mesh` is in board coordinates (0..SQUARE, from the homography of `corners`).

    true_shape: sample a square of real paper from the middle of each cell, so digits keep their
    shape when the drawn cells are wide or tall (whole cells would squeeze them thin)."""
    homography = get_homography(corners)
    cell = SQUARE / 9
    u = np.arange(SQUARE, dtype=np.float32) + 0.5
    index = np.minimum((u // cell).astype(int), 8)
    t = (u - index * cell) / cell
    i, j = index[:, None], index[None, :]  # rows (v), columns (u)
    tv, tu = t[:, None], t[None, :]
    if true_shape:
        width, height = cell_sizes(transform(mesh, homography))
        aspect = (width / height)[i, j]
        tu = 0.5 + (tu - 0.5) * np.minimum(1, 1 / aspect)
        tv = 0.5 + (tv - 0.5) * np.minimum(1, aspect)
    tv, tu = tv[..., None], tu[..., None]
    top = (1 - tu) * mesh[i, j] + tu * mesh[i, j + 1]
    bottom = (1 - tu) * mesh[i + 1, j] + tu * mesh[i + 1, j + 1]
    points = transform((1 - tv) * top + tv * bottom, homography)
    return cv2.remap(gray, points[..., 0], points[..., 1], cv2.INTER_LINEAR)


def cell_sizes(mesh):
    """Width and height of each of the 9 x 9 cells of a mesh (averages of opposite sides)."""
    sides = lambda a, b: np.linalg.norm(a - b, axis=-1)
    width = (sides(mesh[:-1, 1:], mesh[:-1, :-1]) + sides(mesh[1:, 1:], mesh[1:, :-1])) / 2
    height = (sides(mesh[1:, :-1], mesh[:-1, :-1]) + sides(mesh[1:, 1:], mesh[:-1, 1:])) / 2
    return width, height


def find_board(bgr, corners=None):
    """A hand-drawn board in a frame up to FRAME_WIDTH wide: its 4 corners and the mesh of its
    line crossings (board coordinates), or None. With `corners`, look only there."""
    clean = clean_gray(bgr)
    thresh = threshold(clean)
    if corners is None:
        thick = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        candidates = outlines(thick)
    else:
        candidates = [np.float32(corners)]
    for outline in candidates:
        region = grown(outline, OUTLINE_MARGIN)
        ink = cv2.warpPerspective(thresh, get_homography(region), (SQUARE, SQUARE),
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        mesh = find_mesh(ink)
        if mesh is None:
            continue
        mesh = transform(mesh, get_homography(region))  # frame coordinates
        board = np.float32([mesh[0, 0], mesh[0, -1], mesh[-1, -1], mesh[-1, 0]])
        mesh = transform(mesh, np.linalg.inv(get_homography(board)))  # board coordinates
        if grid_score(threshold(even_board(clean, board, mesh))) >= MIN_GRID_SCORE:
            return board, mesh
    return None


def read_board(bgr, corners, mesh):
    """Read a hand-drawn board, from a frame of any size (`corners` in its pixels). Every digit
    on it is part of the puzzle: all are handwritten. Returns givens and confidence."""
    flat = even_board(clean_gray(bgr), np.float32(corners), np.float32(mesh), true_shape=True)
    printed, written, confidence = read_digits(flat)
    givens = [[p or w for p, w in zip(*rows)] for rows in zip(printed, written)]
    return givens, confidence
