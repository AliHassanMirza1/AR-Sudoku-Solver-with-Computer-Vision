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

from .detect import SQUARE, get_homography, grid_score, quad_from_hull, score_corners, threshold, warp_grid
from .digits import read_digits

FRAME_WIDTH = 640  # pixel sizes below are for frames this wide; larger frames are scaled

RULED_VOTES = 0.6  # a ruled line is straight for at least this share of the frame's short side
RULED_WIDTH = 2
COLOUR_RATIO = 0.1  # blue-minus-red shift per unit of darkness above which a pixel is coloured
FAINT = 0.3  # on a ruled line, pixels less dark than this share of the paper are the line alone
MIN_AREA_RATIO = 0.05
STRAY_WIDTH = 25  # strokes thinner than this hanging off the board are not part of its outline
MAX_CANDIDATES = 5
STRIPS = 18  # strips across the lines when following them
REACH = 4  # a line may wander this many pixels within one strip
GAP = 6  # gaps up to twice this long along a line are bridged (where a ruled line crossed it)
LINE_COVER = 0.75  # share of a strip's length a line must cross
SLANTS = np.tan(np.radians([-20, -10, 0, 10, 20]))  # slants a line may cross a strip at
LINK_PIXELS = 12  # a line moves at most this much from one strip to the next
MIN_SPAN = 0.4  # share of the strips a line must be found in (dashed or short lines)
MIN_CELL, MAX_CELL = 0.4, 2.0  # cell size limits, in even cells
MISSING_GAP = 1.6  # a gap this many times the usual spacing has a line missing in it
MIN_SUPPORT = 0.15  # share of a guessed outer line that must run over ink
MIN_GRID_SCORE = 0.35  # lower than detect.py's: drawn lines are dashed where ruled lines crossed
EVEN_SCORE = 0.55  # a board whose even grid scores this well (detect.py) is printed, not drawn
PHOTO_TRIES = [(0, 1), (0, 0.9), (3, 1), (-3, 1), (0, 0.8), (3, 0.9), (-3, 0.9)]  # (degrees, zoom)
NO_GRID_SCORE = 0.12  # below this, no grid at all (drawn boards score 0.16 and more)
OUTLINE_MARGIN = 0.04  # the outline is rough: look a little outside it for the border lines


# the crossings of a straight, even grid, in board coordinates
EVEN_MESH = np.stack(np.meshgrid(np.arange(10), np.arange(10)), axis=-1).astype(np.float32) * SQUARE / 9


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
    darkness = paper.astype(np.float32) - gray
    coloured = np.abs(tint - paper_tint.astype(np.float32) + 128) > COLOUR_RATIO * darkness
    faint = darkness < FAINT * paper  # lighter than pen: only the printed line is there
    return np.where((straight > 0) & (coloured | faint), paper, gray)


def odd(size):
    return 2 * round(size / 2) + 1


# outline

def outlines(thresh):
    """Rough 4-sided outlines of the largest shapes in a threshold image, largest first.

    The strokes are thickened so the board's dashed lines join into one shape. Stray strokes
    can hang off it (a ruled line that was not removed, the page's edge); filled in, the board
    is a solid area and they are thin, so an opening removes them. Both outlines are tried."""
    height, width = thresh.shape
    thick = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (STRAY_WIDTH, STRAY_WIDTH))
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CANDIDATES]:
        if cv2.contourArea(contour) < MIN_AREA_RATIO * height * width:
            return
        solid = np.zeros_like(thick)
        cv2.drawContours(solid, [contour], -1, 255, -1)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, opening)
        trimmed, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for shape in sorted(trimmed, key=cv2.contourArea, reverse=True)[:1] + [contour]:
            corners = quad_from_hull(shape)
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
    rows have ink in, along one of SLANTS (within REACH pixels, and with gaps up to 2 GAP
    bridged, so dashed lines count). Bands are linked from strip to strip, and lines found in
    enough strips become curves.
    """
    size = ink.shape[0]
    strip = size // STRIPS
    near = cv2.dilate(ink, np.ones((1, 2 * REACH + 1), np.uint8))
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((2 * GAP + 1, 1), np.uint8)) > 0
    chains = []  # each a list of (y, x)
    rows = np.arange(strip)[:, None]
    for k in range(STRIPS):
        y = (k + 0.5) * strip
        band = near[k * strip:(k + 1) * strip]
        cover = np.zeros(size)
        for slant in SLANTS:  # a slanted line covers the strip along its slant
            shift = np.round((np.arange(strip) - strip / 2) * slant).astype(int)[:, None]
            cover = np.maximum(cover, band[rows, (np.arange(size) + shift) % size].mean(axis=0))
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


def fill_missing(curves, ink):
    """Up to 2 lines a strip-by-strip search missed (too faint or dashed). Where a gap between
    neighbours is about twice the usual spacing, a line halfway between them; otherwise an
    outer line one spacing beyond the first or last line, on the side with more ink along it."""
    size = ink.shape[0]
    for _ in range(10 - len(curves)):
        if not 8 <= len(curves) < 10:
            break
        middles = [f(size / 2) for f in curves]
        gaps = np.diff(middles)
        spacing = np.median(gaps)
        widest = int(np.argmax(gaps))
        if gaps[widest] >= MISSING_GAP * spacing:
            curves.insert(widest + 1, Between(curves[widest], curves[widest + 1]))
            continue
        before, after = Shifted(curves[0], -spacing), Shifted(curves[-1], spacing)
        best = max((before, after), key=lambda f: support(ink, f))
        if support(ink, best) < MIN_SUPPORT:
            break
        curves.insert(0 if best is before else len(curves), best)
    return curves


def support(ink, curve):
    """Share of a curve x = f(y) that runs over ink (0 where it leaves the image)."""
    size = ink.shape[0]
    t = np.arange(0, size, 2)
    x = np.round([curve(v) for v in t]).astype(int)
    inside = (x >= 0) & (x < size)
    near = cv2.dilate(ink, np.ones((1, 2 * REACH + 1), np.uint8))
    return float((near[t[inside], x[inside]] > 0).sum()) / len(t)


class Shifted:
    """A curve moved sideways."""

    def __init__(self, curve, shift):
        self.curve, self.shift = curve, shift

    def __call__(self, t):
        return self.curve(t) + self.shift


class Between:
    """The curve halfway between two curves."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, t):
        return (self.a(t) + self.b(t)) / 2


def find_mesh(ink):
    """The 10 x 10 crossings of the drawn lines in a square ink image, or None."""
    verticals = fill_missing(follow_lines(ink), ink)  # x = f(y)
    horizontals = fill_missing(follow_lines(ink.T), ink.T)  # y = g(x)
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


def find_board(bgr, corners=None, even=False):
    """A hand-drawn board in a frame up to FRAME_WIDTH wide: its 4 corners and the mesh of its
    line crossings (board coordinates), or None. With `corners`, look only there.

    A board whose lines are straight and even is left to the usual reader (it tells printed
    givens from handwritten answers), unless `even`: the usual reader already failed on it."""
    clean = clean_gray(bgr)
    thresh = threshold(clean)
    if corners is None:
        candidates = outlines(thresh)
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
        if not even and score_corners(thresh, board) >= EVEN_SCORE:
            return None  # straight, even lines: a printed grid, read by the usual path
        if grid_score(threshold(even_board(clean, board, mesh))) >= MIN_GRID_SCORE:
            return board, mesh
    return None


def find_board_in_photo(bgr, even=False):
    """find_board for a single photo, which gets one chance where live video gets many frames:
    also try slightly turned and zoomed copies, and map what is found back to the photo."""
    height, width = bgr.shape[:2]
    for degrees, scale in PHOTO_TRIES:
        turn = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, scale)
        copy = cv2.warpAffine(bgr, turn, (width, height), borderMode=cv2.BORDER_REPLICATE)
        found = find_board(copy, even=even)
        if found is not None:
            corners, mesh = found  # the mesh is in board coordinates: it stays as it is
            back = cv2.invertAffineTransform(turn)
            return cv2.transform(corners.reshape(-1, 1, 2), back).reshape(4, 2), mesh
    return None


def no_grid(bgr, corners):
    """Is there nothing like a grid inside these corners (a face, a table, blank paper)?"""
    return grid_score(threshold(warp_grid(clean_gray(bgr), np.float32(corners)))) < NO_GRID_SCORE


def read_board(bgr, corners, mesh):
    """Read a hand-drawn board, from a frame of any size (`corners` in its pixels). Every digit
    on it is part of the puzzle: all are handwritten. Returns givens and confidence."""
    flat = even_board(clean_gray(bgr), np.float32(corners), np.float32(mesh), true_shape=True)
    printed, written, confidence = read_digits(flat)
    givens = [[p or w for p, w in zip(*rows)] for rows in zip(printed, written)]
    return givens, confidence
