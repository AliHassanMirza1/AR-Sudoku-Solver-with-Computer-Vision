"""The website's interface to sudoku_cv, called from worker.js.

worker.js calls `handle(name, args_json, pixels, width, height)`. If pixels are given they are
turned into an RGBA frame and passed as the first argument of the function `name`. Every function
returns a dict of plain lists and numbers, which `handle` sends back as JSON.

Frame coordinates are pixels of the frame the page sent (at most 640 px wide).
"""

import json

import cv2
import numpy as np

from sudoku_cv import SudokuTracker, handdrawn, process_image, solve
from sudoku_cv.detect import GRID_SCORE_THRESHOLD, grid_score, threshold, warp_grid
from sudoku_cv.digits import cell_masks, get_data, hog_features, knn_classify, normalize_digit, read_digits
from sudoku_cv.tracker import find_drawn, make_result, read_drawn

from generate import generate as generate_puzzle
from track import SurfaceTracker, board_on_sheet, find_sheet

READ_STILL_FRAMES = 5  # pen digits are read once the board has been still this long,
READ_EVERY_FRAMES = 15  # and again every 15 frames while it stays still (reading takes ~25 ms)
HANDDRAWN_EVERY = 5  # while no grid is found, look for a hand-drawn one every 5th frame (~30 ms)

paper_tracker = SudokuTracker()
surface = None
misses = 0  # frames in a row without a grid


def gray_of(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)


# paper puzzle (mode A)

def paper_frame(frame):
    """Live camera: find, read and solve the puzzle (Leonardo's tracker). When it finds no grid,
    look for a hand-drawn one now and then; once found, the surface tracker follows it and the
    result says pinned=True."""
    global misses
    result = paper_tracker.process_frame(frame)
    misses = misses + 1 if result["status"] == "no_grid" else 0
    if misses and misses % HANDDRAWN_EVERY == 0:
        drawn = find_drawn(frame)
        if drawn is not None:
            pin(frame, drawn["corners"], around=False)
            drawn["pinned"] = True
            return drawn
    return result


def photo(frame):
    """A single photo: find, read and solve the puzzle, printed or hand-drawn."""
    return process_image(frame)


def read_at(frame, corners):
    """Read and solve the puzzle inside 4 corners the user placed by hand. If the grid there
    is not even (hand-drawn), its own lines are found and used."""
    corners = np.float32(corners)
    flat = warp_grid(gray_of(frame), corners)
    if grid_score(threshold(flat)) < GRID_SCORE_THRESHOLD:
        found = handdrawn.find_board(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR), corners)
        if found is not None:
            return read_drawn(frame, *found)
    givens, user_digits, confidence = read_digits(flat)
    return make_result(corners, givens, user_digits, confidence, solve(givens, user_digits))


def set_givens(givens, user_digits):
    """The user fixed misread printed digits: keep them for the live tracker and solve again."""
    paper_tracker.set_givens(givens)
    return solve(givens, user_digits)


def reset():
    global surface, misses
    paper_tracker.reset()
    surface = None
    misses = 0
    return {}


# virtual board (mode B) and paper puzzles placed by hand

def snap(frame):
    """Board corners on a sheet of paper in view, or None."""
    sheet = find_sheet(gray_of(frame))
    return {"corners": None if sheet is None else board_on_sheet(sheet).tolist()}


def pin(frame, corners, around=True):
    """Keep the board at these corners on the surface from now on. around=False for a printed
    puzzle: follow only the paper, not what is around it."""
    global surface
    surface = SurfaceTracker(gray_of(frame), corners, around)
    return board_state()


def follow(frame, givens=None):
    """Move the pinned board with the camera. With `givens`, also read pen digits when still."""
    gray = gray_of(frame)
    surface.update(gray)
    ink = None
    still = surface.still_count - READ_STILL_FRAMES
    if (givens is not None and surface.on_sheet and surface.state == "tracking"
            and still >= 0 and still % READ_EVERY_FRAMES == 0):
        ink = read_handwritten(gray, surface.corners, givens)
    return board_state(ink)


def board_state(ink=None):
    return {"track": surface.state, "corners": surface.corners.tolist(),
            "on_sheet": surface.on_sheet, "ink": ink}


def read_handwritten(gray, corners, givens):
    """Digits written with a pen in the empty cells of a board drawn only on the screen."""
    _, _, train_features, train_labels = get_data()
    digits, cells = [], []
    for r, c, mask, _ in cell_masks(warp_grid(gray, np.float32(corners))):
        if not givens[r][c]:
            digits.append(normalize_digit(mask))
            cells.append((r, c))
    ink = [[0] * 9 for _ in range(9)]
    if digits:
        for (r, c), (label, _) in zip(cells, knn_classify(hog_features(np.array(digits)),
                                                          train_features, train_labels)):
            ink[r][c] = label
    return ink


# puzzles

def generate(difficulty):
    return generate_puzzle(difficulty)


def solve_grid(givens, user_digits=None):
    return solve(givens, user_digits)


API = {f.__name__: f for f in [paper_frame, photo, read_at, set_givens, reset, snap, pin, follow,
                                 generate, solve_grid]}


def handle(name, args_json, pixels=None, width=0, height=0):
    args = json.loads(args_json)
    if pixels is not None:
        data = pixels.to_bytes() if hasattr(pixels, "to_bytes") else pixels
        args.insert(0, np.frombuffer(data, np.uint8).reshape(height, width, 4))
    return json.dumps(API[name](*args))
