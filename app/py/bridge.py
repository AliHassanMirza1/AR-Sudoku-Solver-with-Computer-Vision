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
from sudoku_cv.detect import grid_score, threshold, warp_grid
from sudoku_cv.digits import cell_masks, get_data, hog_features, knn_classify, normalize_digit, read_digits
from sudoku_cv.tracker import find_drawn, make_result, read_drawn

from generate import generate as generate_puzzle
from track import SurfaceTracker, board_on_sheet, find_sheet

READ_STILL_FRAMES = 5  # pen digits are read once the board has been still this long,
READ_EVERY_FRAMES = 15  # and again every 15 frames while it stays still (reading takes ~25 ms)
HANDDRAWN_AFTER = 10  # frames without a solved puzzle before looking for a hand-drawn board,
HANDDRAWN_EVERY = 3  # then every 3rd frame (it takes ~70 ms)
REREAD_EVERY = 10  # a pinned hand-drawn board that reads as unsolvable is read again this often

paper_tracker = SudokuTracker()
surface = None
misses = 0  # frames in a row without a solved puzzle
drawn = None  # the pinned hand-drawn board: its mesh, whether it is solved, frames since read


def gray_of(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)


# paper puzzle (mode A)

def paper_frame(frame):
    """Live camera: find, read and solve the puzzle (Leonardo's tracker). When it solves nothing
    for a while (no grid, or a hand-drawn grid it cannot read), look for a hand-drawn board now
    and then; once found, the surface tracker follows it and the result says pinned=True."""
    global misses
    result = paper_tracker.process_frame(frame)
    misses = misses + 1 if result["status"] != "solved" else 0
    if misses >= HANDDRAWN_AFTER and (misses - HANDDRAWN_AFTER) % HANDDRAWN_EVERY == 0:
        found = find_drawn(frame, even=result["status"] == "no_solution")
        if found is not None:
            pin(frame, found["corners"], around=False)
            remember_drawn(found)
            found["pinned"] = True
            return found
    return result


def remember_drawn(result):
    global drawn
    drawn = {"mesh": result["mesh"], "solved": result["status"] == "solved", "frames": 0}


def photo(frame):
    """A single photo: find, read and solve the puzzle, printed or hand-drawn."""
    return process_image(frame)


def read_at(frame, corners):
    """Read and solve the puzzle inside 4 corners the user placed by hand. A grid that is not
    straight and even there is hand-drawn: its own lines are used if they can be traced, else
    straight cells, and every digit on it is a given (none is taken for the user's answer).
    Call pin first when the camera is live, so an unsolvable read is retried (see follow)."""
    corners = np.float32(corners)
    flat = warp_grid(gray_of(frame), corners)
    if grid_score(threshold(flat)) < handdrawn.EVEN_SCORE:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        found = handdrawn.find_board(bgr, corners, even=True)
        if found is None and handdrawn.no_grid(bgr, corners):
            return make_result(None)  # status no_grid: nothing to read here
        result = read_drawn(frame, *(found or (corners, handdrawn.EVEN_MESH)))
        if surface is not None:
            remember_drawn(result)  # pinned: read it again as it is followed, until it solves
        return result
    givens, user_digits, confidence = read_digits(flat)
    return make_result(corners, givens, user_digits, confidence, solve(givens, user_digits))


def set_givens(givens, user_digits):
    """The user fixed misread printed digits: keep them for the live tracker and solve again."""
    paper_tracker.set_givens(givens)
    return solve(givens, user_digits)


def reset():
    global surface, misses, drawn
    paper_tracker.reset()
    surface = None
    misses = 0
    drawn = None
    return {}


# virtual board (mode B) and paper puzzles placed by hand

def snap(frame):
    """Board corners on a sheet of paper in view, or None."""
    sheet = find_sheet(gray_of(frame))
    return {"corners": None if sheet is None else board_on_sheet(sheet).tolist()}


def pin(frame, corners, around=True):
    """Keep the board at these corners on the surface from now on. around=False for a printed
    puzzle: follow only the paper, not what is around it."""
    global surface, drawn
    surface = SurfaceTracker(gray_of(frame), corners, around)
    drawn = None
    return board_state()


def follow(frame, givens=None):
    """Move the pinned board with the camera. With `givens`, also read pen digits when still.
    A pinned hand-drawn board that read as unsolvable is read again now and then (`paper`)."""
    gray = gray_of(frame)
    surface.update(gray)
    ink = paper = None
    still = surface.still_count - READ_STILL_FRAMES
    if (givens is not None and surface.on_sheet and surface.state == "tracking"
            and still >= 0 and still % READ_EVERY_FRAMES == 0):
        ink = read_handwritten(gray, surface.corners, givens)
    if drawn is not None and not drawn["solved"] and surface.state == "tracking":
        drawn["frames"] += 1
        if drawn["frames"] % REREAD_EVERY == 0:
            paper = read_drawn(frame, surface.corners, drawn["mesh"])
            drawn["solved"] = paper["status"] == "solved"
    return board_state(ink, paper)


def board_state(ink=None, paper=None):
    return {"track": surface.state, "corners": surface.corners.tolist(),
            "on_sheet": surface.on_sheet, "ink": ink, "paper": paper}


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
