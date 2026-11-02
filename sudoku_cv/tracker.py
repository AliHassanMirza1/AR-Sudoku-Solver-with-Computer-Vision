import cv2
import numpy as np

from .detect import SQUARE, find_grid_contour, get_homography, threshold, to_bgr, warp_grid
from .digits import find_rotation, read_digits
from .solver import copy_grid, solve

REFERENCE_WIDTH = 640
STABLE_FRAMES = 5
STABLE_PIXELS = 4
JUMP_PIXELS = 20
SMOOTHING = 0.5
LOST_FRAMES = 10
REREAD_FRAMES = 30
NEW_PUZZLE_CELLS = 5


# results

def debug_images(frame, thresh, corners):
    images = {"threshold": thresh}
    if corners is not None:
        contour = frame.copy()
        cv2.polylines(contour, [np.int32(corners)], True, (0, 255, 0), 3)
        images["contour"] = contour
        images["flat"] = warp_grid(frame, corners)
    return {name: cv2.imencode(".png", img)[1].tobytes() for name, img in images.items()}


def make_result(corners, givens=None, user_digits=None, confidence=None, solved=None, debug=None):
    result = {"status": "no_grid", "corners": None, "homography": None, "givens": None,
              "user_digits": None, "confidence": None, "solution": None, "order": [],
              "wrong": [], "debug": debug}
    if corners is None:
        return result

    result["status"] = "reading"
    result["corners"] = np.float32(corners).tolist()
    result["homography"] = get_homography(corners).tolist()
    if givens is None:
        return result

    result["givens"] = copy_grid(givens)
    result["user_digits"] = copy_grid(user_digits)
    result["confidence"] = [list(row) for row in confidence]
    result.update(solved)
    return result


def count_differences(grid1, grid2):
    return sum(grid1[r][c] != grid2[r][c] for r in range(9) for c in range(9))


def upright_grid(gray, corners):
    flat = warp_grid(gray, corners)
    k = find_rotation(flat)
    if k:
        corners = np.roll(corners, -k, axis=0)
        flat = warp_grid(gray, corners)
    return corners, flat, k


# single photo

def process_image(image, debug=False):
    image = to_bgr(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = threshold(gray)
    corners = find_grid_contour(thresh)
    if corners is None:
        return make_result(None, debug=debug_images(image, thresh, None) if debug else None)

    corners, flat, _ = upright_grid(gray, corners)
    debug = debug_images(image, thresh, corners) if debug else None
    givens, user_digits, confidence = read_digits(flat)
    return make_result(corners, givens, user_digits, confidence, solve(givens, user_digits), debug)


# live video

class SudokuTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.corners = None
        self.rotation = 0
        self.stable_count = 0
        self.lost_count = 0
        self.frames_since_read = 0
        self.givens = None
        self.fixed_givens = None
        self.user_digits = None
        self.confidence = None
        self.solved = None

    def process_frame(self, frame, debug=False):
        frame = to_bgr(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = threshold(gray)
        corners = find_grid_contour(thresh)
        scale = max(gray.shape) / REFERENCE_WIDTH

        if corners is None:
            self.lost_count += 1
            self.stable_count = 0
            if self.lost_count > LOST_FRAMES:
                self.reset()
        else:
            corners = np.roll(corners, -self.rotation, axis=0)
            self.lost_count = 0
            self.update_corners(corners, scale)
            self.frames_since_read += 1
            if self.stable_count >= STABLE_FRAMES and (
                    self.givens is None or self.frames_since_read >= REREAD_FRAMES):
                self.read(gray, corners)

        debug = debug_images(frame, thresh, self.corners) if debug else None
        return self.result(debug)

    def update_corners(self, corners, scale):
        if self.corners is None:
            self.corners = corners
            return
        move = np.abs(corners - self.corners).max() / scale
        if move > JUMP_PIXELS:
            self.corners = corners
        else:
            self.corners = SMOOTHING * corners + (1 - SMOOTHING) * self.corners
        self.stable_count = self.stable_count + 1 if move < STABLE_PIXELS else 0

    def read(self, gray, corners):
        _, flat, k = upright_grid(gray, corners)
        if k:
            self.rotation = (self.rotation + k) % 4
            self.corners = np.roll(self.corners, -k, axis=0)
        givens, user_digits, confidence = read_digits(flat)
        self.frames_since_read = 0
        if self.fixed_givens is not None:
            givens = self.fixed_givens
        elif (self.givens is not None and self.solved["status"] == "solved"
              and count_differences(givens, self.givens) < NEW_PUZZLE_CELLS):
            givens = self.givens
        for r in range(9):
            for c in range(9):
                if givens[r][c]:
                    user_digits[r][c] = 0

        if self.givens != givens or self.user_digits != user_digits:
            self.solved = solve(givens, user_digits)
        self.givens, self.user_digits, self.confidence = givens, user_digits, confidence

    def set_givens(self, givens):
        self.fixed_givens = copy_grid(givens)
        self.givens = copy_grid(givens)
        if self.user_digits is None:
            self.user_digits = [[0] * 9 for _ in range(9)]
            self.confidence = [[1.0] * 9 for _ in range(9)]
        self.solved = solve(self.givens, self.user_digits)
        return self.result()

    def result(self, debug=None):
        if self.givens is None:
            return make_result(self.corners, debug=debug)
        return make_result(self.corners, self.givens, self.user_digits, self.confidence,
                           self.solved, debug)
