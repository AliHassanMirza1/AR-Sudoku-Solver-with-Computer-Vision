"""Tests for app/py on synthetic frames. Run from the repo root: python3 -m unittest discover tests"""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "app", "py"), os.path.dirname(__file__)]

import bridge  # noqa: E402
import track  # noqa: E402
from generate import GIVENS, count_solutions, generate  # noqa: E402
from sudoku_cv.solver import is_valid  # noqa: E402
from synthetic import PUZZLE, TILTED, printed_scene, scene, SIZE, CELL  # noqa: E402
from track import SurfaceTracker, transform  # noqa: E402

SHEET = np.float32([[170, 40], [470, 60], [455, 450], [150, 430]])


def rgba(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)


def gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def noisy(gray_frame, rng, sigma=4):
    """Camera sensor noise."""
    return (gray_frame + rng.normal(0, sigma, gray_frame.shape)).clip(0, 255).astype(np.uint8)


def camera_move(bgr, dx, dy, angle):
    """The same scene seen from a slightly moved camera, and the homography of that move."""
    h, w = bgr.shape[:2]
    move = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), [0, 0, 1]])
    move[:2, 2] += (dx, dy)
    return cv2.warpPerspective(bgr, move, (w, h), borderMode=cv2.BORDER_REFLECT), move


class GenerateTest(unittest.TestCase):
    def test_unique_puzzles_with_the_right_number_of_givens(self):
        for difficulty, givens in GIVENS.items():
            puzzle = generate(difficulty, seed=1)
            count = sum(d > 0 for row in puzzle["givens"] for d in row)
            self.assertLessEqual(count, givens + 4, difficulty)
            self.assertTrue(is_valid(puzzle["solution"]))
            self.assertEqual(count_solutions(puzzle["givens"]), 1)
            for r in range(9):
                for c in range(9):
                    if puzzle["givens"][r][c]:
                        self.assertEqual(puzzle["givens"][r][c], puzzle["solution"][r][c])


class TrackTest(unittest.TestCase):
    board = np.float32([[200, 140], [420, 150], [430, 360], [190, 350]])

    def test_follows_a_textured_table(self):
        frame = scene(None)
        tracker = SurfaceTracker(gray(frame), self.board)
        for step in range(1, 16):
            moved, move = camera_move(frame, 2 * step, -step, 0.5 * step)
            tracker.update(gray(moved))
        expected = transform(self.board, move)
        self.assertEqual(tracker.state, "tracking")
        self.assertLess(np.abs(tracker.corners - expected).max(), 3)

    def test_board_is_steady_when_the_camera_is_still(self):
        frame = gray(scene(None))
        tracker = SurfaceTracker(frame, self.board)
        rng = np.random.default_rng(0)
        worst = 0
        for _ in range(40):
            tracker.update(noisy(frame, rng))
            worst = max(worst, np.abs(tracker.corners - self.board).max())
        self.assertLess(worst, 0.5)

    def test_no_drift_after_moving_around(self):
        frame = scene(None)
        tracker = SurfaceTracker(gray(frame), self.board)
        rng = np.random.default_rng(0)
        for step in range(1, 61):
            t = step / 60 * 2 * np.pi  # a loop that ends where it started
            moved, _ = camera_move(frame, 30 * np.sin(t), 15 * np.sin(2 * t), 4 * np.sin(t))
            tracker.update(noisy(gray(moved), rng))
            self.assertNotEqual(tracker.state, "lost", step)
        self.assertLess(np.abs(tracker.corners - self.board).max(), 1)

    def test_keeps_a_printed_puzzle_partly_out_of_view(self):
        frame = printed_scene()
        tracker = SurfaceTracker(gray(frame), TILTED, around=False)
        for step in range(1, 31):
            moved, move = camera_move(frame, 5 * step, 2 * step, 0.3 * step)
            tracker.update(gray(moved))
            self.assertNotEqual(tracker.state, "lost", step)
            self.assertLess(np.abs(tracker.corners - transform(TILTED, move)).max(), 3, step)

    def test_a_hand_over_the_board_does_not_move_it(self):
        frame = scene(None)
        tracker = SurfaceTracker(gray(frame), self.board)
        for step in range(1, 21):
            moved, move = camera_move(frame, step, step / 2, 0.2 * step)
            cv2.ellipse(moved, (150 + 8 * step, 300), (90, 140), 20, 0, 360, (90, 120, 170), -1)
            tracker.update(gray(moved))
        self.assertNotEqual(tracker.state, "lost")
        self.assertLess(np.abs(tracker.corners - transform(self.board, move)).max(), 3)

    def test_a_few_bad_frames_do_not_lose_the_board(self):
        frame = scene(None)
        tracker = SurfaceTracker(gray(frame), self.board)
        blank = np.full_like(gray(frame), 128)
        for _ in range(track.LOST_FRAMES - 1):
            tracker.update(blank)
        self.assertNotEqual(tracker.state, "lost")
        np.testing.assert_allclose(tracker.corners, self.board)
        tracker.update(blank)
        self.assertEqual(tracker.state, "lost")

        moved, move = camera_move(frame, 30, 10, 5)
        tracker.update(gray(moved))
        self.assertEqual(tracker.state, "tracking")
        self.assertLess(np.abs(tracker.corners - transform(self.board, move)).max(), 3)

    def test_plain_surface_keeps_the_board_still(self):
        wall = np.full((480, 640), 200, np.uint8)
        tracker = SurfaceTracker(wall, self.board)
        tracker.update(wall)
        self.assertEqual(tracker.state, "plain")
        np.testing.assert_allclose(tracker.corners, self.board)

    def test_follows_a_blank_sheet(self):
        frame = scene(None, sheet_corners=SHEET)
        board = bridge.board_on_sheet(SHEET)
        tracker = SurfaceTracker(gray(frame), board)
        self.assertTrue(tracker.on_sheet)
        moved, move = camera_move(frame, 15, 5, 3)
        tracker.update(gray(moved))
        self.assertLess(np.abs(tracker.corners - transform(board, move)).max(), 4)


class BridgeTest(unittest.TestCase):
    def test_reads_a_puzzle_inside_corners_set_by_hand(self):
        result = bridge.read_at(rgba(printed_scene()), TILTED.tolist())
        self.assertEqual(result["status"], "solved")
        self.assertGreaterEqual(sum(result["givens"][r][c] == PUZZLE[r][c]
                                    for r in range(9) for c in range(9)), 78)

    def test_handle_decodes_rgba_frames(self):
        frame = rgba(printed_scene())
        h, w = frame.shape[:2]
        result = bridge.json.loads(bridge.handle("photo", "[]", frame.tobytes(), w, h))
        self.assertEqual(result["status"], "solved")

    def test_snaps_to_a_sheet(self):
        result = bridge.snap(rgba(scene(None, sheet_corners=SHEET)))
        self.assertIsNotNone(result["corners"])
        for x, y in result["corners"]:
            self.assertGreaterEqual(cv2.pointPolygonTest(SHEET, (x, y), False), 0)

    def test_reads_pen_digits_on_a_virtual_board(self):
        data = np.load(os.path.join(ROOT, "sudoku_cv", "data", "handwritten.npz"))
        flat = np.full((SIZE, SIZE), 240, np.uint8)
        written = {(0, 2): 4, (1, 1): 7, (4, 4): 5, (8, 6): 3}
        for (r, c), digit in written.items():
            sample = data["digits"][np.flatnonzero(data["labels"] == digit)[0]]
            ink = cv2.resize(sample, (36, 36)) > 100
            cell = flat[r * CELL + 7:r * CELL + 43, c * CELL + 7:c * CELL + 43]
            cell[ink] = 30

        board = bridge.board_on_sheet(SHEET)
        frame = rgba(scene(board, flat, sheet_corners=SHEET))
        bridge.pin(frame, board.tolist())
        givens = [[0] * 9 for _ in range(9)]
        ink = None
        for _ in range(bridge.READ_STILL_FRAMES + 1):
            ink = bridge.follow(frame, givens)["ink"] or ink
        self.assertIsNotNone(ink)
        found = {(r, c): ink[r][c] for r in range(9) for c in range(9) if ink[r][c]}
        self.assertEqual(found, written)


if __name__ == "__main__":
    unittest.main()
