import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sudoku_cv import SudokuTracker

WIDTH = 640


def main():
    video = cv2.VideoCapture(sys.argv[1])
    tracker = SudokuTracker()
    statuses, corners, times = [], [], []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        scale = WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (WIDTH, round(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        t0 = time.time()
        result = tracker.process_frame(frame)
        times.append(time.time() - t0)
        statuses.append(result["status"])
        corners.append(None if result["corners"] is None else np.float32(result["corners"]))

    n = len(statuses)
    found = sum(s != "no_grid" for s in statuses)
    first_solved = statuses.index("solved") if "solved" in statuses else None
    moves = [np.abs(a - b).mean() for a, b in zip(corners, corners[1:])
             if a is not None and b is not None]
    print(f"frames: {n}, grid found: {found}/{n}, first solved at frame {first_solved}")
    print(f"per frame: {1000 * np.mean(times):.1f} ms ({1 / np.mean(times):.0f} fps)")
    print(f"corner motion per frame: mean {np.mean(moves):.2f} px, "
          f"median {np.median(moves):.2f} px")


if __name__ == "__main__":
    main()
