import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sudoku_cv.detect import find_grid_contour, find_grid_hough, threshold
from sudoku_cv.digits import read_digits
from sudoku_cv.solver import solve
from sudoku_cv.tracker import upright_grid

WIDTH = 640
CORNER_TOLERANCE = 8
EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_image(path):
    img = cv2.imread(path)
    scale = WIDTH / img.shape[1]
    return cv2.resize(img, (WIDTH, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def draw_reading(flat, givens, user_digits):
    out = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
    for r in range(9):
        for c in range(9):
            for grid, color in [(givens, (255, 0, 0)), (user_digits, (0, 0, 255))]:
                if grid[r][c]:
                    cv2.putText(out, str(grid[r][c]), (c * 50 + 2, r * 50 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


# labelling

def label(folder):
    labels_path = os.path.join(folder, "labels.json")
    labels = json.load(open(labels_path)) if os.path.exists(labels_path) else {}
    os.makedirs(os.path.join(folder, "check"), exist_ok=True)
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(EXTENSIONS) or name in labels:
            continue
        img = load_image(os.path.join(folder, name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_grid_contour(threshold(gray))
        if corners is None:
            labels[name] = {"corners": None, "givens": None, "user_digits": None}
            print(f"{name}: no grid found, add the corners by hand")
            continue
        _, flat, _ = upright_grid(gray, corners)
        givens, user_digits, _ = read_digits(flat)
        labels[name] = {"corners": np.round(corners).tolist(), "givens": givens,
                        "user_digits": user_digits}
        check = img.copy()
        cv2.polylines(check, [np.int32(corners)], True, (0, 255, 0), 2)
        flat = draw_reading(flat, givens, user_digits)
        check = cv2.resize(check, (450 * check.shape[1] // check.shape[0], 450))
        cv2.imwrite(os.path.join(folder, "check", name + ".png"), np.hstack([check, flat]))
        print(f"{name}: prefilled, check it")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=1)


# evaluation

def evaluate(folder):
    labels = json.load(open(os.path.join(folder, "labels.json")))
    counts = {"images": 0, "read": 0, "contour": 0, "hough": 0, "printed": 0, "printed_ok": 0,
              "handwritten": 0, "handwritten_ok": 0, "as_printed": 0, "wrong": 0,
              "wrong_caught": 0, "solved": 0}
    times = []
    by_condition = {}
    for name, true in labels.items():
        condition = by_condition.setdefault(true.get("condition", "-"), [0, 0, 0])
        condition[0] += 1
        counts["images"] += 1
        if true["corners"] is None:
            continue
        img = load_image(os.path.join(folder, name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        true_corners = np.float32(true["corners"])

        t0 = time.time()
        corners = find_grid_contour(threshold(gray))
        times.append(time.time() - t0)
        if corners is not None and np.abs(corners - true_corners).max() < CORNER_TOLERANCE:
            counts["contour"] += 1
            condition[1] += 1
        hough = find_grid_hough(gray)
        if hough is not None and np.abs(hough - true_corners).max() < CORNER_TOLERANCE:
            counts["hough"] += 1
        if corners is None or true["givens"] is None:
            continue

        counts["read"] += 1
        givens, user_digits, _ = read_digits(upright_grid(gray, corners)[1])
        true_wrong = solve(true["givens"], true["user_digits"])["wrong"]
        result = solve(givens, user_digits)
        solved = givens == true["givens"] and result["status"] == "solved"
        counts["solved"] += solved
        condition[2] += solved
        counts["wrong"] += len(true_wrong)
        counts["wrong_caught"] += sum(cell in result["wrong"] for cell in true_wrong)
        for r in range(9):
            for c in range(9):
                if true["givens"][r][c]:
                    counts["printed"] += 1
                    counts["printed_ok"] += givens[r][c] == true["givens"][r][c]
                if true["user_digits"][r][c]:
                    counts["handwritten"] += 1
                    counts["handwritten_ok"] += user_digits[r][c] == true["user_digits"][r][c]
                    counts["as_printed"] += givens[r][c] != 0

    n = counts["images"]
    print(f"images: {n}")
    print(f"grid found, contour: {counts['contour']}/{n}, hough: {counts['hough']}/{n}")
    print(f"printed digits: {counts['printed_ok']}/{counts['printed']}")
    print(f"handwritten digits: {counts['handwritten_ok']}/{counts['handwritten']}, "
          f"read as printed: {counts['as_printed']}")
    print(f"wrong handwritten caught: {counts['wrong_caught']}/{counts['wrong']}")
    print(f"fully solved: {counts['solved']}/{counts['read']}")
    print(f"contour detection: {1000 * np.mean(times):.1f} ms")
    for condition, (n, found, solved) in by_condition.items():
        print(f"  {condition}: grid {found}/{n}, solved {solved}/{n}")


def main():
    mode, folder = sys.argv[1], sys.argv[2]
    if mode == "label":
        label(folder)
    else:
        evaluate(folder)


if __name__ == "__main__":
    main()
