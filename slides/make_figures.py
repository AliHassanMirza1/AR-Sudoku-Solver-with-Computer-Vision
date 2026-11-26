import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from evaluate import load_image
from sudoku_cv import process_image
from sudoku_cv.detect import SQUARE, find_grid_contour, quad_from_hull, threshold, warp_grid
from sudoku_cv.tracker import upright_grid

PHOTOS = os.path.join(os.path.dirname(__file__), "..", "paper", "photos")
OUT = os.path.join(os.path.dirname(__file__), "figures")
CELL = SQUARE // 9
BLUE = (200, 90, 20)
RED = (40, 40, 220)
GREEN = (60, 170, 60)
MARGIN = 50

with open(os.path.join(PHOTOS, "labels.json")) as f:
    LABELS = json.load(f)


def photo(name):
    return load_image(os.path.join(PHOTOS, f"IMG_{name}.jpg"))


def crop_grid(img, name, margin=MARGIN, corners=None):
    if corners is None:
        corners = LABELS[f"IMG_{name}.jpg"]["corners"]
    corners = np.float32(corners)
    x0, y0 = np.int32(np.min(corners, axis=0)) - margin
    x1, y1 = np.int32(np.max(corners, axis=0)) + margin
    return img[max(y0, 0):y1, max(x0, 0):x1]


def save(name, img):
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if name.endswith(".jpg") else []
    cv2.imwrite(os.path.join(OUT, name), np.uint8(img), params)


def put_digit(img, text, r, c, color, scale=1.1, thickness=2, corner=False):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = c * CELL + (CELL - w) // 2
    y = r * CELL + (CELL + h) // 2
    if corner:
        x, y = c * CELL + 4, r * CELL + h + 4
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_on_paper(frame, homography, cells):
    board = np.zeros((SQUARE, SQUARE, 3), np.uint8)
    mask = np.zeros((SQUARE, SQUARE), np.uint8)
    for r, c, text, color in cells:
        if text == "box":
            top_left = (c * CELL + 3, r * CELL + 3)
            bottom_right = ((c + 1) * CELL - 3, (r + 1) * CELL - 3)
            cv2.rectangle(board, top_left, bottom_right, color, 4)
            cv2.rectangle(mask, top_left, bottom_right, 255, 4)
        elif text.startswith("#"):
            put_digit(board, text[1:], r, c, color, 0.7, 2, corner=True)
            put_digit(mask, text[1:], r, c, 255, 0.7, 2, corner=True)
        else:
            put_digit(board, text, r, c, color)
            put_digit(mask, text, r, c, 255)
    size = (frame.shape[1], frame.shape[0])
    board = cv2.warpPerspective(board, np.float64(homography), size)
    mask = cv2.warpPerspective(mask, np.float64(homography), size)
    out = frame.copy()
    out[mask > 64] = board[mask > 64]
    return out


def solution_cells(result):
    return [(r, c, str(result["solution"][r][c]), BLUE) for r, c in result["order"]]


# goal

def goal():
    frame = photo("0386")
    result = process_image(frame)
    out = draw_on_paper(frame, result["homography"], solution_cells(result))
    save("goal.jpg", crop_grid(out, "0386"))


# finding the grid

def grid_steps():
    frame = photo("0387")
    thresh = threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    corners = find_grid_contour(thresh)
    contour = frame.copy()
    cv2.polylines(contour, [np.int32(corners)], True, GREEN, 3)
    save("grid_threshold.jpg", crop_grid(thresh, "0387"))
    save("grid_contour.jpg", crop_grid(contour, "0387"))
    flat = warp_grid(frame, corners)
    lines = flat.copy()
    for i in range(10):
        cv2.line(lines, (i * CELL, 0), (i * CELL, SQUARE), GREEN, 2)
        cv2.line(lines, (0, i * CELL), (SQUARE, i * CELL), GREEN, 2)
    save("grid_lines.jpg", lines)


def merged_contour():
    frame = photo("0399")
    thresh = threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    out = frame.copy()
    cv2.drawContours(out, [contour], -1, RED, 2)
    save("merged_contour.jpg", crop_grid(out, "0399"))
    out = frame.copy()
    cv2.polylines(out, [np.int32(quad_from_hull(contour))], True, RED, 2)
    cv2.polylines(out, [np.int32(find_grid_contour(thresh))], True, GREEN, 2)
    save("merged_refined.jpg", crop_grid(out, "0399"))


# digits and solving

def empty_grid():
    grid = np.full((SQUARE, SQUARE, 3), 255, np.uint8)
    for i in range(10):
        width = 3 if i % 3 == 0 else 1
        cv2.line(grid, (i * CELL, 0), (i * CELL, SQUARE), (0, 0, 0), width)
        cv2.line(grid, (0, i * CELL), (SQUARE, i * CELL), (0, 0, 0), width)
    return grid


def digits():
    frame = photo("0382")
    result = process_image(frame)
    save("digits_flat.jpg", warp_grid(frame, np.float32(result["corners"])))
    read = empty_grid()
    for r in range(9):
        for c in range(9):
            if result["givens"][r][c]:
                put_digit(read, str(result["givens"][r][c]), r, c, BLUE)
            if result["user_digits"][r][c]:
                put_digit(read, str(result["user_digits"][r][c]), r, c, RED)
    save("digits_read.jpg", read)


def templates():
    path = os.path.join(os.path.dirname(__file__), "..", "sudoku_cv", "data", "templates.npz")
    data = np.load(path)
    rows = []
    for digit in [2, 4, 7]:
        fonts = data["digits"][data["labels"] == digit]
        picks = fonts[np.linspace(0, len(fonts) - 1, 12).astype(int)]
        rows.append(np.hstack([np.pad(255 - t, 2, constant_values=255) for t in picks]))
    out = np.vstack(rows)
    save("templates.png", cv2.resize(out, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))


def mistakes():
    frame = photo("0382")
    result = process_image(frame)
    cells = [(r, c, "box", RED) for r, c in result["wrong"]]
    out = draw_on_paper(frame, result["homography"], cells)
    save("mistakes.jpg", crop_grid(out, "0382", 10))


def reveal_order():
    frame = photo("0373")
    result = process_image(frame)
    cells = []
    for i, (r, c) in enumerate(result["order"][:6]):
        cells += [(r, c, str(result["solution"][r][c]), BLUE), (r, c, f"#{i + 1}", RED)]
    out = draw_on_paper(frame, result["homography"], cells)
    save("order.jpg", crop_grid(out, "0373", 10))


# what works and what does not

def examples():
    # bent paper and covered border have no labelled corners: the found ones, or a box by hand
    boxes = {"0402": None, "0397": [[110, 165], [530, 585]]}
    for name in ["0383", "0388", "0392", "0416", "0402", "0397"]:
        frame = photo(name)
        result = process_image(frame)
        if result["homography"] is not None and result["solution"]:
            frame = draw_on_paper(frame, result["homography"], solution_cells(result))
        corners = boxes.get(name, LABELS[f"IMG_{name}.jpg"]["corners"]) or result["corners"]
        save(f"example_{name}.jpg", crop_grid(frame, name, corners=corners))
        print(f"{name}: {result['status']}")


def main():
    os.makedirs(OUT, exist_ok=True)
    goal()
    grid_steps()
    merged_contour()
    digits()
    templates()
    mistakes()
    reveal_order()
    examples()


if __name__ == "__main__":
    main()
