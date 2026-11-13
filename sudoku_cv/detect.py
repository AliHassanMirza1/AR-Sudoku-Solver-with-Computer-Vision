import math

import cv2
import numpy as np

SQUARE = 450
CELL = SQUARE // 9
BLOCK_SIZE = 15
THRESHOLD_C = 5
MAX_CANDIDATES = 10
MIN_AREA_RATIO = 0.05
GRID_SCORE_THRESHOLD = 0.6
LINE_BAND = 4
HOUGH_THRESHOLD_RATIO = 0.2
MERGE_PIXELS = 8
SPACING_TOLERANCE = 0.2
MIN_INNER_LINES = 5
MAX_PAIRS = 3
HULL_EPSILONS = [0.02, 0.04, 0.06, 0.08]
REFINE_THRESHOLD = 0.2
REFINE_SIZE = 225
REFINE_STEPS = [32, 16, 8, 4, 2, 1]
TRACK_STEPS = [2, 1]


# thresholding

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def threshold(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, BLOCK_SIZE, THRESHOLD_C)


# rectification

def order_corners(pts):
    pts = np.float32(pts).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def get_homography(corners):
    square = np.float32([[0, 0], [SQUARE, 0], [SQUARE, SQUARE], [0, SQUARE]])
    return cv2.getPerspectiveTransform(square, np.float32(corners))


def warp_grid(img, corners):
    homography = get_homography(corners)
    return cv2.warpPerspective(img, homography, (SQUARE, SQUARE),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def grid_score(flat_thresh):
    size = flat_thresh.shape[0]
    band = max(1, round(LINE_BAND * size / SQUARE))
    rows, cols = [], []
    for k in range(10):
        pos = min(round(k * size / 9), size - 1)
        lo, hi = max(pos - band, 0), min(pos + band + 1, size)
        rows.append((flat_thresh[lo:hi, :] > 0).mean(axis=1).max())
        cols.append((flat_thresh[:, lo:hi] > 0).mean(axis=0).max())
    return float(min(np.mean(rows), np.mean(cols)))


def score_corners(thresh, corners, size=SQUARE):
    square = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    homography = cv2.getPerspectiveTransform(square, np.float32(corners))
    flat = cv2.warpPerspective(thresh, homography, (size, size),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    return grid_score(flat)


# grid detection with contours

def quad_from_hull(contour):
    hull = cv2.convexHull(contour)
    for epsilon in HULL_EPSILONS:
        approx = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return order_corners(approx)
    return None


def refine_corners(thresh, corners, steps=REFINE_STEPS):
    best = score_corners(thresh, corners, REFINE_SIZE)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                    trial = corners.copy()
                    trial[i] += (dx, dy)
                    score = score_corners(thresh, trial, REFINE_SIZE)
                    if score > best:
                        corners, best, improved = trial, score, True
    return corners


def find_grid_contour(thresh, previous=None):
    height, width = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CANDIDATES]
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA_RATIO * height * width:
            break
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        merged = len(approx) != 4 or not cv2.isContourConvex(approx)
        corners = quad_from_hull(contour) if merged else order_corners(approx)
        if corners is None:
            continue
        score = score_corners(thresh, corners)
        if score > GRID_SCORE_THRESHOLD and not merged:
            return corners
        if score > REFINE_THRESHOLD:
            if previous is not None:
                corners = refine_corners(thresh, previous.copy(), TRACK_STEPS)
            else:
                corners = refine_corners(thresh, corners)
            if score_corners(thresh, corners) > GRID_SCORE_THRESHOLD:
                return corners
    return None


# grid detection with hough lines

def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a = np.array([[math.cos(theta1), math.sin(theta1)], [math.cos(theta2), math.sin(theta2)]])
    x, y = np.linalg.solve(a, [rho1, rho2])
    return [x, y]


def merge_close_lines(lines):
    merged = []
    for pos, line in sorted(lines):
        if not merged or pos - merged[-1][0] > MERGE_PIXELS:
            merged.append((pos, line))
    return merged


def outer_line_pairs(lines, min_size):
    lines = merge_close_lines(lines)
    pairs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            first, last = lines[i][0], lines[j][0]
            spacing = (last - first) / 9
            if last - first < min_size:
                continue
            matches = 0
            for k in range(1, 9):
                expected = first + k * spacing
                if any(abs(pos - expected) < SPACING_TOLERANCE * spacing for pos, _ in lines):
                    matches += 1
            if matches >= MIN_INNER_LINES:
                pairs.append((matches, last - first, lines[i][1], lines[j][1]))
    pairs.sort(key=lambda pair: pair[:2], reverse=True)
    return [(first, last) for _, _, first, last in pairs[:MAX_PAIRS]]


def find_grid_hough(gray):
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(HOUGH_THRESHOLD_RATIO * min(height, width)))
    if lines is None:
        return None

    horizontal, vertical = [], []
    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.pi / 2) < np.pi / 6:
            y = (rho - width / 2 * math.cos(theta)) / math.sin(theta)
            horizontal.append((y, (rho, theta)))
        elif theta < np.pi / 6 or theta > 5 * np.pi / 6:
            x = (rho - height / 2 * math.sin(theta)) / math.cos(theta)
            vertical.append((x, (rho, theta)))

    thresh = threshold(gray)
    best, best_score = None, GRID_SCORE_THRESHOLD
    for top, bottom in outer_line_pairs(horizontal, MIN_AREA_RATIO ** 0.5 * height):
        for left, right in outer_line_pairs(vertical, MIN_AREA_RATIO ** 0.5 * width):
            corners = np.float32([line_intersection(top, left),
                                  line_intersection(top, right),
                                  line_intersection(bottom, right),
                                  line_intersection(bottom, left)])
            score = grid_score(warp_grid(thresh, corners))
            if score > best_score:
                best, best_score = corners, score
    return best
