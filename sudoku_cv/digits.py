import os

import cv2
import numpy as np

from .detect import CELL

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BLOCK_SIZE = 25
THRESHOLD_C = 10
MARGIN = 5
MIN_COMPONENT_AREA = 8
MIN_DIGIT_AREA = 40
MIN_DIGIT_HEIGHT = 0.35
DIGIT_SIZE = 28
DIGIT_BOX = 20
CENTER_RATIO = 0.25
PRINTED_THRESHOLD = 0.75
STRONG_THRESHOLD = 0.85
MIN_REFERENCES = 5
CONTRAST_RATIO = 0.8
HEIGHT_TOLERANCE = 0.15
HOG_CELL = 7
HOG_BINS = 9
K = 5


# cells

def threshold_cells(flat_gray):
    blur = cv2.GaussianBlur(flat_gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, BLOCK_SIZE, THRESHOLD_C)


def extract_digit(cell_thresh):
    size = cell_thresh.shape[0]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cell_thresh)
    mask = np.zeros_like(cell_thresh)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w == size or y + h == size
        centered = (abs(x + w / 2 - size / 2) < CENTER_RATIO * size
                    and abs(y + h / 2 - size / 2) < CENTER_RATIO * size)
        if area >= MIN_COMPONENT_AREA and (centered or not touches_border):
            mask[labels == i] = 255
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_DIGIT_AREA or ys.max() - ys.min() + 1 < MIN_DIGIT_HEIGHT * size:
        return None
    return mask


def normalize_digit(mask):
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)
    h, w = crop.shape
    digit = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.float32)
    y0, x0 = (DIGIT_SIZE - h) // 2, (DIGIT_SIZE - w) // 2
    digit[y0:y0 + h, x0:x0 + w] = crop

    m = cv2.moments(digit)
    shift = np.float32([[1, 0, DIGIT_SIZE / 2 - m["m10"] / m["m00"]],
                        [0, 1, DIGIT_SIZE / 2 - m["m01"] / m["m00"]]])
    digit = cv2.warpAffine(digit, shift, (DIGIT_SIZE, DIGIT_SIZE))
    return digit / digit.max()


# printed digits: template matching

def zero_mean_unit(vectors):
    vectors = vectors - vectors.mean(axis=1, keepdims=True)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6)


def prepare_templates(digits):
    blurred = np.array([cv2.GaussianBlur(d, (3, 3), 0) for d in digits])
    return zero_mean_unit(blurred.reshape(len(digits), -1))


def match_templates(digit, templates, template_labels):
    vector = zero_mean_unit(cv2.GaussianBlur(digit, (3, 3), 0).reshape(1, -1))[0]
    scores = templates @ vector
    best = np.argmax(scores)
    return int(template_labels[best]), float(scores[best])


# handwritten digits: hog + k-nearest neighbours

def hog_features(digits):
    n = len(digits)
    gx = np.zeros_like(digits)
    gy = np.zeros_like(digits)
    gx[:, :, 1:-1] = digits[:, :, 2:] - digits[:, :, :-2]
    gy[:, 1:-1, :] = digits[:, 2:, :] - digits[:, :-2, :]
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) % np.pi
    bins = np.minimum((angle / np.pi * HOG_BINS).astype(int), HOG_BINS - 1)

    cells = DIGIT_SIZE // HOG_CELL
    hist = np.zeros((n, cells, cells, HOG_BINS), np.float32)
    for b in range(HOG_BINS):
        m = np.where(bins == b, magnitude, 0)
        hist[..., b] = m.reshape(n, cells, HOG_CELL, cells, HOG_CELL).sum(axis=(2, 4))
    hist = hist.reshape(n, -1)
    return hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-6)


def knn_classify(features, train_features, train_labels):
    results = []
    for f in features:
        distances = ((train_features - f) ** 2).sum(axis=1)
        nearest = train_labels[np.argsort(distances)[:K]]
        votes = np.bincount(nearest, minlength=10)
        results.append((int(np.argmax(votes)), float(votes.max() / K)))
    return results


# reading the grid

def ink_contrast(cell_gray, mask):
    background = np.median(cell_gray[mask == 0])
    ink = cell_gray[mask > 0].mean()
    return (background - ink) / max(background, 1)


def digit_height(mask):
    ys = np.nonzero(mask)[0]
    return ys.max() - ys.min() + 1


def fit_plane(points):
    a = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    return np.linalg.lstsq(a, points[:, 2], rcond=None)[0]


def fit_reference(cells):
    strong = np.array([(r, c, contrast, height) for r, c, _, _, score, contrast, height in cells
                       if score >= STRONG_THRESHOLD], np.float64)
    if len(strong) < MIN_REFERENCES:
        return None
    plane = fit_plane(strong)
    expected = plane @ np.array([np.ones(len(strong)), strong[:, 0], strong[:, 1]])
    inliers = strong[strong[:, 2] >= CONTRAST_RATIO * expected]
    if len(inliers) >= MIN_REFERENCES:
        plane = fit_plane(inliers)
    return plane, np.median(strong[:, 3])


def is_printed(r, c, score, contrast, height, reference):
    if score < PRINTED_THRESHOLD:
        return False
    if reference is None:
        return True
    plane, ref_height = reference
    return (contrast >= CONTRAST_RATIO * (plane @ [1, r, c])
            and abs(height / ref_height - 1) <= HEIGHT_TOLERANCE)


def cell_masks(flat_gray):
    thresh = threshold_cells(flat_gray)
    for r in range(9):
        for c in range(9):
            rows = slice(r * CELL + MARGIN, (r + 1) * CELL - MARGIN)
            cols = slice(c * CELL + MARGIN, (c + 1) * CELL - MARGIN)
            mask = extract_digit(thresh[rows, cols])
            if mask is not None:
                yield r, c, mask, flat_gray[rows, cols]


def find_rotation(flat_gray):
    templates, template_labels, _, _ = get_data()
    digits = [normalize_digit(mask) for _, _, mask, _ in cell_masks(flat_gray)]
    if not digits:
        return 0
    scores = []
    for k in range(4):
        scores.append(np.mean([match_templates(np.rot90(d, k), templates, template_labels)[1]
                               for d in digits]))
    return int(np.argmax(scores))


def read_digits(flat_gray):
    templates, template_labels, train_features, train_labels = get_data()
    cells = []
    for r, c, mask, cell_gray in cell_masks(flat_gray):
        digit = normalize_digit(mask)
        label, score = match_templates(digit, templates, template_labels)
        contrast = ink_contrast(cell_gray, mask)
        cells.append((r, c, digit, label, score, contrast, digit_height(mask)))

    reference = fit_reference(cells)

    givens = [[0] * 9 for _ in range(9)]
    user_digits = [[0] * 9 for _ in range(9)]
    confidence = [[1.0] * 9 for _ in range(9)]
    handwritten, handwritten_cells = [], []
    for r, c, digit, label, score, contrast, height in cells:
        if is_printed(r, c, score, contrast, height, reference):
            givens[r][c] = label
            confidence[r][c] = round(score, 2)
        else:
            handwritten.append(digit)
            handwritten_cells.append((r, c))

    if handwritten:
        features = hog_features(np.array(handwritten))
        results = knn_classify(features, train_features, train_labels)
        for (r, c), (label, conf) in zip(handwritten_cells, results):
            user_digits[r][c] = label
            confidence[r][c] = round(conf, 2)
    return givens, user_digits, confidence


# data

def load_data():
    templates = np.load(os.path.join(DATA_DIR, "templates.npz"))
    handwritten = np.load(os.path.join(DATA_DIR, "handwritten.npz"))
    train_digits = handwritten["digits"].astype(np.float32) / 255
    return (prepare_templates(templates["digits"].astype(np.float32) / 255),
            templates["labels"], hog_features(train_digits), handwritten["labels"])


DATA = None


def get_data():
    global DATA
    if DATA is None:
        DATA = load_data()
    return DATA
