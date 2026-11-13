"""Surface tracker: keeps the 4 corners of a board fixed on a surface while the camera moves.

Three classical methods, from cheapest to most robust:
  * a sheet of paper under the board is found again in every frame (contours),
  * Lucas-Kanade optical flow follows surface points from frame to frame (RANSAC homography),
  * ORB features are matched to the reference frame every few frames, which removes the drift
    of optical flow and finds the board again after it left the view.
"""

import cv2
import numpy as np

from sudoku_cv.detect import order_corners

MAX_POINTS = 150
MIN_POINTS = 12
POINT_QUALITY = 0.01
POINT_DISTANCE = 8
REGION_MARGIN = 0.5
FLOW_WINDOW = (21, 21)
FLOW_LEVELS = 3
RANSAC_PIXELS = 3.0
ORB_FEATURES = 500
MIN_MATCHES = 15
REFRESH_FRAMES = 10
OCCLUDED_RATIO = 0.6
STILL_PIXELS = 1.5
MIN_BOARD_AREA = 0.01
SHEET_MIN_AREA = 0.08
SHEET_MAX_JUMP = 0.15
SHEET_FILL = 0.85
SHEET_BORDER = 3


# geometry

def transform(points, homography):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), homography).reshape(-1, 2)


def plausible(corners, shape):
    """A convex quadrilateral that is not tiny."""
    quad = np.float32(corners).reshape(-1, 1, 2)
    area = cv2.contourArea(quad)
    return cv2.isContourConvex(np.int32(quad)) and area > MIN_BOARD_AREA * shape[0] * shape[1]


def region_mask(shape, corners, margin=REGION_MARGIN):
    """The board grown by `margin` of its size: points around the board count too."""
    corners = np.float32(corners)
    center = corners.mean(axis=0)
    grown = center + (corners - center) * (1 + 2 * margin)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.int32(grown), 255)
    return mask


def find_points(gray, corners):
    points = cv2.goodFeaturesToTrack(gray, MAX_POINTS, POINT_QUALITY, POINT_DISTANCE,
                                     mask=region_mask(gray.shape, corners))
    return np.empty((0, 1, 2), np.float32) if points is None else points


# sheet of paper

def find_sheet(gray, inside=None):
    """Corners of a bright 4-sided region (a sheet of paper), optionally one that contains `inside`."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < SHEET_MIN_AREA * gray.size:
            break
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) != 4:
            continue
        corners = order_corners(approx)
        if touches_border(corners, gray.shape):
            continue
        if inside is None or all(cv2.pointPolygonTest(corners, (float(x), float(y)), False) >= 0
                                 for x, y in inside):
            return corners
    return None


def touches_border(corners, shape):
    """A sheet cut by the frame edge: its real corners are not in view."""
    height, width = shape[:2]
    x, y = corners[:, 0], corners[:, 1]
    return (x.min() < SHEET_BORDER or y.min() < SHEET_BORDER
            or x.max() > width - 1 - SHEET_BORDER or y.max() > height - 1 - SHEET_BORDER)


def board_on_sheet(sheet):
    """A square-looking board centred on the sheet, covering most of its short side."""
    tl, tr, br, bl = np.float32(sheet)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    w = SHEET_FILL * min(1, height / width)
    h = SHEET_FILL * min(1, width / height)
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    board = 0.5 + (unit - 0.5) * [w, h]
    return transform(board, cv2.getPerspectiveTransform(unit, np.float32(sheet)))


# tracker

class SurfaceTracker:
    """Call `update` with every new grey frame; read `corners`, `state` and `still_count`.

    state is one of: tracking, occluded (part of the board hidden), plain (too few surface
    points, the board stays where it is on the screen) and lost.
    """

    def __init__(self, gray, corners):
        self.ref_corners = np.float32(corners)
        self.corners = self.ref_corners.copy()
        self.prev = gray
        self.frame_count = 0
        self.still_count = 0

        self.orb = cv2.ORB_create(ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(
            gray, region_mask(gray.shape, corners))

        self.ref_sheet = find_sheet(gray, inside=corners)
        self.sheet = self.ref_sheet
        self.points = find_points(gray, corners)
        self.plain = len(self.points) < MIN_POINTS and self.ref_sheet is None
        self.state = "plain" if self.plain else "tracking"

    @property
    def on_sheet(self):
        return self.ref_sheet is not None

    def update(self, gray):
        self.frame_count += 1
        corners, state = None, "lost"
        if self.on_sheet:
            corners = self.follow_sheet(gray)
            state = "tracking" if corners is not None else state
        if corners is None and not self.plain:
            corners, state = self.follow_points(gray)
            if corners is None or self.frame_count % REFRESH_FRAMES == 0:
                matched = self.match_reference(gray)
                if matched is not None:
                    corners, state = matched, "tracking" if corners is None else state
        if self.plain:
            state = "plain"

        if corners is not None and plausible(corners, gray.shape):
            moved = np.abs(corners - self.corners).max()
            self.still_count = self.still_count + 1 if moved < STILL_PIXELS else 0
            self.corners = np.float32(corners)
        else:
            self.still_count = self.still_count + 1 if state == "plain" else 0
            state = "plain" if self.plain else "lost"

        if state == "tracking" and (self.frame_count % REFRESH_FRAMES == 0
                                    or len(self.points) < MAX_POINTS // 2):
            self.points = find_points(gray, self.corners)
        self.prev = gray
        self.state = state
        return state

    def follow_sheet(self, gray):
        sheet = find_sheet(gray)
        if sheet is None or self.sheet is None:
            return None
        jump = np.abs(sheet - self.sheet).max() / max(gray.shape)
        if jump > SHEET_MAX_JUMP:
            return None
        self.sheet = sheet
        return transform(self.ref_corners, cv2.getPerspectiveTransform(self.ref_sheet, sheet))

    def follow_points(self, gray):
        if len(self.points) < MIN_POINTS:
            return None, "lost"
        new, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.points, None,
                                                  winSize=FLOW_WINDOW, maxLevel=FLOW_LEVELS)
        ok = status.ravel() == 1
        old, new = self.points[ok], new[ok]
        if len(new) < MIN_POINTS:
            self.points = new
            return None, "lost"
        homography, inliers = cv2.findHomography(old, new, cv2.RANSAC, RANSAC_PIXELS)
        if homography is None:
            return None, "lost"
        inliers = inliers.ravel() == 1
        ratio = inliers.sum() / len(self.points)
        self.points = new[inliers].reshape(-1, 1, 2)
        state = "occluded" if ratio < OCCLUDED_RATIO else "tracking"
        return transform(self.corners, homography), state

    def match_reference(self, gray):
        if self.ref_descriptors is None or len(self.ref_keypoints) < MIN_MATCHES:
            return None
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None:
            return None
        matches = self.matcher.match(self.ref_descriptors, descriptors)
        if len(matches) < MIN_MATCHES:
            return None
        src = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PIXELS)
        if homography is None or inliers.sum() < MIN_MATCHES:
            return None
        return transform(self.ref_corners, homography)
