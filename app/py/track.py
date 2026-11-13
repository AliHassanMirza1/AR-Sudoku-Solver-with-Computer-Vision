"""Surface tracker: keeps the 4 corners of a board fixed on a surface while the camera moves.

Every frame, Lucas-Kanade optical flow follows surface points, and a RANSAC homography from
their positions in the reference frame (when the board was placed) to the new frame moves the
board. Measuring from the reference, not from the last frame, means errors do not add up.
When there are too few points, two fallbacks give the same homography:
  * the sheet of paper under the board, found again with contours,
  * ORB features matched to the reference frame, which also finds the board again after it
    left the view.
"""

import cv2
import numpy as np

from sudoku_cv.detect import order_corners

MAX_POINTS = 150
MIN_POINTS = 12
POINT_QUALITY = 0.01
POINT_DISTANCE = 8
REGION_MARGIN = 0.5
PRINTED_MARGIN = 0.05
FLOW_WINDOW = (21, 21)
FLOW_LEVELS = 3
RANSAC_PIXELS = 3.0
FLOW_CHECK_PIXELS = 1.0
ORB_FEATURES = 500
MIN_MATCHES = 25
MATCH_RATIO = 0.75
OCCLUDED_RATIO = 0.6
LOST_FRAMES = 5
STILL_PIXELS = 1.5
FOLLOW_PIXELS = 1.5
MIN_SMOOTHING = 0.25
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


def find_points(gray, corners, margin):
    points = cv2.goodFeaturesToTrack(gray, MAX_POINTS, POINT_QUALITY, POINT_DISTANCE,
                                     mask=region_mask(gray.shape, corners, margin))
    return np.empty((0, 1, 2), np.float32) if points is None else points


# sheet of paper

def find_sheet(gray, inside=None, near=None):
    """Corners of a bright 4-sided region (a sheet of paper), optionally one that contains the
    points `inside`, or one whose corners are close to `near` (the sheet followed so far)."""
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
        if inside is not None and not all(
                cv2.pointPolygonTest(corners, (float(x), float(y)), False) >= 0 for x, y in inside):
            continue
        if near is not None and np.abs(corners - near).max() > SHEET_MAX_JUMP * max(gray.shape):
            continue
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

    around: also follow surface points around the board (a virtual board on a table). Off for
    a printed puzzle, which has texture of its own and may be held in the hand.
    """

    def __init__(self, gray, corners, around=True):
        self.ref_corners = np.float32(corners)
        self.corners = self.ref_corners.copy()
        self.homography = np.eye(3)  # reference frame -> current frame
        self.margin = REGION_MARGIN if around else PRINTED_MARGIN
        self.prev = gray
        self.missed = 0
        self.still_count = 0

        mask = region_mask(gray.shape, corners, self.margin)
        self.orb = cv2.ORB_create(ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray, mask)

        self.ref_sheet = find_sheet(gray, inside=corners)
        self.sheet = self.ref_sheet
        self.points = find_points(gray, corners, self.margin)
        self.ref_points = self.points.copy()
        self.plain = len(self.points) < MIN_POINTS and self.ref_sheet is None
        self.state = "plain" if self.plain else "tracking"

    @property
    def on_sheet(self):
        return self.ref_sheet is not None

    def update(self, gray):
        homography, state = self.follow_points(gray)
        if homography is None and self.on_sheet:
            homography, state = self.follow_sheet(gray), "tracking"
        if homography is None and not self.plain:
            homography, state = self.match_reference(gray), "tracking"

        corners = None if homography is None else transform(self.ref_corners, homography)
        if corners is not None and plausible(corners, gray.shape):
            self.homography = homography
            self.missed = 0
            self.move_corners(corners)
            if len(self.points) < MAX_POINTS // 2:
                self.add_points(gray)
        elif self.plain:
            state = "plain"
            self.still_count += 1
        else:
            self.missed += 1
            self.still_count = 0
            state = "lost" if self.missed >= LOST_FRAMES else self.state
        self.prev = gray
        self.state = state
        return state

    def move_corners(self, corners):
        """Follow big moves at once, smooth small ones (sensor noise) away."""
        moved = np.abs(corners - self.corners).max()
        weight = min(1.0, max(MIN_SMOOTHING, moved / FOLLOW_PIXELS))
        self.corners = np.float32(weight * corners + (1 - weight) * self.corners)
        self.still_count = self.still_count + 1 if moved < STILL_PIXELS else 0

    def follow_points(self, gray):
        """Optical flow for every point, checked by tracking it back; homography from the reference."""
        if len(self.points) < MIN_POINTS:
            return None, "lost"
        flow = dict(winSize=FLOW_WINDOW, maxLevel=FLOW_LEVELS)
        new, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.points, None, **flow)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, new, None, **flow)
        error = np.abs(back - self.points).reshape(-1, 2).max(axis=1)
        good = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < FLOW_CHECK_PIXELS)
        tracked = len(self.points)
        self.points, self.ref_points = new[good], self.ref_points[good]
        if len(self.points) < MIN_POINTS:
            return None, "lost"

        homography, inliers = cv2.findHomography(self.ref_points, self.points, cv2.RANSAC, RANSAC_PIXELS)
        if homography is None or inliers.sum() < MIN_POINTS:
            return None, "lost"
        inliers = inliers.ravel() == 1
        self.points, self.ref_points = self.points[inliers], self.ref_points[inliers]
        return homography, "occluded" if inliers.sum() < OCCLUDED_RATIO * tracked else "tracking"

    def add_points(self, gray):
        """New points to follow; their reference positions come from the current homography."""
        points = find_points(gray, self.corners, self.margin)
        if len(points) == 0:
            return
        self.points = points
        self.ref_points = transform(points, np.linalg.inv(self.homography)).reshape(-1, 1, 2)

    def follow_sheet(self, gray):
        sheet = find_sheet(gray, near=self.sheet)
        if sheet is None:
            return None
        self.sheet = sheet
        return cv2.getPerspectiveTransform(self.ref_sheet, sheet)

    def match_reference(self, gray):
        """ORB matches with Lowe's ratio test, so the repeated look of grid corners is not matched."""
        if self.ref_descriptors is None or len(self.ref_keypoints) < MIN_MATCHES:
            return None
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 2:
            return None
        pairs = self.matcher.knnMatch(self.ref_descriptors, descriptors, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < MATCH_RATIO * p[1].distance]
        if len(matches) < MIN_MATCHES:
            return None
        src = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PIXELS)
        if homography is None or inliers.sum() < MIN_MATCHES:
            return None
        return homography
