import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sudoku_cv.digits import DATA_DIR, normalize_digit

FONT_DIR = "/usr/share/fonts/truetype"
FONTS = [
    "dejavu/DejaVuSans.ttf",
    "dejavu/DejaVuSans-Bold.ttf",
    "dejavu/DejaVuSerif.ttf",
    "liberation/LiberationSans-Regular.ttf",
    "liberation/LiberationSans-Bold.ttf",
    "liberation/LiberationSerif-Regular.ttf",
    "liberation/LiberationSerif-Bold.ttf",
    "liberation/LiberationSansNarrow-Regular.ttf",
    "liberation/LiberationSansNarrow-Bold.ttf",
    "freefont/FreeSans.ttf",
    "freefont/FreeSerif.ttf",
]
GOOGLE_FONTS = os.path.join(os.path.dirname(__file__), "fonts.txt")
GOOGLE_URL = "https://raw.githubusercontent.com/google/fonts/main/"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_SIZE = 100


def load_font(path, weight):
    font = ImageFont.truetype(path, FONT_SIZE)
    if weight:
        axes = font.get_variation_axes()
        font.set_variation_by_axes([weight if a["name"] in (b"Weight", "Weight") else a["default"]
                                    for a in axes])
    return font


def google_fonts():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fonts = []
    for line in open(GOOGLE_FONTS):
        path, weight = line.split()
        local = os.path.join(CACHE_DIR, os.path.basename(path))
        if not os.path.exists(local):
            url = GOOGLE_URL + urllib.request.quote(path)
            urllib.request.urlretrieve(url, local)
        fonts.append((local, int(weight)))
    return fonts


def render_digit(digit, font):
    img = Image.new("L", (2 * FONT_SIZE, 2 * FONT_SIZE), 0)
    ImageDraw.Draw(img).text((FONT_SIZE // 2, FONT_SIZE // 4), str(digit), fill=255, font=font)
    return (np.array(img) > 127).astype(np.uint8) * 255


def main():
    fonts = [(os.path.join(FONT_DIR, f), 0) for f in FONTS] + google_fonts()
    digits, labels = [], []
    for path, weight in fonts:
        font = load_font(path, weight)
        for d in range(1, 10):
            digits.append(normalize_digit(render_digit(d, font)))
            labels.append(d)
    digits = np.uint8(np.round(np.array(digits) * 255))
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, "templates.npz"), digits=digits, labels=labels)
    print(f"templates: {len(digits)} ({len(fonts)} fonts)")


if __name__ == "__main__":
    main()
