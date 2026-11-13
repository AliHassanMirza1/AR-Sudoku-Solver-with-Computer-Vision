# AR Sudoku Solver with Computer Vision

Point your camera at a Sudoku. The website finds the puzzle, solves it and writes the missing
digits on the paper. With no puzzle on paper, it draws a board in AR on any flat surface for you
to play on.

**Live:** https://alihassanmirza1.github.io/AR-Sudoku-Solver-with-Computer-Vision/

Classical computer vision only (no neural networks). Everything runs on the device that opens the
page: Python, numpy and OpenCV run in the browser with [Pyodide](https://pyodide.org), so the video
never leaves the device and the site works offline after the first visit.

## Two modes

- **Paper puzzle.** *Live* reveals every missing digit; *Step* shows one per tap. *Check* freezes
  the picture to fix a misread digit. Handwritten digits are outlined green (right) or red (wrong).
  If the grid is not found, set its 4 corners by hand.
- **Virtual board.** Pick a puzzle (easy, medium, hard, or type one in), drag the 4 corners onto a
  table, wall or sheet of paper, and play by tapping cells or by writing on the paper. The board
  stays on the surface while the phone moves (optical flow, ORB matching, RANSAC homography).

## Repository

```
app/            the website: index.html, app.js (screens and frame loop), worker.js (Pyodide),
                python.js (calls into the worker), overlay.js (homography and drawing),
                virtual.js (corner handles and the game), sw.js (offline), manifest.json (install)
app/py/         bridge.py (the worker's interface to sudoku_cv), track.py (surface tracker),
                generate.py (puzzle generator)
sudoku_cv/      grid detection, digit reading and solver
scripts/        build.py: builds the static site in _site/
tests/          unit tests for app/py on synthetic frames
```

## Run it locally

```sh
python3 scripts/build.py              # app + sudoku_cv + a pinned Pyodide into _site/ (~28 MB)
python3 -m http.server -d _site 8000  # open http://localhost:8000
```

Browsers only allow the camera on `https://` or `localhost`. To try a phone on the same Wi-Fi,
use the deployed site. Run the tests with:

```sh
pip install -r requirements.txt
python3 -m unittest discover tests
```

## Deploy

`.github/workflows/pages.yml` builds and publishes the site on every push to `main`. One-time
setup: in the repository's **Settings → Pages**, set **Source** to **GitHub Actions**.
