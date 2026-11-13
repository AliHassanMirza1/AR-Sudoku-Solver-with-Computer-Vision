"""Build the static website in _site/ (used locally and by the GitHub Action).

_site/
  app/*           the page, worker and styles
  py.zip          sudoku_cv/ (code and .npz data) and app/py/*.py, unpacked by the worker
  pyodide/        a pinned copy of Pyodide with numpy and OpenCV, served from our own site
  vendor/         pinned browser libraries (a HEIC photo decoder, loaded only when needed)

Downloads are kept in .cache/ so later builds work offline.
Run: python3 scripts/build.py, then python3 -m http.server -d _site 8000
"""

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

PYODIDE_VERSION = "314.0.7"
PYODIDE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
PYODIDE_CORE = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip",
                "pyodide-lock.json"]
PYTHON_PACKAGES = ["numpy", "opencv-python"]
VENDOR = {  # for browsers that cannot open iPhone (HEIC) photos themselves
    "heic-to.min.js": "https://cdn.jsdelivr.net/npm/heic-to@1.6.7/dist/heic-to.min.js",
}

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
SITE = ROOT / "_site"
CACHE = ROOT / ".cache" / f"pyodide-{PYODIDE_VERSION}"
VENDOR_CACHE = ROOT / ".cache" / "vendor"


def download(name, url=None, cache=CACHE):
    """A file from Pyodide's release (or from `url`), downloaded once into the cache."""
    path = cache / name
    if not path.exists():
        print("downloading", name)
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url or PYODIDE_URL + name) as response:
            path.with_suffix(".part").write_bytes(response.read())
        path.with_suffix(".part").rename(path)
    return path


def copy_vendor():
    target = SITE / "vendor"
    target.mkdir()
    for name, url in VENDOR.items():
        shutil.copy(download(name, url, VENDOR_CACHE), target / name)


def package_files(lock, names):
    """Wheel file names of the packages and everything they depend on."""
    files, todo, seen = [], list(names), set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        package = lock["packages"][name]
        files.append(package["file_name"])
        todo += package["depends"]
    return files


def copy_pyodide():
    for name in PYODIDE_CORE:
        download(name)
    lock = json.loads((CACHE / "pyodide-lock.json").read_text())
    target = SITE / "pyodide"
    target.mkdir()
    for name in PYODIDE_CORE + package_files(lock, PYTHON_PACKAGES):
        shutil.copy(download(name), target / name)


def zip_python():
    with zipfile.ZipFile(SITE / "py.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "sudoku_cv").rglob("*")):
            if path.suffix in (".py", ".npz"):
                archive.write(path, path.relative_to(ROOT))
        for path in sorted((APP / "py").glob("*.py")):
            archive.write(path, path.name)


def check_service_worker():
    """sw.js names its Pyodide cache after the version, so an upgrade replaces the cached files."""
    if f'PYODIDE_VERSION = "{PYODIDE_VERSION}"' not in (APP / "sw.js").read_text():
        raise SystemExit(f"Set PYODIDE_VERSION in app/sw.js to {PYODIDE_VERSION}")


def main():
    check_service_worker()
    shutil.rmtree(SITE, ignore_errors=True)
    shutil.copytree(APP, SITE, ignore=shutil.ignore_patterns("py", ".DS_Store", "__pycache__"))
    zip_python()
    copy_pyodide()
    copy_vendor()
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"built {SITE.relative_to(ROOT)}/ ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
