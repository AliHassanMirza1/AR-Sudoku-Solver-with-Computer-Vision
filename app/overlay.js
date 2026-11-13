// Drawing on the paper. The board is drawn flat on a SIZE x SIZE canvas, then bent onto its
// 4 corners on screen with a homography (a CSS matrix3d transform).

export const SIZE = 450;
const CELL = SIZE / 9;

export const COLORS = {
  ink: "#18181b",
  app: "#2563eb",
  next: "#ea580c",
  right: "#16a34a",
  wrong: "#dc2626",
  placing: "#ffffff",
};

// ---------- geometry (3x3 matrices are arrays of rows; corners are TL, TR, BR, BL) ----------

const SQUARE = [[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]];

/** Homography that maps the SIZE x SIZE square onto `corners`. */
export function squareToQuad(corners) {
  const a = [];
  const b = [];
  SQUARE.forEach(([x, y], i) => {
    const [u, v] = corners[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  });
  const h = solveLinear(a, b);
  return [h.slice(0, 3), h.slice(3, 6), [h[6], h[7], 1]];
}

/** Gaussian elimination with partial pivoting. */
function solveLinear(a, b) {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let row = col + 1; row < n; row++) {
      const f = m[row][col] / m[col][col];
      for (let k = col; k <= n; k++) m[row][k] -= f * m[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = m[row][n];
    for (let k = row + 1; k < n; k++) sum -= m[row][k] * x[k];
    x[row] = sum / m[row][row];
  }
  return x;
}

export function invert([[a, b, c], [d, e, f], [g, h, i]]) {
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [C / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

export function apply(m, [x, y]) {
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w];
}

function matrix3d([[a, b, c], [d, e, f], [g, h, i]]) {
  return `matrix3d(${a},${d},0,${g},${b},${e},0,${h},0,0,1,0,${c},${f},0,${i})`;
}

/** The cell [row, col] at a point of the board (in the same coordinates as `corners`), or null. */
export function cellAt(corners, point) {
  const [x, y] = apply(invert(squareToQuad(corners)), point);
  const r = Math.floor(y / CELL);
  const c = Math.floor(x / CELL);
  return r >= 0 && r < 9 && c >= 0 && c < 9 ? [r, c] : null;
}

// ---------- the camera picture on screen ----------

/**
 * Maps frame pixels (the frames sent to Python) to stage pixels. The picture is shown with
 * object-fit: contain, so it is scaled to fit and centred.
 */
export class Stage {
  constructor(element, boardCanvas) {
    this.element = element;
    this.board = boardCanvas;
    this.frame = { width: 640, height: 480 };
  }

  fit() {
    const { clientWidth: w, clientHeight: h } = this.element;
    const scale = Math.min(w / this.frame.width, h / this.frame.height);
    return { scale, x: (w - this.frame.width * scale) / 2, y: (h - this.frame.height * scale) / 2 };
  }

  toScreen([x, y]) {
    const fit = this.fit();
    return [fit.x + x * fit.scale, fit.y + y * fit.scale];
  }

  toFrame([x, y]) {
    const fit = this.fit();
    return [(x - fit.x) / fit.scale, (y - fit.y) / fit.scale];
  }

  /** Show the board canvas on these frame corners, or hide it. */
  showBoard(corners) {
    this.board.hidden = !corners;
    if (corners) this.board.style.transform = matrix3d(squareToQuad(corners.map((p) => this.toScreen(p))));
  }
}

// ---------- the flat board image ----------

/**
 * Draw the board: an optional background and grid lines, and a list of cells
 * {r, c, digit, color, fill, outline}.
 */
export function drawBoard(canvas, { lines = null, background = null, cells = [] } = {}) {
  const ctx = canvas.getContext("2d");
  const k = canvas.width / SIZE;
  ctx.setTransform(k, 0, 0, k, 0, 0);
  ctx.clearRect(0, 0, SIZE, SIZE);
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, SIZE, SIZE);
  }

  for (const { r, c, fill, outline } of cells) {
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
    }
    if (outline) {
      ctx.strokeStyle = outline;
      ctx.lineWidth = 4;
      ctx.strokeRect(c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6);
    }
  }

  if (lines) {
    ctx.save();
    ctx.strokeStyle = lines;
    ctx.shadowColor = "rgba(0, 0, 0, 0.6)"; // keeps white lines visible on white paper
    ctx.shadowBlur = 3;
    for (let i = 0; i <= 9; i++) {
      ctx.lineWidth = i % 3 === 0 ? 3 : 1;
      const p = Math.min(Math.max(i * CELL, 1.5), SIZE - 1.5);
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, SIZE);
      ctx.moveTo(0, p);
      ctx.lineTo(SIZE, p);
      ctx.stroke();
    }
    ctx.restore();
  }

  ctx.font = "600 34px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const { r, c, digit, color } of cells) {
    if (!digit) continue;
    ctx.fillStyle = color;
    ctx.fillText(String(digit), c * CELL + CELL / 2, r * CELL + CELL / 2 + 2);
  }
}

/** Tint a colour for a cell background, e.g. tint(COLORS.next, 0.25). */
export function tint(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
