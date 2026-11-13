// The virtual board: placing it on a surface, and the puzzle the user plays on it.

import { COLORS, SIZE, apply, invert, squareToQuad, tint } from "./overlay.js";

const GRAB_RADIUS = 36;
const MIN_SIDE = 60; // smallest board side on screen, in pixels
const CORNER_SIGNS = [[-1, -1], [1, -1], [1, 1], [-1, 1]]; // TL, TR, BR, BL from the centre

/**
 * The board while it is placed, in stage pixels. The board is a square lying on a surface:
 * `surface` is the homography from surface coordinates to the screen, and `board` is the
 * square's centre and side on the surface. Moving and resizing change only `board`, so the
 * board slides and grows on the surface instead of turning in 3D.
 *
 *   drag inside     move the board on the surface
 *   drag a corner   resize it, the opposite corner stays; in tilt mode, move that corner alone,
 *                   which sets the surface's perspective
 *   two fingers     resize it
 *
 * `onChange(corners)` is called at most once per animation frame.
 */
export class Placement {
  constructor(layer, onChange) {
    this.layer = layer;
    this.onChange = onChange;
    this.tilt = false;
    this.touched = false;
    this.pointers = new Map(); // pointer id -> [x, y]
    this.gesture = null;
    this.frame = 0;
    this.dots = CORNER_SIGNS.map(() => {
      const dot = document.createElement("div");
      dot.className = "handle";
      return layer.appendChild(dot);
    });
    layer.addEventListener("pointerdown", (e) => this.grab(e));
    layer.addEventListener("pointermove", (e) => this.drag(e));
    layer.addEventListener("pointerup", (e) => this.drop(e));
    layer.addEventListener("pointercancel", (e) => this.drop(e));
  }

  get corners() {
    const { x, y, side } = this.board;
    return CORNER_SIGNS.map(([sx, sy]) => apply(this.surface, [x + (sx * side) / 2, y + (sy * side) / 2]));
  }

  get dragging() {
    return this.gesture !== null;
  }

  /** Put the board on a surface (homography from surface coordinates to stage pixels). */
  place(surface, board, touched = false) {
    this.surface = surface;
    this.inverse = invert(surface);
    this.board = board;
    this.touched = touched;
    this.changed();
  }

  /** Put the board exactly on 4 stage corners; the surface's perspective follows from them. */
  placeQuad(corners, touched = false) {
    this.place(squareToQuad(corners), { x: SIZE / 2, y: SIZE / 2, side: SIZE }, touched);
  }

  /** A centred square covering most of the stage. */
  centred() {
    const { clientWidth: w, clientHeight: h } = this.layer;
    const half = 0.3 * Math.min(w, h);
    return CORNER_SIGNS.map(([sx, sy]) => [w / 2 + sx * half, h / 2 + sy * half]);
  }

  point(e) {
    const rect = this.layer.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  toSurface(p) {
    return apply(this.inverse, p);
  }

  spread() {
    const [a, b] = [...this.pointers.values()];
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  grab(e) {
    const p = this.point(e);
    this.pointers.set(e.pointerId, p);
    this.layer.setPointerCapture(e.pointerId);
    if (this.pointers.size === 2) {
      this.gesture = { kind: "pinch", spread: this.spread(), side: this.board.side };
    } else if (this.pointers.size === 1) {
      const corners = this.corners;
      const distances = corners.map(([x, y]) => Math.hypot(x - p[0], y - p[1]));
      const i = distances.indexOf(Math.min(...distances));
      if (distances[i] < GRAB_RADIUS) this.gesture = { kind: "corner", i, anchor: this.toSurface(corners[(i + 2) % 4]) };
      else if (inside(corners, p)) this.gesture = { kind: "move", last: this.toSurface(p) };
    }
    if (this.gesture) this.touched = true;
    this.changed();
  }

  drag(e) {
    if (!this.pointers.has(e.pointerId)) return;
    const p = this.point(e);
    this.pointers.set(e.pointerId, p);
    const g = this.gesture;
    if (!g) return;
    const board = { ...this.board };
    if (g.kind === "move") {
      const q = this.toSurface(p);
      board.x += q[0] - g.last[0];
      board.y += q[1] - g.last[1];
      g.last = q;
    } else if (g.kind === "pinch") {
      board.side = (g.side * this.spread()) / g.spread;
    } else if (this.tilt) {
      const corners = this.corners;
      corners[g.i] = p;
      if (valid(corners)) this.placeQuad(corners, true);
      return;
    } else {
      const [sx, sy] = CORNER_SIGNS[g.i];
      const q = this.toSurface(p);
      board.side = Math.max(0, (sx * (q[0] - g.anchor[0]) + sy * (q[1] - g.anchor[1])) / 2);
      board.x = g.anchor[0] + (sx * board.side) / 2;
      board.y = g.anchor[1] + (sy * board.side) / 2;
    }
    const previous = this.board;
    this.board = board;
    if (!valid(this.corners)) this.board = previous;
    this.changed();
  }

  drop(e) {
    this.pointers.delete(e.pointerId);
    if (this.pointers.size === 0) this.gesture = null;
    else if (this.gesture?.kind === "pinch") this.gesture = null; // one finger left: wait for a new grab
    this.changed();
  }

  /** Draw the handles and report the corners on the next animation frame. */
  changed() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      const corners = this.corners;
      this.dots.forEach((dot, i) => {
        dot.style.transform = `translate(${corners[i][0]}px, ${corners[i][1]}px)`;
        dot.classList.toggle("active", this.gesture?.i === i);
      });
      this.onChange(corners);
    });
  }
}

/** A convex quadrilateral whose sides are all at least MIN_SIDE long. */
function valid(corners) {
  if (!corners.flat().every(Number.isFinite)) return false;
  const sides = corners.map(([x, y], i) => Math.hypot(corners[(i + 1) % 4][0] - x, corners[(i + 1) % 4][1] - y));
  return Math.min(...sides) >= MIN_SIDE && convex(corners);
}

function convex(corners) {
  const turns = corners.map(([x1, y1], i) => {
    const [x2, y2] = corners[(i + 1) % 4];
    const [x3, y3] = corners[(i + 2) % 4];
    return Math.sign((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2));
  });
  return turns.every((t) => t > 0) || turns.every((t) => t < 0);
}

/** Is the point inside the convex quadrilateral? */
function inside(corners, [x, y]) {
  const signs = corners.map(([x1, y1], i) => {
    const [x2, y2] = corners[(i + 1) % 4];
    return Math.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1));
  });
  return signs.every((s) => s >= 0) || signs.every((s) => s <= 0);
}

const emptyGrid = () => Array.from({ length: 9 }, () => Array(9).fill(0));

/**
 * A puzzle on the virtual board. The user's digits come from taps (`typed`) or from a pen on
 * paper (`ink`, read by Python). Taps win over ink.
 */
export class Game {
  constructor({ givens, solution }, difficulty = "") {
    this.givens = givens;
    this.solution = solution;
    this.difficulty = difficulty;
    this.typed = emptyGrid();
    this.ink = emptyGrid();
    this.hints = new Set();
    this.showSolution = false;
  }

  /** Clear the tapped digits, hints and the shown solution (pen digits stay: they are on paper). */
  reset() {
    this.typed = emptyGrid();
    this.hints.clear();
    this.showSolution = false;
  }

  isGiven(r, c) {
    return this.givens[r][c] > 0;
  }

  digit(r, c) {
    return this.typed[r][c] || this.ink[r][c];
  }

  set(r, c, digit) {
    this.typed[r][c] = digit;
  }

  setInk(ink) {
    this.ink = ink;
  }

  hint(r, c) {
    this.hints.add(r * 9 + c);
  }

  isHinted(r, c) {
    return this.hints.has(r * 9 + c);
  }

  /** The user's digits as one grid, for the solver. */
  userDigits() {
    return this.givens.map((row, r) => row.map((given, c) => (given ? 0 : this.digit(r, c))));
  }

  stats() {
    let written = 0;
    let wrong = 0;
    this.userDigits().forEach((row, r) => row.forEach((d, c) => {
      if (!d) return;
      written++;
      if (d !== this.solution[r][c]) wrong++;
    }));
    return { written, wrong };
  }

  get done() {
    return this.givens.every((row, r) => row.every((g, c) => g || this.digit(r, c) === this.solution[r][c]));
  }

  /** Cells to draw, for drawBoard. */
  cells() {
    const cells = [];
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const answer = this.solution[r][c];
        const color = this.digit(r, c) === answer ? COLORS.right : COLORS.wrong;
        if (this.isGiven(r, c)) cells.push({ r, c, digit: this.givens[r][c], color: COLORS.ink });
        else if (this.typed[r][c]) cells.push({ r, c, digit: this.typed[r][c], color });
        else if (this.ink[r][c]) cells.push({ r, c, outline: color });
        else if (this.isHinted(r, c)) {
          cells.push({ r, c, digit: answer, color: COLORS.next, fill: tint(COLORS.next, 0.2), outline: COLORS.next });
        } else if (this.showSolution) cells.push({ r, c, digit: answer, color: COLORS.app });
      }
    }
    return cells;
  }
}
