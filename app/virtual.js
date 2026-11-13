// The virtual board: corner handles to place it, and the puzzle the user plays on it.

import { COLORS, tint } from "./overlay.js";

const GRAB_RADIUS = 36;

/**
 * Four draggable corner handles (stage pixels). Dragging inside the board moves all 4.
 * `onChange(corners)` is called on every move.
 */
export class Handles {
  constructor(layer, onChange) {
    this.layer = layer;
    this.onChange = onChange;
    this.corners = [];
    this.active = null;
    this.touched = false;
    this.dots = [0, 1, 2, 3].map(() => {
      const dot = document.createElement("div");
      dot.className = "handle";
      return layer.appendChild(dot);
    });
    layer.addEventListener("pointerdown", (e) => this.grab(e));
    layer.addEventListener("pointermove", (e) => this.drag(e));
    layer.addEventListener("pointerup", () => this.drop());
    layer.addEventListener("pointercancel", () => this.drop());
  }

  /** A centred square covering most of the stage. */
  centred() {
    const { clientWidth: w, clientHeight: h } = this.layer;
    const half = 0.3 * Math.min(w, h);
    const [x, y] = [w / 2, h / 2];
    return [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]];
  }

  set(corners, touched = false) {
    this.corners = corners.map(([x, y]) => [x, y]);
    this.touched = touched;
    this.render();
    this.onChange(this.corners);
  }

  point(e) {
    const rect = this.layer.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  grab(e) {
    const p = this.point(e);
    const distances = this.corners.map(([x, y]) => Math.hypot(x - p[0], y - p[1]));
    const nearest = distances.indexOf(Math.min(...distances));
    if (distances[nearest] < GRAB_RADIUS) this.active = nearest;
    else if (inside(this.corners, p)) this.active = "all";
    else return;
    this.last = p;
    this.touched = true;
    this.layer.setPointerCapture(e.pointerId);
    this.render();
  }

  drag(e) {
    if (this.active === null) return;
    const p = this.point(e);
    const [dx, dy] = [p[0] - this.last[0], p[1] - this.last[1]];
    this.last = p;
    this.corners.forEach((corner, i) => {
      if (this.active === "all" || this.active === i) {
        corner[0] += dx;
        corner[1] += dy;
      }
    });
    this.render();
    this.onChange(this.corners);
  }

  drop() {
    this.active = null;
    this.render();
  }

  render() {
    this.dots.forEach((dot, i) => {
      const [x, y] = this.corners[i] ?? [0, 0];
      dot.style.transform = `translate(${x}px, ${y}px)`;
      dot.classList.toggle("active", this.active === i);
    });
  }
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
