// AR Sudoku: the screens, the camera, and the loop that sends frames to Python and draws the
// answers on the paper. Views (body[data-view]): home, paper, check, place, play, typing.

import { call, ready } from "./python.js";
import { COLORS, Stage, cellAt, drawBoard, tiltedSurface, tint } from "./overlay.js";
import { Game, Placement } from "./virtual.js";

const FRAME_SIZE = 640; // longest side of the frames sent to Python
const STILL_SIZE = 1600; // longest side of photos shown on screen
const REVEAL_MS = 120; // Live mode: time between two revealed digits
const SNAP_MS = 500; // placing a board: how often to look for a sheet of paper
const FOCAL = 0.78; // camera focal length / longest side of the picture (a 65° wide phone camera)
const MAX_TABLE_TILT = 70; // phone tilt (degrees from pointing straight down) still read as a table

const MESSAGES = {
  loading: ["Loading the solver…", ""],
  photo: ["Reading the photo…", "accent"],
  no_grid: ["Point the camera at a Sudoku", ""],
  no_grid_photo: ["No puzzle found. Set the corners by hand.", "bad"],
  reading: ["Hold still…", "accent"],
  no_solution: ["Some digits look wrong. Tap Check.", "bad"],
  solved: ["Solved. Watch the paper.", "good"],
  checking: ["Tap a digit the app read wrong", "accent"],
  placing: ["Drag to move, corners to resize", "accent"],
  placing_tilt: ["Drag each corner into place", "accent"],
  snapped: ["Corners snapped to the sheet. Tap Done.", "accent"],
  tracking: ["Tap a cell to play", "accent"],
  tracking_sheet: ["Tap a cell, or write on the paper", "accent"],
  occluded: ["Board kept; checking when your hand moves away", "accent"],
  plain: ["Plain surface: hold the phone still", "accent"],
  lost: ["Point back at the board, or tap Adjust", "bad"],
  done: ["Solved. Well done!", "good"],
};

const LEVELS = { easy: "Easy", medium: "Medium", hard: "Hard", typed: "Your puzzle" };

const $ = (id) => document.getElementById(id);
const video = $("video");
const still = $("still");
const boardCanvas = $("board");
const stage = new Stage($("stage"), boardCanvas);
const placement = new Placement($("handles"), showPlacing);
const motion = { tilt: null }; // degrees between the back camera and straight down (tilt sensor)

const state = {
  token: 0, // changes with the view, so late results from Python are dropped
  pythonReady: false,
  busy: false, // a frame is in Python
  source: "video", // "video" (camera) or "still" (photo, or a frame frozen by Check)
  frozen: false,
  corners: null, // where the board is, in frame pixels
  pinned: false, // the surface tracker follows the board (virtual board, or paper corners set by hand)
  autoPinned: false, // pinned because Python found a hand-drawn board; back to searching if lost
  track: null, // surface tracker state: tracking, occluded, plain, lost
  onSheet: false,
  // paper puzzle
  result: null,
  puzzleKey: null,
  shown: 0,
  step: false,
  lastReveal: 0,
  // placing and playing a virtual board
  placing: null, // "paper" or "virtual": what Done does
  snapped: false,
  lastSnap: 0,
  game: null,
  typed: null,
};

// ---------- views ----------

const view = () => document.body.dataset.view;

function show(name) {
  document.body.dataset.view = name;
  state.token++;
  stage.measure();
  render();
}

function render() {
  switch (view()) {
    case "paper":
    case "check":
      return renderPaper();
    case "place":
      return placement.board && showPlacing(placement.corners);
    case "play":
      return renderPlay();
    case "typing":
      return renderTyping();
  }
}

function setStatus(key, pill = $("status")) {
  const [text, tone] = MESSAGES[key];
  if (pill.textContent === text) return; // unchanged text would still cost a layout
  pill.textContent = text;
  pill.dataset.tone = tone;
}

function goHome(message = "") {
  stopCamera();
  setSource("video");
  Object.assign(state, {
    corners: null, pinned: false, autoPinned: false, track: null, result: null, puzzleKey: null, game: null,
  });
  call("reset");
  $("home-message").textContent = message;
  $("home-message").hidden = !message;
  show("home");
}

/** Run an async action from a button; errors go back to the home screen with a message. */
function run(action) {
  Promise.resolve()
    .then(action)
    .catch((error) => {
      console.error(error);
      const messages = {
        NotAllowedError: "Camera blocked. Allow the camera in the browser settings, or upload a photo.",
        NotFoundError: "No camera found. Upload a photo instead.",
      };
      goHome(messages[error.name] ?? `Something went wrong: ${error.message}`);
    });
}

/**
 * Open a <dialog> and resolve with the value of the button that closed it ("" if cancelled).
 * Uses submit and cancel, which fire at once, not close, which fires later and could reach
 * the next question when the user taps fast.
 */
function ask(dialog) {
  const form = dialog.querySelector("form");
  dialog.showModal();
  return new Promise((resolve) => {
    const answer = (value) => {
      form.onsubmit = dialog.oncancel = null;
      resolve(value);
    };
    form.onsubmit = (e) => answer(e.submitter?.value ?? "");
    dialog.oncancel = () => answer("");
  });
}

async function pickDigit() {
  const value = await ask($("picker"));
  return value === "" ? null : Number(value);
}

// ---------- camera and frames ----------

let stream = null;

async function startCamera() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await video.play();
  setSource("video");
  $("flash").hidden = !stream.getVideoTracks()[0].getCapabilities?.().torch;
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  $("flash").classList.remove("on");
}

function setSource(source) {
  state.source = source;
  $("stage").dataset.source = source;
  const size = frameSize();
  if (size.width) stage.frame = size;
}

/** Size of the frames sent to Python: the picture shrunk so its longest side is FRAME_SIZE. */
function frameSize() {
  const [w, h] = state.source === "video" ? [video.videoWidth, video.videoHeight] : [still.width, still.height];
  const scale = Math.min(1, FRAME_SIZE / Math.max(w, h));
  return { width: Math.round(w * scale), height: Math.round(h * scale) };
}

const grabCanvas = document.createElement("canvas");
const grabContext = grabCanvas.getContext("2d", { willReadFrequently: true });

function grabFrame() {
  const { width, height } = (stage.frame = frameSize());
  if (grabCanvas.width !== width || grabCanvas.height !== height) {
    grabCanvas.width = width;
    grabCanvas.height = height;
  }
  grabContext.drawImage(state.source === "video" ? video : still, 0, 0, width, height);
  return grabContext.getImageData(0, 0, width, height);
}

// The camera gives about 30 new pictures a second and the screen redraws 60 times or more;
// only new pictures are worth sending to Python.
let newPicture = true;
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  const onPicture = () => {
    newPicture = true;
    video.requestVideoFrameCallback(onPicture);
  };
  video.requestVideoFrameCallback(onPicture);
}

/** Every animation frame: reveal digits in Live mode, and send a new picture when Python is free. */
function loop(now) {
  requestAnimationFrame(loop);
  revealNext(now);
  if (state.busy || !newPicture || !state.pythonReady || state.source !== "video" || !video.videoWidth) return;
  const task = frameTask(now);
  if (!task) return;
  newPicture = !("requestVideoFrameCallback" in video);
  state.busy = true;
  task(grabFrame())
    .catch(console.error)
    .finally(() => (state.busy = false));
}

function frameTask(now) {
  if (view() === "paper") return state.pinned ? followBoard : readPaper;
  if (view() === "play") return followBoard;
  if (view() === "place" && state.placing === "virtual" && !placement.touched && now - state.lastSnap > SNAP_MS) {
    state.lastSnap = now;
    return snapToSheet;
  }
  return null;
}

async function readPaper(frame) {
  const token = state.token;
  const result = await call("paper_frame", [], frame);
  if (token === state.token) setPaperResult(result);
}

async function followBoard(frame) {
  const token = state.token;
  const givens = view() === "play" ? state.game.givens : null;
  const result = await call("follow", givens ? [givens] : [], frame);
  if (token !== state.token) return;
  Object.assign(state, { corners: result.corners, track: result.track, onSheet: result.on_sheet });
  if (result.ink) state.game?.setInk(result.ink);
  if (state.autoPinned && result.track === "lost") Object.assign(state, { pinned: false, autoPinned: false });
  render();
}

async function snapToSheet(frame) {
  const token = state.token;
  const { corners } = await call("snap", [], frame);
  if (token !== state.token || placement.touched || !corners) return;
  state.snapped = true;
  placement.placeQuad(corners.map((p) => stage.toScreen(p)));
}

// ---------- paper puzzle ----------

function startPaper() {
  Object.assign(state, {
    result: null, puzzleKey: null, shown: 0, corners: null, pinned: false, autoPinned: false, track: null,
  });
  call("reset");
  show("paper");
}

function setPaperResult(result) {
  const key = JSON.stringify(result.givens);
  if (key !== state.puzzleKey) Object.assign(state, { puzzleKey: key, shown: 0 });
  state.result = result;
  state.corners = result.corners;
  if (result.pinned) Object.assign(state, { pinned: true, autoPinned: true, track: "tracking" });
  render();
}

function revealNext(now) {
  const order = state.result?.order;
  if (view() !== "paper" || state.step || !order || state.shown >= order.length) return;
  if (now - state.lastReveal < REVEAL_MS) return;
  state.shown++;
  state.lastReveal = now;
  render();
}

function paperStatus() {
  if (view() === "check") return "checking";
  if (!state.pythonReady) return "loading";
  if (state.pinned && state.track === "lost") return "lost";
  const status = state.result?.status;
  if (state.source === "still" && !status) return "photo";
  if (state.source === "still" && status === "no_grid") return "no_grid_photo";
  return status ?? "no_grid";
}

function renderPaper() {
  const result = state.result;
  const order = result?.order ?? [];
  const solved = result?.status === "solved";
  setStatus(paperStatus());
  $("counter").textContent = solved ? `${state.shown} of ${order.length} digits shown` : "";
  $("back").disabled = !state.step || state.shown === 0;
  $("next").disabled = !state.step || state.shown >= order.length;
  $("check").disabled = !result?.givens;
  drawBoard(boardCanvas, paperBoard(result, view() === "check"));
  stage.showBoard(state.pinned && state.track === "lost" ? null : state.corners);
}

/**
 * The board drawn on the paper. A hand-drawn board (result.drawn) gets a clean board with
 * straight lines over it, hiding its uneven lines, with its digits redrawn in black. With
 * corners set by hand the grid is drawn too, so the user sees where the app thinks it is.
 */
function paperBoard(result, checking) {
  const cells = paperCells(result, checking);
  if (result?.drawn) {
    if (!checking) cells.unshift(...givenCells(result.givens));
    return { lines: COLORS.app, background: "rgba(255,255,255,0.92)", cells };
  }
  return { lines: state.pinned ? COLORS.app : null, cells };
}

function givenCells(givens) {
  const cells = [];
  givens.forEach((row, r) => row.forEach((d, c) => d && cells.push({ r, c, digit: d, color: COLORS.ink })));
  return cells;
}

/** The app's digits in blue, the next one in orange, and the user's digits outlined green or red. */
function paperCells(result, checking) {
  if (!result?.givens) return [];
  const cells = [];
  const wrong = new Set(result.wrong.map(([r, c]) => r * 9 + c));
  result.user_digits.forEach((row, r) =>
    row.forEach((d, c) => d && cells.push({ r, c, outline: wrong.has(r * 9 + c) ? COLORS.wrong : COLORS.right })),
  );
  if (checking) {
    result.givens.forEach((row, r) =>
      row.forEach((d, c) => d && cells.push({ r, c, digit: d, color: COLORS.ink, fill: "rgba(255,255,255,0.9)" })),
    );
    return cells;
  }
  if (result.status !== "solved") return cells;
  for (const [r, c] of result.order.slice(0, state.shown)) {
    cells.push({ r, c, digit: result.solution[r][c], color: COLORS.app });
  }
  const next = result.order[state.shown];
  if (state.step && next) cells.push({ r: next[0], c: next[1], fill: tint(COLORS.next, 0.3), outline: COLORS.next });
  return cells;
}

async function openPhoto(blob) {
  const image = await createImageBitmap(blob);
  const scale = Math.min(1, STILL_SIZE / Math.max(image.width, image.height));
  still.width = Math.round(image.width * scale);
  still.height = Math.round(image.height * scale);
  still.getContext("2d").drawImage(image, 0, 0, still.width, still.height);
  stopCamera();
  setSource("still");
  startPaper();
  await ready;
  const token = state.token;
  const result = await call("photo", [], grabFrame());
  if (token === state.token) setPaperResult(result);
}

/** Check: freeze the picture so the user can fix misread digits. */
function startCheck() {
  if (state.source === "video") {
    still.width = video.videoWidth;
    still.height = video.videoHeight;
    still.getContext("2d").drawImage(video, 0, 0);
    setSource("still");
    state.frozen = true;
  }
  show("check");
}

function endCheck() {
  if (state.frozen) setSource("video");
  state.frozen = false;
  show("paper");
}

async function fixDigit(r, c) {
  const digit = await pickDigit();
  if (digit === null) return;
  const { givens, user_digits: userDigits } = state.result;
  givens[r][c] = digit;
  userDigits[r][c] = 0;
  Object.assign(state.result, await call("set_givens", [givens, userDigits]));
  Object.assign(state, { puzzleKey: JSON.stringify(givens), shown: 0 });
  render();
}

// ---------- placing a board by hand ----------

/**
 * Place the board by hand. A board already in view starts where it is (on its surface).
 * A new virtual board starts on the surface guessed by surfaceGuess; a sheet of paper in view
 * takes over (snapToSheet). Paper puzzles start in tilt mode: corners go onto the grid.
 */
function startPlacing(kind) {
  Object.assign(state, { placing: kind, snapped: false });
  show("place");
  setTilt(kind === "paper");
  if (state.corners && state.track !== "lost") placement.placeQuad(state.corners.map((p) => stage.toScreen(p)), true);
  else resetPlacement();
}

function resetPlacement() {
  state.snapped = false;
  if (state.placing === "paper") placement.placeQuad(placement.centred());
  else placement.place(...surfaceGuess());
}

/**
 * The surface a new board starts on: the table below the phone when its tilt sensor says it
 * points down at one, else a surface facing the camera (a wall, or a laptop's view).
 */
function surfaceGuess() {
  const { scale } = stage.fit();
  const { width, height } = stage.frame;
  const tilt = motion.tilt !== null && motion.tilt <= MAX_TABLE_TILT ? Math.max(0, motion.tilt) : 0;
  const focal = FOCAL * Math.max(width, height) * scale;
  const surface = tiltedSurface((tilt * Math.PI) / 180, focal, stage.toScreen([width / 2, height / 2]));
  return [surface, { x: 0, y: 0, side: 0.6 * Math.min(width, height) * scale }];
}

function watchTilt({ beta, gamma }) {
  if (beta === null) return;
  const portrait = (screen.orientation?.angle ?? 0) % 180 === 0;
  const tilt = portrait ? beta : Math.abs(gamma);
  if (motion.tilt !== null && Math.abs(tilt - motion.tilt) < 1) return;
  motion.tilt = tilt;
  // Until the user touches it, a new board follows the phone's tilt.
  if (view() === "place" && state.placing === "virtual" && !placement.touched && !state.snapped) {
    placement.place(surfaceGuess()[0], placement.board);
  }
}

/** iOS shows the tilt sensor only after the user allows it, asked from a tap. */
function askForTilt() {
  window.DeviceOrientationEvent?.requestPermission?.().catch(() => {});
}

function setTilt(on) {
  placement.tilt = on;
  $("mode-move").classList.toggle("on", !on);
  $("mode-tilt").classList.toggle("on", on);
  render();
}

function showPlacing(screenCorners) {
  if (view() !== "place") return;
  drawBoard(boardCanvas, { lines: COLORS.placing, background: "rgba(255,255,255,0.15)" });
  stage.showBoard(screenCorners.map((p) => stage.toFrame(p)));
  if (state.snapped && !placement.touched) setStatus("snapped");
  else setStatus(placement.tilt ? "placing_tilt" : "placing");
}

async function finishPlacing() {
  const corners = placement.corners.map((p) => stage.toFrame(p));
  await ready;
  if (state.placing === "paper") {
    setPaperResult(await call("read_at", [corners], grabFrame()));
    if (state.source === "video") await pin(corners, false);
    return show("paper");
  }
  await pin(corners);
  show("play");
}

/** Start the surface tracker. around=false follows only the board (a printed puzzle). */
async function pin(corners, around = true) {
  const result = await call("pin", [corners, around], grabFrame());
  Object.assign(state, { pinned: true, autoPinned: false, corners: result.corners, track: result.track, onSheet: result.on_sheet });
}

// ---------- virtual board ----------

async function newPuzzle() {
  const level = await ask($("difficulty"));
  if (!level) return;
  if (level === "type") return startTyping();
  startGame(new Game(await call("generate", [level]), level));
}

async function startGame(game) {
  state.game = game;
  if (state.pinned) return show("play");
  await startCamera();
  Object.assign(state, { corners: null, track: null });
  startPlacing("virtual");
}

function renderPlay() {
  const game = state.game;
  drawBoard(boardCanvas, { lines: COLORS.app, background: "rgba(255,255,255,0.4)", cells: game.cells() });
  stage.showBoard(state.track === "lost" ? null : state.corners);
  const { written, wrong } = game.stats();
  $("play-info").textContent = `${LEVELS[game.difficulty]} • ${written} written, ${wrong} wrong`;
  if (game.done) setStatus("done");
  else if (state.track === "tracking") setStatus(state.onSheet ? "tracking_sheet" : "tracking");
  else setStatus(state.track);
}

async function playDigit(r, c) {
  if (state.game.isGiven(r, c)) return;
  const digit = await pickDigit();
  if (digit === null) return;
  state.game.set(r, c, digit);
  render();
}

async function hint() {
  const game = state.game;
  const { order } = await call("solve_grid", [game.givens, game.userDigits()]);
  const cell = order.find(([r, c]) => !game.isHinted(r, c));
  if (cell) game.hint(...cell);
  render();
}

// ---------- typing in a puzzle ----------

function startTyping() {
  state.typed = Array.from({ length: 9 }, () => Array(9).fill(0));
  $("typing-status").textContent = "Tap a cell to type the puzzle";
  $("typing-status").dataset.tone = "";
  show("typing");
}

function renderTyping() {
  const cells = [];
  state.typed.forEach((row, r) => row.forEach((d, c) => d && cells.push({ r, c, digit: d, color: COLORS.ink })));
  drawBoard($("flat"), { lines: COLORS.ink, background: "#ffffff", cells });
}

async function typeDigit(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const r = Math.min(8, Math.floor(((e.clientY - rect.top) / rect.height) * 9));
  const c = Math.min(8, Math.floor(((e.clientX - rect.left) / rect.width) * 9));
  const digit = await pickDigit();
  if (digit === null) return;
  state.typed[r][c] = digit;
  render();
}

async function useTyped() {
  const solved = await call("solve_grid", [state.typed]);
  if (solved.status !== "solved") {
    $("typing-status").textContent = "This puzzle has no solution. Check the digits.";
    $("typing-status").dataset.tone = "bad";
    return;
  }
  startGame(new Game({ givens: state.typed, solution: solved.solution }, "typed"));
}

// ---------- events ----------

function tapStage(e) {
  if (e.target.closest(".top, .controls") || !state.corners) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const cell = cellAt(state.corners, stage.toFrame([e.clientX - rect.left, e.clientY - rect.top]));
  if (!cell) return;
  if (view() === "check") run(() => fixDigit(...cell));
  if (view() === "play") run(() => playDigit(...cell));
}

function setStep(step) {
  state.step = step;
  $("live").classList.toggle("on", !step);
  $("step").classList.toggle("on", step);
  render();
}

function toggleFlash() {
  const on = !$("flash").classList.contains("on");
  stream?.getVideoTracks()[0].applyConstraints({ advanced: [{ torch: on }] });
  $("flash").classList.toggle("on", on);
}

$("start-camera").onclick = () => run(async () => {
  await startCamera();
  startPaper();
});
$("draw-board").onclick = () => {
  askForTilt();
  run(newPuzzle);
};
$("try-sample").onclick = () => run(async () => openPhoto(await (await fetch("sample.jpg")).blob()));
$("upload").onchange = (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (file) run(() => openPhoto(file));
};

$("close").onclick = () => goHome();
$("flash").onclick = toggleFlash;
$("stage").addEventListener("click", tapStage);
$("stage").addEventListener("gesturestart", (e) => e.preventDefault()); // Safari pinch zoom

$("live").onclick = () => setStep(false);
$("step").onclick = () => setStep(true);
$("next").onclick = () => { state.shown++; render(); };
$("back").onclick = () => { state.shown--; render(); };
$("check").onclick = startCheck;
$("reset-digits").onclick = () => {
  state.shown = 0;
  setStep(true); // in Live mode the digits would come back at once
};
$("corners").onclick = () => startPlacing("paper");
$("continue").onclick = endCheck;

$("mode-move").onclick = () => setTilt(false);
$("mode-tilt").onclick = () => setTilt(true);
$("reset-corners").onclick = resetPlacement;
$("done").onclick = () => run(finishPlacing);

$("new").onclick = () => run(newPuzzle);
$("hint").onclick = () => run(hint);
$("solve").onclick = () => { state.game.showSolution = true; render(); };
$("reset-game").onclick = () => { state.game.reset(); render(); };
$("adjust").onclick = () => startPlacing("virtual");

$("flat").onclick = (e) => run(() => typeDigit(e));
$("type-use").onclick = () => run(useTyped);
$("type-cancel").onclick = () => (state.pinned ? show("play") : goHome());

window.addEventListener("resize", () => {
  stage.measure();
  render();
});
window.addEventListener("deviceorientation", watchTilt);

ready.then(
  () => {
    state.pythonReady = true;
    $("python-status").textContent = "Ready. Everything runs on this device.";
    render();
  },
  (error) => ($("python-status").textContent = `The solver could not load: ${error.message}`),
);

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js");
requestAnimationFrame(loop);
