// Calls the Python functions in app/py/bridge.py, which run in a background thread (worker.js).

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 0;
let onReady;
let onFailed;

/** Resolves when Python, numpy, OpenCV and sudoku_cv are loaded. */
export const ready = new Promise((resolve, reject) => {
  onReady = resolve;
  onFailed = reject;
});

worker.onmessage = ({ data }) => {
  if (data.type === "ready") return onReady();
  if (data.type === "failed") return onFailed(new Error(data.error));
  const { resolve, reject } = pending.get(data.id);
  pending.delete(data.id);
  if (data.error) reject(new Error(data.error));
  else resolve(data.result);
};

/**
 * bridge.<fn>(frame, ...args) when a frame (ImageData) is given, else bridge.<fn>(...args).
 * The frame's pixels are moved to the worker, not copied.
 */
export function call(fn, args = [], frame = null) {
  const id = nextId++;
  const message = { id, fn, args };
  const transfer = [];
  if (frame) {
    message.frame = { data: frame.data, width: frame.width, height: frame.height };
    transfer.push(frame.data.buffer);
  }
  worker.postMessage(message, transfer);
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}
