// Background thread: loads Pyodide (Python, numpy, OpenCV) and our Python code, then runs
// bridge.handle(name, args, pixels) for every message from the page.

import { loadPyodide } from "./pyodide/pyodide.mjs";

async function start() {
  const pyodide = await loadPyodide({ indexURL: new URL("./pyodide/", import.meta.url).href });
  await pyodide.loadPackage(["numpy", "opencv-python"], { messageCallback: () => {} });
  const code = await fetch(new URL("./py.zip", import.meta.url));
  pyodide.unpackArchive(await code.arrayBuffer(), "zip");
  return pyodide.pyimport("bridge");
}

const bridge = start();
bridge.then(
  () => postMessage({ type: "ready" }),
  (error) => postMessage({ type: "failed", error: String(error) }),
);

onmessage = async ({ data: { id, fn, args, frame } }) => {
  try {
    const py = await bridge;
    const json = frame
      ? py.handle(fn, JSON.stringify(args), frame.data, frame.width, frame.height)
      : py.handle(fn, JSON.stringify(args));
    postMessage({ id, result: JSON.parse(json) });
  } catch (error) {
    postMessage({ id, error: String(error) });
  }
};
