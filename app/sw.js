// Offline support. App files are fetched fresh when online and come from the cache when offline.
// The pinned Pyodide files never change, so they come from the cache once downloaded.

const PYODIDE_VERSION = "314.0.7"; // keep in sync with scripts/build.py (the build checks it)
const APP_CACHE = "app";
const PYODIDE_CACHE = `pyodide-${PYODIDE_VERSION}`;

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => {
  const keep = [APP_CACHE, PYODIDE_CACHE];
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.filter((name) => !keep.includes(name)).map((name) => caches.delete(name))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET" || !request.url.startsWith(self.registration.scope)) return;
  const pinned = new URL(request.url).pathname.includes("/pyodide/");
  event.respondWith(pinned ? cacheFirst(request, PYODIDE_CACHE) : networkFirst(request, APP_CACHE));
});

async function cacheFirst(request, cacheName) {
  return (await caches.match(request)) ?? fetchAndCache(request, cacheName);
}

async function networkFirst(request, cacheName) {
  try {
    return await fetchAndCache(request, cacheName);
  } catch (error) {
    return (await caches.match(request)) ?? Response.error();
  }
}

async function fetchAndCache(request, cacheName) {
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(cacheName);
    await cache.put(request, response.clone());
  }
  return response;
}
