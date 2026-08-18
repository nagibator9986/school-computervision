// Refresh the live previews.
//
// Two rules, both learned from the legacy dashboard:
//
//   1. Never build DOM from strings. This file only swaps an <img> src and sets
//      textContent. A pupil named `<img src=x onerror=...>` must be inert.
//   2. Never poll the server for the whole world. The legacy re-fetched the entire
//      events table and the entire canteen log every 2.5 seconds, per client.
//      Here each <img> refetches its own JPEG, and one small JSON call carries status.

(function () {
  "use strict";

  const REFRESH_MS = 1000;

  function refreshImages() {
    document.querySelectorAll(".cam img[data-src]").forEach(function (img) {
      // The cache-buster is what makes the browser actually re-request the frame;
      // the endpoint itself sends no-store.
      img.src = img.dataset.src + "?t=" + Date.now();
    });
  }

  async function refreshStatus() {
    let payload;
    try {
      const response = await fetch("/api/cameras", { credentials: "same-origin" });
      if (!response.ok) return;
      payload = await response.json();
    } catch (err) {
      return; // the page keeps showing the last known state rather than blanking
    }

    payload.cameras.forEach(function (camera) {
      const card = document.querySelector('.cam[data-camera="' + CSS.escape(camera.name) + '"]');
      if (!card) return;

      const badge = card.querySelector('[data-role="status"]');
      if (badge) {
        badge.textContent = camera.status; // textContent, never innerHTML
        badge.className = "badge " + camera.status;
      }
      card.classList.toggle("offline", !camera.live);
    });
  }

  function tick() {
    refreshImages();
    refreshStatus();
  }

  // Only refresh while the tab is visible: a background tab does not need frames,
  // and the operator's laptop does not need to decode them.
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) tick();
  });

  setInterval(function () {
    if (!document.hidden) tick();
  }, REFRESH_MS);

  tick();
})();
