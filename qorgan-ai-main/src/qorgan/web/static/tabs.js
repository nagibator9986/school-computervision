/* In-page tabs, and the sidebar's expandable sections.
 *
 * **Both work without this file.** The tab panels are ordinary sections and the sidebar's
 * sub-lists are ordinary lists; with scripting off every panel is visible and every link
 * reachable, which is the behaviour a page of school records has to have. This only
 * collapses what is not being looked at.
 */
(function () {
  "use strict";

  document.querySelectorAll('[role="tablist"]').forEach(function (list) {
    var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
    if (tabs.length < 2) return;

    function show(target) {
      tabs.forEach(function (tab) {
        var selected = tab === target;
        tab.setAttribute("aria-selected", selected ? "true" : "false");
        tab.tabIndex = selected ? 0 : -1;
        var panel = document.getElementById(tab.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
      });
    }

    // Only now, once the script is known to be running, is hiding a panel safe.
    show(tabs.find(function (t) { return t.getAttribute("aria-selected") === "true"; }) || tabs[0]);

    list.addEventListener("click", function (event) {
      var tab = event.target.closest('[role="tab"]');
      if (tab) show(tab);
    });
    // Left/right walk the strip, which is what a screen reader user expects of a tablist.
    list.addEventListener("keydown", function (event) {
      var step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      var here = tabs.indexOf(document.activeElement);
      if (here === -1) return;
      var next = tabs[(here + step + tabs.length) % tabs.length];
      next.focus();
      show(next);
      event.preventDefault();
    });
  });
})();
