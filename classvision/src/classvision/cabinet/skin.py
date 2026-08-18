"""The platform's own skin, worn by the cabinet: `app.css` verbatim, plus the masthead.

--------------------------------------------------------------------------------
**WHY THE CSS IS A COPY IN A PYTHON CONSTANT AND NOT AN IMPORT OR A LINK.**

`DESIGN.md` is the contract: the cabinet is shown next to `qorgan-ai-main` and to a viewer
it has to be the same product, so its design is not invented — it is copied from
`qorgan-ai-main/src/qorgan/web/static/app.css`, token for token. Three ways to obtain that
string, and two of them are wrong here:

  * **Import it from the platform.** `classvision` does not depend on `qorgan` and must not
    start: they are separate packages, the analysis side runs on a machine where the web
    application is not installed, and `cabinet report` has to work there.
  * **Link it as a sibling file.** Cheaper to keep in sync, and it fails the way this
    surface must not fail: a psychologist mails ONE page to a colleague, or copies one out
    of the shared folder onto a stick, and the copy arrives with no stylesheet. An unstyled
    page of caveats is a page nobody reads to the end, and an unread caveat protects
    nobody — the same argument that put every long block behind a `<details>` rather than
    deleting it. `tests/test_cabinet.py::test_every_page_is_self_contained` asserts the
    rule this choice keeps: no page reaches outside itself for anything, ever.
  * **A verbatim copy, inlined into every page.** ~9 KB per page, 21 pages. That is the
    price, and it buys a folder of files that render identically on a machine with no
    network, this year and in five years.

The copy is therefore a copy, and the risk of a copy is drift. `PLATFORM_CSS` below is
BYTE-FOR-BYTE `app.css`, never edited — everything this surface adds is in `CABINET_CSS`
after it, so a diff against the platform file is a one-command check and
`tests/test_cabinet_skin.py` performs it whenever the platform tree is present beside this
one. Do not "tidy" the first constant; change the second.

--------------------------------------------------------------------------------
**THE MASTHEAD IS STATIC, AND ITS NAV LINKS ONLY WHAT THIS EXPORT CAN SERVE.**

`templates/base.html` draws every nav item from the capability the corresponding route is
gated on, so that nobody is ever offered a link into a 403. A static export has no session,
no capabilities and no routes at all, and the same rule applies with more force: a link to
`/events` in a folder of files opens nothing. So the nav carries the cabinet's OWN pages
and nothing else, `.who` says what this is instead of naming a user, and the brand points
at `index.html` rather than at the platform's landing page.

There is deliberately no separate «Уроки» item beside «Что сравнимо между уроками»: the
platform's `/lessons` is a route this export cannot serve, and the cabinet's own
cross-lesson page is that one page — listing it twice under two names would be exactly the
fake link the rule above forbids. Where a page belongs to one class, the nav gains that
class as a third item, because that IS a page in this folder.
"""

from __future__ import annotations

import html as html_escape
from typing import Any

# `qorgan-ai-main/src/qorgan/web/static/app.css`, VERBATIM. See the module docstring: this
# constant is a copy and its only job is to stay one. Nothing below the marker at its end
# belongs to the platform.
PLATFORM_CSS = r"""/* ГЕНЕРИРУЕМЫЙ ФАЙЛ — не правьте вручную. Источник: src/qorgan/web/tailwind.css, сборка: npm run css. */
/*! tailwindcss v4.3.3 | MIT License | https://tailwindcss.com */
@layer properties;
@layer theme, base, components, utilities;
@layer theme {
  :root, :host {
    --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans",
    "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace;
    --color-white: #fff;
    --spacing: 0.25rem;
    --font-weight-medium: 500;
    --font-weight-semibold: 600;
    --font-weight-bold: 700;
    --radius-md: 0.375rem;
    --default-font-family: var(--font-sans);
    --default-mono-font-family: var(--font-mono);
    --color-bg: #0c1015;
    --color-panel: #151b23;
    --color-raise: #1d252f;
    --color-line: #263040;
    --color-line-soft: #1e2733;
    --color-text: #e7ecf3;
    --color-muted: #8e9aab;
    --color-ok: #3fb27a;
    --color-alert: #e08c33;
    --color-critical: #e0524f;
    --color-offline: #4d5768;
    --color-ok-text: #7fd3a5;
    --color-alert-text: #f0b26b;
    --color-critical-text: #f3a09d;
    --color-accent: #5389e8;
    --color-accent-text: #a8c6ff;
    --color-accent-strong: #3b74dd;
    --color-accent-hover: #4d84ec;
    --text-fine: 0.75rem;
    --text-small: 0.8125rem;
    --text-body: 0.9375rem;
    --text-h2: 1.0625rem;
    --text-h1: 1.625rem;
    --radius-surface: 0.625rem;
    --radius-control: 0.375rem;
    --shadow-surface: 0 1px 2px rgb(0 0 0 / 0.30), 0 0 0 1px rgb(255 255 255 / 0.025) inset;
    --shadow-raised: 0 4px 16px -4px rgb(0 0 0 / 0.45), 0 0 0 1px rgb(255 255 255 / 0.04) inset;
    --shadow-float: 0 18px 48px -12px rgb(0 0 0 / 0.65), 0 0 0 1px rgb(255 255 255 / 0.05) inset;
    --ease-out-soft: cubic-bezier(0.22, 1, 0.36, 1);
  }
}
@layer base {
  *, ::after, ::before, ::backdrop, ::file-selector-button {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    border: 0 solid;
  }
  html, :host {
    line-height: 1.5;
    -webkit-text-size-adjust: 100%;
    tab-size: 4;
    font-family: var(--default-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji");
    font-feature-settings: var(--default-font-feature-settings, normal);
    font-variation-settings: var(--default-font-variation-settings, normal);
    -webkit-tap-highlight-color: transparent;
  }
  hr {
    height: 0;
    color: inherit;
    border-top-width: 1px;
  }
  abbr:where([title]) {
    -webkit-text-decoration: underline dotted;
    text-decoration: underline dotted;
  }
  h1, h2, h3, h4, h5, h6 {
    font-size: inherit;
    font-weight: inherit;
  }
  a {
    color: inherit;
    -webkit-text-decoration: inherit;
    text-decoration: inherit;
  }
  b, strong {
    font-weight: bolder;
  }
  code, kbd, samp, pre {
    font-family: var(--default-mono-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace);
    font-feature-settings: var(--default-mono-font-feature-settings, normal);
    font-variation-settings: var(--default-mono-font-variation-settings, normal);
    font-size: 1em;
  }
  small {
    font-size: 80%;
  }
  sub, sup {
    font-size: 75%;
    line-height: 0;
    position: relative;
    vertical-align: baseline;
  }
  sub {
    bottom: -0.25em;
  }
  sup {
    top: -0.5em;
  }
  table {
    text-indent: 0;
    border-color: inherit;
    border-collapse: collapse;
  }
  :-moz-focusring:where(:not(iframe)) {
    outline: auto;
  }
  progress {
    vertical-align: baseline;
  }
  summary {
    display: list-item;
  }
  ol, ul, menu {
    list-style: none;
  }
  img, svg, video, canvas, audio, iframe, embed, object {
    display: block;
    vertical-align: middle;
  }
  img, video {
    max-width: 100%;
    height: auto;
  }
  button, input, select, optgroup, textarea, ::file-selector-button {
    font: inherit;
    font-feature-settings: inherit;
    font-variation-settings: inherit;
    letter-spacing: inherit;
    color: inherit;
    border-radius: 0;
    background-color: transparent;
    opacity: 1;
  }
  :where(select:is([multiple], [size])) optgroup {
    font-weight: bolder;
  }
  :where(select:is([multiple], [size])) optgroup option {
    padding-inline-start: 20px;
  }
  ::file-selector-button {
    margin-inline-end: 4px;
  }
  ::placeholder {
    opacity: 1;
  }
  @supports (not (-webkit-appearance: -apple-pay-button))  or (contain-intrinsic-size: 1px) {
    ::placeholder {
      color: currentcolor;
      @supports (color: color-mix(in lab, red, red)) {
        color: color-mix(in oklab, currentcolor 50%, transparent);
      }
    }
  }
  textarea {
    resize: vertical;
  }
  ::-webkit-search-decoration {
    -webkit-appearance: none;
  }
  ::-webkit-date-and-time-value {
    min-height: 1lh;
    text-align: inherit;
  }
  ::-webkit-datetime-edit {
    display: inline-flex;
  }
  ::-webkit-datetime-edit-fields-wrapper {
    padding: 0;
  }
  ::-webkit-datetime-edit, ::-webkit-datetime-edit-year-field, ::-webkit-datetime-edit-month-field, ::-webkit-datetime-edit-day-field, ::-webkit-datetime-edit-hour-field, ::-webkit-datetime-edit-minute-field, ::-webkit-datetime-edit-second-field, ::-webkit-datetime-edit-millisecond-field, ::-webkit-datetime-edit-meridiem-field {
    padding-block: 0;
  }
  ::-webkit-calendar-picker-indicator {
    line-height: 1;
  }
  :-moz-ui-invalid {
    box-shadow: none;
  }
  button, input:where([type="button"], [type="reset"], [type="submit"]), ::file-selector-button {
    appearance: button;
  }
  ::-webkit-inner-spin-button, ::-webkit-outer-spin-button {
    height: auto;
  }
  [hidden]:where(:not([hidden="until-found"])) {
    display: none !important;
  }
}
@layer utilities {
  .visible {
    visibility: visible;
  }
  .block {
    display: block;
  }
  .grid {
    display: grid;
  }
  .hidden {
    display: none;
  }
  .inline {
    display: inline;
  }
  .table {
    display: table;
  }
  .ordinal {
    --tw-ordinal: ordinal;
    font-variant-numeric: var(--tw-ordinal,) var(--tw-slashed-zero,) var(--tw-numeric-figure,) var(--tw-numeric-spacing,) var(--tw-numeric-fraction,);
  }
  .filter {
    filter: var(--tw-blur,) var(--tw-brightness,) var(--tw-contrast,) var(--tw-grayscale,) var(--tw-hue-rotate,) var(--tw-invert,) var(--tw-saturate,) var(--tw-sepia,) var(--tw-drop-shadow,);
  }
}
@layer base {
  html {
    color-scheme: dark;
  }
  body {
    background: var(--color-bg);
    color: var(--color-text);
    font-family: var(--font-sans);
    font-size: var(--text-body);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    overflow-x: clip;
  }
  main {
    max-width: 87.5rem;
    padding: 1.75rem 2rem 4rem;
  }
  h1 {
    font-size: var(--text-h1);
    line-height: 1.2;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin: 0 0 1.375rem;
    text-wrap: balance;
  }
  h2 {
    font-size: var(--text-h2);
    line-height: 1.35;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 1.875rem 0 0.75rem;
    text-wrap: balance;
  }
  h3 {
    font-size: var(--text-body);
    font-weight: 600;
    margin: 1.25rem 0 0.5rem;
  }
  p {
    text-wrap: pretty;
  }
  :where(a, button, input, select, textarea, summary, [tabindex]):focus-visible {
    outline: 2px solid var(--color-accent-text);
    outline-offset: 2px;
    border-radius: 3px;
  }
  :where(a, button, input, select, summary, .tag, .cam, .tile) {
    transition: color 140ms var(--ease-out-soft), background-color 140ms var(--ease-out-soft), border-color 140ms var(--ease-out-soft), box-shadow 140ms var(--ease-out-soft);
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      transition-duration: 0.01ms !important;
    }
  }
  .cv-notes, .cv-checklist, .warning ul, .warning ol, .blank ul, .meta ul, .cv-reading ul {
    list-style: disc;
    padding-inline-start: 1.25rem;
  }
  .cv-notes > li, .cv-checklist > li, .warning ul > li, .warning ol > li, .blank ul > li, .meta ul > li, .cv-reading ul > li {
    display: list-item;
  }
  .warning ol {
    list-style: decimal;
  }
  code {
    font-family: var(--font-mono);
    font-size: 0.88em;
    color: var(--color-muted);
    background: color-mix(in srgb, #8e9aab 12%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-muted) 12%, transparent);
    }
    border-radius: 4px;
    padding: 0.0625rem 0.3125rem;
  }
  @media print {
    :root {
      --color-bg: #fff;
      --color-panel: #fff;
      --color-raise: #fff;
      --color-line: #b9bfc7;
      --color-line-soft: #d7dbe0;
      --color-text: #000;
      --color-muted: #45505f;
    }
    body {
      font-size: 10.5pt;
    }
    .topbar, .pages, .review, form.inline {
      display: none !important;
    }
    main {
      max-width: none;
      padding: 0;
    }
    a {
      color: inherit;
      text-decoration: underline;
    }
    .tile, .cv-reading, .cv-trend, .cv-footnotes, .blank, .block {
      break-inside: avoid;
    }
  }
}
@layer components {
  .shell {
    display: grid;
    grid-template-columns: 15.5rem minmax(0, 1fr);
    min-block-size: 100dvh;
  }
  .shell-bare {
    grid-template-columns: minmax(0, 1fr);
  }
  .shell-bare main {
    margin-inline: auto;
  }
  .sidebar {
    position: sticky;
    top: 0px;
    display: flex;
    flex-direction: column;
    gap: var(--spacing);
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 4);
    block-size: 100dvh;
    overflow-y: auto;
    background: var(--color-panel);
    border-inline-end: 1px solid var(--color-line);
  }
  .brand {
    margin-inline-start: calc(var(--spacing) * 2);
    margin-bottom: calc(var(--spacing) * 4);
    display: flex;
    align-items: center;
    gap: calc(var(--spacing) * 2.5);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    white-space: nowrap;
    text-decoration-line: none;
    color: var(--color-text);
    font-size: 0.9375rem;
    letter-spacing: -0.01em;
  }
  .brand::before {
    content: "";
    inline-size: 1.125rem;
    block-size: 1.25rem;
    flex: none;
    background: linear-gradient(160deg, var(--color-accent-text), var(--color-accent-strong));
    clip-path: polygon(50% 0, 100% 24%, 100% 64%, 50% 100%, 0 64%, 0 24%);
  }
  .sidenav {
    flex: auto;
  }
  .sidenav-group {
    margin-inline-start: calc(var(--spacing) * 2);
    margin-top: calc(var(--spacing) * 4);
    margin-bottom: calc(var(--spacing) * 1.5);
    text-transform: uppercase;
    color: var(--color-muted);
    font-size: 0.6875rem;
    letter-spacing: 0.08em;
    font-weight: 600;
  }
  .sidenav ul {
    margin: 0px;
    list-style-type: none;
    padding: 0px;
  }
  .sidenav a {
    position: relative;
    display: flex;
    align-items: center;
    gap: calc(var(--spacing) * 2);
    border-radius: var(--radius-md);
    padding-inline: calc(var(--spacing) * 2.5);
    padding-block: calc(var(--spacing) * 1.5);
    text-decoration-line: none;
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  .sidenav a:hover {
    color: var(--color-text);
    background: var(--color-raise);
  }
  .sidenav a[aria-current="page"] {
    color: var(--color-text);
    background: var(--color-raise);
    font-weight: 500;
  }
  .sidenav a[aria-current="page"]::before {
    content: "";
    position: absolute;
    inset-inline-start: calc(var(--spacing) * 0);
    border-start-end-radius: calc(infinity * 1px);
    border-end-end-radius: calc(infinity * 1px);
    inset-block: 0.375rem;
    inline-size: 2px;
    background: var(--color-accent);
  }
  .sidenav-sub > ul {
    margin-inline-start: calc(var(--spacing) * 4);
    overflow: hidden;
    padding-inline-start: calc(var(--spacing) * 2);
    border-inline-start: 1px solid var(--color-line);
    max-block-size: 0;
    opacity: 0;
    transition: max-block-size 200ms var(--ease-out-soft), opacity 140ms var(--ease-out-soft);
  }
  .sidenav-sub.open > ul {
    max-block-size: 8rem;
    opacity: 1;
  }
  .sidenav-sub a {
    font-size: var(--text-fine);
  }
  .sidenav-sub a[aria-current="page"]::before {
    inline-size: 0;
  }
  .session {
    margin-top: calc(var(--spacing) * 4);
    display: flex;
    flex-direction: column;
    gap: calc(var(--spacing) * 2);
    padding-inline: calc(var(--spacing) * 2);
    padding-top: calc(var(--spacing) * 3);
    border-block-start: 1px solid var(--color-line);
  }
  .who {
    display: flex;
    align-items: center;
    gap: calc(var(--spacing) * 2);
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .who-name {
    flex: auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .who-role {
    border-radius: calc(infinity * 1px);
    padding-inline: calc(var(--spacing) * 2);
    padding-block: 1px;
    white-space: nowrap;
    border: 1px solid var(--color-line);
    font-size: 0.625rem;
  }
  .link {
    cursor: pointer;
    border-style: var(--tw-border-style);
    border-width: 0px;
    background-color: transparent;
    padding: 0px;
    text-align: start;
    white-space: nowrap;
    color: var(--color-muted);
    font: inherit;
    font-size: var(--text-small);
  }
  .link:hover {
    color: var(--color-text);
  }
  .skip {
    position: absolute;
    border-radius: 0.25rem;
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 2);
    inset-inline-start: -100vw;
    background: var(--color-accent-strong);
    color: #fff;
    z-index: 50;
  }
  .skip:focus {
    inset-inline-start: 0.5rem;
    top: 0.5rem;
  }
  @media (width <= 60rem) {
    .shell {
      display: block;
    }
    .sidebar {
      flex-direction: row;
      align-items: center;
      gap: calc(var(--spacing) * 3);
      overflow-x: auto;
      padding-block: calc(var(--spacing) * 2);
      block-size: auto;
      position: static;
      border-inline-end: 0;
      border-block-end: 1px solid var(--color-line);
    }
    .brand {
      margin-bottom: 0px;
    }
    .sidenav {
      display: flex;
      align-items: center;
      gap: calc(var(--spacing) * 3);
    }
    .sidenav-group {
      display: none;
    }
    .sidenav ul {
      display: flex;
      align-items: center;
      gap: var(--spacing);
    }
    .sidenav-sub, .session form {
      display: none;
    }
    .session {
      margin: 0px;
      flex-direction: row;
      border-style: var(--tw-border-style);
      border-width: 0px;
      padding: 0px;
    }
  }
  .login {
    margin-inline: auto;
    padding: calc(var(--spacing) * 8);
    padding-bottom: calc(var(--spacing) * 7);
    max-inline-size: 22rem;
    margin-block-start: 13vh;
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-float);
  }
  .login h1 {
    margin: 0px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: calc(var(--spacing) * 2.5);
    font-size: 1.3125rem;
  }
  .login h1::before {
    content: "";
    inline-size: 1.125rem;
    block-size: 1.25rem;
    flex: none;
    background: linear-gradient(160deg, var(--color-accent-text), var(--color-accent-strong));
    clip-path: polygon(50% 0, 100% 24%, 100% 64%, 50% 100%, 0 64%, 0 24%);
  }
  .login .lede {
    margin-top: calc(var(--spacing) * 1.5);
    margin-bottom: calc(var(--spacing) * 6);
    text-align: center;
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .login label {
    margin-bottom: calc(var(--spacing) * 4);
    display: block;
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  .login input {
    margin-top: calc(var(--spacing) * 1.5);
    width: 100%;
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 2.5);
  }
  .login input:focus {
    border-color: var(--color-accent);
  }
  .login button {
    margin-top: calc(var(--spacing) * 2.5);
    width: 100%;
    cursor: pointer;
    border-style: var(--tw-border-style);
    border-width: 0px;
    padding-block: calc(var(--spacing) * 2.5);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    color: var(--color-white);
    background: linear-gradient(180deg, var(--color-accent-hover), var(--color-accent-strong));
    border-radius: var(--radius-control);
    font: inherit;
    font-weight: 600;
    box-shadow: 0 1px 0 rgb(255 255 255 / 0.14) inset;
  }
  .login button:hover {
    filter: brightness(1.08);
  }
  select, textarea, input:not([type="hidden"], [type="checkbox"], [type="radio"], [type="submit"]) {
    background: var(--color-bg);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-control);
    color: var(--color-text);
    font: inherit;
    font-size: var(--text-small);
    padding: 0.375rem 0.5rem;
    color-scheme: dark;
  }
  select:hover, textarea:hover, input:not([type="hidden"]):hover {
    border-color: var(--color-muted);
  }
  select {
    cursor: pointer;
  }
  textarea {
    field-sizing: content;
    min-block-size: 4lh;
  }
  .filters {
    margin-bottom: calc(var(--spacing) * 5);
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: calc(var(--spacing) * 3.5);
  }
  .filters label {
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .filters select, .filters input {
    margin-top: calc(var(--spacing) * 1.5);
    display: block;
    background: var(--color-panel);
  }
  .button {
    display: inline-flex;
    cursor: pointer;
    align-items: center;
    gap: calc(var(--spacing) * 2);
    padding-inline: calc(var(--spacing) * 3.5);
    white-space: nowrap;
    text-decoration-line: none;
    min-block-size: 2.125rem;
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-control);
    color: var(--color-text);
    font: inherit;
    font-size: var(--text-small);
  }
  .button:hover {
    border-color: var(--color-muted);
    background: var(--color-raise);
  }
  .button:disabled, .button[aria-disabled="true"] {
    color: var(--color-muted);
    background: var(--color-bg);
    cursor: default;
  }
  .button:disabled:hover, .button[aria-disabled="true"]:hover {
    border-color: var(--color-line);
    background: var(--color-bg);
  }
  .filters button.confirm, .review button {
    cursor: pointer;
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 1.5);
    background: var(--color-bg);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-control);
    color: var(--color-text);
    font: inherit;
    font-size: var(--text-fine);
  }
  .review {
    margin-top: calc(var(--spacing) * 3);
    display: flex;
    gap: calc(var(--spacing) * 2);
  }
  .review .confirm:hover {
    border-color: var(--color-critical);
    color: var(--color-critical-text);
  }
  .review .reject:hover {
    border-color: var(--color-ok);
    color: var(--color-ok-text);
  }
  form.inline {
    display: inline-flex;
    align-items: center;
    gap: calc(var(--spacing) * 1.5);
  }
  form.inline select {
    font-size: var(--text-fine);
  }
  form.inline button {
    cursor: pointer;
    border-style: var(--tw-border-style);
    border-width: 0px;
    background-color: transparent;
    padding: 0px;
    text-decoration-line: underline;
    color: var(--color-muted);
    font: inherit;
    font-size: var(--text-fine);
  }
  form.inline button.reject:hover {
    color: var(--color-critical-text);
  }
  form.inline button.confirm:hover {
    color: var(--color-ok-text);
  }
  .pages {
    margin-top: calc(var(--spacing) * 6);
    display: flex;
    align-items: center;
    gap: calc(var(--spacing) * 2.5);
  }
  .pages a {
    display: inline-flex;
    align-items: center;
    padding-inline: calc(var(--spacing) * 3);
    text-decoration-line: none;
    min-block-size: 2rem;
    border: 1px solid var(--color-line);
    border-radius: var(--radius-control);
    color: var(--color-text);
    font-size: var(--text-small);
  }
  .pages a:hover {
    border-color: var(--color-muted);
    background: var(--color-raise);
  }
  .pages .muted {
    border-style: var(--tw-border-style);
    border-width: 0px;
    padding: 0px;
  }
  .crumbs {
    margin-bottom: calc(var(--spacing) * 3);
    display: flex;
    align-items: center;
    gap: calc(var(--spacing) * 2);
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .crumbs a {
    color: var(--color-muted);
  }
  .crumbs a:hover {
    color: var(--color-text);
  }
  .crumbs [aria-current="page"] {
    color: var(--color-text);
  }
  .card-grid {
    display: grid;
    gap: calc(var(--spacing) * 4);
    grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
  }
  .class-card {
    display: flex;
    flex-direction: column;
    gap: calc(var(--spacing) * 2.5);
    padding: calc(var(--spacing) * 4);
    text-decoration-line: none;
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-block-start: 3px solid var(--color-offline);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
    color: var(--color-text);
  }
  .class-card:hover {
    border-color: var(--color-muted);
    box-shadow: var(--shadow-raised);
  }
  .class-card.signed {
    border-block-start-color: var(--color-ok);
  }
  .class-card-key {
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    font-size: 1.375rem;
    letter-spacing: -0.02em;
  }
  .class-card-rooms {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing);
  }
  .class-card-facts {
    margin: 0px;
    display: grid;
    column-gap: calc(var(--spacing) * 3);
    row-gap: var(--spacing);
    grid-template-columns: max-content 1fr;
    font-size: var(--text-fine);
  }
  .class-card-facts dt {
    color: var(--color-muted);
  }
  .class-card-facts dd {
    margin: 0px;
    font-variant-numeric: tabular-nums;
  }
  .class-card-plan {
    margin-top: auto;
    padding-top: calc(var(--spacing) * 2.5);
    font-size: var(--text-fine);
    border-block-start: 1px solid var(--color-line-soft);
  }
  .class-card-plan.ok {
    color: var(--color-ok-text);
  }
  .class-card-plan.part {
    color: var(--color-alert-text);
  }
  .class-card-plan.none {
    color: var(--color-muted);
  }
  .tabs {
    margin-bottom: calc(var(--spacing) * 4);
    display: flex;
    gap: var(--spacing);
    border-block-end: 1px solid var(--color-line);
  }
  .tab {
    position: relative;
    cursor: pointer;
    border-style: var(--tw-border-style);
    border-width: 0px;
    background-color: transparent;
    padding-inline: calc(var(--spacing) * 3.5);
    padding-block: calc(var(--spacing) * 2);
    color: var(--color-muted);
    font: inherit;
    font-size: var(--text-small);
  }
  .tab:hover {
    color: var(--color-text);
  }
  .tab[aria-selected="true"] {
    color: var(--color-text);
    font-weight: 500;
  }
  .tab[aria-selected="true"]::after {
    content: "";
    position: absolute;
    inset-inline: 0px;
    border-top-left-radius: calc(infinity * 1px);
    border-top-right-radius: calc(infinity * 1px);
    inset-block-end: -1px;
    block-size: 2px;
    background: var(--color-accent);
  }
  .room {
    margin-bottom: calc(var(--spacing) * 7);
  }
  .room h2 {
    margin-top: 0px;
  }
  main a {
    color: var(--color-accent-text);
    text-decoration-line: none;
  }
  main a:hover {
    text-decoration-line: underline;
  }
  .muted {
    color: var(--color-muted);
    font-weight: 400;
  }
  h1 .muted {
    font-size: var(--text-body);
  }
  h2 .muted {
    font-size: var(--text-small);
  }
  .meta {
    color: var(--color-muted);
    font-size: var(--text-small);
    max-inline-size: 96ch;
  }
  .cv-lede {
    margin: 0px;
    margin-bottom: calc(var(--spacing) * 4);
    color: var(--color-text);
    max-inline-size: 90ch;
  }
  .cv-note {
    margin-block: calc(var(--spacing) * 2);
    margin-bottom: calc(var(--spacing) * 4);
    color: var(--color-muted);
    font-size: var(--text-small);
    max-inline-size: 100ch;
  }
  .cv-notes {
    margin-block: calc(var(--spacing) * 2);
    margin-bottom: calc(var(--spacing) * 4);
    padding-inline-start: calc(var(--spacing) * 5);
    max-inline-size: 100ch;
  }
  .cv-notes li {
    margin-block: var(--spacing);
    font-size: var(--text-small);
  }
  .table-scroll, .cv-scroll {
    overflow-x: auto;
    container-type: inline-size;
  }
  table {
    margin-bottom: calc(var(--spacing) * 2);
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: var(--text-small);
  }
  table th, table td {
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 2.5);
    text-align: start;
    border-block-end: 1px solid var(--color-line-soft);
  }
  table th {
    color: var(--color-muted);
    font-weight: 500;
    white-space: nowrap;
  }
  table td {
    font-variant-numeric: tabular-nums;
  }
  table thead th {
    position: sticky;
    top: 0px;
    z-index: 10;
    background: color-mix(in srgb, #0c1015 92%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-bg) 92%, transparent);
    }
    backdrop-filter: blur(8px);
    border-block-end: 1px solid var(--color-line);
  }
  .cv-scroll thead th, .camera-config thead th {
    background: color-mix(in srgb, #151b23 92%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-panel) 92%, transparent);
    }
  }
  tbody tr:hover td {
    background: color-mix(in srgb, #8e9aab 6%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-muted) 6%, transparent);
    }
  }
  td.empty-cell {
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 6);
    text-align: center;
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  td.rowmsg {
    color: var(--color-text);
    font-size: var(--text-small);
  }
  td.rowmsg.critical {
    color: var(--color-critical-text);
  }
  td.rowmsg.alert {
    color: var(--color-alert-text);
  }
  td.error {
    color: var(--color-muted);
    font-size: var(--text-fine);
    font-family: var(--font-mono);
    max-inline-size: 26rem;
    overflow-wrap: anywhere;
  }
  td.wrap {
    max-inline-size: 38rem;
    overflow-wrap: anywhere;
  }
  td.nowrap {
    white-space: nowrap;
    color: var(--color-muted);
  }
  .journal {
    table-layout: fixed;
  }
  .journal th:nth-child(1) {
    inline-size: 16ch;
  }
  .journal th:nth-child(2) {
    inline-size: 11ch;
  }
  .journal th:nth-child(3) {
    inline-size: 15ch;
  }
  .journal th:nth-child(4) {
    inline-size: 20ch;
  }
  .journal .fields {
    margin-inline-start: calc(var(--spacing) * 1.5);
    display: inline-flex;
    flex-wrap: wrap;
    gap: var(--spacing);
  }
  .journal pre {
    margin-top: calc(var(--spacing) * 2);
    overflow-x: auto;
    padding: calc(var(--spacing) * 2);
    margin-block-end: 0;
    background: var(--color-bg);
    border: 1px solid var(--color-line);
    border-radius: 4px;
    font-size: var(--text-fine);
    white-space: pre;
  }
  .journal summary {
    margin-top: calc(var(--spacing) * 1.5);
    cursor: pointer;
    font-size: var(--text-fine);
  }
  .tiles {
    margin-bottom: calc(var(--spacing) * 6);
    display: flex;
    flex-wrap: wrap;
    gap: calc(var(--spacing) * 3.5);
  }
  .tile {
    flex: 1;
    flex-basis: 0px;
    padding: calc(var(--spacing) * 4);
    min-inline-size: 9.5rem;
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-block-start: 3px solid var(--color-offline);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
  }
  .tile.ok {
    border-block-start-color: var(--color-ok);
  }
  .tile.alert {
    border-block-start-color: var(--color-alert);
  }
  .tile.critical {
    border-block-start-color: var(--color-critical);
  }
  .tile .value {
    display: block;
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    font-size: 1.8125rem;
    line-height: 1.1;
    letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums;
  }
  .tile .label {
    margin-top: calc(var(--spacing) * 1.5);
    display: block;
    color: var(--color-muted);
    font-size: var(--text-fine);
    line-height: 1.35;
  }
  .blank {
    margin-top: var(--spacing);
    margin-bottom: calc(var(--spacing) * 5);
    padding: calc(var(--spacing) * 5);
    background: var(--color-panel);
    border: 1px dashed var(--color-line);
    border-radius: var(--radius-surface);
    color: var(--color-muted);
    font-size: var(--text-small);
    max-inline-size: 78ch;
  }
  .blank strong {
    margin-bottom: calc(var(--spacing) * 1.5);
    display: block;
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text);
    font-size: var(--text-body);
  }
  .warning {
    margin-bottom: calc(var(--spacing) * 5);
    padding-inline: calc(var(--spacing) * 4);
    padding-block: calc(var(--spacing) * 3);
    background: color-mix(in srgb, #e08c33 12%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-alert) 12%, transparent);
    }
    border: 1px solid color-mix(in srgb, #e08c33 55%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      border: 1px solid color-mix(in oklab, var(--color-alert) 55%, transparent);
    }
    border-radius: var(--radius-surface);
    color: var(--color-alert-text);
    font-size: var(--text-small);
    max-inline-size: 110ch;
  }
  .warning h2 {
    margin-top: 0px;
    color: inherit;
  }
  .warning a {
    color: inherit;
  }
  .warning .meta {
    color: inherit;
    opacity: 0.82;
  }
  .error {
    margin-bottom: calc(var(--spacing) * 4);
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 2.5);
    background: color-mix(in srgb, #e0524f 15%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-critical) 15%, transparent);
    }
    border: 1px solid color-mix(in srgb, #e0524f 60%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      border: 1px solid color-mix(in oklab, var(--color-critical) 60%, transparent);
    }
    border-radius: var(--radius-control);
    color: var(--color-critical-text);
    font-size: var(--text-small);
  }
  .tag {
    margin-inline-end: var(--spacing);
    display: inline-block;
    border-radius: 0.25rem;
    padding-inline: calc(var(--spacing) * 1.5);
    padding-block: 1px;
    border: 1px solid var(--color-line);
    color: var(--color-muted);
    font-size: 0.6875rem;
  }
  .tag.ok, .tag.running, .tag.sent {
    border-color: var(--color-ok);
    color: var(--color-ok-text);
  }
  .tag.alert, .tag.degraded {
    border-color: var(--color-alert);
    color: var(--color-alert-text);
  }
  .tag.off, .tag.crashed, .tag.failed, .tag.critical, .tag.error, .tag.confirmed {
    border-color: var(--color-critical);
    color: var(--color-critical-text);
  }
  .tag.new, .tag.suspicion, .tag.reviewed {
    border-color: var(--color-muted);
    color: var(--color-muted);
  }
  .tag.false_positive {
    border-color: var(--color-ok);
    color: var(--color-ok-text);
  }
  .tag.queued {
    border-color: var(--color-alert);
    color: var(--color-alert-text);
  }
  .tag.camera {
    border-color: var(--color-alert);
    color: var(--color-alert-text);
  }
  .tag.profile {
    border-color: var(--color-accent);
    color: var(--color-accent-text);
  }
  .tag.default {
    border-color: var(--color-muted);
    color: var(--color-muted);
  }
  tr.notification .tag.queued {
    border-color: var(--color-muted);
    color: var(--color-muted);
  }
  tr.notification .tag.failed {
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
  }
  tr.notification.failed td {
    background: color-mix(in srgb, #e0524f 8%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-critical) 8%, transparent);
    }
  }
  tr.notification.failed:hover td {
    background: color-mix(in srgb, #e0524f 15%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-critical) 15%, transparent);
    }
  }
  .grid {
    display: grid;
    gap: calc(var(--spacing) * 4);
    grid-template-columns: repeat(auto-fill, minmax(19rem, 1fr));
  }
  .cam {
    overflow: hidden;
    padding-bottom: calc(var(--spacing) * 3.5);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
    container-type: inline-size;
  }
  .cam:hover {
    box-shadow: var(--shadow-raised);
  }
  .cam .frame {
    position: relative;
    aspect-ratio: 16 / 9;
    background: #000;
  }
  .cam img {
    display: block;
    height: 100%;
    width: 100%;
    object-fit: cover;
  }
  .cam.offline img {
    opacity: 25%;
    filter: grayscale(1);
  }
  .cam img {
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .cam h2 {
    margin-inline: calc(var(--spacing) * 3.5);
    margin-top: calc(var(--spacing) * 3.5);
    margin-bottom: var(--spacing);
    font-size: var(--text-body);
  }
  .cam .meta {
    margin-inline: calc(var(--spacing) * 3.5);
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .cam .meta.error {
    color: var(--color-critical-text);
    font-size: var(--text-small);
  }
  .badge {
    position: absolute;
    inset-inline-end: calc(var(--spacing) * 2);
    top: calc(var(--spacing) * 2);
    border-radius: calc(infinity * 1px);
    padding-inline: calc(var(--spacing) * 2);
    padding-block: calc(var(--spacing) * 0.5);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    color: var(--color-white);
    text-transform: uppercase;
    font-size: 0.6875rem;
    letter-spacing: 0.03em;
    background: var(--color-ok);
    box-shadow: 0 1px 6px rgb(0 0 0 / 0.45);
  }
  .badge.alert {
    background: var(--color-alert);
  }
  .badge.critical {
    background: var(--color-critical);
  }
  .badge.offline {
    background: var(--color-offline);
  }
  .camlist {
    display: flex;
    flex-direction: column;
    gap: calc(var(--spacing) * 3.5);
  }
  .camrow {
    display: grid;
    align-items: flex-start;
    gap: calc(var(--spacing) * 4);
    grid-template-columns: 17.5rem 1fr;
    padding-bottom: 0;
  }
  .camrow .frame {
    aspect-ratio: 16 / 9;
  }
  .camrow .camfacts {
    padding-inline-start: 0px;
    padding-inline-end: 0px;
    padding-top: calc(var(--spacing) * 3);
    padding-bottom: calc(var(--spacing) * 3.5);
  }
  .camrow h2 {
    margin-inline: 0px;
    margin-top: 0px;
    margin-bottom: var(--spacing);
  }
  .camrow .meta {
    margin-inline: 0px;
    margin-bottom: calc(var(--spacing) * 2);
  }
  .camrates {
    margin-top: calc(var(--spacing) * 2.5);
    display: grid;
    column-gap: calc(var(--spacing) * 3.5);
    row-gap: var(--spacing);
    grid-template-columns: max-content 1fr;
    font-size: var(--text-fine);
  }
  .camrates dt {
    color: var(--color-muted);
  }
  .camrates dd {
    margin: 0px;
  }
  @media (width <= 40rem) {
    .camrow {
      grid-template-columns: 1fr;
    }
    .camrow .camfacts {
      padding-inline: calc(var(--spacing) * 3.5);
      padding-top: 0px;
      padding-bottom: calc(var(--spacing) * 3.5);
    }
  }
  .events {
    display: flex;
    flex-direction: column;
    gap: calc(var(--spacing) * 3.5);
  }
  .event {
    display: grid;
    gap: calc(var(--spacing) * 4);
    padding: calc(var(--spacing) * 4);
    grid-template-columns: minmax(0, 18.75rem) minmax(0, 1fr);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-inline-start: 3px solid var(--color-offline);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
  }
  .event.alert {
    border-inline-start-color: var(--color-alert);
  }
  .event.critical {
    border-inline-start-color: var(--color-critical);
  }
  .event .evidence {
    overflow: hidden;
    aspect-ratio: 16/9;
    background: #000;
    border-radius: var(--radius-control);
  }
  .event .evidence img, .event .evidence video {
    height: 100%;
    width: 100%;
    object-fit: cover;
  }
  .event .no-media {
    display: flex;
    height: 100%;
    align-items: center;
    justify-content: center;
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .event h2 {
    margin-top: 0px;
    margin-bottom: calc(var(--spacing) * 1.5);
    font-size: var(--text-body);
  }
  @media (width <= 47.5rem) {
    .event {
      grid-template-columns: 1fr;
    }
  }
  .merge-card {
    margin-bottom: calc(var(--spacing) * 3.5);
    padding: calc(var(--spacing) * 4);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-inline-start: 3px solid var(--color-offline);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
  }
  .merge-card h3 {
    margin-top: 0px;
    margin-bottom: calc(var(--spacing) * 1.5);
    font-size: var(--text-body);
  }
  .pair {
    margin-block: calc(var(--spacing) * 3);
    display: flex;
    flex-wrap: wrap;
    gap: calc(var(--spacing) * 4);
  }
  .side {
    flex: 1;
    min-inline-size: 11rem;
  }
  .side img {
    width: 100%;
    overflow: hidden;
    padding: calc(var(--spacing) * 2);
    max-inline-size: 13.75rem;
    aspect-ratio: 3/4;
    object-fit: cover;
    background: #000;
    border-radius: var(--radius-control);
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .side .no-media {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--spacing) * 2.5);
    text-align: center;
    max-inline-size: 13.75rem;
    aspect-ratio: 3/4;
    background: var(--color-panel);
    border: 1px dashed var(--color-line);
    border-radius: var(--radius-control);
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .new-account {
    margin-bottom: calc(var(--spacing) * 6);
  }
  .new-account h2 {
    margin: 0px;
    margin-bottom: calc(var(--spacing) * 2.5);
    --tw-font-weight: var(--font-weight-medium);
    font-weight: var(--font-weight-medium);
    color: var(--color-muted);
    font-size: 0.875rem;
  }
  .camera-config {
    margin-bottom: calc(var(--spacing) * 2.5);
    padding-inline: calc(var(--spacing) * 4);
    padding-block: calc(var(--spacing) * 3);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-surface);
  }
  .camera-config summary {
    display: flex;
    cursor: pointer;
    align-items: center;
    gap: calc(var(--spacing) * 2.5);
    list-style: none;
  }
  .camera-config summary::-webkit-details-marker {
    display: none;
  }
  .camera-config summary::before {
    content: "\25B8";
    color: var(--color-muted);
    flex: none;
    font-size: 0.6875rem;
    transition: rotate 140ms var(--ease-out-soft);
  }
  .camera-config[open] > summary::before {
    rotate: 90deg;
  }
  .camera-config h3 {
    margin-top: calc(var(--spacing) * 4);
    margin-bottom: calc(var(--spacing) * 1.5);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  .camera-config code {
    word-break: break-all;
  }
  .blocks {
    margin-bottom: calc(var(--spacing) * 7);
    display: grid;
    gap: calc(var(--spacing) * 4);
    grid-template-columns: repeat(auto-fit, minmax(21rem, 1fr));
  }
  .block {
    padding-inline: calc(var(--spacing) * 4.5);
    padding-top: calc(var(--spacing) * 4);
    padding-bottom: calc(var(--spacing) * 1.5);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-inline-start: 3px solid var(--color-offline);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
  }
  .block.live {
    border-inline-start-color: var(--color-ok);
  }
  .block.anonymous {
    border-inline-start-color: var(--color-alert);
  }
  .block h2 {
    margin-top: 0px;
    margin-bottom: calc(var(--spacing) * 2.5);
    font-size: var(--text-h2);
  }
  .block .meta {
    margin: 0px;
    margin-bottom: calc(var(--spacing) * 2.5);
    font-size: var(--text-small);
  }
  .cv-unmeasured {
    font-style: italic;
    color: var(--color-muted);
    font-size: var(--text-fine);
    border-block-end: 1px dotted var(--color-muted);
  }
  .tile .value .cv-unmeasured {
    font-style: normal;
    font-size: 0.875rem;
    font-style: italic;
  }
  .cv-brief {
    color: var(--color-text);
  }
  .cv-brief sup {
    color: var(--color-alert);
    font-size: 0.625rem;
    margin-inline-start: 1px;
  }
  .cv-cov {
    display: inline-block;
    border-radius: 0.25rem;
    padding-inline: calc(var(--spacing) * 1.5);
    padding-block: 1px;
    min-inline-size: 2.875rem;
    background: linear-gradient(to right, color-mix(in srgb, #3fb27a 40%, transparent) var(--cov), color-mix(in srgb, #8e9aab 14%, transparent) var(--cov));
    @supports (color: color-mix(in lab, red, red)) {
      background: linear-gradient(to right, color-mix(in oklab, var(--color-ok) 40%, transparent) var(--cov), color-mix(in oklab, var(--color-muted) 14%, transparent) var(--cov));
    }
    font-variant-numeric: tabular-nums;
  }
  .cv-footnotes {
    margin-block: calc(var(--spacing) * 2.5);
    margin-bottom: calc(var(--spacing) * 5);
    padding-inline: calc(var(--spacing) * 3.5);
    padding-block: calc(var(--spacing) * 2.5);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-surface);
    max-inline-size: 100ch;
  }
  .cv-footnotes summary {
    cursor: pointer;
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  .cv-footnotes p, .cv-footnotes li {
    font-size: var(--text-small);
  }
  .cv-adult td {
    background: color-mix(in srgb, #8e9aab 7%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-muted) 7%, transparent);
    }
  }
  .cv-reading {
    margin-bottom: calc(var(--spacing) * 6);
    padding-inline: calc(var(--spacing) * 4.5);
    padding-block: calc(var(--spacing) * 3.5);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-inline-start: 3px solid var(--color-accent);
    border-radius: var(--radius-surface);
    max-inline-size: 110ch;
    box-shadow: var(--shadow-surface);
  }
  .cv-reading h2 {
    margin-top: 0px;
  }
  .cv-reading h3 {
    margin-top: calc(var(--spacing) * 4);
    margin-bottom: calc(var(--spacing) * 1.5);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  .cv-overview {
    margin: 0px;
    font-size: var(--text-body);
    line-height: 1.65;
  }
  .cv-reading details {
    margin-top: calc(var(--spacing) * 3.5);
  }
  .cv-reading summary {
    cursor: pointer;
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .cv-reading details p {
    color: var(--color-muted);
    font-size: var(--text-fine);
    max-inline-size: 100ch;
  }
  .cv-trend {
    margin-bottom: calc(var(--spacing) * 6);
    padding-inline: calc(var(--spacing) * 4.5);
    padding-block: calc(var(--spacing) * 3.5);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-inline-start: 3px solid var(--color-offline);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
  }
  .cv-trend-demo_only, .cv-trend-identity_not_established {
    border-inline-start-color: var(--color-alert);
  }
  .cv-trend-available {
    border-inline-start-color: var(--color-ok);
  }
  .cv-trend h2 {
    margin-top: 0px;
  }
  .cv-trend h3 {
    margin-top: calc(var(--spacing) * 4.5);
    margin-bottom: calc(var(--spacing) * 1.5);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    color: var(--color-muted);
    font-size: var(--text-small);
  }
  .cv-trend p {
    font-size: var(--text-small);
    max-inline-size: 100ch;
  }
  .cv-headline {
    margin: 0px;
    margin-bottom: calc(var(--spacing) * 2);
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    font-size: 1.0625rem;
  }
  .gates {
    margin-block: calc(var(--spacing) * 2);
    list-style-type: none;
    padding: 0px;
    max-inline-size: 110ch;
  }
  .gates li {
    position: relative;
    margin-block: calc(var(--spacing) * 2);
    padding-inline-start: calc(var(--spacing) * 6);
    font-size: var(--text-small);
  }
  .gates li::before {
    position: absolute;
    inset-inline-start: calc(var(--spacing) * 0);
    --tw-font-weight: var(--font-weight-bold);
    font-weight: var(--font-weight-bold);
  }
  .gates li.ok::before {
    content: "\2713";
    color: var(--color-ok);
  }
  .gates li.no::before {
    content: "\2014";
    color: var(--color-alert);
  }
  .gates .why {
    margin-top: calc(var(--spacing) * 0.5);
    display: block;
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .cv-checklist {
    padding-inline-start: calc(var(--spacing) * 6);
    max-inline-size: 110ch;
  }
  .cv-checklist li {
    margin-block: calc(var(--spacing) * 3);
    font-size: var(--text-small);
  }
  .cv-checklist .why, .cv-checklist .how {
    margin-top: var(--spacing);
    display: block;
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .cv-lesson-strip {
    margin-bottom: calc(var(--spacing) * 5);
  }
  .cv-strip-head {
    margin-bottom: calc(var(--spacing) * 1.5);
    font-size: var(--text-small);
  }
  .cv-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: calc(var(--spacing) * 2);
  }
  .cv-seg {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--spacing);
    padding-inline: calc(var(--spacing) * 1.5);
    padding-block: calc(var(--spacing) * 2);
    min-inline-size: 4.875rem;
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-control);
  }
  .cv-seg-value {
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    font-size: 1.125rem;
    font-variant-numeric: tabular-nums;
  }
  .cv-seg-bar {
    width: 100%;
    block-size: 3.375rem;
    border-radius: 3px;
    background: linear-gradient(to top, var(--color-accent) var(--h), color-mix(in srgb, #8e9aab 15%, transparent) var(--h));
    @supports (color: color-mix(in lab, red, red)) {
      background: linear-gradient(to top, var(--color-accent) var(--h), color-mix(in oklab, var(--color-muted) 15%, transparent) var(--h));
    }
  }
  .cv-seg-label {
    text-align: center;
    color: var(--color-muted);
    font-size: 0.6875rem;
  }
  .cv-seg-off .cv-seg-value {
    font-style: italic;
    color: var(--color-muted);
    font-size: var(--text-fine);
  }
  .cv-bar {
    display: inline-block;
    border-radius: 0.25rem;
    padding-inline: calc(var(--spacing) * 1.5);
    padding-block: 1px;
    min-inline-size: 3.75rem;
    font-variant-numeric: tabular-nums;
    background: linear-gradient(to right, color-mix(in srgb, #5389e8 50%, transparent) var(--w), color-mix(in srgb, #8e9aab 14%, transparent) var(--w));
    @supports (color: color-mix(in lab, red, red)) {
      background: linear-gradient(to right, color-mix(in oklab, var(--color-accent) 50%, transparent) var(--w), color-mix(in oklab, var(--color-muted) 14%, transparent) var(--w));
    }
  }
  .cv-legend {
    margin-bottom: calc(var(--spacing) * 4);
    display: flex;
    flex-wrap: wrap;
    gap: calc(var(--spacing) * 2);
  }
  .cv-key {
    display: inline-block;
    border-radius: 0.25rem;
    padding-inline: calc(var(--spacing) * 2);
    padding-block: calc(var(--spacing) * 0.5);
    font-size: 0.6875rem;
    border: 1px solid var(--color-line);
    color: var(--color-muted);
  }
  .cv-action {
    border-color: var(--color-accent);
    color: var(--color-accent-text);
  }
  .cv-posture {
    border-color: var(--color-alert);
    color: var(--color-alert-text);
  }
  .cv-moved {
    border-color: #8a6fd4;
    color: #b9a4f0;
  }
  .cv-seated {
    border-color: var(--color-ok);
    color: var(--color-ok-text);
  }
  .cv-unknown {
    border-color: var(--color-offline);
    color: var(--color-muted);
  }
  .cv-frame {
    margin-bottom: calc(var(--spacing) * 5);
    display: grid;
    align-items: flex-start;
    gap: calc(var(--spacing) * 4);
    padding: calc(var(--spacing) * 3.5);
    grid-template-columns: minmax(21rem, 3fr) minmax(18.75rem, 2fr);
    background: var(--color-panel);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-surface);
    box-shadow: var(--shadow-surface);
  }
  .cv-shot {
    position: relative;
    line-height: 0;
    background: #000;
    border-radius: var(--radius-control);
  }
  .cv-shot img {
    display: block;
    height: auto;
    width: 100%;
    border-radius: var(--radius-control);
  }
  .cv-box {
    position: absolute;
    border: 2px solid var(--color-accent-text);
    border-radius: 3px;
  }
  .cv-box.cv-action {
    border-color: var(--color-accent);
    box-shadow: 0 0 0 1px color-mix(in srgb, #5389e8 40%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      box-shadow: 0 0 0 1px color-mix(in oklab, var(--color-accent) 40%, transparent);
    }
  }
  .cv-box.cv-posture {
    border-color: var(--color-alert);
  }
  .cv-box.cv-moved {
    border-color: #8a6fd4;
  }
  .cv-box.cv-seated {
    border-color: var(--color-ok);
  }
  .cv-box.cv-unknown {
    border-color: var(--color-offline);
  }
  .cv-box-unmeasured {
    border-style: dashed;
    opacity: 80%;
  }
  .cv-box-label {
    position: absolute;
    padding-inline: calc(var(--spacing) * 1.5);
    padding-block: 1px;
    white-space: nowrap;
    inset-inline-start: -2px;
    inset-block-end: 100%;
    border-radius: 3px 3px 0 0;
    background: color-mix(in srgb, #0c1015 88%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-bg) 88%, transparent);
    }
    color: var(--color-text);
    font: 500 0.625rem/1.4 var(--font-sans);
  }
  .cv-shot-facts h2 {
    margin-top: 0px;
  }
  .cv-happening {
    margin: 0px;
    margin-bottom: calc(var(--spacing) * 2.5);
    font-size: var(--text-small);
  }
  .cv-caveat {
    margin-top: calc(var(--spacing) * 2);
    color: var(--color-muted);
    font-size: var(--text-fine);
    max-inline-size: 70ch;
  }
  @media (width <= 56.25rem) {
    .cv-frame {
      grid-template-columns: 1fr;
    }
  }
  .cv-scroll a, .cv-note a, .cv-strip-head a, .cv-lede a, .cv-reading a, .cv-trend a {
    color: var(--color-accent-text);
    text-decoration-line: none;
  }
  .cv-scroll a:hover, .cv-note a:hover, .cv-strip-head a:hover, .cv-lede a:hover, .cv-reading a:hover, .cv-trend a:hover {
    text-decoration-line: underline;
  }
  .tag.demo {
    border-color: var(--color-alert);
    color: var(--color-alert-text);
    background: color-mix(in srgb, #e08c33 18%, transparent);
    @supports (color: color-mix(in lab, red, red)) {
      background: color-mix(in oklab, var(--color-alert) 18%, transparent);
    }
    --tw-font-weight: var(--font-weight-semibold);
    font-weight: var(--font-weight-semibold);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .warning.cv-demo {
    border-width: 2px;
  }
}
@property --tw-ordinal {
  syntax: "*";
  inherits: false;
}
@property --tw-slashed-zero {
  syntax: "*";
  inherits: false;
}
@property --tw-numeric-figure {
  syntax: "*";
  inherits: false;
}
@property --tw-numeric-spacing {
  syntax: "*";
  inherits: false;
}
@property --tw-numeric-fraction {
  syntax: "*";
  inherits: false;
}
@property --tw-blur {
  syntax: "*";
  inherits: false;
}
@property --tw-brightness {
  syntax: "*";
  inherits: false;
}
@property --tw-contrast {
  syntax: "*";
  inherits: false;
}
@property --tw-grayscale {
  syntax: "*";
  inherits: false;
}
@property --tw-hue-rotate {
  syntax: "*";
  inherits: false;
}
@property --tw-invert {
  syntax: "*";
  inherits: false;
}
@property --tw-opacity {
  syntax: "*";
  inherits: false;
}
@property --tw-saturate {
  syntax: "*";
  inherits: false;
}
@property --tw-sepia {
  syntax: "*";
  inherits: false;
}
@property --tw-drop-shadow {
  syntax: "*";
  inherits: false;
}
@property --tw-drop-shadow-color {
  syntax: "*";
  inherits: false;
}
@property --tw-drop-shadow-alpha {
  syntax: "<percentage>";
  inherits: false;
  initial-value: 100%;
}
@property --tw-drop-shadow-size {
  syntax: "*";
  inherits: false;
}
@property --tw-font-weight {
  syntax: "*";
  inherits: false;
}
@property --tw-border-style {
  syntax: "*";
  inherits: false;
  initial-value: solid;
}
@layer properties {
  @supports ((-webkit-hyphens: none) and (not (margin-trim: inline))) or ((-moz-orient: inline) and (not (color:rgb(from red r g b)))) {
    *, ::before, ::after, ::backdrop {
      --tw-ordinal: initial;
      --tw-slashed-zero: initial;
      --tw-numeric-figure: initial;
      --tw-numeric-spacing: initial;
      --tw-numeric-fraction: initial;
      --tw-blur: initial;
      --tw-brightness: initial;
      --tw-contrast: initial;
      --tw-grayscale: initial;
      --tw-hue-rotate: initial;
      --tw-invert: initial;
      --tw-opacity: initial;
      --tw-saturate: initial;
      --tw-sepia: initial;
      --tw-drop-shadow: initial;
      --tw-drop-shadow-color: initial;
      --tw-drop-shadow-alpha: 100%;
      --tw-drop-shadow-size: initial;
      --tw-font-weight: initial;
      --tw-border-style: solid;
    }
  }
}
/* --- end of the verbatim copy of the platform stylesheet --- */
"""

# Everything the cabinet adds, and nothing else. Each rule says which platform component it
# extends and why that component was not enough; a rule that cannot say so is a local
# invention and `DESIGN.md` says to delete it rather than defend it.
CABINET_CSS = """
/* h3/h4, lists and in-flow links: the platform's pages have headings, tables and tiles and
   no running prose, so `app.css` never needed these. This surface is prose plus tables, and
   an unstyled `<a>` inside a table renders as browser-default blue — a colour that is in no
   token and means nothing here. Links take --text and keep their underline, so the page
   survives being printed in grey (DESIGN.md rule 4). */
h3 { font-size: 14px; margin: 22px 0 8px; font-weight: 600; }
ul { margin: 8px 0; padding-left: 20px; }
li { margin: 4px 0; }
main a { color: var(--color-text); }
main p { margin: 0 0 12px; }

/* The nav item for the page you are already on. Not a colour with a meaning: it is the
   same --text every hovered nav link takes, so the row reads as "you are here" and prints
   as bold-nothing. */
.topbar nav a[aria-current] { color: var(--color-text); }

/* A raised surface holding SENTENCES. The platform's raised surfaces are `.tile` (one
   number under a 3px accent) and `.cam` (a video frame with a caption); a paragraph needs
   neither the accent nor the frame, so this is the same --panel/--line/8px surface with
   nothing added to it. DESIGN.md names --panel "any raised surface: card, tile, topbar". */
.panel { background: var(--color-panel); border: 1px solid var(--color-line); border-radius: 8px;
  padding: 12px 16px; margin: 0 0 16px; }
.panel > :first-child { margin-top: 0; }
.panel > :last-child { margin-bottom: 0; }

/* `<details>` is the whole of the reorganisation. NOT ONE CAVEAT WAS DELETED; the long ones
   moved in here, behind a summary that states what it is about to say. No script, by
   construction: the element opens itself, and a caveat that needed JavaScript to appear is
   a caveat that disappears the day the JavaScript does. */
details { background: var(--color-panel); border: 1px solid var(--color-line); border-radius: 8px;
  padding: 10px 16px; margin: 0 0 16px; }
details > summary { cursor: pointer; color: var(--color-muted); font-size: 13px; }
details[open] > summary { margin-bottom: 10px; padding-bottom: 8px;
  border-bottom: 1px solid var(--color-line); }
/* The amber variant exists for exactly one kind of content -- a limitation of the
   measurement -- because --alert is the only accent a caveat may take and nothing else on
   this surface may take it (DESIGN.md rule 3). Matches `.warning`'s values. */
details.warn { background: rgba(217, 130, 43, .12); border-color: var(--color-alert); }
details.warn > summary { color: #e0a460; }
details.warn[open] > summary { border-bottom-color: var(--color-alert); }

/* «Нет значения», in the same colour as `.muted` under a different name, because the two
   mean different things in a table cell: `.muted` is prose that matters less, `.none` is a
   number that does not exist. `.void` is the stronger case -- the camera cannot witness
   this event at all -- and is italic so that a column of refusals cannot be skimmed as a
   column of small numbers. */
.none { color: var(--color-muted); }
.void, td.void, th.void { color: var(--color-muted); font-style: italic; }
sup { color: var(--color-alert); font-size: 10px; }

/* A tile whose value is a refusal instead of a number. `.tile .value` is 30px/600 because a
   number that size is readable from across a room; «не измерялось» at 30px would be the
   loudest thing on the page. Same tile, same accent, smaller words.

   `.tile strong` exists because the per-counter tiles are rendered as
   `<strong>N</strong><br><span class="none">label</span>` and a test pins that exact
   markup -- it is the test that proves «выходил к доске» never prints as a confident 0. */
.tile strong { display: block; font-size: 30px; font-weight: 600; }
/* The label half of that pinned markup is a `<span class="none">` rather than the platform's
   `.label`, for the same reason: the test pins it. Same 12px, so the two kinds of tile line
   up in one row. */
.tile .none { display: block; margin-top: 3px; font-size: 12px; }
.tile.void strong, .tile.void .value { font-size: 15px; font-weight: 400; font-style: italic;
  color: var(--color-muted); }
.tile .sub { display: block; margin-top: 3px; color: var(--color-muted); font-size: 12px; }
/* A tile that is a link into another page of the cabinet: the class cards, the place list.
   Same surface, so a reader who has learned that a tile is one fact does not have to learn
   a second card shape. */
a.tile { text-decoration: none; color: var(--color-text); }
a.tile:hover { border-color: var(--color-muted); }
a.tile strong { font-size: 15px; }

/* The machine-written orientation note. Dashed and unfilled, because every surface here
   that holds a measurement is a filled --panel: the one block that holds none must not be
   able to pass for one at a glance. A reader scanning a page reads shapes before words. */
.machine { border: 1px dashed var(--color-line); border-radius: 8px; padding: 12px 16px;
  margin: 0 0 12px; background: none; }
.machine blockquote { margin: 10px 0; padding-left: 12px;
  border-left: 3px solid var(--color-line); white-space: pre-wrap; }
.machine .who { color: var(--color-muted); font-size: 12px; margin: 6px 0 0; }

/* Wide tables -- six event counters beside five properties of the recording -- scroll in
   their own box. A page that scrolls sideways as a whole is a page whose first column, the
   one naming the row, is the first thing to leave the screen. */
.wrap { overflow-x: auto; margin-bottom: 12px; }
.wrap table { min-width: 700px; margin-bottom: 0; }

/* What must be true before a weekly trend can be computed. The tick is --ok because a
   passed gate is a passed gate; the missing one is a dash and a word, never a red cross:
   nothing is broken, the ingredient has not arrived. */
.gates { list-style: none; padding: 0; margin: 8px 0; }
.gates li { margin: 6px 0; padding-left: 22px; position: relative; }
.gates li::before { position: absolute; left: 0; font-weight: 700; }
.gates li.ok::before { content: "\\2713"; color: var(--color-ok); }
.gates li.no::before { content: "\\2014"; color: var(--color-muted); }
.gates .why { display: block; color: var(--color-muted); font-size: 12px; }

/* The refusal turned into a list a school can act on. Nested inside a `.panel`, so it takes
   the --bg surface to stay distinguishable from its container. */
.req { background: var(--color-bg); border: 1px solid var(--color-line); border-radius: 5px;
  padding: 10px 12px; margin: 8px 0; }
.req .what { display: block; font-weight: 600; }
.req .why, .req .how { display: block; color: var(--color-muted); font-size: 12px; margin-top: 4px; }

/* Index per lesson, oldest left. No axis, no gridline, no tooltip: there is a table
   directly underneath, and this is a shape beside a number rather than a chart to read
   values off. Drawn in --offline, the platform's neutral accent, because a direction this
   page has refused to state must not arrive as a colour. */
.spark { background: var(--color-panel); border: 1px solid var(--color-line); border-radius: 8px;
  color: var(--color-offline); }
"""


def _e(value: Any) -> str:
    return html_escape.escape("" if value is None else str(value))


# What the masthead says instead of naming a user. The platform's `.who` carries
# `username · role`, read from a session; this export has none, and inventing one would be
# the only false sentence on an otherwise measured page.
WHO_RU = "кабинет психолога · статическая выгрузка"


def masthead(*, active: str = "", section: tuple[str, str] | None = None,
             comparable: bool = True) -> str:
    """`templates/base.html`'s topbar, with only the links this folder can serve.

    `active` is the page's own nav key, so the row can say "you are here" without a second
    source of truth about which page this is; `section` is the (href, label) of the class
    page a lesson or a place belongs to, which is a real file in this same directory.

    `comparable=False` is the empty cabinet, and it is not a cosmetic case: `report.render`
    writes `index.html` ALONE when nothing has been imported, so on that one page the
    «Что сравнимо между уроками» item would point at a file that is not there. In a folder
    of static files there is no server to report the 404 — the link simply does nothing,
    which reads as a broken product rather than as an empty one.
    """
    items = [("index", "index.html", "Кабинет")]
    if comparable:
        items.append(("lessons", "lessons.html", "Что сравнимо между уроками"))
    if section is not None:
        items.append(("class", section[0], section[1]))
    # The attribute is built outside the f-string on purpose: an escaped quote inside one is
    # Python 3.12 syntax and `pyproject.toml` says `requires-python = ">=3.11"`.
    links = []
    for key, href, label in items:
        current = ' aria-current="page"' if key == active else ""
        links.append(f'<a href="{_e(href)}"{current}>{_e(label)}</a>')
    return (f'<header class="topbar"><a class="brand" href="index.html">Qorgan AI</a>'
            f'<nav>{"".join(links)}<span class="who">{_e(WHO_RU)}</span></nav></header>')


def page(title: str, body: str, *, active: str = "",
         section: tuple[str, str] | None = None, comparable: bool = True) -> str:
    """One self-contained document: platform stylesheet, platform masthead, then the page."""
    bar = masthead(active=active, section=section, comparable=comparable)
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(title)} — Qorgan AI</title>"
            f"<style>{PLATFORM_CSS}{CABINET_CSS}</style></head>"
            f"<body>{bar}<main>{body}</main></body></html>")
