# The cabinet's design contract

The psychologist's cabinet is shown next to `qorgan-ai-main`, and to a viewer it has to be
the same product rather than a tool that was bolted on. So its design is not invented here:
it is **copied from the platform**, token for token, out of
`qorgan-ai-main/src/qorgan/web/static/app.css` and `templates/base.html`.

Anything in the cabinet that does not appear below is a local invention and should be
deleted rather than defended.

## Tokens — verbatim from the platform

```css
--bg:       #0f1216   /* page */
--panel:    #171c22   /* any raised surface: card, tile, topbar */
--raise:    #1d232b   /* one step above --panel: a header sitting on a panel, a hover */
--line:     #262d36   /* every border and table rule */
--text:     #e6e9ee
--muted:    #8b96a4   /* labels, secondary prose, table headers */
--ok:       #3fa86b
--alert:    #d9822b   /* the ONLY colour a caveat may use */
--critical: #d64545
--offline:  #4a5462   /* the neutral accent, and the default tile top-border */

/* Interactive, and «the model is speaking». NOT a caveat colour -- see rule 3. */
--accent:              #4a7fd4
--accent-text:         #9dc0f5   /* the same colour as readable text/link */
--accent-strong:       #2f6fdb   /* a primary button's fill */
--accent-strong-hover: #3d7ce8

/* Lightened states, for TEXT on a tinted surface. The state colours themselves are
   calibrated as borders and fills and cannot be read as body copy on --panel. */
--critical-text: #f0a0a0
--alert-text:    #e0a460
--ok-text:       #7fc79b
```

Type is a scale, not a set of literals: `--t-h1: 26px`, `--t-h2: 17px`, `--t-body: 15px`,
`--t-small: 13px`, `--t-fine: 12px`, on `"Segoe UI", system-ui, sans-serif` at 1.55.
`h1` uses `--t-h1` with margin `0 0 22px`; `h2` uses `--t-h2`/600 with margin `30px 0 12px`.
A counter inside a heading (`<span class="muted">`) drops to `--t-body` / `--t-small`.

Layout: `main { padding: 28px 24px 64px; max-width: 1400px; margin: 0 auto; }`.
Radius `--radius` (8px) on surfaces, `--radius-sm` (5px) on controls. Borders are always
1px `--line`. Every interactive element takes ONE focus ring, defined once on
`:focus-visible`; nothing may set `outline: none` without replacing it.

## Components that already exist — use these names, do not re-invent

| class | what it is |
|---|---|
| `.topbar` + `.brand` + `nav` + `.who` | the masthead. `Qorgan AI` on the left, links, identity on the right |
| `.tiles` / `.tile` | a row of headline numbers. `.tile .value` 30px/600, `.tile .label` 12px muted, `border-top: 3px` accent (`.ok` / `.alert` / `.critical`, default `--offline`) |
| `table` | 13px, `th` muted 500, cells `padding: 8px 12px`, `border-bottom: 1px solid var(--line)` |
| `.tag` | inline pill: 11px, 1px border, muted. Variants recolour the border and the text, never the background. **A variant class built from a data value must exist for every value that data can hold** — `.tag.ok` did not, so «активен» and «отключён» painted identically. And it may never be put on a `<td>` or a `<p>`: `display: inline-block` stops a cell being a cell and silently voids `colspan` |
| `.blank` | an empty list, as a surface: dashed border, a `<strong>` lede, then one sentence saying what would put something in it. Never a bare paragraph. NOT called `.empty` — that word is a `SignalState` value and collides with `class="tag {{ state }}"` |
| `.rowmsg` | a message spanning a whole table row. The class `.tag` cannot do this job (see above) |
| `.warning` | amber-tinted block, `rgba(217,130,43,.12)` on a `--alert` border, text `#e0a460` |
| `.error` | the same shape in `--critical` |
| `.muted` | secondary text |
| `.button` | `--panel` surface, `--line` border, 13px |
| `.grid` | `repeat(auto-fill, minmax(300px, 1fr))`, gap 16px |
| `.pages` | a row of links under content |

## Rules for the cabinet specifically

1. **The numbers lead; the reasoning is one click away.** Every caveat currently in the
   cabinet is there because it prevents a real misreading, and **none may be deleted**. But
   a page that opens with six paragraphs before its first number is a page nobody reads to
   the end, and an unread caveat protects nobody. So: the one sentence that changes how a
   number is read stays visible; the full argument goes into a `<details>` whose summary
   states what it will say.

2. **Say a thing once per page.** The cross-room warning is currently repeated verbatim on
   fifteen pages. It belongs, in full, on the one page that compares rooms; everywhere else
   it is a single line and a link. Repetition at that volume reads as boilerplate and gets
   skipped — which costs the warning exactly the readers it was written for.

3. **`--alert` is the only accent a caveat may take, and nothing else may take it.** If
   everything is amber, amber means nothing.

4. **No colour carries meaning on its own.** A direction is a word first (`ниже`, `в
   пределах разброса`, `не установлено`); colour only reinforces it. Copies of this page get
   printed in grey.

5. **A table cell that was not measured says «не измерялось», never `—` alone and never
   blank.** That distinction is the whole discipline of this project and it must survive
   the visual tidy-up.

6. **No decoration that implies precision the data lacks.** No progress bars on an index
   whose thresholds are unvalidated, no traffic lights on a child, no sparkline through a
   single point.
