# eval/

`labels.csv`, `baseline.json` and `school_labels_2026-07.csv` are results and belong in
git. **`clips/`, `candidates.csv`, and its coverage manifest `candidates.coverage.csv` are
video of children (or name it) and never do** — see `.gitignore`.

`school_labels_2026-07.csv` is the school's own labelling spreadsheet
(`Разметка_буллинг.xlsx`, gitignored) rendered as data: all 48 rows, each with the original
`Тип` and `Комментарий` and the `LabelKind` we mapped it to. It exists because
`labels.csv` cannot hold it — `append_label` refuses any header that is not exactly its
five columns, so a `comment` column there would break `qorgan eval label`. The comments
describe children by clothing and position and contain **no names and no IDs** (checked);
that is the same standard as the clip filenames already tracked in `labels.csv`.

`eval scan` writes `candidates.coverage.csv` beside `candidates.csv` in the same run: it
lists every clip the detector actually decoded and ran to completion, whether or not it
fired. `eval sample` needs it to tell a **covered-but-silent** clip (proven: the detector
saw nothing) from one it **never scanned** (unknown). A clip missing from the manifest is a
hard error — never a guessed "silent" — because a real fight in an unscanned clip filed as
silent would never be sampled. An old `candidates.csv` with no manifest makes `eval sample`
refuse until you re-run `eval scan`.

---

## The three human-named clips, and which camera they came from

Three clips in the corpus were named by a person, not by the recorder:

```
1.2 - двое мужчин просто стоят и один слегка обнимает второго.mp4
1.2 - нет буллинга.mp4
1.2 - ученики стоят разговаривают и резко начинают толкаться, подозрение на буллинг.mp4
```

`evaluation/clips.py` cannot attribute them by filename, and it refuses to guess — correctly,
because `hall_left` carries a `mirror_ignore` zone over a reflective column that `hall_right`
cannot see, so scoring footage against the wrong camera's zones blanks a region of the frame
and returns a plausible, wrong number.

**But the third clip is the only confirmed fight in all 663.** Losing it would leave the
corpus with zero positives.

### So the scene was measured instead of guessed

`hall_left` and `hall_right` look at different places. Build a median background per camera
from known clips (the room, with the people averaged out), then ask which room each unknown
clip is actually in. Zero-mean normalised cross-correlation, so brightness does not matter
and structure does.

**Control first, because a method nobody has tested on a known answer is not a method.** Eight
held-out clips from cameras we *do* know:

```
control accuracy: 8/8      margins +0.54 to +0.81
```

### The result

| clip | hall_left | hall_right | verdict |
|---|---|---|---|
| `нет буллинга` | +0.099 | **+0.745** | **hall_right** (margin +0.646) |
| `…резко начинают толкаться` | +0.093 | **+0.721** | **hall_right** (margin +0.628) |
| `двое мужчин … обнимает` | +0.062 | +0.133 | **UNRESOLVED** (margin +0.070) |

The first two match `hall_right` exactly as strongly as the controls match their own camera.

**The third matches nothing.** Its best score is +0.242 against `hall_right` when tested
against every camera in the corpus (the stairs score *negative*) — an order of magnitude
below every control margin. It is not from either hall camera, or the lighting differs so
much that this method cannot say. **So no camera is asserted for it.**

That costs us nothing: it is a *negative* example ("two men standing, one lightly hugging the
other") and the corpus holds 663 negatives. Guessing its camera, to gain one more, would mean
scoring it against zones that may not be its own — which is the exact failure this refusal
exists to prevent.

### What this means for `labels.csv`

The two resolved clips get an explicit `camera` value: `hall_right`. That column exists
precisely so a human can state a truth a parser cannot infer — and here the "human" is a
measurement with a validated control, which is recorded above rather than asserted.

**The school should still confirm it**, and the third clip needs an answer from them. It is
in `docs/questions-for-school.md`.

`t_start` / `t_end` for the fight must be set by a person watching the clip. `qorgan eval
label` exists for that. Nothing here guesses a timestamp.
