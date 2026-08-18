# What a human sees in D14, by eye — the check the teacher module must pass

Read off frames of `D14_20260815103136.mp4` (starts 2026-08-15 10:31:36). These are eyeball
observations, not model output, and they exist so that a teacher module claiming "90 % at the board"
or "0 % at the board" is caught by a person rather than believed.

| video t | wall clock | what a human sees |
|---|---|---|
| 60 s | 10:32:36 | **Teacher STANDING AT THE BOARD**, dark shirt, writing/explaining. Board already has chalk. ~6 pupils seated. |
| 600 s | 10:41:36 | Teacher **seated at the front-left desk** with a laptop. One pupil walking across the room. |
| 1200 s | 10:51:36 | Teacher **not visible in frame**. Pupils at desks working. More chalk on the board. |
| 1800 s | 11:01:36 | Teacher **seated AMONG the pupils** at a middle desk. **A PUPIL is standing at the board**, pointing. |
| 2100 s | 11:06:36 | Teacher **not visible**. Pupils clustered at one desk, working together. |
| 2700 s | 11:16:36 | **Teacher STANDING AT THE BOARD** again, right-hand panel. |

## What this means for the module

* The teacher **alternates** between the board, the front desk, sitting among pupils, and out of
  frame. Any state distribution that is dominated by one state is wrong.
* `AT_BOARD` must fire around t=60 and t=2700, and must **not** fire at t=600 or t=1800.
* A **pupil** stands at the board at t≈1800. So `AT_BOARD` is not teacher-only: the pupil state
  taxonomy needs it too, and the adult/pupil split must not be done by "who is at the board".
* Out-of-frame is common here and must never be reported as absence — the teacher leaves the frame
  toward the near wall, not the room.

## Room geometry, approximate, in 2560x1440 coordinates

The green chalkboard occupies `x 1024..1603, y 137..313` — measured, and recorded once in
`configs/camera_d14.yaml` as `board_surface`, which is the only place that rectangle is written down.
The floor a person stands on while at the board is BELOW that: a person at the board has their feet
around `y 400..600`.

**The board zone in the room layout is NOT that floor polygon.** This paragraph used to say it was —
"the board zone in the room layout is the FLOOR polygon, because `zones.point_in_polygon` is tested
against a foot point, not against chalk" — and that sentence is now false in both halves. Zones are
tested against the **shoulder anchor** (`geometry.anchor`), and the floor strip in front of the board
is the single trap `room/layout_io.py` rule 6 exists to refuse: it is a perfectly correct polygon
that no shoulder line ever enters, so it yields «у доски: 0.0 мин» for the life of an install with
nothing in the artefact to say why. Since `board_surface` became mandatory alongside a non-null
`board_zone`, a profile built from the old sentence is refused at load time rather than run for a
term. The sentence is kept here, struck through in prose, because this file is the human check and a
silently corrected ground truth teaches nobody why the check exists.

---

## The measured version of the table above

Shoulder-anchor y of every person the pose model found, at each ground-truth moment. Produced from
the cached detection pass, not by eye. This is the fixture the board zone and the teacher module are
checked against.

| video t | truth | topmost shoulder y | other people (y) |
|---|---|---|---|
| 60 s | teacher AT BOARD | **216** | 419, 439, 489, 580, 594 |
| 600 s | at the front desk | 342 | 386, 465, 471, 565, 574, 585, 589 |
| 1200 s | out of frame | 465 | 481, 504, 554, 558, 560 |
| 1800 s | pupil AT BOARD | **315** | 414, 463, 559, 568, 575 |
| 2100 s | out of frame | 466 | 472, 488, 562 |
| 2700 s | teacher AT BOARD | **215** | 440, 442, 487, 558, 560, 581 |

**A shoulder-anchor threshold around y ≈ 330 separates all six cases with no errors.** The three
at-board moments sit at 215–315; the three non-board moments have nobody above 342. That is a gap of
27 px at the tightest, between a pupil at the board (315) and the adult seated at his own desk (342)
— narrow, and the reason the zone is a polygon rather than a horizontal line: the desk sits at a
different x from the board, so the two are far apart in the plane even where they are close in y.

**The finding that kills scale-based adult identification on this camera.** At t = 60 s the adult is
at the board with a shoulder width of **16.8 px**, while pupils seated nearer the lens measure
63–96 px. Standing at the board he is the SMALLEST person in the frame, not the largest —
the exact inverse of the first camera, where the adult sat nearest the lens at ~220 px and
`identify_adult`'s scale ratio worked. On D14 that heuristic will nominate a front-row pupil.
The configured `teacher_zone` is therefore not a convenience here, it is the only correct route,
and `identify_adult` must fall back to NONE rather than to scale when no zone is configured.

**A second consequence, for the pupil half.** A person at the board is 16–52 px across the
shoulders, against the 70 px p25 measured on the first camera. Every shoulder-normalised threshold
is noisier there, so states assigned to someone at the board carry more uncertainty than the same
states assigned to someone at a desk. That is a property of the geometry, not a bug, and the report
should not present the two as equally well measured.
