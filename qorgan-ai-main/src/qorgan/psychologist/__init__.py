"""The psychologist's cabinet (client §13): facts with numbers, and no recommendation.

**THE DISTINCTION THIS WHOLE PACKAGE TURNS ON.** Two statements in this repository look
contradictory and are not:

  * `qorgan.classroom` and `classroom/reports.py`: «Никаких диагнозов и никаких
    направлений к психологу ОТ СИСТЕМЫ». That is quoted from what the school was promised
    in writing (`docs/questions-for-school.md` §8) and it stands, unweakened, here.
  * Client §9: an operator must be able to mark an event «передано психологу».

**A HUMAN deciding to refer is the product. THE SYSTEM deciding is forbidden.** Every
signal in this cabinet is a fact with a number beside it; a referral is an act a named
person took, recorded with their name and the minute they took it. Nothing in this package
computes a recommendation — there is no score, no threshold that fires, no ranking, and no
sort order that implies one. If a line reading «рекомендуется проверка школьным
психологом» ever seems to belong here, that is exactly the line §8 forbids, and §12.3 of
the client's own spec asking for it is the request the school was answered "no" to in
writing.

--------------------------------------------------------------------------------
**WHERE THE SIGNAL IS REAL AND WHERE IT IS EMPTY — the point of `SignalState`.**

The old system was not killed by missing features. It was killed by features that looked
like they were working. So every block on every page in this package carries its own state
and says it out loud:

  * **Referrals** — REAL from the day this shipped. The mechanism is a person and a
    button; it needs no camera and no model.
  * **Canteen attendance, per named pupil, over weeks** — REAL identity, no classroom
    recognition needed, and the one longitudinal signal in this system that is honestly
    about a named child. It is EMPTY today because the canteen camera is still pointed at
    the wrong place; the table and the accumulation exist anyway, so that counting does not
    start from zero on the morning of the pilot.
  * **Classroom metrics** — arriving, and permanently ANONYMOUS. They are per track, and
    a track lives minutes. They can never be attached to a pupil (`qorgan.classroom`), so
    the cabinet shows that they exist and refuses to imply they are about anybody.

--------------------------------------------------------------------------------
**THE CONTRADICTION INSIDE §8, WHICH THIS PACKAGE DOES NOT RESOLVE.**

§8 promises comparing a child against **their own norm over the previous four weeks**.
That requires knowing today's child is the same child as three weeks ago — identification
— which the same paragraph forbids inside a classroom, and which this school's own
corridor footage says would not work there anyway: 14 970 faces, median 11.5 px, **zero**
recognised. A track lives minutes and dies to the first occlusion.

**This package does not resolve that by quietly adding a nullable reference to a pupil.**
It builds the longitudinal signal where the identity is genuine — the canteen, where the
child is recognised at a door at conversational distance — and it states on the page that
the classroom half is anonymous and will stay so. The question of whether the school wants
per-child classroom trends, and at what price, is the school's: either children are
enrolled by some means that is not a classroom camera, or there is no trend.
`docs/questions-for-school.md` §10 is where it is asked.
"""
