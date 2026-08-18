"""Identity: the four-file answer to «а кто это на месте 3?», and its refusals.

Read in this order:

  * `roster.py`  — the closed list of children who may be named at all.
  * `seatmap.py` — a human's signed, dated statement about who sits where. **The only
    route by which a name enters this system.**
  * `faces.py`   — seat-level aggregated face evidence. May corroborate or contradict the
    statement above; may never create a name. The measurement that forced this is
    `MEASUREMENTS.md` §4: median best cosine 0.30 with a 0.10 margin.
  * `assign.py`  — the evidence standard, the gates, and `NOT_ESTABLISHED` as a
    first-class result.

Nothing here is imported by `report/artefact.py`, so the web project never sees it.
"""
