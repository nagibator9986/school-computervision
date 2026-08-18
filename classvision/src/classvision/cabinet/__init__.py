"""The accumulation layer: many lessons, one psychologist's cabinet.

`report/artefact.py` describes ONE lesson. This package is the other half of what the
client asked for — «эта вся аналитика копится в кабинете психолога ... первая неделя
хорошо, вторая неделя не очень» — and it is deliberately a separate package because it
answers a different question with a different unit.

  * `store.py`   — a stdlib-`sqlite3` store that ingests artefacts idempotently and
                   solves the one hard problem accumulation has: a `seat_id` is per-run,
                   so it is NOT a key across lessons. `places` is.
  * `weekly.py`  — per place / per attested pupil / per ISO week counters, the activity
                   index recomputed by `metrics/activity.py` over summed ledgers, and the
                   trend delegated to `metrics/trend.py`.
  * `report.py`  — the self-contained HTML the psychologist actually opens.

Nothing here imports torch, ultralytics, opencv or insightface, and nothing here needs
them: it reads JSON documents and writes SQLite and HTML. That is what makes it the piece
`qorgan` can mirror without inheriting a 2 GB dependency tree.
"""
