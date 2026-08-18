# The three modules the client asked for, before the pilot

> **For agentic workers:** each module below is a SEPARATE plan and a separate branch.
> Do not build two of them in one worktree. Steps are boundaries and refusals, not
> transcribed code — see "Why this plan has no code in it".

**Goal:** everything the client named in §12.1, §13 and §14 exists before the pilot, and
no module pretends to be more than it is.

**Decision, 2026-07-28.** The owner has heard the argument for closing the basics first
(detector accuracy, canteen camera placement, 24 h run) and decided otherwise: the client
asked for these modules from the beginning and wants them present at the pilot. That is
the owner's call and this plan implements it in full. The basics run in parallel; they are
not a precondition here.

**Tech stack:** Python 3.11, FastAPI, SQLAlchemy 2.0 + Alembic, Pydantic (`extra="forbid"`),
Ultralytics YOLO, OpenCV, Jinja + HTMX.

---

## Why this plan has no code in it

The house style for plans is bite-sized steps carrying literal code. This one deliberately
does not, and the reason is measured rather than aesthetic.

Every agent dispatched on this codebase today was given boundaries, refusals and the
reasons behind them — not transcribed implementations. What they returned was *better than
the brief*: one found that a migration would cascade-delete every event the school has and
that the obvious guard against it is a documented no-op; one measured that OpenCV
serialises the FFmpeg open, which overturned its own design; one found that §8 of the
questions to the school promises a signal that its own constraints forbid. None of those
was in the brief. A brief that prescribed the code would have prescribed past all three.

So each module below states: what it is for, what it may not do, what it must refuse, what
"done" means, and which existing decisions it is forbidden to weaken. The implementer
chooses the code and writes the tests.

---

## Global constraints — every task inherits these

Verbatim from the repository's own rules; none is negotiable.

- **R1** — no file over ~500 lines, no function over ~50. Enforced over `src/` **and**
  `tests/` by `tests/test_code_limits.py`. **Split, never loosen.**
- **R2** — one source of truth for detection logic. The production worker and the eval
  harness call the *same* function objects; `tests/test_evaluation.py` asserts identity.
- **R4** — no secret in code, config, database, log line or debug image. Redaction happens
  at the WRITE site, not at the render site.
- **R5** — every endpoint authenticated by default. `tests/test_web_auth.py` walks the real
  route table; a new route is protected unless explicitly listed public.
- **R6** — no absolute path in the database. `RelPath` refuses one at bind time.
- **R7** — no worker thread dies silently.
- **R8** — bounded memory. No unbounded dict; TTL eviction on every cache.
- **R10** — config is a validated schema, `extra="forbid"`. A new key must be in the schema
  **and** read by the layer whose behaviour it claims to control. `test_config_deadkeys.py`
  is an owner-resolving AST scan with an empty exemption list. **Its subtle half is not
  covered:** a key can be "read" by a derived property while nothing acts on it. Ask what
  actually consumes the value.
- **Capabilities arrive WITH the routes they guard.** `src/qorgan/roles.py`: "a permission
  guarding nothing is a guess". Add to the union in `ROLE_CAPABILITIES`; never replace it —
  five branches each rewrote those two lines and any single version would have silently
  revoked the others' pages, with a green suite.
- **Test counting:** the fixed cost of a new `.py` file is **+2** `test_code_limits`
  parameters, in `src/` as in `tests/`. Decompose a delta by **diffing test ids**, never by
  matching a total to an expectation.
- **Media on disk is photographs and video of children.** Never open them. Count in Python,
  never in the shell. Never `git add -A`; check with `git add -An`.
- **Nothing merges to `main`.** Branch and push, yes. Merge is the owner's.

Baseline at the time of writing: `main` = `a88cbf6`, **2319 tests, 0 failures, ruff clean.**

---

## Module A — Weapons (client §12.1)

**Branch:** `feat/weapons-detection` · **Worktree:** `../q.ai-weapons`

### What it is

A second detection session with its own class map, at a reduced frame rate, raising an
alert a **human then confirms**. Never an autonomous trigger: a false gun alert in a school
has consequences of its own.

The client's shape, which is correct and must be kept: object detected → tracked across
several frames → confidence checked → **checked for being near a person** → confirmed again
→ snapshot and clip → critical notification.

### The wall, and how this module is honest about it

**The client has no model.** Their `best.pt` is 0 bytes and was a violence *classifier*, not
a weapons *detector* — so the module "worked" for months with no model in it at all. That
is the failure this module exists to make impossible.

`REWRITE_SPEC.md` §5.1 already reserves the seam: *"violence model (optional, currently
absent — keep the plug-in point)"*. Build against that seam.

**Refusals this module must implement:**

- **No model ⇒ refuse to start**, loudly, naming the missing file. Never fall back to
  motion analysis, never log a warning and continue. Silent fallback is the exact defect.
- **The panel must show which weights are loaded and what they were evaluated on.** A
  module that cannot say what it is running is a module nobody can audit.
- **A weapon alert is never auto-actioned.** It is confirmed by a person, and the record
  says who confirmed it.

### Zones

Corridor and kitchen are different rules, not different thresholds: in a kitchen a knife is
a tool. Zones are stored as fractions of the frame (resolution-independent) — reuse the
existing zone machinery, do not invent a second one.

### Honest limits to write into the code

Distance decides feasibility and no threshold changes it: a knife in a hand at the school
entrance is a 100+ px object and will work; the same knife down a corridor at 15 m is 15 px
and will never work. Say so where the camera is configured, in the same spirit as
`identity camera-report`.

### Done when

Pipeline exists and is tested end to end against a synthetic detector; refuses to start
without weights; names its weights on screen; kitchen and corridor rules diverge and both
are tested; a smoke run over the school's corridor corpus finds **nothing**, and that
negative is reported as a measured result rather than assumed.

### Open question for the client, to be recorded not guessed

Knife or firearm; where the camera goes; whether the kitchen is in scope.

---

## Module B — The psychologist's cabinet (client §13)

**Branch:** `feat/psychologist-cabinet` · **Worktree:** `../q.ai-psy`

### The distinction this module turns on — read this first

Two statements in this repository look contradictory and are not:

- `src/qorgan/classroom/__init__.py` and `classroom/reports.py`: **"No diagnosis, and no
  referral to a psychologist FROM THE SYSTEM."** That is quoted from what was promised to
  the school in writing (`docs/questions-for-school.md` §8) and it stands.
- Client §9: an operator must be able to mark an event **"передано психологу"**.

**A human deciding to refer is the product. The system deciding is forbidden.** Every
signal this cabinet shows is a fact with a number beside it; the referral is an action a
named person took, recorded with their name. Nothing in this module may compute a
recommendation.

### What it may show, today

`EventStatus` currently has `new · reviewed · confirmed · false_positive` and **no referral
value** — §9 requires one, so it is missing and this module adds it.

Content that exists now, without waiting for anything:

1. **Incidents an operator referred**, with who referred them and when.
2. **Per-pupil canteen attendance over time.** This is the part I under-rated when I first
   said there was nothing to show: the canteen carries **real identity**, so "this child
   ate every day and then stopped" is a genuine longitudinal signal per named pupil, and it
   needs none of the classroom identity that §8 forbids. It becomes live when the canteen
   camera is moved; **the table and the accumulation must exist before that**, or counting
   starts from zero on pilot day.
3. **The psychologist's own notes**, confidential — an operator must not be able to read
   them. §13 says so explicitly and it is a capability boundary, not a UI preference.
4. **Classroom metrics** as they accumulate, from the module built on 2026-07-28.

### The contradiction inside §8, which is not this module's to resolve

§8 promises comparing a child against **their own norm over the previous four weeks**. That
requires knowing today's child is the same child as three weeks ago — identification — which
the same paragraph forbids in a classroom, and which the corridor measurement (14 970 faces,
zero recognised) says would not work there anyway. A track lives minutes.

**Do not resolve this by quietly adding a nullable foreign key to a pupil.** Build the
signals that have real identity (the canteen ones), show plainly that the classroom half is
anonymous, and leave the question where it belongs — with the school.

### The role

`UserRole` has `OPERATOR · ADMIN · DEVELOPER · CANTEEN_STAFF`. `accounts.py` carries a
deliberate note that there is **no psychologist role**, and a test asserts a role that does
not exist changes nothing. **Both are correct as written and must be updated
deliberately**, in the same change that adds the routes — not worked around.

### Done when

The role exists with capabilities that arrive with their routes; a referral status exists
and an operator can set it; the cabinet shows referred incidents, per-pupil canteen trend,
and confidential notes an operator provably cannot read; every page states what is
accumulating and what is not yet; **no computed recommendation anywhere.**

---

## Module C — Superadmin and several schools (client §14)

**Branch:** `feat/multi-school` · **Worktree:** `../q.ai-tenancy`

### The guard comes first, and it must be able to fail

Mechanically this is a migration, `school_id` on the root tables, composite uniques and a
filter in the queries. Small.

**The day `school_id` lands, every unfiltered query becomes a cross-tenant leak of
children's data.** So this module is built in the order this codebase uses for exactly this
shape of risk — the same order `test_web_auth.py` uses for routes:

**Task 1 is the guard, not the feature.** A test that enumerates every query against a root
table and **fails on any that carries no school filter**. Deny by default: a query nobody
thought about is caught, not exempt. Its exemption list starts empty and every entry needs
a written reason.

**Prove the guard can fail before building on it.** Write it, watch it go red against the
unfiltered code that exists today, and only then add the column. A guard first seen green is
a guard nobody has tested.

### Then

Migration and `school_id` on the root tables; composite uniqueness (an `external_id` is
unique **within a school**, not globally — two schools may both have a pupil numbered 7);
the filter applied everywhere the guard demands; the `SUPERADMIN` role **with** its page,
never before it.

### Done when

The guard exists, was seen red, and is green; every root-table query is filtered; two
schools in one database cannot see each other's pupils, events or canteen records, and a
test proves it by trying; the superadmin role has a page and no capability guards nothing.

---

## Verification — identical for all three

- `pytest --junitxml=<unique path>`, parse the XML, **confirm collected is non-zero**.
  `pytest -q` silently drops its summary line; a wrong path exits having collected NOTHING
  and looks exactly like success.
- `ruff check .` as its own command. "X clean" and "Y passes" are two different claims.
- **Sabotage your own fix**: break it, confirm the *specific* test goes red, revert, and
  confirm both directions with `grep`. A sabotage that stays green means you broke the wrong
  lever.
- Do not run a suite concurrently with anything else you can avoid. Known contention flakes:
  `test_analysis_rate` throughput, `test_det_every_is_honoured` timing, `test_web_pages.py`
  zmq bind. If one of those is red, re-run **it alone** before believing it.
- **A test can assert the CLIENT's behaviour and look exactly like server coverage.** The
  `/media` traversal test passed — and kept passing with the defence sabotaged — because
  httpx collapses `..` before the request is sent. And every login test passed while logging
  in from a browser was impossible, because `TestClient` fetches exactly the paths a test
  names. Ask what each of your tests actually exercises.

## Report, for each module

What was built and why that shape; **which numbers were chosen rather than measured**, named
explicitly; the junit figures with the path; `ruff` on its own line; the sabotage evidence;
what was deliberately NOT built and why; and every place the implementer had to guess,
stated plainly rather than smoothed over.
