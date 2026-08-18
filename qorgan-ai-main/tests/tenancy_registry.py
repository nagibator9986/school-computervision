"""WHO OWNS WHICH TABLE, and every query allowed to ignore the answer.

Lifted out of `test_tenancy_guard.py` when that file crossed the repo's 500-line limit. The
split is along the seam the file already had: this is the REGISTER — a list of claims a human
made and can be held to — and what stayed behind is the machinery that checks them against
the code. Neither half means anything alone, which is why they import each other rather than
being merged back the first time one of them looks short.

Adding a model here is a decision, not bookkeeping: `test_every_mapped_class_says_whose_it_is`
fails on any mapped class this file does not mention, so a new table cannot reach production
without somebody stating whose data it holds.
"""

from __future__ import annotations

SCHOOL_COLUMN = "school_id"

# The four tables nothing else can answer for. These carry the column.
# `ClassvisionLesson`/`ClassvisionPlace` carry `school_id` NOT NULL instead of reaching one
# through a camera: their `camera_key` is a string read off a recording's filename, not a row
# in `cameras`. These files arrive from equipment this installation never talked to.
ROOT_MODELS = frozenset({"Camera", "Person", "User", "MealWindow",
                         "ClassvisionLesson", "ClassvisionPlace"})

# School data that reaches its school through a foreign key that cannot be null. No
# column, but every query against one must still join the root that knows.
DERIVED_MODELS = frozenset(
    {
        "PersonPhoto",  # -> persons
        "FaceEmbedding",  # -> persons
        "Event",  # -> cameras
        "Notification",  # -> events
        "CanteenSession",  # -> cameras (entry_camera_id)
        "RecognitionAttempt",  # -> cameras
        "Lesson",  # -> cameras
        "LessonTrack",  # -> lessons
        "PsychologistNote",  # -> persons, via person_id (NOT NULL), never the author
        # The classroom analyses hang off a lesson or a place, never off a camera row.
        # `ClassvisionPlaceLesson.place_id` is nullable on purpose (a seat matching no stable
        # place is the point of the unmatched row), so its LESSON answers for it.
        # `ClassvisionAttestation` anchors on the PLACE, not the person it names: a school-A
        # chair bound to a school-B child is the leak, and the chair is what catches it.
        "ClassvisionRun",  # -> classvision_lessons
        "ClassvisionPlaceLesson",  # -> classvision_lessons
        "ClassvisionFrame",  # -> classvision_lessons
        "ClassvisionReading",  # -> classvision_runs
        "ClassvisionTeacherLesson",  # -> classvision_lessons
        "ClassvisionAttestation",  # -> classvision_places
    }
)

TENANT_MODELS = ROOT_MODELS | DERIVED_MODELS

# The INSTALLATION, not a school -- the server the schools are hosted on. §14 gives
# "управление серверами" to the суперадминистратор and to nobody else, which is the same
# boundary. A school filter on these would be meaningless, and pretending otherwise would
# put four fake entries in the exemption list on day one.
#
#   * `School` is the tenancy register itself. Its rows ARE the schools, so it cannot be
#     filtered by one; who may read it is a capability question (`MANAGE_SCHOOLS`), not a
#     query question, and `test_web_capability_roles` is where that is enforced.
#   * `WorkerHeartbeat` and `ModeLog` are one process's health and one installation's
#     mode. A camera worker is per-school; the supervisor that restarts it is not.
#   * `AppSetting` is read by nothing in `src/` today (see `test_config_deadkeys` for what
#     this project thinks of that). If it ever holds a school's setting it becomes tenant
#     data and moves, and `test_every_mapped_class_says_whose_it_is` is what will force
#     the decision rather than letting it drift.
INSTALLATION_MODELS = frozenset({"School", "WorkerHeartbeat", "ModeLog", "AppSetting"})

# Every query that touches a school's table without naming a school, and why that is
# right. **Empty is the starting position and the goal.** An entry here is a promise that
# a human read the statement and can defend it; `test_no_exemption_outlives_its_reason`
# collects the debt when the statement changes.
#
# Format: `"<file>::<function>::<n>": "why"`. The key is what `Site.name` prints in the
# failure, and it survives adding a filter -- it moves only if the function gains or loses
# a query, which is exactly when a human should look again.
#
# **THIS LIST IS NOT EMPTY, AND SAYING SO IN THE FIRST LINE IS THE HONEST THING.**
# `test_config_deadkeys.py` keeps an empty one and earned it. This one could only be empty
# by writing filters that NAME a school without constraining anything -- and that exact
# shape (`Person.school_id == Person.school_id`) was measured against this guard: it passes
# here and leaks in `test_tenancy_isolation`. A defended sentence is worth more than a
# decorative `.where()`.
#
# **THE COUNT, SO THAT NOBODY HAS TO ESTIMATE IT.**
# UNSCOPED_QUERIES holds 24 entries, UNATTRIBUTED_QUERIES holds 2, total exempted 26.
# Of those, 11 are the worker sub-block below. Quoting that 11 as though it were the whole
# list understates the exemptions by more than half, which is exactly what happened once
# when this branch was summarised for its owner.
#
# Those four numbers are ASSERTED against this comment by
# `tests/test_the_exemption_block_says_what_is_true.py`, which reads this file as text --
# because a number in a comment is a number nobody re-measures, and this repository has
# already paid for one of those.
#
# Every entry is one of three kinds, and none of them is "we did not get to it":
#   1. THE QUERY THAT ESTABLISHES THE TENANT -- it cannot be filtered by a school, because
#      its result is what decides which school the caller is in.
#   2. THE QUERY WHOSE CORRECTNESS REQUIRES THE WHOLE INSTALLATION -- scoping it would make
#      it wrong rather than safer, and in one case below would delete another school's photos.
#   3. A CORRELATED SUBQUERY already constrained by the scoped outer statement it sits in.
UNSCOPED_QUERIES: dict[str, str] = {
    # -- 1. establishes the tenant -------------------------------------------------
    "qorgan/web/security.py::authenticate::1": (
        "The login lookup. A login form carries no school -- the person typing it has not "
        "been identified yet -- so there is nothing to filter by, and the row this returns "
        "is what DECIDES the school for every query afterwards. `users.username` is "
        "deliberately globally unique (migration 0009), so this resolves to exactly one "
        "account on the installation: the caller's own. Filtering it by school would mean "
        "knowing the answer before asking the question."
    ),
    "qorgan/web/security.py::load_user::1": (
        "Re-loads the logged-in account on every request from the id in the SIGNED session "
        "cookie. Same argument as `authenticate`: this row is the source of `school_id` "
        "for the whole request and so cannot be constrained by it. It returns the caller's "
        "own account and no other; `web.security.school_of` is what then confines every "
        "downstream query to that account's school."
    ),
    "qorgan/accounts.py::create_account::1": (
        "Checks whether a username is already taken. `users.username` is globally unique "
        "ACROSS the installation on purpose, so the check must be global too: scoped to "
        "one school it would pass, and the INSERT behind it would then hit the global "
        "UNIQUE index and reach a headteacher as a 500 on a form they filled in honestly. "
        "It discloses nothing -- a boolean about a name the caller has just typed."
    ),
    # -- 2. correctness requires the whole installation ----------------------------
    "qorgan/maintenance/janitor.py::_remove_orphans::1": (
        "Builds the set of media paths ANY event still points at, to decide which files on "
        "disk are referenced by nothing. **Scoping this by school would delete another "
        "school's photographs of children.** A file referenced only by another school's "
        "event would be missing from a scoped set, be classified an orphan, and be "
        "unlinked. There is one media root per installation, so `is this file referenced` "
        "is an installation question and every narrower answer is destructive."
    ),
    "qorgan/maintenance/janitor.py::_remove_orphans::2": (
        "The clip half of the same set, for the same reason and with the same consequence."
    ),
    "qorgan/maintenance/janitor.py::_expire_media::1": (
        "Retention. Drops the media of events older than the cutoff, across the "
        "installation, because the disk is the installation's and so is the retention "
        "period (`DEFAULT_MEDIA_DAYS`, one config value). Scoping it by school would run "
        "the identical work once per school and delete the identical rows -- ceremony, not "
        "safety. Nothing it reads reaches a user: the result goes to `unlink()`. NOTE for "
        "the day a school negotiates its own retention period: that is when this becomes a "
        "per-school sweep, and this entry is what should stop it being written by accident."
    ),
    "qorgan/maintenance/janitor.py::_prune_attempts::1": (
        "Recognition attempts older than the cutoff, on the same argument as `_expire_"
        "media`: one installation-wide retention period, rows that reach no user, and an "
        "identical outcome from a per-school loop."
    ),
    # -- 3b. THE DETECTION WORKER, WHICH REFUSES TO RUN AT ALL ON TWO SCHOOLS --------
    #
    # `canteen/sessions.py` and `classroom/store.py` are reached ONLY from the worker
    # process and the supervisor sweep -- checked, not assumed: nothing under `qorgan/web/`
    # imports either (`web/routes/lessons.py` imports `classroom.reports`, which IS
    # scoped). So no id below ever arrives from a URL or a form.
    #
    # **NINE OF THE ELEVEN rest on a measured refusal; the other TWO do not, and saying
    # "all eleven" was wrong.** The worker chain REFUSES on a two-school database instead
    # of guessing: measured, against a database holding two schools with nothing plumbed,
    # `ensure_cameras`, `camera_id_for`, `load_gallery`, `resolve_external`, `day_report`
    # and `recent_lessons` EVERY one raise `UndecidedSchool` rather than returning a row.
    # A worker cannot start against a multi-school installation -- it stops, loudly, at its
    # first database call.
    #
    # **The two exceptions are `close_sessions_nobody_exited._sweep::1` and
    # `close_stale_lessons._sweep::1`.** They run in the SUPERVISOR process, which imports
    # none of those six entry points and reads its sweep rules from YAML rather than from
    # the database -- so on a two-school installation the supervisor keeps sweeping the
    # whole installation quite happily while the workers crash-loop. Nothing about the
    # refusal protects them.
    #
    # That costs nothing here, because both already carry a SELF-SUFFICIENT argument -- an
    # installation-wide janitor on an installation-wide rule, closing identical rows and
    # reaching no user, exactly like the `maintenance/janitor.py` entries above. They do
    # not need the refusal and never did. The defect was the over-claim, not the excuse.
    #
    # **THE EXPIRY CONDITION IS A TEST, NOT THIS PARAGRAPH.**
    # `tests/test_the_exemption_block_says_what_is_true.py` asserts that all six entry
    # points still raise, with a one-school control proving the refusal is about the SECOND
    # school rather than about an empty database -- and separately asserts that the
    # supervisor still imports none of them, which is what keeps the split above honest. It
    # goes RED the moment somebody plumbs a school through the worker, which is exactly
    # when these entries must be re-read. Red there is not "somebody broke something good";
    # it is "nine of these excuses are now unbacked, resolve them before deleting the test".
    #
    # It is a test and not a note on purpose. `HANDOFF.md` records a previous watchdog note
    # in this repository that asked the reader to notice -- and twelve commits later nobody
    # had re-measured the number, "because a guard that only asks you to notice is not a
    # check". Eleven exemptions over children's meal records and classrooms are too much to
    # hang on a paragraph that nothing executes.
    #
    # Until then, scoping these would add a predicate already implied by the
    # id being passed in -- and a filter that constrains nothing new while LOOKING like a
    # school filter is the exact shape measured to pass this guard and leak anyway
    # (`Person.school_id == Person.school_id`; see `test_tenancy_isolation`).
    "qorgan/canteen/sessions.py::SessionManager._cooldown_block::1": (
        "Keyed on `person_id`, which comes from the face gallery -- and the gallery is now "
        "loaded for ONE school (`faces/gallery.py`). A person id therefore determines the "
        "school by itself: `persons.school_id` answers for that row. Decides whether a "
        "second meal RECORD is made for a child already seen."
    ),
    "qorgan/canteen/sessions.py::SessionManager._active_for::1": (
        "Same key, same argument: this school's `person_id` from this school's gallery, "
        "finding that person's own open session."
    ),
    "qorgan/canteen/sessions.py::SessionManager.confirm_inside._confirm::1": (
        "`session.get(CanteenSession, session_id)` where `session_id` came from "
        "`active_session_id` on this same manager moments earlier. Never parsed from "
        "input; an inside camera can neither open nor close a session."
    ),
    "qorgan/canteen/sessions.py::SessionManager.late_bind._bind::1": (
        "Same in-process `session_id`, and it only ever fills a NULL identity -- it "
        "refuses to overwrite one already present, which is the legacy defect (spec §5.2) "
        "of attaching a recognised pupil to somebody else's session."
    ),
    "qorgan/canteen/sessions.py::SessionManager.close._close::1": (
        "Keyed on `person_id` from this school's gallery, closing that person's own open "
        "session. Same argument as `_active_for`."
    ),
    "qorgan/canteen/sessions.py::close_sessions_nobody_exited._sweep::1": (
        "The supervisor's janitor: force-closes meal sessions nobody exited, across the "
        "installation, on one installation-wide rule (`max_session_minutes`). Same shape "
        "as the janitor entries above -- a per-school loop would close the identical rows "
        "and nothing it reads reaches a user. It writes an UNKNOWN outcome, which "
        "`_count_forced_unknown` then reports PER SCHOOL, scoped."
    ),
    "qorgan/classroom/store.py::open_or_resume._open::1": (
        "Keyed on `camera_id`, which the worker owns and which determines the school by "
        "itself (`cameras.school_id`). `ensure_cameras` -- now scoped -- is where that id "
        "comes from, and a camera belongs to exactly one worker group."
    ),
    "qorgan/classroom/store.py::flush._write::1": (
        "`session.get(Lesson, lesson_id)` where `lesson_id` is what `open_or_resume` "
        "returned to this worker. Never parsed from input."
    ),
    "qorgan/classroom/store.py::_upsert::1": (
        "The (lesson, track) upsert inside that same flush, under the same `lesson_id`."
    ),
    "qorgan/classroom/store.py::close._close::1": (
        "`session.get(Lesson, lesson_id)` for the lesson this worker opened. Idempotent, "
        "and never parsed from input."
    ),
    "qorgan/classroom/store.py::close_stale_lessons._sweep::1": (
        "The supervisor's janitor over lessons nobody ended, on one installation-wide rule "
        "(`max_lesson_minutes`). Same argument as `close_sessions_nobody_exited`: a "
        "per-school loop closes the identical rows and nothing it reads reaches a user."
    ),
    # -- 4. a gap this module is NOT pretending to have closed -----------------------
    "qorgan/notify/queue.py::NotificationWorker._due::1": (
        "Drains the delivery queue for the whole installation. **This is a real multi-"
        "school gap and it is recorded here rather than papered over with a filter.** "
        "Telegram is configured once per installation (`settings.telegram_bot_token`, one "
        "chat id), so there is exactly ONE destination: scoping this query by school would "
        "not route a second school's alerts anywhere better, it would only stop them being "
        "sent at all, which on a bullying alert is the worse failure. Multi-school "
        "notification ROUTING -- a bot and a chat per school -- is not built, and until it "
        "is, a second school on this installation must not be given cameras that raise "
        "alerts. That is a deployment constraint, and it belongs in writing where somebody "
        "adding the second school will meet it."
    ),
    "qorgan/notify/queue.py::NotificationWorker._settle._update::1": (
        "Writes the outcome back onto a notification this same worker took out of `_due` "
        "one call earlier, by the id `_due` returned. It reads no wider population than "
        "`_due` already did and carries the same gap, above."
    ),
    # -- 3. the row this process minted, by an id that never came from outside -------
    #
    # These three are `session.get(Event, event_id)` where `event_id` is the RETURN VALUE
    # of `record_event` earlier in the same worker, for a camera that worker owns. That is
    # a different thing from `/events/{id}/review`, which takes the id out of a URL and is
    # scoped (see `web/routes/events.py`) -- the distinction is where the number came from,
    # and it is the whole of the argument. An id minted in-process cannot be pointed at
    # another school by anybody; a URL can be, by typing.
    "qorgan/events/store.py::attach_media._update::1": (
        "Back-fills the burst clip onto the event `record_event` returned an id for, in "
        "the same worker. The id is never parsed from input."
    ),
    "qorgan/events/store.py::raise_confidence._update::1": (
        "Raises the same worker's own event to the incident's worst moment, by the id it "
        "holds from `record_event`. Never parsed from input."
    ),
    "qorgan/events/store.py::record_telegram_decision._update::1": (
        "Writes whether a human was told, onto the event this worker just decided about, "
        "by the id it holds. Never parsed from input."
    ),
    # -- 4. correlated with a scoped outer statement -------------------------------
    "qorgan/identity/registry.py::_query::1": (
        "The `recognisable` EXISTS, correlated by `FaceEmbedding.person_id == Person.id` "
        "to the register query it sits inside -- which IS scoped (`_query::2`). It is "
        "evaluated once per outer row and every outer row is already one school's. A "
        "school filter here would be a second answer to a question the outer statement has "
        "already answered, which is the shape this project has been bitten by twice."
    ),
}

# A query the scan could not attribute to any table at all -- `update(model)` where
# `model` is a parameter. It cannot be judged, so it must be answered for BY NAME, the
# same way `test_config_deadkeys.UNRESOLVED_KEYS` answers for an access it cannot type.
# Quietly skipping these is how a guard becomes decoration.
UNATTRIBUTED_QUERIES: dict[str, str] = {
    "qorgan/db/models/school.py::sole_school_id::1": (
        "`select(School.id)` -- and the scan cannot attribute it because `School` is "
        "DEFINED in that module rather than imported from `qorgan.db.models`, so it is not "
        "in the module's alias table. The statement is the tenancy register itself: it "
        "counts the schools on the installation in order to answer 'which school is this, "
        "when nobody said?', and it returns at most two rows so that it can RAISE on the "
        "second rather than choose. `School` is in INSTALLATION_MODELS: a table whose rows "
        "ARE the schools cannot be filtered by one, and a school filter here would make "
        "the refusal that protects every other query impossible to compute."
    ),
    "qorgan/identity/merge.py::_repoint::1": (
        "`update(model).where(model.person_id == drop_id)` where `model` is a `type` "
        "PARAMETER -- called three times with PersonPhoto, FaceEmbedding and CanteenSession "
        "-- so the scan cannot know which table it is and correctly refuses to judge it.\n\n"
        "**The invariant that keeps it inside one school is on `drop_id`, not on the "
        "statement.** Every row it touches is selected by `person_id == drop_id`, and "
        "`drop_id` reached `_merge` through `_require`, which fetches a person only within "
        "ONE school (it was a bare `session.get` until this branch). `keep_id` is proved "
        "to be that same school's, in the same transaction, by the same function.\n\n"
        "**Which school that is depends on the caller, and the distinction is worth "
        "stating precisely rather than loosely.** When the caller names a school -- as "
        "`web/routes/duplicates.py` does, from the session of the person clicking -- it is "
        "the acting user's school. When the caller names none, `resolve_school_id` falls "
        "back to the only school there is and RAISES on two, so the unnamed case cannot "
        "silently pick one. An earlier version of this sentence said 'the acting school' "
        "flatly; that is true of the web path and not of the CLI, and the two coincide "
        "only while there is one school -- which is the precise sort of claim this project "
        "keeps being bitten by.\n\n"
        "Making it resolvable by naming the three classes literally would mean three "
        "near-identical statements, and the reason it is one is that a fourth table "
        "hanging off `persons` must not be able to be forgotten here -- which is a real "
        "hazard: forgetting one would silently strand a child's photographs on a retired "
        "id. The dynamic form is the safer shape, and this entry is the price of it."
    )
}
