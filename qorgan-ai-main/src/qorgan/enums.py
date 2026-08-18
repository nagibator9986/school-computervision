"""Enumerations shared by the config schema, the database models, and the workers.

Defined once. The legacy system re-derived person_type from substring heuristics in
five different files, so the same person could be staff in the database and a pupil
at runtime (audit M-28).
"""

from __future__ import annotations

from enum import StrEnum


class CameraType(StrEnum):
    BULLYING = "bullying"
    CANTEEN = "canteen"
    CLASSROOM = "classroom"
    # §12.1. A SECOND detection session with its own class map and its own frame rate,
    # not a block bolted onto a bullying camera -- the discriminated union in
    # `config/camera.py` is what stops a weapons camera silently carrying 25 bullying
    # thresholds nothing reads, which is the mistake the legacy made on two canteen
    # cameras.
    WEAPONS = "weapons"


class CameraRole(StrEnum):
    MAIN_HALL = "main_hall"
    STAIRS = "stairs"
    OUTDOOR = "outdoor"
    CANTEEN_ENTRY = "canteen_entry"
    CANTEEN_EXIT = "canteen_exit"
    CANTEEN_INSIDE = "canteen_inside"
    # §12.4: "отдельная камера с обзором учеников" -- one role, because the classroom
    # camera has exactly one job. The canteen needed three roles because entry, exit and
    # inside cameras do genuinely different things to a meal record; a second classroom
    # camera would do the same thing as the first.
    CLASSROOM = "classroom"
    # §12.1: ONE role, for the same reason the classroom has one -- a second weapons
    # camera does the same job as the first. It is deliberately NOT split into
    # "entrance" and "corridor" even though the physics differs enormously between them
    # (100+ px against ~15 px for the same knife): nothing in the code would BEHAVE
    # differently, and a role that changes nothing is a distinction somebody will read as
    # a guarantee. Where a camera can and cannot work is answered by
    # `qorgan weapons camera-report`, which measures it, and by the docstring in
    # `config/weapons.py`, which states it beside the settings.
    WEAPONS = "weapons"


BULLYING_ROLES = frozenset({CameraRole.MAIN_HALL, CameraRole.STAIRS, CameraRole.OUTDOOR})
CANTEEN_ROLES = frozenset(
    {CameraRole.CANTEEN_ENTRY, CameraRole.CANTEEN_EXIT, CameraRole.CANTEEN_INSIDE}
)
CLASSROOM_ROLES = frozenset({CameraRole.CLASSROOM})
WEAPONS_ROLES = frozenset({CameraRole.WEAPONS})


class ClipEnd(StrEnum):
    """What a camera backed by a recorded file does when the recording runs out.

    An RTSP stream never ends; a file does. There is no default that is right for both
    uses of a file-backed camera, so it is asked rather than guessed:

      LOOP  the clip starts again at its first frame and the stream never ends. For a
            demonstration -- the hall keeps moving for as long as anybody is watching.
      STOP  the source ends and the camera's thread finishes. For a run over the corpus,
            where "the clip is over" is the answer, and replaying it would count the same
            children walking past twice.
    """

    LOOP = "loop"
    STOP = "stop"


class PersonType(StrEnum):
    STUDENT = "student"
    STAFF = "staff"


class ExternalIdSource(StrEnum):
    """Where a person's external_id came from.

    There used to be a GENERATED member, for ids derived from a name at photo import.
    There is no such thing now: the school issues the ids and we read them. An id we
    invented is an id we can be wrong about, and being wrong about identity is how a child
    eats someone else's lunch.
    """

    ROSTER = "roster"  # authoritative, from the school's own directory tree


class EventType(StrEnum):
    BULLYING = "bullying"
    # §12.1. A separate type rather than a severity, because the two are not the same
    # KIND of record and three of the columns beside them mean different things:
    # `skeleton_confirmed` is not applicable to a weapon (there is no pose tier in that
    # pipeline) and `/events` would render its False as "the skeleton disagreed", which
    # is a statement nobody made. `/events` therefore filters to BULLYING and `/weapons`
    # shows these -- see `qorgan/weapons/store.py` for what each column holds on a
    # weapon row.
    WEAPON = "weapon"


class EventStatus(StrEnum):
    """**Was the detector right?** — and nothing else.

    Every member here is a verdict on the DETECTION. `/events/{id}/review` is the only
    channel through which the school tells us we were wrong, and a detector nobody
    corrects never improves, so this column has exactly one job and must keep it.

    **«Передано психологу» is deliberately NOT a member, and it was one for a day.**
    §9 lists five marks a member of staff must be able to make — «подтверждено; ложное
    срабатывание; требует доп. проверки; передано психологу; закрыто» — and it is
    tempting to read that as the contents of this enum. It is not: two of those five
    («требует доп. проверки», «закрыто») have never been members either. §9 lists what an
    operator can RECORD, and this column answers one of those questions rather than all
    of them.

    A referral was added here on 2026-07-28 and removed on 2026-07-29, having been
    measured: because `refer()` wrote the token into `status`, and the review controls are
    drawn only for an unreviewed event, an operator who pressed «Передать психологу» could
    never afterwards mark that event confirmed or false — **every referred incident fell
    silently out of the loop that corrects the detector.** The reverse order lost more: a
    confirmed event, once referred, no longer said it had been confirmed, and the verdict
    was unrecoverable, because `reviewed_at`/`reviewed_by_id` record who and when but
    never WHAT was decided.

    The two facts are orthogonal — «был ли детектор прав» and «куда это ушло» — so they
    are two columns. The referral lives in `events.referred_at` / `events.referred_by_id`
    and is written by exactly one function (`psychologist.referrals.refer`), which cannot
    write it without a name and a minute. That was always where the cabinet read it from:
    every query in `qorgan.psychologist` selects on `referred_at IS NOT NULL`, so the
    token this enum used to carry was already redundant on the day it was added — it
    changed nothing except which controls the page would still draw.
    """

    NEW = "new"
    REVIEWED = "reviewed"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"


class Severity(StrEnum):
    SUSPICION = "suspicion"
    ALERT = "alert"
    CRITICAL = "critical"


class SessionState(StrEnum):
    OPEN = "open"
    INSIDE_CONFIRMED = "inside_confirmed"
    CLOSED = "closed"


class SessionOutcome(StrEnum):
    ATE = "ate"
    NOT_ATE = "not_ate"
    # LEFT_IMMEDIATELY ("came in and left") DELETED with the quick-return path. It was
    # reachable only through close(quick_return=True), which production never passes
    # (see canteen/sessions.py), so the exit guard meant no session ever closed younger
    # than 30s and classify() never returned it. An outcome nothing can produce is a
    # database column that lies about the range of a meal record.
    UNKNOWN = "unknown"


class IdentitySource(StrEnum):
    """How a session learned who it belongs to."""

    ENTRY = "entry"
    INSIDE_LATE_BIND = "inside_late_bind"
    EXIT = "exit"
    MANUAL = "manual"


class CloseReason(StrEnum):
    EXIT_CAMERA = "exit_camera"
    # QUICK_RETURN DELETED with the quick-return path (see canteen/sessions.py): production
    # never closes a session younger than the exit guard, so no close was ever reasoned
    # QUICK_RETURN. A close is now either the exit camera, the 90-minute timeout, or manual.
    TIMEOUT = "timeout"
    MANUAL = "manual"


class LessonState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class LessonCloseReason(StrEnum):
    """How a lesson ended. Two members, because there are two ways it can.

    There is no BELL and no SCHEDULED, because the school has given us no timetable and
    the system has no bell signal -- a member nothing can produce is a column that lies
    about the range of a record, which is why `SessionOutcome.LEFT_IMMEDIATELY` and
    `CloseReason.QUICK_RETURN` were deleted rather than kept for symmetry. There is no
    MANUAL either: nothing in the web layer closes a lesson, and the day something does
    is the day this gains a third member.
    """

    # The room emptied and stayed empty. The ordinary ending.
    EMPTY_ROOM = "empty_room"
    # Still open after `max_lesson_minutes`. Recorded honestly as "we lost track of the
    # end of this lesson" rather than left open to absorb tomorrow's tracks.
    TIMEOUT = "timeout"


class MealKind(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    SNACK = "snack"


class UserRole(StrEnum):
    """A name for a set of capabilities, not a rank. See `qorgan.roles`.

    What each of these may do is written in `ROLE_CAPABILITIES` and nowhere else. The
    comments here describe the job, not an ordering: none of these implies another.
    """

    OPERATOR = "operator"  # view, review events
    ADMIN = "admin"  # settings, pupils, cameras
    DEVELOPER = "developer"  # debug views, ROI calibration
    CANTEEN_STAFF = "canteen_staff"  # §14: the canteen journal, БЕЗ доступа к буллингу
    # §14: «сигналы по ученикам; подтвержденные случаи; история наблюдений; свои
    # комментарии». Deliberately absent until 2026-07-28, and the absence was correct as
    # written: `roles.py` refuses a permission that guards nothing, so a role whose pages
    # do not exist is a guess. It arrives HERE, in the same change that adds `/psychologist`
    # and the notes routes — which is the only order that keeps that rule true.
    PSYCHOLOGIST = "psychologist"
    # §14: "управление школами, серверами". **The only role whose `users.school_id` is
    # NULL, because it is not inside a school** -- and the only role a form can never
    # assign (`accounts.ASSIGNABLE_ROLES`), because a page one school's headteacher can
    # reach must not be able to mint an account that reaches every other school.
    #
    # It holds the schools register and the server diagnostics, and NOT one child-facing
    # capability: not the corridor previews, not the bullying log, not a photograph. Whose
    # data it is does not change because somebody administers the machine it sits on, and
    # a role that manages twenty schools would otherwise put twenty schools of children
    # behind one password.
    SUPERADMIN = "superadmin"


class NotificationChannel(StrEnum):
    TELEGRAM = "telegram"


class NotificationStatus(StrEnum):
    QUEUED = "queued"
    SENT = "sent"
    FAILED = "failed"


class TelegramSkipReason(StrEnum):
    """Why a recorded event was deliberately NOT sent to Telegram (client §7).

    This answers a different question from `NotificationStatus`, and the difference is the
    whole point. A notification row exists only for an alert somebody tried to deliver, so
    FAILED means "we tried and the message did not arrive" — a broken router, a bad token.
    These members mean "we decided not to try", which has no row at all and, until this
    enum existed, left no trace anywhere: the school asked "why was I not told about the
    fight?" and the system could not answer.

    **An enumeration, not free text.** The legacy's `_telegram_skip_reason` was a sentence
    assembled at each call site, so the same cause was worded three ways and none of them
    could be filtered or counted. Every member below corresponds to exactly one branch of
    the real decision path — `detection.validation.judge` and the merger's at-most-once
    claim — and `tests/test_telegram_skip_reason.py` drives each one through that path.
    A reason nothing can produce is a dead handle, and this project has had one before
    (`display_fps`, which passed the dead-key check for its whole life).

    Three of §7's seven causes are deliberately absent; see that test module for why.
    """

    # The pose model could not look at all: disabled, too few crops, or it raised.
    # Distinct from NO_SKELETON_CONFIRMATION on purpose — `detection.validation` insists
    # the two are different, and only one of them is an operational fault.
    SKELETON_NOT_RUN = "skeleton_not_run"
    # It looked, and what it saw did not reach the aggression bar.
    NO_SKELETON_CONFIRMATION = "no_skeleton_confirmation"
    # It looked, it scored, and every reason it gave was weak: gesturing, standing close,
    # legs swinging. §5.9's `_weak_evidence_social_contact_block`. That is a playground.
    WEAK_EVIDENCE_ONLY = "weak_evidence_only"
    # The skeleton confirmed and the blended confidence still fell short of the bar.
    LOW_CONFIDENCE = "low_confidence"
    # A human has already been told about THIS incident. §7 calls this "cooldown"; in v2
    # the mechanism is `detection.merging.EventMerger`, not a timer.
    ALREADY_NOTIFIED = "already_notified"


class WorkerState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    CRASHED = "crashed"


class SystemMode(StrEnum):
    BULLYING = "bullying"
    CANTEEN = "canteen"
    BOTH = "both"
    IDLE = "idle"


class ModeSource(StrEnum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    STARTUP = "startup"
