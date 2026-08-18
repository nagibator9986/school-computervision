"""Meal sessions: the domain core, as a persisted state machine.

**A session is a ROW, not a dict entry.** In the legacy, sessions lived in a RAM
dictionary inside a module-global singleton, so a process restart silently lost every
open session — every child who had walked in but not yet walked out simply ceased to
exist. Worse, one code path popped a session from that dict *without ever writing it to
the database*, so the pupil vanished from the record entirely (spec §5.2).

The rules below were earned in a real canteen. They look arbitrary until you stand in the
doorway:

  * **Only the entry camera opens a session; only the exit camera closes it.** Inside
    cameras confirm presence and may late-bind an identity to a session opened as Unknown.
    They never open and never close.

  * **The exit camera will not close a session younger than 30 seconds.** It is pointed at
    the backs of heads, and it sees the back of a child who has *just walked in*. Without
    this rule, a pupil is marked as having left the moment they arrive.

  * **A per-pupil cooldown of 60 seconds on each door**, because one child lingering in a
    doorway is one meal, not forty.

  * **Staff never create meal sessions.** They are tracked separately, with a TTL.

  * **A session nobody ever exited is force-closed after 90 minutes** as `unknown`. It is
    better to record honestly that we lost track of a child than to leave a session open
    for the rest of the year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from qorgan.config.canteen import MealOutcomeRules, SessionRules
from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import CanteenSession
from qorgan.db.types import utcnow
from qorgan.enums import CloseReason, IdentitySource, SessionOutcome, SessionState
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

# `close()` refused because the session is not old enough YET. **This refusal is TRANSIENT**
# and it is the only one that is: wait, and the same call succeeds. Every other refusal is
# final. A caller that treats this as a verdict spends its one attempt on a "not yet" and
# the session never closes at all -- see `CanteenPipeline._on_exit`. Named, because a
# caller matching on the string "session_too_young" is an agreement two modules keep by
# luck.
SESSION_TOO_YOUNG = "session_too_young"


@dataclass(frozen=True, slots=True)
class OpenResult:
    session_id: int | None
    opened: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CloseResult:
    session_id: int | None
    closed: bool
    outcome: SessionOutcome | None
    dwell_seconds: float | None
    reason: str


class SessionManager:
    """Opens, closes and force-closes meal sessions. Every decision is a row."""

    def __init__(
        self,
        rules: SessionRules,
        outcomes: MealOutcomeRules,
        entry_camera_id: int,
        exit_camera_id: int | None = None,
    ) -> None:
        self.rules = rules
        self.outcomes = outcomes
        self.entry_camera_id = entry_camera_id
        self.exit_camera_id = exit_camera_id

    # -- entry ---------------------------------------------------------------

    def open(
        self, person_id: int | None, now: datetime | None = None, *, is_staff: bool = False
    ) -> OpenResult:
        """A pupil walked in."""
        moment = now or utcnow()

        if is_staff:
            # Staff eat too, but they do not have a meal record. Counting them as pupils
            # is how a cook ends up on the "did not eat" report.
            return OpenResult(None, False, "staff_do_not_open_sessions")

        if person_id is None and not self.rules.allow_unknown_sessions:
            return OpenResult(None, False, "unknown_sessions_disabled")

        def _open() -> OpenResult:
            with session_scope() as session:
                if person_id is not None:
                    blocked = self._cooldown_block(session, person_id, moment)
                    if blocked:
                        return blocked

                    already = self._active_for(session, person_id, moment)
                    if already is not None:
                        return OpenResult(already.id, False, "already_inside")

                row = CanteenSession(
                    person_id=person_id,
                    entry_camera_id=self.entry_camera_id,
                    state=SessionState.OPEN,
                    opened_at=moment,
                    identity_source=IdentitySource.ENTRY if person_id else None,
                )
                session.add(row)
                session.flush()
                return OpenResult(row.id, True, "opened")

        return with_retry(_open)

    def _cooldown_block(
        self, session, person_id: int, moment: datetime
    ) -> OpenResult | None:
        """One child lingering in a doorway is one meal, not forty.

        Only the ENTRY door has a cooldown, and the asymmetry is real, not an oversight. A
        cooldown stops a second RECORD being made for the same child, and only `open` makes
        records. `close` mutates an existing row and is already idempotent -- a person has
        one open session at a time and a closed session never reopens -- so there is no exit
        double-count for a cooldown to prevent.
        """
        cutoff = moment - timedelta(seconds=self.rules.entry_cooldown_seconds)

        recent = session.scalar(
            select(CanteenSession)
            .where(
                CanteenSession.person_id == person_id,
                CanteenSession.opened_at >= cutoff,
            )
            .order_by(CanteenSession.opened_at.desc())
        )
        if recent is None:
            return None
        return OpenResult(recent.id, False, "cooldown")

    def _active_for(self, session, person_id: int, moment: datetime) -> CanteenSession | None:
        return session.scalar(
            select(CanteenSession)
            .where(
                CanteenSession.person_id == person_id,
                CanteenSession.state != SessionState.CLOSED,
            )
            .order_by(CanteenSession.opened_at.desc())
        )

    # -- inside --------------------------------------------------------------

    def active_session_id(self, person_id: int) -> int | None:
        """This pupil's open session, if they have one."""
        with session_scope() as session:
            row = self._active_for(session, person_id, utcnow())
            return row.id if row is not None else None

    def confirm_inside(self, session_id: int) -> bool:
        """An inside camera saw them. It cannot open or close anything."""

        def _confirm() -> bool:
            with session_scope() as session:
                row = session.get(CanteenSession, session_id)
                if row is None or row.state is SessionState.CLOSED:
                    return False
                row.state = SessionState.INSIDE_CONFIRMED
                return True

        return with_retry(_confirm)

    def late_bind(self, session_id: int, person_id: int, score: float) -> bool:
        """An inside camera worked out who the Unknown session belongs to.

        Only ever fills in a blank. The legacy's `resolve_exit_session` would attach a
        recognised pupil to somebody ELSE'S oldest Unknown session, handing them another
        child's dwell time and meal status (spec §5.2).
        """

        def _bind() -> bool:
            with session_scope() as session:
                row = session.get(CanteenSession, session_id)
                if row is None or row.state is SessionState.CLOSED:
                    return False
                if row.person_id is not None:
                    return False  # never overwrite an identity we already had

                row.person_id = person_id
                row.identity_source = IdentitySource.INSIDE_LATE_BIND
                row.identity_score = score
                return True

        return with_retry(_bind)

    # -- exit ----------------------------------------------------------------

    def close(
        self,
        person_id: int,
        now: datetime | None = None,
    ) -> CloseResult:
        """A pupil walked out.

        The exit camera is pointed at the backs of heads, so `close()` refuses a session
        younger than `exit_min_session_age_seconds`: the back it most often sees is a child
        who has just walked IN, and without the guard a pupil is marked as having left the
        moment they arrive. That guard has NO bypass. A genuine "walked in, saw the queue,
        turned round and walked out" cannot be told from a just-entered child's face by any
        signal this pipeline produces, so the state machine does not guess one -- it waits
        until the session is old enough to close honestly, or the janitor times it out. The
        quick-return escape hatch and its `left_immediately` outcome are gone with it.
        """
        moment = now or utcnow()

        def _close() -> CloseResult:
            with session_scope() as session:
                row = session.scalar(
                    select(CanteenSession)
                    .where(
                        CanteenSession.person_id == person_id,
                        CanteenSession.state != SessionState.CLOSED,
                    )
                    .order_by(CanteenSession.opened_at.desc())
                )
                if row is None:
                    return CloseResult(None, False, None, None, "no_open_session")

                age = (moment - row.opened_at).total_seconds()
                blocked = self._too_young(row.id, age)
                if blocked is not None:
                    return blocked

                # Past the guard, so age >= exit_min_session_age_seconds: a real close is
                # always the exit camera's. (CloseReason.QUICK_RETURN went with the path.)
                outcome = SessionOutcome(self.outcomes.classify(age))
                _finalise(row, moment, outcome, CloseReason.EXIT_CAMERA, age, self.exit_camera_id)
                return CloseResult(row.id, True, outcome, age, "closed")

        return with_retry(_close)

    def _too_young(self, session_id: int, age: float) -> CloseResult | None:
        """The exit camera looks at the backs of heads, and the back it most often sees is
        that of a child who has just walked IN. Without this guard a pupil is marked as
        having left the moment they arrive, and the whole meal record is corrupt. The guard
        is absolute: there is no quick-return bypass, because nothing can supply the strong
        evidence a bypass would demand."""
        if age >= self.rules.exit_min_session_age_seconds:
            return None
        return CloseResult(session_id, False, None, age, SESSION_TOO_YOUNG)

    # -- the janitor ---------------------------------------------------------

    def force_close_stale(self, now: datetime | None = None) -> int:
        """This canteen's view of `close_sessions_nobody_exited`."""
        return close_sessions_nobody_exited(self.rules, now)


def close_sessions_nobody_exited(rules: SessionRules, now: datetime | None = None) -> int:
    """Close sessions nobody ever exited.

    Better to record honestly that we lost track of a child than to leave a session
    open for the rest of the school year, quietly blocking every future meal for them.

    It takes the RULES, not a `SessionManager`. The sweep reads `max_session_minutes` and
    nothing else — no cameras, no outcome ladder — but for five releases it could only be
    reached through a manager, which needs an `entry_camera_id` this code never looks at.
    So the only callers it ever had were tests that already had a manager to hand. The
    process that must run it (`supervisor`) has no camera rows and no business making any.
    """
    moment = now or utcnow()
    cutoff = moment - timedelta(minutes=rules.max_session_minutes)

    def _sweep() -> int:
        with session_scope() as session:
            stale = session.scalars(
                select(CanteenSession).where(
                    CanteenSession.state != SessionState.CLOSED,
                    CanteenSession.opened_at < cutoff,
                )
            ).all()

            for row in stale:
                age = (moment - row.opened_at).total_seconds()
                _finalise(row, moment, SessionOutcome.UNKNOWN, CloseReason.TIMEOUT, age, None)

            if stale:
                logger.warning(
                    "force-closed sessions with no exit recorded",
                    extra={"count": len(stale)},
                )
            return len(stale)

    return with_retry(_sweep)


def _finalise(
    row: CanteenSession,
    moment: datetime,
    outcome: SessionOutcome,
    reason: CloseReason,
    dwell: float,
    exit_camera_id: int | None,
) -> None:
    row.state = SessionState.CLOSED
    row.outcome = outcome
    row.close_reason = reason
    row.closed_at = moment
    row.dwell_seconds = dwell
    if exit_camera_id is not None:
        row.exit_camera_id = exit_camera_id
