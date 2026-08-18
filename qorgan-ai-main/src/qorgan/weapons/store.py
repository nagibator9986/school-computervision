"""Writing a weapon alert, and recording the person who ruled on it.

**A weapon alert is never auto-actioned.** §12.1 and `docs/questions-for-school.md` §7 say
so in the same words -- «тревогу всегда подтверждает человек» -- and the reason is not
squeamishness: a false gun alert in a school has consequences of its own, and they land on
a child. So what this module writes is a QUESTION, and the answer is somebody's name.

Three things follow, and they are all here rather than described anywhere:

  * the row is written at `EventStatus.NEW` and **nothing in `src/` moves it off NEW
    except `rule_on_weapon_alert`, which cannot be called without a user id**;
  * the summary is worded as a possibility requiring confirmation, not as a finding. The
    row is read back out by the Telegram notifier, so the wording on a teacher's phone is
    this string and no other -- and «Обнаружено оружие» on a phone IS an automatic action,
    whatever the database says afterwards;
  * the weights that produced it are recorded ON THE ROW. Not for tidiness: this project's
    signature failure is a value that was true in one layer and quietly wrong in the next,
    and "which model said this?" asked six months later cannot be answered from a config
    file that has been edited since.

## What the bullying-shaped columns hold on a weapon row

`events` was built for the bullying tier and three of its columns need saying plainly,
because a column that means two things is the defect `migrations/0005` exists about:

  * `candidate_probability` -- the count of observations the track collected, as a
    fraction of what `min_track_observations` demanded, capped at 1.0. It is genuinely a
    "how much of the first gate did this clear" number, which is what the column is for.
  * `validation_score` -- the same for the reconfirmation gate. The second tier's
    evidence, which is what this column means on a bullying row too.
  * `skeleton_confirmed` -- **always False, and it means NOT APPLICABLE.** There is no
    pose tier in this pipeline. `/events` renders False as "the skeleton did not confirm",
    which would be a statement nobody made, and that is the reason `/events` filters to
    `EventType.BULLYING` and `/weapons` is a separate page rather than a filter on it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import Event
from qorgan.db.tenancy import resolve_school_id, scope
from qorgan.db.types import utcnow
from qorgan.enums import EventStatus, EventType, Severity
from qorgan.events.reasons import pack_reasons
from qorgan.logging_setup import get_logger
from qorgan.weapons.pipeline import WeaponAlert
from qorgan.weapons.weights import LoadedWeights

logger = get_logger(__name__)

# §12.1's target classes, in the language of the person holding the phone.
CLASS_LABELS: dict[str, str] = {
    "knife": "нож",
    "axe": "топор",
    "bat": "дубинка",
    "metal_object": "металлический предмет",
    "firearm": "огнестрельное оружие",
}

# The verdicts a human may return. Deliberately the two that already exist: a weapon alert
# is confirmed or it is a false positive, and there is no third answer that changes
# anything. `REVIEWED` is not offered -- "somebody looked" is not a ruling, and on a
# weapon it would leave the row in a state that reads as handled and asserts nothing.
WEAPON_VERDICTS = (EventStatus.CONFIRMED, EventStatus.FALSE_POSITIVE)


def label_for(class_name: str) -> str:
    """A weapon class in Russian. An unknown class prints as itself -- ugly beats silent,
    and a class the module has no word for still has to reach a person."""
    return CLASS_LABELS.get(class_name, class_name)


def summarise_weapon(alert: WeaponAlert, camera_display_name: str) -> str:
    """The words a human reads first, and they are a question.

    Read back out of the row by the notifier, so this string is what appears on a phone.
    «Возможное» and «требуется подтверждение» are load-bearing: §12.1 forbids an automatic
    action, and a message asserting that a weapon was found IS the action -- everything a
    school does in the next sixty seconds follows from the sentence, not from the status
    column somebody sets afterwards.
    """
    return (
        f"Возможное оружие: {label_for(alert.class_name)} — {camera_display_name} "
        f"({alert.confidence:.0%}). Требуется подтверждение человека"
    )


def record_weapon_alert(
    *,
    camera_id: int,
    occurred_at: datetime,
    alert: WeaponAlert,
    weights: LoadedWeights,
    summary_text: str,
    min_observations: int,
    reconfirm_observations: int,
    snapshot_path: str | None = None,
    clip_path: str | None = None,
) -> int:
    """Insert one weapon alert at status NEW and return its id.

    `with_retry`, exactly as `events.store.record_event`: under WAL a concurrent writer
    can still produce `database is locked`, and in the legacy that exception killed the
    event-writing thread outright and forever (audit H-10).

    Severity is always CRITICAL, and that is not a claim about certainty. Severity here is
    how fast somebody has to look, and the answer for a possible weapon in a school is
    "now". How SURE the system is lives in `confidence` and in the wording of the summary,
    which says «возможное» and asks for a confirmation.
    """

    def _insert() -> int:
        with session_scope() as session:
            event = _weapon_row(
                camera_id=camera_id,
                occurred_at=occurred_at,
                alert=alert,
                weights=weights,
                summary_text=summary_text,
                min_observations=min_observations,
                reconfirm_observations=reconfirm_observations,
                snapshot_path=snapshot_path,
                clip_path=clip_path,
            )
            session.add(event)
            session.flush()
            return event.id

    return with_retry(_insert)


def _weapon_row(
    *,
    camera_id: int,
    occurred_at: datetime,
    alert: WeaponAlert,
    weights: LoadedWeights,
    summary_text: str,
    min_observations: int,
    reconfirm_observations: int,
    snapshot_path: str | None,
    clip_path: str | None,
) -> Event:
    """The row itself. Split out only so `record_weapon_alert` fits R1's 50 lines."""
    return Event(
        camera_id=camera_id,
        event_type=EventType.WEAPON,
        occurred_at=occurred_at,
        confidence=alert.confidence,
        candidate_probability=_gate_fraction(alert.observations, min_observations),
        validation_score=_gate_fraction(alert.strong_observations, reconfirm_observations),
        # NOT APPLICABLE, not "the pose model disagreed". See the module docstring.
        skeleton_confirmed=False,
        severity=Severity.CRITICAL,
        summary_text=summary_text,
        reasons=pack_reasons(alert.reasons + _provenance(weights)),
        snapshot_path=snapshot_path,
        clip_path=clip_path,
        # The weapon track and the person it was beside. An operator opening the clip is
        # looking for one of these two numbers.
        track_ids=f"{alert.track_id},{alert.person_track_id or 0}",
        # Explicit rather than left to the column default: this is the whole of "never
        # auto-actioned", and a default is a thing that can be changed somewhere else.
        status=EventStatus.NEW,
    )


def rule_on_weapon_alert(
    event_id: int,
    verdict: EventStatus,
    user_id: int,
    username: str,
    school_id: int | None = None,
) -> bool:
    """A person's ruling on a weapon alert. **The only way one ever leaves NEW.**

    `user_id` is required and is not optional anywhere up the call chain. Every other
    argument could be defaulted; this one is the point of the function, and §12.1's
    «в записи остаётся, кто подтвердил» is a column that must not be NULL on a ruled row.

    Returns False when there is no such weapon event -- a caller must not be able to turn
    a bullying event into a confirmed weapon by guessing an id, and must not be able to
    declare ANOTHER SCHOOL'S child armed by guessing one either. The lookup is scoped
    and type-checked for those two reasons, and the caller cannot tell the refusals
    apart: which ids exist elsewhere on the installation is not this school's business.
    """
    if verdict not in WEAPON_VERDICTS:
        raise ValueError(
            f"{verdict!r} is not a ruling on a weapon alert. A person answers "
            f"{[v.value for v in WEAPON_VERDICTS]}: anything else leaves the row reading "
            "as handled while asserting nothing."
        )

    def _update() -> bool:
        with session_scope() as session:
            school = resolve_school_id(session, school_id)
            mine = scope(select(Event), Event, school)
            event = session.scalar(mine.where(Event.id == event_id))
            if event is None or event.event_type is not EventType.WEAPON:
                return False
            event.status = verdict
            event.reviewed_by_id = user_id
            event.reviewed_at = utcnow()
            return True

    ruled = with_retry(_update)
    logger.warning(
        "weapon alert ruled on by a person",
        extra={
            "event_id": event_id,
            "verdict": verdict.value,
            "by": username,
            "found": ruled,
        },
    )
    return ruled


def _gate_fraction(reached: int, required: int) -> float:
    """How much of a counting gate this track cleared, as 0..1.

    Capped at 1.0 because the columns are probabilities on every other row and a 2.3 in a
    `Float` column that reads as a probability everywhere else is a number that will be
    averaged with the others one day.
    """
    if required <= 0:
        return 1.0
    return min(1.0, reached / required)


def _provenance(weights: LoadedWeights) -> tuple[str, ...]:
    """Which weights said this, as slugs on the row.

    A fingerprint rather than a path: the path is in the config and the config gets
    edited, and `best.pt` is what every Ultralytics training run in the world calls its
    output. `pack_reasons` refuses a comma, so nothing here may contain one -- the
    fingerprint is hex and the token is fixed.
    """
    return (f"weights:{weights.file.fingerprint}",)
