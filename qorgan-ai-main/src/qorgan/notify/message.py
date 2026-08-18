"""The words on a teacher's phone.

The message used to be `summarise()` and nothing else:

    Зафиксирована агрессия — Холл слева (93%)

which is severity, camera and confidence — and the school asked for three other things
(§7), none of which were there:

  * **Why.** The reasons existed on the `Verdict` all along and went to a log line nobody
    reads. "Someone went down" and "two children waved their arms" carry the same
    severity and the same confidence, and they are not the same emergency.

  * **When.** The message had no time at all, so it was only usable if you happened to be
    holding the phone at the instant it buzzed. An alert read ten minutes later described
    an incident at an unknown moment.

  * **Which event.** Nothing tied the message to a row. There was no way to open the
    dashboard and find the snapshot for the alert in your hand.

Two rules this module exists to keep:

  * **The school's time, not UTC.** Every timestamp in the database is UTC by column type
    and the school is UTC+5. A message saying 09:12 for a fight at 14:12 sends a teacher
    to the wrong five minutes of CCTV, where they find an empty corridor and conclude the
    system is lying to them.

  * **The message is delivered even when this module is surprised.** An unknown reason
    slug prints as itself rather than raising: a `KeyError` here happens inside the
    notification queue, which would retry an assault alert five times and then dead-letter
    it. Nobody loses an alert over a missing dictionary entry.
"""

from __future__ import annotations

from datetime import datetime

from qorgan.settings import get_settings

# Every reason slug this system can put on a row, in the language of the person holding
# the phone. A slug that is missing here is printed raw, never raised over.
#
# **There are TWO producers and ONE table, on purpose.** `describe_reasons` is called by
# the notifier, which reads a row and cannot know which pipeline wrote it; a second lookup
# beside this one would be a second place to forget, and the failure mode is a phone
# message full of `weapon_near_a_person`. The keys remain the contract with whichever
# module produced them.
REASON_LABELS: dict[str, str] = {
    # The five skeleton features (`qorgan.detection.validation.REASON_EVIDENCE`).
    "body_fall_or_low_posture": "падение или низкая поза",
    "sudden_body_displacement": "резкое смещение тела",
    "rapid_hand_motion": "резкое движение рук",
    "close_upper_body_contact": "контакт корпусом",
    "kick_like_leg_motion": "удар ногой",
    # §12.1's three gates (`qorgan.weapons.pipeline.EVIDENCE`). A weapon alert also
    # carries a `weights:<fingerprint>` slug, which is deliberately NOT translated: it is
    # a hex identifier, and a made-up Russian phrase for it would be worse than the hex.
    "weapon_tracked_across_frames": "объект отслеживался несколько кадров",
    "weapon_near_a_person": "объект рядом с человеком",
    "weapon_reconfirmed": "повторное подтверждение моделью",
}

# DD.MM.YYYY, which is what this school reads. Not ISO, and certainly not MM/DD.
TIME_FORMAT = "%d.%m.%Y %H:%M:%S"


def compose_caption(
    *, summary: str, occurred_at: datetime, event_id: int, reasons: tuple[str, ...]
) -> str:
    """The full alert: what, when, which row, and why.

    Kept comfortably inside Telegram's 1024-character photo-caption limit by the shape of
    its inputs — a caption over the limit is a 400, which this system treats as permanent,
    which would silently downgrade every alert to text and throw the picture away.
    """
    lines = [
        summary,
        f"Время: {local_time(occurred_at)}",
        f"Событие №{event_id}",
    ]
    if reasons:
        # Omitted entirely rather than left dangling: "Признаки:" with nothing after it
        # reads as a bug, and an event the skeleton could not look at is normal.
        lines.append(f"Признаки: {describe_reasons(reasons)}")
    return "\n".join(lines)


def local_time(moment: datetime) -> str:
    """UTC in, the school's wall clock out."""
    if moment.tzinfo is None:
        # Never guessed at. Naive could mean UTC or local, the two are five hours apart,
        # and being five hours wrong about an assault is worse than failing loudly. Every
        # timestamp here is tz-aware by column type, so this means something upstream is
        # broken.
        raise ValueError(
            f"refusing to put a timezone-less timestamp on an alert: {moment!r}. "
            "Every timestamp in this system is UTC and tz-aware; a naive one here means "
            "something upstream dropped the zone."
        )
    return moment.astimezone(get_settings().tz).strftime(TIME_FORMAT)


def describe_reasons(reasons: tuple[str, ...]) -> str:
    """Slugs in, Russian out. An unknown slug prints as itself — ugly beats undelivered."""
    return ", ".join(REASON_LABELS.get(reason, reason) for reason in reasons)
