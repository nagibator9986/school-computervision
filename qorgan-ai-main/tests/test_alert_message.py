"""What a teacher actually reads when their phone buzzes.

Client §7 asks for three things in the message: the reasons, the time, and the event id.
The message used to be `summarise()` and nothing else -- "Зафиксирована агрессия — Холл
слева (93%)" -- which carries none of them. A teacher reading it knows something happened
somewhere in the hall, at no stated time, with no way to look it up afterwards, and no
idea whether a child is on the floor or two children waved their arms.
"""

from __future__ import annotations

from datetime import UTC, datetime

from qorgan.notify.message import compose_caption
from qorgan.settings import Settings

# 09:12:30 UTC is 14:12:30 in Almaty. The gap is the whole point of the timezone test
# below: five hours is the difference between the right CCTV footage and the wrong.
MOMENT = datetime(2026, 3, 4, 9, 12, 30, tzinfo=UTC)


def _caption(settings: Settings, **kwargs) -> str:
    base = {
        "summary": "Зафиксирована агрессия — Холл слева (93%)",
        "occurred_at": MOMENT,
        "event_id": 41,
        "reasons": ("body_fall_or_low_posture", "rapid_hand_motion"),
    }
    return compose_caption(**{**base, **kwargs})


def test_the_message_still_leads_with_what_it_always_led_with(settings: Settings) -> None:
    """The severity line is what a human reads first, and it stays first."""
    assert _caption(settings).startswith("Зафиксирована агрессия — Холл слева (93%)")


def test_the_message_says_WHY(settings: Settings) -> None:
    """The reasons existed on the Verdict all along and never left the log line."""
    caption = _caption(settings)

    assert "падение или низкая поза" in caption
    assert "резкое движение рук" in caption


def test_the_reasons_are_readable_by_a_teacher_not_by_a_developer(settings: Settings) -> None:
    """`body_fall_or_low_posture` is a slug in a Python dict. The person holding the phone
    is a Russian-speaking teacher deciding whether to run."""
    caption = _caption(settings)

    assert "body_fall_or_low_posture" not in caption
    assert "rapid_hand_motion" not in caption


def test_the_message_says_WHEN_in_the_school_s_own_time_not_UTC(settings: Settings) -> None:
    """THE one that matters. Timestamps are stored UTC; the school is UTC+5. A teacher
    reading "09:12" for a fight that happened at 14:12 scrolls to the wrong five minutes
    of CCTV and finds an empty corridor, and concludes the system is lying."""
    caption = _caption(settings)

    assert "14:12:30" in caption
    assert "09:12:30" not in caption, "the message is in UTC; nobody at this school lives in UTC"


def test_the_message_says_WHEN_in_a_format_this_school_reads(settings: Settings) -> None:
    """04.03.2026, not 2026-03-04 and emphatically not 03/04/2026."""
    assert "04.03.2026" in _caption(settings)


def test_the_message_carries_the_event_id(settings: Settings) -> None:
    """Without it the alert cannot be looked up in the dashboard afterwards. It is the
    only handle between the phone in a corridor and the row in the database."""
    assert "41" in _caption(settings, event_id=41)


def test_a_naive_timestamp_is_refused_rather_than_guessed_at(settings: Settings) -> None:
    """A datetime with no timezone could be UTC or could be local; the two are five hours
    apart and guessing wrong puts the wrong time on an assault alert. Every timestamp in
    this system is tz-aware by column type -- if a naive one ever reaches here, something
    upstream is broken and must say so."""
    import pytest

    with pytest.raises(ValueError, match="timezone"):
        _caption(settings, occurred_at=datetime(2026, 3, 4, 9, 12, 30))


def test_an_unknown_reason_does_not_cost_the_school_an_alert(settings: Settings) -> None:
    """The five skeleton features are a closed set today. If a sixth is ever added and
    nobody updates the label table, the alert must still be delivered -- with an ugly
    slug in it -- rather than raising a KeyError inside the notification queue and
    retrying an assault alert to death."""
    caption = _caption(settings, reasons=("newly_invented_feature",))

    assert "newly_invented_feature" in caption
    assert "Зафиксирована агрессия" in caption


def test_an_event_with_nothing_to_say_still_sends_a_clean_message(settings: Settings) -> None:
    """No reasons is a real state: the skeleton is often unable to look. The message must
    not end with a dangling "Признаки:" and nothing after it."""
    caption = _caption(settings, reasons=())

    assert "Признаки" not in caption
    assert "Зафиксирована агрессия" in caption
    assert "14:12:30" in caption
    assert "41" in caption


def test_the_message_fits_in_a_telegram_photo_caption(settings: Settings) -> None:
    """sendPhoto rejects a caption over 1024 characters with a 400, which this system
    treats as a PERMANENT failure -- so an over-long caption would silently downgrade
    every alert to text and throw the picture away. All five reasons at once is the worst
    case and it is nowhere near the limit; this test is here so it stays that way."""
    from qorgan.detection.validation import REASON_EVIDENCE

    caption = _caption(settings, reasons=tuple(REASON_EVIDENCE), summary="Х" * 200)

    assert len(caption) <= 1024
