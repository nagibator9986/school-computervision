"""Why a recorded fight raised no Telegram — the decision. Client §7's `_telegram_skip_reason`.

The school's first question about this system is *"почему мне не пришло уведомление о
драке?"*, and until this branch nothing anywhere could answer it. `/notifications` answers
the neighbouring question — an alert we TRIED to deliver and did not — because a delivery
attempt is a row. A decision not to try is not a row, and left no trace of any kind. We
said so to the school in writing (`docs/client-note-2026-07-17.md` §6).

That the reason survives to the database and onto the screen is
`test_telegram_skip_reason_recorded.py`. This file is about what is decided, and about
which reasons may exist at all.

**Three of §7's seven listed causes cannot happen in v2, and are deliberately absent from
`TelegramSkipReason`. A reason nothing can produce is a dead handle** — this project has
shipped one before (`display_fps`, which passed the dead-key check for its entire life).

  * **«обычная ходьба».** Ordinary walking is stopped by the ten suppression gates in
    `detection/gates.py`, which run in the FAST tier inside `BullyingDetector.process`. A
    suppressed pair never becomes a `Candidate`, so it never reaches the slow tier, never
    gets a `Verdict`, and never gets an event row. There is nothing to write a reason onto,
    and the suppression is already recorded — per frame, with the gate's name, in
    `DetectionResult.suppressions`. A `NORMAL_WALKING` member would be unreachable.

  * **«отсутствует видеоклип».** v2 never withholds an alert for want of media, on purpose:
    `NotificationWorker._send` sends the caption as text when there is no snapshot, and the
    clip is a best-effort extra afterwards. The recorded-side file holds that open with
    `test_missing_media_is_never_a_reason_to_withhold_an_alert`.

  * **«уровень ниже `alert`».** `should_notify` compares the confidence with
    `notify_threshold` and with nothing else; `alert_threshold` only picks a word for the
    summary. Both default to 0.85, and `Severity` never enters the notification path. A
    second reason keyed off a second number is exactly how the legacy came to hold 0.85 in
    the worker, 0.85 in the service and 0.90 in the config comments (§5.4).
    `test_the_reason_does_not_depend_on_the_severity_threshold` holds that open.

The other four §7 causes map onto members, and «нет skeleton-подтверждения» maps onto two:
`detection/validation.py` is built on the distinction between a pose model that could not
look and one that looked and disagreed, and only the first is a fault anyone can fix.
"""

from __future__ import annotations

import pytest

from qorgan.config.bullying import Confidence
from qorgan.detection.validation import SkeletonResult, judge
from qorgan.enums import TelegramSkipReason
from qorgan.web.routes.events import SKIP_REASON_LABELS
from tests.telegram_skip_support import (
    CONFIG,
    CONFIRMED,
    FROM_A_JUDGEMENT,
    PROBABILITY,
)


@pytest.mark.parametrize("expected", list(FROM_A_JUDGEMENT), ids=lambda r: r.value)
def test_each_reason_is_produced_by_a_real_judgement(expected: TelegramSkipReason) -> None:
    """Driven through `judge`, not asserted against a hand-written mapping.

    The point is reachability. A reason no judgement can produce is a control on a
    dashboard wired to nothing, and it reads as an explanation while explaining nothing.
    """
    probability, skeleton = FROM_A_JUDGEMENT[expected]
    verdict = judge(probability, skeleton.score, skeleton, CONFIG)

    assert verdict.telegram_skip_reason is expected
    assert not verdict.should_notify(CONFIG), "this fixture no longer withholds anything"


def test_no_reason_exists_that_nothing_can_produce() -> None:
    """The guard against the next dead handle.

    Adding a member to `TelegramSkipReason` fails here until somebody shows a judgement
    that reaches it — which is the whole difference between this enum and the client's
    seven-item list. §7 describes the LEGACY system; three of its causes are structurally
    unreachable in v2 (see the module docstring), and inventing branches for them would put
    three explanations in the schema that no event can ever carry.
    """
    covered = set(FROM_A_JUDGEMENT) | {TelegramSkipReason.ALREADY_NOTIFIED}
    assert covered == set(TelegramSkipReason), (
        "every skip reason must be produced by the real decision path; unaccounted for: "
        f"{set(TelegramSkipReason) - covered}"
    )


@pytest.mark.parametrize("case", [*FROM_A_JUDGEMENT.values(), (PROBABILITY, CONFIRMED)])
def test_the_recorded_reason_and_the_notify_decision_are_one_answer(
    case: tuple[float, SkeletonResult],
) -> None:
    """`telegram_skip_reason is None` **iff** the alert goes out. Nothing else may be true.

    The worker reads the recorded reason instead of asking `should_notify` a second time,
    so this equivalence is what makes that safe. If the two could disagree, an event would
    be sent while its row explained why it was not — or worse, the row would say nothing
    while nobody had been told.
    """
    probability, skeleton = case
    verdict = judge(probability, skeleton.score, skeleton, CONFIG)

    assert (verdict.telegram_skip_reason is None) is verdict.should_notify(CONFIG)


def test_a_convincing_judgement_carries_no_reason_at_all() -> None:
    """The other direction of the contract, stated on its own because it is the one that
    breaks silently: a reason left on an event that WAS sent contradicts the message
    already in a teacher's hand."""
    verdict = judge(PROBABILITY, CONFIRMED.score, CONFIRMED, CONFIG)

    assert verdict.should_notify(CONFIG), "the fixture must actually raise an alert"
    assert verdict.telegram_skip_reason is None


def test_the_reason_does_not_depend_on_the_severity_threshold() -> None:
    """§7's «уровень ниже `alert`» is not a separate decision in v2, and must not become one.

    `alert_threshold` chooses the word at the top of the message; `notify_threshold` decides
    whether there is a message at all. Moving the first must not change this answer, or the
    system has grown the legacy's second copy of its own threshold.
    """
    probability, skeleton = FROM_A_JUDGEMENT[TelegramSkipReason.LOW_CONFIDENCE]
    lenient = Confidence(alert_threshold=0.1, critical_threshold=0.95)

    assert lenient.notify_threshold == CONFIG.notify_threshold, "only severity may differ here"
    assert (
        judge(probability, skeleton.score, skeleton, lenient).telegram_skip_reason
        is judge(probability, skeleton.score, skeleton, CONFIG).telegram_skip_reason
    )


def test_every_reason_has_words_a_teacher_can_read() -> None:
    """A token in the schema is not an answer to a person. Missing labels degrade to the
    token rather than raising — deliberately, since a KeyError on the page whose job is to
    explain a silence would replace the explanation with a 500 — so this is the only thing
    that catches a forgotten one."""
    assert set(SKIP_REASON_LABELS) == set(TelegramSkipReason)
