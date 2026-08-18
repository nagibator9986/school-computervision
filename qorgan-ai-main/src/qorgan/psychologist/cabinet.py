"""The cabinet index: what has been handed over, and what is and is not accumulating.

**Every block states its own state, and the states are computed from rows rather than
asserted in prose.** «Сигнал по столовой активен» is a claim, and a claim on a dashboard
that nobody recomputes is how the previous system stayed convincing for months while
producing nothing. So `canteen_block()` counts attributed meal sessions and says EMPTY
when there are none; it does not read a flag somebody set at install time.

**No block ranks anything and no block sums across blocks.** Referrals, meals and lessons
measure different things by different means; a single "signals" total would be a number
whose parts cannot be added, which is the mistake `DayReport` documents for
`unknown_sessions` and `forced_unknown` one layer down.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select

from qorgan.db.engine import session_scope
from qorgan.db.models import CanteenSession, Lesson
from qorgan.db.tenancy import resolve_school_id, scope
from qorgan.psychologist.referrals import Referral, referral_count, referred_incidents
from qorgan.psychologist.signals import Block, SignalState

# **The caption under the empty canteen block, and the owner's decision it carries.**
#
# The block is SHOWN while it is empty rather than hidden, so that the school can see the
# mechanism exists and read what it is waiting for. Hiding it would be the friendlier page
# and the dishonest one: it makes "we never built this" and "this has nothing to say yet"
# look identical, which is the exact failure the system this one replaces died of.
#
# It is ONE NAMED STRING because that decision is marked for review before the pilot and
# somebody will want to reword it. It is deliberately NOT a config key: R10 says a key must
# be read by the layer whose behaviour it controls, and a caption controls no behaviour --
# it would be a dead handle in a schema, which is worse than a line of text somebody edits.
# `tests/test_psychologist_cabinet.py` asserts the block and this sentence are on the page.
CANTEEN_IS_WAITING_FOR_THE_CAMERA = (
    "Ни одна запись пока не привязана к ученику: камеру столовой ещё не перевесили, "
    "поэтому узнать почти никого не удаётся. Таблица и накопление уже работают — чтобы "
    "в день пилота счёт не начинался с нуля."
)


@dataclass(frozen=True, slots=True)
class Cabinet:
    referrals: tuple[Referral, ...]
    blocks: tuple[Block, ...]


def cabinet_view(*, limit: int | None = None, school_id: int | None = None) -> Cabinet:
    """The whole index, in one transaction for the counts and one for the referral rows.

    **Every number here is one school's.** The three blocks answer "is this signal
    accumulating for us yet", and an installation-wide count would answer it with another
    school's data -- a psychologist told the canteen signal is alive when their own school
    has recognised nobody. The referral list is scoped for the stronger reason: those rows
    name children.
    """
    rows = (
        referred_incidents(limit, school_id=school_id)
        if limit is not None
        else referred_incidents(school_id=school_id)
    )

    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        referrals = referral_count(session, school)
        attributed = _attributed_sessions(session, school)
        lessons = _lessons(session, school)

    return Cabinet(
        referrals=rows,
        blocks=(
            _referral_block(referrals),
            _canteen_block(attributed),
            _classroom_block(lessons),
        ),
    )


def _referral_block(count: int) -> Block:
    """LIVE from the day this shipped: the mechanism is a person and a button.

    It needs no camera, no model and no threshold, which is exactly why it is the one part
    of §13 that could be finished rather than started.
    """
    lines = [
        "Сюда попадает то, что ЧЕЛОВЕК передал психологу, с его именем и временем. "
        "Система никогда не направляет ребёнка сама и не считает, кого стоит проверить: "
        "это обещано школе письменно (вопросы школе, §8).",
        "Инцидент остаётся в этом списке, даже если оператор потом отметил его "
        "подтверждённым или ложным: передача состоялась, и её нельзя отменить задним "
        "числом сменой статуса.",
    ]
    if not count:
        lines.append("Пока никто ничего не передавал. Кнопка есть на странице «События».")
    return Block(
        key="referrals",
        title="Переданные инциденты",
        state=SignalState.LIVE if count else SignalState.EMPTY,
        count=count,
        lines=tuple(lines),
    )


def _canteen_block(attributed: int) -> Block:
    """The one longitudinal signal with a real identity behind it -- and it is empty.

    EMPTY here is a statement about the CAMERA, not about the children, and the wording
    says so: the school must not read "0" as "nobody eats".
    """
    lines = [
        "Столовая узнаёт ребёнка по лицу у двери, по списку самой школы — поэтому "
        "«ходил обедать каждый день и перестал» здесь относится к КОНКРЕТНОМУ ученику, "
        "а не к анонимному треку. Это единственный продольный сигнал в системе, который "
        "честно про названного ребёнка.",
    ]
    if attributed:
        lines.append(
            f"Записей, привязанных к ученику: {attributed}. Открывается со страницы "
            "ученика: «Посещаемость по неделям»."
        )
    else:
        lines.append(CANTEEN_IS_WAITING_FOR_THE_CAMERA)
    return Block(
        key="canteen",
        title="Посещаемость столовой",
        state=SignalState.LIVE if attributed else SignalState.EMPTY,
        count=attributed,
        lines=tuple(lines),
    )


def _classroom_block(lessons: int) -> Block:
    """ANONYMOUS whatever the count, and that is not a state waiting to change.

    `lesson_tracks` has no `person_id` and may never gain one (`qorgan.classroom`), so no
    number of lessons turns these into a statement about a named child. Showing this as
    EMPTY would promise the school that waiting fixes it; showing it as LIVE would be a
    lie about what the numbers are.
    """
    return Block(
        key="classroom",
        title="Уроки",
        state=SignalState.ANONYMOUS,
        count=lessons,
        lines=(
            f"Записано уроков: {lessons}. Все показатели урока — по АНОНИМНОМУ ТРЕКУ, а "
            "не по ученику: система не узнаёт лица в классе и не будет. Это измерено, а "
            "не предположено — из 14 970 лиц с камер школы не узнано ни одно.",
            "Поэтому «сравнить ребёнка с его собственной нормой за 4 недели» (вопросы "
            "школе, §8) на классе НЕ РАБОТАЕТ и здесь не сделано: для этого надо узнавать "
            "личность, а тот же §8 это в классе запрещает. Решение — за школой; вопрос "
            "задан в docs/questions-for-school.md §10.",
        ),
    )


def _attributed_sessions(session, school_id: int) -> int:
    """Meal sessions that name a pupil. `person_id IS NULL` is an entry nobody was
    recognised at, which is the state almost every session is in today -- counting those
    would make the canteen signal look alive when it is not.

    Scoped through the ENTRY CAMERA, which is the route `db.tenancy` gives this table: a
    session belongs to the school whose canteen door opened it.
    """
    return int(
        session.scalar(
            scope(
                select(func.count(CanteenSession.id)), CanteenSession, school_id
            ).where(CanteenSession.person_id.is_not(None))
        )
        or 0
    )


def _lessons(session, school_id: int) -> int:
    return int(session.scalar(scope(select(func.count(Lesson.id)), Lesson, school_id)) or 0)
