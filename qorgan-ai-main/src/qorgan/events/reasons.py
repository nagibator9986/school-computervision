"""The skeleton's reasons, on their way to and from the database.

`verdict.reasons` is the answer to the only question a teacher actually has when their
phone buzzes: *why*. "Someone went down" and "someone waved their arms" are the same
severity and the same confidence, and they are not the same message. Until this module
existed the reasons went to the log line and stopped there — the events table had no
column for them, so the alert could say how sure it was and never what it saw.

**One column, not a join table.** The reasons are a closed set of five short slugs
(`qorgan.detection.validation.REASON_EVIDENCE`) attached to exactly one event and never
queried on their own. A `reasons` table would be three joins to answer "why", and the
gain — searching events by reason — is not a thing anyone has asked for. If it is ever
asked for, `unpack_reasons` is the only place that knows the format.

**The separator is enforced, not assumed.** A comma inside a reason would split one
reason into two on the way back out, and the corruption would be invisible: the message
would simply name evidence that was never found. `pack_reasons` refuses instead.
"""

from __future__ import annotations

SEPARATOR = ","


def pack_reasons(reasons: tuple[str, ...]) -> str:
    """Reasons -> the column value. Order is preserved: it is the pose model's."""
    for reason in reasons:
        if SEPARATOR in reason:
            # Never silently: a reason carrying the separator would come back out as two
            # reasons, and the alert would name evidence nobody found.
            raise ValueError(
                f"a skeleton reason may not contain {SEPARATOR!r}: {reason!r}. "
                "It is the separator this column is stored with."
            )
    return SEPARATOR.join(reasons)


def unpack_reasons(packed: str | None) -> tuple[str, ...]:
    """The column value -> reasons. Total: every row is readable, including the old ones.

    NULL and "" both mean "the skeleton had nothing to say", which is the normal state of
    every event written before this column existed and of every event the skeleton could
    not look at. Neither is an error, and neither may be allowed to break an alert.
    """
    if not packed:
        return ()
    return tuple(part for part in packed.split(SEPARATOR) if part)
