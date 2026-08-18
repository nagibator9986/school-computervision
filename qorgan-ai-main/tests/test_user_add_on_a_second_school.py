"""`qorgan user add` on an installation that serves more than one school.

**This is the configuration the whole branch exists to enable, and the day-one command
crashed on it.** Measured before the fix, on a database holding two schools:

    qorgan user add someone --role admin      -> UNCAUGHT UndecidedSchool (a traceback)
    qorgan user add someone --role operator   -> UNCAUGHT UndecidedSchool (a traceback)
    qorgan user add someone --role superadmin -> rc 0

`db.models.school.UndecidedSchool` is a `RuntimeError`, and `_cmd_user_add` caught
`AccountError` and `PasswordRejected` -- so the refusal that exists precisely to stop a row
being filed under a guessed school reached the installer as a stack trace instead of a
sentence. `resolve_school_id` was right to refuse; nothing was there to say so in words.

Two halves, and this file asks for both:

  * **the refusal is readable and says what to type** -- naming the schools, because "name
    a school" is not actionable to somebody who did not create them; and
  * **there is something to type**: `--school <slug>`. That flag is not invented here.
    `db/models/school.py` named it on `School.slug` before it was built, and
    `schools.rename_school` refuses to edit a slug *because* this flag carries it. It was
    declared in two docstrings and implemented in none -- the same shape as the
    `--superadmin` flag that opened this whole task.

The single-school half is asserted too, in `test_the_superadmin_can_be_created.py` and by
the rest of the suite: `--school` is optional while there is only one school, because
`resolve_school_id` falls back to the only one there is. It becomes obligatory exactly when
a default would have to guess.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan import cli
from qorgan.db.models import School, User
from qorgan.enums import UserRole
from qorgan.settings import Settings

PASSWORD = "correct-horse-battery"
OTHER_SLUG = "gymnasium-4"


@pytest.fixture
def two_schools(settings: Settings, session: Session, monkeypatch: pytest.MonkeyPatch) -> Session:
    """The installation the moment a superadmin adds the second school."""
    del settings  # applied by the fixture
    session.add(School(slug=OTHER_SLUG, name="Гимназия №4"))
    session.commit()
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: PASSWORD)
    return session


@pytest.mark.parametrize("role", ["admin", "operator", "developer", "canteen_staff"])
def test_it_refuses_in_words_instead_of_crashing(
    two_schools: Session, capsys: pytest.CaptureFixture[str], role: str
) -> None:
    """The crash itself. **`rc == 1`, not an exception**, and the message has to be usable.

    Every school-bound role is asked, not just `admin`: the fault was in
    `resolve_school_id`, which every one of them goes through, so testing one would have
    proved something about `admin` rather than about the command.
    """
    code = cli.main(["user", "add", f"new_{role}", "--role", role])

    assert code == 1, (
        f"`--role {role}` on a two-school installation did not exit 1. It used to raise "
        "UndecidedSchool, a RuntimeError nothing caught, and the installer got a traceback."
    )
    refusal = capsys.readouterr().err
    assert OTHER_SLUG in refusal and "default" in refusal, (
        f"the refusal does not name the schools to choose between:\n{refusal}\n"
        "Somebody at a shell on a machine whose schools they did not create cannot act on "
        "'name a school' without being told what the names are."
    )
    assert "--school" in refusal, (
        f"the refusal does not say HOW to name one:\n{refusal}\n"
        "An error that states a rule without stating the remedy is the shape of "
        "unhelpfulness this project keeps paying for."
    )


def test_nothing_is_written_when_it_refuses(two_schools: Session) -> None:
    """A refused command must leave no half-made account behind."""
    cli.main(["user", "add", "ghost", "--role", "admin"])

    two_schools.expire_all()
    assert two_schools.scalar(select(User).where(User.username == "ghost")) is None, (
        "the command refused and still wrote a row"
    )


def test_naming_the_school_by_slug_works(two_schools: Session) -> None:
    """The other half: there is something to type, and typing it succeeds.

    The account must land in the school that was NAMED, not in whichever one the fallback
    would have picked -- so the assertion is on the id of `gymnasium-4` specifically.
    """
    wanted = two_schools.scalar(select(School.id).where(School.slug == OTHER_SLUG))

    code = cli.main(["user", "add", "head4", "--role", "admin", "--school", OTHER_SLUG])

    assert code == 0, f"`--school {OTHER_SLUG}` did not create the account"
    two_schools.expire_all()
    created = two_schools.scalar(select(User).where(User.username == "head4"))
    assert created is not None
    assert created.school_id == wanted, (
        f"the account went to school {created.school_id} and {OTHER_SLUG} is {wanted}. A "
        "command that accepts a school and then files the row somewhere else is worse "
        "than one that refuses."
    )


def test_an_unknown_slug_is_refused_and_lists_the_real_ones(
    two_schools: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo must not become a new school, and must not become a traceback either."""
    code = cli.main(["user", "add", "nobody", "--role", "admin", "--school", "gymnasium-5"])

    assert code == 1, "an unknown school slug was accepted"
    refusal = capsys.readouterr().err
    assert OTHER_SLUG in refusal, f"the refusal does not list the real slugs:\n{refusal}"
    two_schools.expire_all()
    assert two_schools.scalar(select(User).where(User.username == "nobody")) is None


def test_the_superadmin_still_needs_no_school(two_schools: Session) -> None:
    """The role that belongs to no school is unaffected by there being two of them.

    This is the one branch that never resolves a school, so it was the one role that did
    NOT crash before the fix -- and it must go on working, without `--school`, on the
    installation where every other role now requires it.
    """
    assert cli.main(["user", "add", "root", "--role", "superadmin"]) == 0

    two_schools.expire_all()
    created = two_schools.scalar(select(User).where(User.username == "root"))
    assert created is not None
    assert created.role is UserRole.SUPERADMIN
    assert created.school_id is None, "the superadmin was given one of the two schools"


def test_school_is_refused_for_the_superadmin_before_the_password_prompt(
    two_schools: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--school` with `--role superadmin` is a contradiction, and it is caught early.

    **Before the prompt**, which is the point of the test: there is nothing to type twice
    for a command that cannot succeed, and a refusal arriving after two password prompts
    reads as the password having been wrong. `getpass` is replaced with something that
    fails the test if it is called at all.
    """

    def _must_not_be_asked(_prompt: str) -> str:
        raise AssertionError("the password was prompted for before the arguments were checked")

    monkeypatch.setattr(cli.getpass, "getpass", _must_not_be_asked)

    code = cli.main(["user", "add", "root", "--role", "superadmin", "--school", OTHER_SLUG])

    assert code == 1, "--school was accepted for the role that belongs to no school"
    assert "--school" in capsys.readouterr().err
