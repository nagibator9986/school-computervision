"""What a password must be, and the one way it becomes a stored value.

**Why this is not in `qorgan.web.security`, where it used to live.** Three things need it
now -- the login check, `qorgan user add` at a shell, and `qorgan.accounts` behind the
/users page -- and only one of them is the web. `qorgan/web/__init__.py` builds the whole
FastAPI app on import, so a domain module reaching into the web package for `hash_password`
was a genuine import cycle (accounts -> web.security -> web -> app -> routes.users ->
accounts), not merely untidy. The CLI had been reaching across that line since day one.

**bcrypt directly, not passlib**: passlib has been unmaintained since 2020 and its bcrypt
backend breaks outright against bcrypt >= 4.1. This module is the whole adapter.

**The two bounds are counted in different units, deliberately.** The maximum is a fact
about bcrypt, which reads 72 BYTES and silently discards the rest; the minimum is a fact
about guessing, and what a guesser guesses is CHARACTERS. Getting either unit wrong is the
same mistake pointed in a different direction, and one of those directions lets a weak
password in. See `hash_password`.

Both rules live at the point where a password turns into something storable, rather than in
whichever front door happens to be asking, because a rule you can bypass by calling the
layer underneath it is a rule for whoever remembers it.
"""

from __future__ import annotations

import bcrypt

# bcrypt silently truncates the input at 72 bytes, so a long passphrase would have its tail
# ignored. Reject rather than truncate: a password that is quietly not the password you
# typed is worse than one that is refused.
BCRYPT_MAX_BYTES = 72

# One minimum, not one per front door. `qorgan user add` and the accounts page are two doors
# into the same table; a minimum restated in each is two numbers nobody ever sees side by
# side, and the day they drift the laxer door silently becomes how weak accounts get made.
MIN_PASSWORD_LENGTH = 10


class PasswordRejected(ValueError):
    """Refused, and never quoting what it refused.

    These messages are rendered into a page and written to a log. One that echoed the value
    would put the password in both, which is the whole thing being avoided.
    """


class PasswordTooLong(PasswordRejected):
    pass


class PasswordTooShort(PasswordRejected):
    pass


def hash_password(raw: str) -> str:
    """The only way a password becomes a stored value, and therefore the only place the
    rules have to be written. A check in a route is a check the CLI does not have."""
    encoded = raw.encode("utf-8")

    # THE MAXIMUM IS BYTES, because that is what bcrypt reads. Cyrillic is two bytes a
    # character in UTF-8, so a 40-letter Russian passphrase is 80 bytes -- counted in
    # characters it would pass here and be truncated one layer down, and the 72-byte prefix
    # would then be a working password for that account that nobody was ever told about.
    if len(encoded) > BCRYPT_MAX_BYTES:
        raise PasswordTooLong(f"password must be at most {BCRYPT_MAX_BYTES} bytes")

    # THE MINIMUM IS CHARACTERS, because that is what a guesser guesses. In bytes, a
    # five-letter Russian word is ten and would clear a bar of ten while being half of it:
    # the same confusion of units, pointed the other way, and this one lets a weak password
    # IN. `qorgan user add` has always measured `len(password)`; this is that rule moved,
    # not a new one.
    if len(raw) < MIN_PASSWORD_LENGTH:
        raise PasswordTooShort(f"password must be at least {MIN_PASSWORD_LENGTH} characters")

    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def verify_password(raw: str, hashed: str) -> bool:
    """Does this password match that hash?

    Deliberately NOT the place the policy above is applied. A minimum that was raised, or a
    rule added, must never lock out somebody whose existing password predates it -- the
    question here is "is this the password", not "would we accept it today".
    """
    encoded = raw.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode("ascii"))
    except ValueError:
        return False  # a malformed hash in the row is a failed login, not a crash
