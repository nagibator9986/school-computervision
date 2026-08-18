"""Logging in the way a browser does, for tests.

Every mutating request now needs this session's CSRF token (`qorgan.web.csrf`), and that
includes `POST /login` -- login CSRF is a real attack, so the login form is not an
exception. A test that posts without a token is not "a test with an annoying extra step",
it is a test simulating something a browser never does.

Centralised so the next test to be written gets it right by default. Sixteen call sites
each doing their own two-step would be sixteen chances to write the one that skips it.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

FIELD = "csrf_token"


def csrf_token(client: TestClient) -> str:
    """This session's token, read off a page the server rendered -- where a browser finds it.

    Works logged OUT and logged IN, and that is not incidental: an authenticated client
    asking for `/login` is REDIRECTED to its landing page (a login screen shown to someone
    already logged in is how a redirect loop starts), so there is no form to read. Every
    rendered page carries the token in a `<meta>` tag for exactly this reason, so the
    fallback reads a real page instead of asserting that the login form was there.
    """
    for path in ("/login", "/", "/canteen"):
        page = client.get(path, follow_redirects=True)
        if page.status_code != 200:
            continue
        found = re.search(rf'name="{FIELD}"\s+value="([^"]+)"', page.text) or re.search(
            r'name="csrf-token"\s+content="([^"]+)"', page.text
        )
        if found:
            return found.group(1)
    raise AssertionError("no CSRF token on any reachable page; nothing could post safely")


def token_for(client: TestClient, path: str = "/login") -> str:
    """The token as carried by an arbitrary rendered page (an authenticated one, say)."""
    page = client.get(path)
    found = re.search(rf'name="{FIELD}"\s+value="([^"]+)"', page.text) or re.search(
        r'name="csrf-token"\s+content="([^"]+)"', page.text
    )
    assert found, f"no CSRF token on {path}"
    return found.group(1)


def with_token(client: TestClient, data: dict | None = None) -> dict:
    """`data` plus this session's token."""
    return {**(data or {}), FIELD: csrf_token(client)}
