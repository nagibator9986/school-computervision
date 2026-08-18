"""The cabinet wears the platform's stylesheet, and this is the check that it still does.

`DESIGN.md` is a contract with one clause that no reviewer can hold in their head: the
cabinet's design is **copied from `qorgan-ai-main/src/qorgan/web/static/app.css`, token for
token**. `cabinet/skin.py` holds that copy, because `classvision` does not depend on `qorgan`
and must not start (see that module's docstring for why it is inlined rather than linked).

A copy drifts. The platform is a sibling checkout rather than a dependency, so the drift
cannot be caught by a version pin and would show up as the cabinet slowly stopping looking
like the product it is shown next to — the exact failure the contract was written against.
This test compares the two files whenever the platform tree is present beside this one, and
skips when it is not: `classvision` has to remain installable and testable on its own.

It deliberately asserts BYTE equality of the vendored half. `PLATFORM_CSS` is not a place to
fix a colour: everything this surface adds lives in `CABINET_CSS` after it, so the diff is
always empty or always a real drift, never a judgement call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from classvision.cabinet import skin, store


@pytest.fixture()
def db(tmp_path):
    connection = store.connect(tmp_path / "cabinet.sqlite3")
    yield connection
    connection.close()

# Where the platform lives when both trees are checked out side by side. `parents[2]` is the
# directory holding `classvision/` and `qorgan-ai-main/`.
PLATFORM_CSS_PATH = (Path(__file__).resolve().parents[2]
                     / "qorgan-ai-main/src/qorgan/web/static/app.css")

# The line `skin.py` appends to the copy so that the vendored half has an unambiguous end.
END_MARKER = "/* --- end of the verbatim copy of the platform stylesheet --- */"


def test_the_vendored_stylesheet_is_still_byte_for_byte_the_platforms():
    if not PLATFORM_CSS_PATH.exists():
        pytest.skip(f"the platform tree is not checked out at {PLATFORM_CSS_PATH}")
    assert skin.PLATFORM_CSS.endswith(END_MARKER + "\n"), (
        "skin.PLATFORM_CSS no longer ends with its own end marker, so this test cannot tell "
        "the vendored copy from the cabinet's own additions. Restore the marker.")
    vendored = skin.PLATFORM_CSS[: -len(END_MARKER) - 1]
    assert vendored == PLATFORM_CSS_PATH.read_text(encoding="utf-8"), (
        "cabinet/skin.py::PLATFORM_CSS has drifted from the platform's app.css. DESIGN.md "
        "makes that file the contract: copy it over verbatim again, and put whatever the "
        "cabinet needs into CABINET_CSS instead of editing the copy.")


def test_the_masthead_offers_no_link_this_export_cannot_serve():
    """A static folder has no routes. A nav item pointing at one opens nothing.

    `templates/base.html` draws every item from the capability its route is gated on, so
    that nobody is ever handed a 403; the same rule with no session at all means the nav may
    name only files that this build writes next to the pages.
    """
    html = skin.masthead(active="index", section=("class-d14.html", "8-А · D14"))
    for route in ("/events", "/canteen", "/psychologist", "/logs", "/pupils", "/lessons",
                  "/cameras", "/settings", "/users", "/schools", "/logout"):
        assert f'href="{route}"' not in html, f"the export cannot serve {route}"
    assert 'href="index.html"' in html
    assert 'href="lessons.html"' in html
    assert 'href="class-d14.html"' in html
    # ...and it says what it is instead of naming a user who is not logged in.
    assert "статическая выгрузка" in html
    assert "Qorgan AI" in html


def test_the_class_item_appears_only_on_pages_that_belong_to_a_class():
    assert "class-" not in skin.masthead(active="lessons")


def test_the_empty_cabinet_does_not_link_the_page_it_did_not_write(db, tmp_path):
    """`render` writes `index.html` alone when nothing has been imported.

    So on that page the «Что сравнимо между уроками» item would point at a file that is not
    in the folder. There is no server here to answer with a 404: the link just does nothing,
    and a masthead item that does nothing reads as a broken product rather than an empty one.
    """
    from classvision.cabinet import report

    result = report.render(db, tmp_path / "html")
    assert result["state"] == "empty"
    assert [Path(p).name for p in result["pages"]] == ["index.html"]
    page = Path(result["index"]).read_text(encoding="utf-8")
    assert 'href="lessons.html"' not in page
    assert 'class="topbar"' in page          # ...and it still wears the masthead
