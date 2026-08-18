"""eval/labels.csv: the schema, and what it refuses.

Split out of test_evaluation.py, which had grown past the 500-line limit. Labels are their
own concern: what a human is allowed to say, and what the loader must refuse to guess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.evaluation import load_labels
from qorgan.evaluation.labels import LabelError, LabelKind, write_template


def _labels_file(
    tmp_path: Path, rows: str, header: str = "video_id,t_start,t_end,label\n"
) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(header + rows, encoding="utf-8")
    return path


def test_labels_load(tmp_path: Path) -> None:
    labels = load_labels(_labels_file(tmp_path, "hall.mp4,12.5,18.0,bullying\n"))

    assert len(labels) == 1
    assert labels.videos == ("hall.mp4",)
    assert labels.positives("hall.mp4")[0].kind is LabelKind.BULLYING


def test_a_video_with_no_label_is_an_error_not_an_assumption(tmp_path: Path) -> None:
    """The legacy derived ground truth from Russian words in the FILENAME and silently
    labelled anything unmatched as `normal` -- so a fight nobody remembered to annotate
    counted as a correct rejection when the detector missed it. The benchmark rewarded
    blindness."""
    path = tmp_path / "labels.csv"
    path.write_text("video_id,t_start,t_end,label\n", encoding="utf-8")

    with pytest.raises(LabelError, match="no labels"):
        load_labels(path)


def test_an_unknown_label_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="unknown label"):
        load_labels(_labels_file(tmp_path, "hall.mp4,1,2,драка\n"))


def test_a_backwards_interval_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="must be after"):
        load_labels(_labels_file(tmp_path, "hall.mp4,18.0,12.0,bullying\n"))


def test_a_missing_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("video_id,t_start,label\nhall.mp4,1,bullying\n", encoding="utf-8")

    with pytest.raises(LabelError, match="t_end"):
        load_labels(path)


def test_the_template_it_writes_is_one_it_can_read(tmp_path: Path) -> None:
    path = tmp_path / "template.csv"
    write_template(path)
    assert len(load_labels(path)) == 2


def test_the_template_documents_the_optional_camera_column(tmp_path: Path) -> None:
    path = tmp_path / "template.csv"
    write_template(path)
    text = path.read_text(encoding="utf-8")

    assert "camera" in text.splitlines()[1]  # the header row, after the comment
    assert text.lstrip().startswith("#"), "no comment explaining when to fill it in"


def test_an_explicit_camera_column_is_optional_and_backward_compatible(tmp_path: Path) -> None:
    """The schema stays `video_id,t_start,t_end,label` for everyone who never asks for
    a camera column -- a missing column means the same thing as an empty cell."""
    labels = load_labels(_labels_file(tmp_path, "hall.mp4,12.5,18.0,bullying\n"))
    assert labels.camera_for("hall.mp4") is None


def test_an_explicit_camera_value_is_read_from_its_own_column(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "video_id,t_start,t_end,label,camera\n"
        "human_named.mp4,1.0,2.0,bullying,hall_left\n",
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert labels.camera_for("human_named.mp4") == "hall_left"


def test_an_empty_camera_cell_means_infer_from_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "video_id,t_start,t_end,label,camera\nhall.mp4,1.0,2.0,bullying,\n",
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert labels.camera_for("hall.mp4") is None




def test_a_pending_row_carries_its_camera_and_asserts_no_fight(tmp_path: Path) -> None:
    """`pending` exists so a clip stays scannable while its timestamp is unknown -- the only
    confirmed fight was written as `ignore` for want of it. It must carry the camera, and it
    is NOT a fight, so it is no positive. A separate kind, never a re-used one."""
    path = tmp_path / "labels.csv"
    path.write_text(
        "video_id,t_start,t_end,label,camera\nfight.mp4,0.0,17.6,pending,hall_right\n",
        encoding="utf-8",
    )
    labels = load_labels(path)

    assert labels.camera_for("fight.mp4") == "hall_right"
    assert labels.positives("fight.mp4") == ()
    assert labels.pending("fight.mp4")[0].kind is LabelKind.PENDING


def test_a_clip_labelled_with_two_different_cameras_is_a_contradiction(tmp_path: Path) -> None:
    """A clip comes from exactly one camera. Two answers is a human contradicting himself.

    Taking the first row and moving on would pick a winner SILENTLY -- and the entire reason
    this column exists is that scoring footage against the wrong camera's config blanks a
    region of the frame with no error and a perfectly plausible result. hall_left carries a
    mirror_ignore zone over a reflective column that hall_right cannot see.
    """
    path = _labels_file(
        tmp_path,
        "clip.mp4,1.0,2.0,bullying,hall_left\nclip.mp4,5.0,6.0,normal,hall_right\n",
        header="video_id,t_start,t_end,label,camera\n",
    )
    labels = load_labels(path)

    with pytest.raises(LabelError, match="more than one camera"):
        labels.camera_for("clip.mp4")
