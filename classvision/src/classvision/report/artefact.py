"""The analysis artefact: one versioned JSON document per processed recording.

**This file is the contract.** The web project imports this and nothing else — not the
models, not torch, not OpenCV. That is the rule that keeps a 2 GB CUDA-shaped dependency
out of a school's web process, and it only holds if everything the dashboard needs is
here, in plain types, with units.

**A re-run is a NEW document, never an overwrite.** `run_id` is derived from the video's
content hash *and* every setting that could change a number — model weights, imgsz,
sampling rate, every threshold, the room layout. Two runs of the same lesson under
different thresholds are two different measurements of it, and letting the second quietly
replace the first is how a term's trend acquires a step change that nobody can explain.
The importer keys on `run_id` and is therefore idempotent for free.

**Provenance is not metadata, it is part of the result.** «Ребёнок стал менее активен»
is only meaningful if both weeks were measured the same way. So the thresholds actually
used are written into the document rather than referenced, the model version is recorded,
and `clock_source` says whether the date was read off the picture or typed in by a human.

**Every total is accompanied by what we could not see.** `coverage`, `absent_observations`,
`unreadable_observations` and `hand_unmeasurable_observations` travel with the counts they
qualify, and `caveats` carries the sentences that must appear on any surface rendering
them. A number without them is not a smaller truth, it is a different and false one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 1.1 renamed the ambiguous per-state `seconds` into `observed_seconds_by_state` +
# `episode_seconds`, and put the denominator into the teacher's share names. A consumer
# written against 1.0 that reads `seconds` now gets a KeyError instead of the wrong
# number, which is the entire point of bumping rather than aliasing.
SCHEMA_VERSION = "classvision/1.1"

# The sentences that must appear beside these numbers wherever they are shown. They live
# here, in one place, for the reason the neighbouring codebase learned the hard way: four
# hand-written copies of a caveat is four chances for one of them to quietly stop being
# true. The importer copies them into the web project; the web project does not retype
# them.
CAVEATS_RU: tuple[str, ...] = (
    "Система не ставит диагнозов и не даёт рекомендаций. Это счётчики наблюдаемых "
    "фактов, которые решает интерпретировать человек.",
    "Единица учёта — МЕСТО в классе, а не ребёнок. Если двое поменялись местами, их "
    "истории поменялись вместе с ними, и это может заметить только человек с планом "
    "рассадки.",
    "«Вовлечённость» здесь не измеряется и не выводится. Показатель активности — это "
    "сумма наблюдаемых признаков, и все его составляющие показаны отдельно.",
    "Ни один порог в этом отчёте не проверен на размеченном вручную уроке. Значения "
    "получены из геометрии этой записи и подлежат уточнению.",
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything that could change a number, recorded so two lessons can be compared."""

    schema: str
    video_path: str
    video_sha256: str
    video_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    # Wall clock. `clock_source` is never inferred from the value -- see `video/clock.py`.
    started_at: str | None
    clock_source: str
    clock_drift_seconds: float | None

    analysed_at: str
    sample_fps: float
    sample_interval_seconds: float
    analysed_frames: int
    window_start_seconds: float
    window_end_seconds: float | None

    model: dict[str, Any]
    thresholds: dict[str, Any]
    room: dict[str, Any]
    software: dict[str, Any]

    # How this artefact was ASSEMBLED, when it was assembled from more than one recording.
    #
    # `null` on an ordinary run and non-null on a session, so "was this one file?" is a
    # field rather than an inference. When it is present, `video_sha256` is the digest of
    # the assembly and each constituent file's own hash lives in `session.parts[].sha256`,
    # `video_path` names the FIRST part only, and `duration_seconds` is the span of the
    # joined timeline. That is stated here because a consumer that reads those three fields
    # without reading this one would describe a 62-minute lesson as a 13-minute file.
    #
    # Additive, so `classvision/1.1` consumers keep working: nothing was renamed, nothing
    # was removed, and a reader that ignores this key gets exactly the document it got
    # before -- for single-file runs, byte for byte.
    session: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        """Content hash of the video plus every setting that could change a number.

        For a session, `video_sha256` is the assembly digest — it covers every part's hash
        AND where each was placed — so the same files joined under a different tolerance are
        a different run rather than an overwrite of the first.
        """
        payload = json.dumps({
            "schema": self.schema, "video": self.video_sha256,
            "sample_fps": self.sample_fps, "window": [self.window_start_seconds,
                                                      self.window_end_seconds],
            "model": self.model, "thresholds": self.thresholds, "room": self.room,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SeatRecord:
    """One place, for one lesson. `pupil` is None until identity clears its own bar."""

    seat_id: int
    label: str
    centre: tuple[float, float]
    scale_px: float
    occupancy: float
    role: str                      # "pupil" | "adult" | "excluded"
    pupil: dict[str, Any] | None   # external_id, full_name, method, confidence -- or None
    ledger: dict[str, Any]
    metrics: dict[str, Any]
    timeline: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TeacherRecord:
    """The adult, measured with the same honesty and none of the judgement.

    Deliberately a description of position over time. There is no "teaching quality"
    field and there is nowhere to put one: «сидит» is not «не ведёт урок», and a camera
    cannot tell the difference.
    """

    seat_id: int | None
    identification: dict[str, Any]
    ledger: dict[str, Any]
    metrics: dict[str, Any]
    # Where the adult's place is, in the same terms as a `SeatRecord`. Present because the
    # verification overlay draws every place, and without it the adult's circle was drawn
    # at the frame origin -- a visible-in-hindsight bug that a reader of the JSON alone
    # would never have found. Anything that consumes seats has to be able to consume this
    # one too, or the adult is a hole in every downstream view.
    centre: tuple[float, float] | None = None
    scale_px: float | None = None
    # WHERE IN THE ROOM he was, span by span: the `TeacherState` episodes from
    # `metrics/teacher.py`. This is the strip the teacher section draws, because the
    # question this half answers is «где он был».
    timeline: list[dict[str, Any]] = field(default_factory=list)
    # WHAT HIS BODY WAS DOING at a settled place, span by span: the `PupilState` episodes
    # from his `SeatLedger`, present only when he had a seat at all. A separate field and
    # not merged into `timeline`, because the two use different enums for different
    # questions and a consumer that concatenated them would produce a strip in which
    # «сидит» and «у доски» look like alternatives to each other. Empty is a real value
    # here and means "he never settled anywhere long enough to have a place".
    pose_timeline: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """What the run could not see, at the level of the whole lesson."""

    analysed_frames: int
    frames_with_no_person: int
    observations_total: int
    observations_unassigned: int      # a person at no discovered seat
    observations_unreadable: int
    seats_never_settled: int
    rejected_clusters: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Artefact:
    """One lesson, analysed. The only thing the web project ever reads."""

    provenance: Provenance
    seats: list[SeatRecord]
    teacher: TeacherRecord | None
    uncertainty: Uncertainty
    lesson: dict[str, Any]
    caveats: tuple[str, ...] = CAVEATS_RU

    @property
    def run_id(self) -> str:
        return self.provenance.run_id

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_id"] = self.run_id
        return data

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
        return path


def sha256_of(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Content hash of the recording. Read in chunks because these files are ~1 GB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    """What was installed when this ran. A model's output changes with its library."""
    versions: dict[str, str] = {}
    for name in ("ultralytics", "torch", "cv2", "numpy", "insightface", "onnxruntime"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "absent"
    versions["python"] = ".".join(str(v) for v in __import__("sys").version_info[:3])
    return versions


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
