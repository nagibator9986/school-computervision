"""Faces — corroboration only, aggregated over a whole lesson, never a source of names.

--------------------------------------------------------------------------------
**THE MEASUREMENT THIS MODULE IS BUILT AROUND SAYS IT DOES NOT WORK.**

`MEASUREMENTS.md` §4, on this camera and this roster:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| face height (px) | 30.8 | 53.5 | **64.0** | 73.7 | 95.2 |
| best cosine vs the 141-pupil gallery | 0.1 | 0.3 | **0.30** | 0.4 | 0.6 |
| margin over the runner-up | 0.0 | 0.0 | **0.10** | 0.2 | 0.3 |

ArcFace wants ≈0.4–0.5 to call two faces the same person. 0.30 with a 0.10 margin is a
preference, not a recognition — and all 141 roster photos embedded without a single
failure, so this is not a data problem that better photographs would fix. It is what a
64-pixel face from a ceiling camera is worth.

The module exists anyway, because that measurement has two known slacknesses and both are
addressable here rather than by hoping:

  1. **It was single-frame.** One observation against one gallery. A lesson gives this
     module *hundreds* of observations of the same seat. Averaging unit vectors suppresses
     per-frame pose and blur noise as roughly 1/√n, and a vote across hundreds of frames
     has a completely different failure mode than one frame's argmax. This is why every
     public entry point here is **seat-level** and there is deliberately no
     `identify_this_face()` — a per-frame answer is exactly the thing that measured 0.30.
  2. **It was 141 candidates wide.** The room holds nine. `roster.restrict_to(class)`
     turns a 141-way choice into a ~12-way one before this module sees it.

**And it still may not create a name.** Both fixes make the evidence better; neither makes
it sufficient. Everything below produces *evidence*, and `assign.py` is where it is allowed
to agree with a human's seating plan or to contradict it. There is no path from this file
to a name.

--------------------------------------------------------------------------------
**BOTH THE MEAN AND THE VOTE ARE REPORTED, BECAUSE THEY FAIL DIFFERENTLY.**

`AggregateMatch` carries a mean-embedding match *and* a vote distribution over per-frame
argmaxes, and `assign.py` requires them to agree. They are not two views of one number:

  * The **mean embedding** is the sensitive one. It uses the full geometry of every
    observation, so it recovers a signal too weak to win any single frame — but a handful
    of embeddings of the *wrong* person (a neighbour leaning into the crop, the adult
    walking past) drag the mean toward a stranger, silently.
  * The **vote** is the robust one. One intruding frame is one vote out of three hundred.
    But it throws away every runner-up, so a seat whose true pupil is second-best in every
    single frame votes unanimously for the wrong child with no sign of distress.

A contaminated crop set moves the mean and not the vote; a systematically mis-galleried
pupil moves the vote and not the mean. Requiring both, as `assign.py` does, costs recall
in a module that has no business optimising for recall.

--------------------------------------------------------------------------------
**MEASURED ON THIS MACHINE, 2026-08-12, AND IT CHANGED TWO DEFAULTS.**

  * **CoreML is not the fast path, and at det_size 320 it does not run at all.** With
    `providers=["CoreMLExecutionProvider"]`, SCRFD raises
    `CoreML static output shape ({1,1,1,800,1}) and inferred shape ({3200,1}) have
    different ranks` at 320×320. At 640×640 it runs — and takes **0.396 s** against the
    CPU provider's **0.213 s** on the same image. CPU is therefore the CHOSEN default:
    faster here and it does not crash at the size this module actually wants.
  * **`cv2.imread` is not used for photographs.** The photo tree is
    `student_photos/5-А/72.jpg` — Cyrillic path components. `np.fromfile` +
    `cv2.imdecode` sidesteps the whole question of how OpenCV's C++ layer handles a
    non-ASCII path on a given platform, at a cost of one line.
  * **Faces are cropped from the person box, not detected in the full frame.** SCRFD
    rescales its input to fit `det_size`, so what matters is the face's share of the image
    it is given. A 64 px face in a 2560-wide frame at det 1280 arrives as **32 px**; the
    same face inside a 400 px-tall person crop at det 320 arrives as **51 px**; inside the
    upper 55 % of that crop, as **~93 px**. The head crop is tried first and the full
    person box is the fallback, because a pupil bent over their desk has their head
    outside the top of the box.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "buffalo_l"

# CPU first. Measured above: CoreML is 1.9x SLOWER on this machine and raises a shape-rank
# error at det_size 320. Left overridable because a different host may invert this, and a
# provider list is exactly the kind of thing that should be an argument rather than a
# belief.
DEFAULT_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)

# The detector's input square. 320 is CHOSEN for the crop path: the crops are 150-500 px
# tall, so 320 neither throws away resolution nor spends time upscaling noise. Gallery
# photographs are 3000-4000 px and get their own, larger size.
CROP_DET_SIZE = 320
PHOTO_DET_SIZE = 640

# Below this the observation is discarded rather than weighted down. MEASURED: the face
# height distribution on this footage runs 30.8 / 53.5 / 64.0 / 73.7 / 95.2 px. 40 px sits
# between the minimum and p25, so it drops the smallest tail — the pupil furthest from the
# lens, whose embedding is the one most likely to be a smear — and keeps ~3/4 of the data.
# CHOSEN at that shoulder of the distribution.
MIN_FACE_PX = 40.0

# MEASURED: the detector's own score never went below 0.5 on this footage, so this rejects
# nothing that was seen properly and everything the detector itself was unsure about.
MIN_DET_SCORE = 0.5

# Fraction of the person box height searched for a head first. CHOSEN from the geometry of
# a seated pupil: shoulders sit at roughly 0.25-0.35 of box height in this view, so 0.55
# clears the head with margin while excluding the desk, the hands and the neighbour's
# shoulder. The full box is retried when this finds nothing.
HEAD_FRACTION = 0.55

# The crop is grown by this fraction of the box on each side before cutting. A pose box is
# tight around the visible keypoints and habitually clips the top of the head, which is
# where SCRFD's landmark regression looks.
CROP_PAD = 0.12


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One face, seen once. Never carries an identity — that is not decided per frame."""

    embedding: np.ndarray          # (512,), L2-normalised
    det_score: float
    height_px: float
    video_seconds: float | None = None

    @property
    def usable(self) -> bool:
        return self.height_px >= MIN_FACE_PX and self.det_score >= MIN_DET_SCORE


@dataclass(frozen=True, slots=True)
class Match:
    """The result of comparing one vector to the gallery. A score, not a decision."""

    best_id: str | None
    best_score: float
    second_id: str | None
    second_score: float

    @property
    def margin(self) -> float:
        return self.best_score - self.second_score

    def to_dict(self) -> dict[str, Any]:
        return {"best": self.best_id, "score": round(self.best_score, 3),
                "runner_up": self.second_id, "runner_up_score": round(self.second_score, 3),
                "margin": round(self.margin, 3)}


class FaceEncoder:
    """A thin wrapper over InsightFace `buffalo_l`, loaded lazily and never at import.

    Lazy because the whole pipeline must run with faces switched off — that is the default
    — and because `import insightface` costs ~6 s of ONNX session setup. A module that
    paid that on import would make every unrelated unit test slower.
    """

    def __init__(self, *, model_name: str = MODEL_NAME,
                 providers: Sequence[str] = DEFAULT_PROVIDERS,
                 det_size: int = CROP_DET_SIZE, det_thresh: float = MIN_DET_SCORE) -> None:
        self.model_name = model_name
        self.providers = tuple(providers)
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._app: Any | None = None

    def _ensure(self) -> Any:
        if self._app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=self.model_name,
                               # Landmarks-3d, landmarks-2d and gender/age are loaded by
                               # default and used by nothing here. Skipping them removes
                               # three ONNX sessions and their memory.
                               allowed_modules=["detection", "recognition"],
                               providers=list(self.providers))
            app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size),
                        det_thresh=self.det_thresh)
            self._app = app
        return self._app

    def provenance(self) -> dict[str, Any]:
        return {"model": self.model_name, "providers": list(self.providers),
                "det_size": self.det_size, "det_thresh": self.det_thresh,
                "min_face_px": MIN_FACE_PX, "min_det_score": MIN_DET_SCORE}

    # -- primitives -------------------------------------------------------------------

    def faces_in(self, image: np.ndarray, *, video_seconds: float | None = None
                 ) -> list[FaceObservation]:
        """Every face in an image, as normalised embeddings. BGR in, as OpenCV gives it."""
        if image is None or image.size == 0 or min(image.shape[:2]) < 16:
            return []
        found = self._ensure().get(image)
        out: list[FaceObservation] = []
        for face in found:
            vector = np.asarray(face.normed_embedding, dtype=np.float32)
            box = np.asarray(face.bbox, dtype=np.float32)
            out.append(FaceObservation(embedding=vector, det_score=float(face.det_score),
                                       height_px=float(box[3] - box[1]),
                                       video_seconds=video_seconds))
        return out

    def embed_photo(self, path: str | Path) -> FaceObservation | None:
        """One roster photograph -> one embedding, or None if no face was found.

        Uses `np.fromfile` + `imdecode` rather than `imread`: the photo tree has Cyrillic
        directory names. Takes the LARGEST face, because a school portrait with a second
        person in the background is a portrait of the person in front.
        """
        import cv2

        path = Path(path)
        try:
            buffer = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            return None
        if image is None:
            return None

        app = self._ensure()
        previous = self.det_size
        if previous != PHOTO_DET_SIZE:
            app.prepare(ctx_id=0, det_size=(PHOTO_DET_SIZE, PHOTO_DET_SIZE),
                        det_thresh=self.det_thresh)
        try:
            faces = self.faces_in(image)
        finally:
            if previous != PHOTO_DET_SIZE:
                app.prepare(ctx_id=0, det_size=(previous, previous),
                            det_thresh=self.det_thresh)
        if not faces:
            return None
        return max(faces, key=lambda f: f.height_px)

    def embed_person(self, frame: np.ndarray, box: Sequence[float], *,
                     video_seconds: float | None = None) -> FaceObservation | None:
        """The face of the person in this box, from this frame, or None.

        Head crop first, whole box as fallback — see the module docstring for the pixel
        arithmetic that makes the crop path worth the extra call. When several faces land
        in one crop (a neighbour leaning in) the one with the largest area is taken, on the
        reasoning that the crop is centred on its own person and an intruder is at its
        edge and further from the lens. This is a heuristic and it is the reason
        `assign.py` requires a vote as well as a mean: an occasional intruding crop must
        not be able to move the answer on its own.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        box_height = max(y2 - y1, 1.0)
        pad_x = CROP_PAD * max(x2 - x1, 1.0)
        pad_y = CROP_PAD * box_height

        for fraction in (HEAD_FRACTION, 1.0):
            top = max(int(y1 - pad_y), 0)
            bottom = min(int(y1 + fraction * box_height + pad_y), height)
            left = max(int(x1 - pad_x), 0)
            right = min(int(x2 + pad_x), width)
            if bottom - top < 16 or right - left < 16:
                continue
            faces = self.faces_in(frame[top:bottom, left:right],
                                  video_seconds=video_seconds)
            if faces:
                return max(faces, key=lambda f: f.height_px)
        return None


# --------------------------------------------------------------------------------------
# The gallery
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gallery:
    """The registered pupils' embeddings, and the arithmetic for comparing against them.

    `missing` is a first-class field, not a log line: a pupil whose photograph produced no
    face is a pupil this gallery **cannot** return, and a matcher that does not know that
    will happily hand their seat to the second-best candidate. It goes into the artefact.
    """

    external_ids: tuple[str, ...]
    matrix: np.ndarray                     # (N, 512), rows L2-normalised
    model: str
    missing: tuple[str, ...] = ()          # pupils whose photo yielded no face
    class_name: str | None = None

    def __len__(self) -> int:
        return len(self.external_ids)

    def match(self, vector: np.ndarray) -> Match:
        """Cosine against every gallery row. Both the best and the runner-up, always.

        The runner-up is returned by construction rather than on request because the
        margin is the number that decided this module's fate — a best score without the
        gap to the next candidate is the statistic that made 0.30 look like a result.
        """
        if len(self) == 0:
            return Match(None, 0.0, None, 0.0)
        norm = float(np.linalg.norm(vector))
        unit = vector / norm if norm > 0 else vector
        scores = self.matrix @ unit
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        if len(order) == 1:
            return Match(self.external_ids[best], float(scores[best]), None, 0.0)
        second = int(order[1])
        return Match(self.external_ids[best], float(scores[best]),
                     self.external_ids[second], float(scores[second]))

    def score_for(self, external_id: str, vector: np.ndarray) -> float | None:
        """This specific pupil's cosine — what `assign.py` needs to test a seat map's claim.

        The question a seat map poses is not "who is this?" but "is this the child the
        plan says it is?", and those have different answers: a pupil can be a clear best
        match at 0.31 and be, in absolute terms, no match at all.
        """
        if external_id not in self.external_ids:
            return None
        index = self.external_ids.index(external_id)
        norm = float(np.linalg.norm(vector))
        unit = vector / norm if norm > 0 else vector
        return float(self.matrix[index] @ unit)

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "pupils": len(self), "class_name": self.class_name,
                "without_face_in_photo": list(self.missing)}


def build_gallery(roster: Any, encoder: FaceEncoder | None = None, *,
                  cache_dir: str | Path = "classvision/.cache") -> Gallery:
    """Embed every roster photograph, caching per photo by its content hash.

    Keyed on the **sha256 of the file**, not its path or mtime. A school that re-uploads
    the same picture under a new name gets a cache hit; a school that replaces a child's
    photograph with a different one gets a miss, which is the behaviour that matters —
    a path-keyed cache would keep serving the embedding of the old photo forever.

    The cache is one `.npz` per model, so a change of model cannot be mistaken for a
    change of photograph.
    """
    encoder = encoder or FaceEncoder()
    cache_path = Path(cache_dir) / f"faces_{encoder.model_name}.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                cached = {key: data[key] for key in data.files}
        except Exception:
            cached = {}   # a corrupt cache is a cache miss, never an error

    ids: list[str] = []
    rows: list[np.ndarray] = []
    missing: list[str] = []
    fresh = False

    for pupil in roster.pupils:
        if pupil.photo is None:
            missing.append(pupil.external_id)
            continue
        key = _file_sha(pupil.photo)
        vector = cached.get(key)
        if vector is None:
            observation = encoder.embed_photo(pupil.photo)
            if observation is None:
                # An empty row is stored so a photograph with no detectable face is not
                # re-attempted on every run. Its emptiness is what marks it.
                vector = np.zeros(512, dtype=np.float32)
            else:
                vector = observation.embedding.astype(np.float32)
            cached[key] = vector
            fresh = True
        if float(np.linalg.norm(vector)) < 1e-6:
            missing.append(pupil.external_id)
            continue
        ids.append(pupil.external_id)
        rows.append(vector / float(np.linalg.norm(vector)))

    if fresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cached)

    matrix = (np.vstack(rows).astype(np.float32) if rows
              else np.zeros((0, 512), dtype=np.float32))
    return Gallery(external_ids=tuple(ids), matrix=matrix, model=encoder.model_name,
                   missing=tuple(missing), class_name=getattr(roster, "class_name", None))


def _file_sha(path: Path, *, chunk: int = 1 << 18) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return "p" + digest.hexdigest()[:24]     # npz keys must be valid identifiers-ish


# --------------------------------------------------------------------------------------
# Seat-level aggregation — the only interface `assign.py` is allowed to use
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AggregateMatch:
    """A whole lesson of one seat's faces, matched two independent ways.

    Everything here is `assign.py`'s input. Nothing here is a decision, and there is
    deliberately no `identity` or `name` field to reach for.
    """

    seat_id: int
    observations: int              # faces found at this seat
    usable: int                    # ...that passed the size/score gate
    median_face_px: float
    median_det_score: float
    mean_match: Match | None       # match of the averaged embedding
    votes: dict[str, int]          # per-frame argmax, counted
    top_vote: str | None
    top_vote_share: float
    median_frame_score: float      # median of the per-frame best cosines
    median_frame_margin: float     # median of the per-frame margins -- the 0.10 above

    def agree(self) -> bool:
        """Do the two methods name the same child? Necessary, nowhere near sufficient."""
        return (self.mean_match is not None and self.mean_match.best_id is not None
                and self.mean_match.best_id == self.top_vote)

    def score_of(self, external_id: str) -> float | None:
        return self._per_pupil.get(external_id) if self._per_pupil else None

    # Populated by `aggregate()`; kept out of the constructor signature so this object
    # stays trivially serialisable.
    _per_pupil: dict[str, float] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "observations": self.observations,
            "usable": self.usable,
            "median_face_px": round(self.median_face_px, 1),
            "median_det_score": round(self.median_det_score, 2),
            "mean_embedding_match": self.mean_match.to_dict() if self.mean_match else None,
            "votes": dict(sorted(self.votes.items(), key=lambda kv: -kv[1])),
            "top_vote": self.top_vote,
            "top_vote_share": round(self.top_vote_share, 3),
            "single_frame_median_score": round(self.median_frame_score, 3),
            "single_frame_median_margin": round(self.median_frame_margin, 3),
            "methods_agree": self.agree(),
        }


@dataclass(slots=True)
class SeatFaceEvidence:
    """Accumulates faces per seat across a lesson. The unit is a SEAT, as everywhere else.

    Not per track. A track dies to occlusion and is reborn with a new id, so track-level
    aggregation would produce a dozen short, weak collections instead of one long strong
    one — which is the same argument `room/seats.py` makes for the whole design, applied
    to the one quantity where averaging actually buys accuracy.
    """

    by_seat: dict[int, list[FaceObservation]] = field(default_factory=dict)
    frames_examined: int = 0
    crops_attempted: int = 0

    def add(self, seat_id: int, observation: FaceObservation | None) -> None:
        self.crops_attempted += 1
        if observation is not None:
            self.by_seat.setdefault(int(seat_id), []).append(observation)

    def aggregate(self, gallery: Gallery) -> dict[int, AggregateMatch]:
        """Per seat: the mean-embedding match, the vote distribution, and the raw quality.

        The mean is taken over **unit** vectors and re-normalised, which is the standard
        way to average directions on a sphere and the reason the noise falls as ~1/√n.
        `_per_pupil` keeps the mean vector's cosine against every gallery member so that
        `assign.py` can ask about the specific child the seating plan names, rather than
        only about the winner.
        """
        out: dict[int, AggregateMatch] = {}
        for seat_id, observations in sorted(self.by_seat.items()):
            usable = [o for o in observations if o.usable]
            heights = sorted(o.height_px for o in observations) or [0.0]
            scores = sorted(o.det_score for o in observations) or [0.0]

            if not usable or len(gallery) == 0:
                out[seat_id] = AggregateMatch(
                    seat_id=seat_id, observations=len(observations), usable=len(usable),
                    median_face_px=heights[len(heights) // 2],
                    median_det_score=scores[len(scores) // 2],
                    mean_match=None, votes={}, top_vote=None, top_vote_share=0.0,
                    median_frame_score=0.0, median_frame_margin=0.0,
                )
                continue

            stack = np.vstack([o.embedding for o in usable]).astype(np.float32)
            mean = stack.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            mean = mean / norm if norm > 0 else mean
            mean_match = gallery.match(mean)
            per_pupil = {external_id: float(gallery.matrix[i] @ mean)
                         for i, external_id in enumerate(gallery.external_ids)}

            per_frame = [gallery.match(o.embedding) for o in usable]
            votes = Counter(m.best_id for m in per_frame if m.best_id is not None)
            top_vote, top_count = (votes.most_common(1)[0] if votes else (None, 0))
            frame_scores = sorted(m.best_score for m in per_frame)
            frame_margins = sorted(m.margin for m in per_frame)

            out[seat_id] = AggregateMatch(
                seat_id=seat_id, observations=len(observations), usable=len(usable),
                median_face_px=heights[len(heights) // 2],
                median_det_score=scores[len(scores) // 2],
                mean_match=mean_match, votes=dict(votes), top_vote=top_vote,
                top_vote_share=top_count / len(per_frame) if per_frame else 0.0,
                median_frame_score=frame_scores[len(frame_scores) // 2],
                median_frame_margin=frame_margins[len(frame_margins) // 2],
                _per_pupil=per_pupil,
            )
        return out

    def summary(self) -> dict[str, Any]:
        return {"frames_examined": self.frames_examined,
                "crops_attempted": self.crops_attempted,
                "faces_found": sum(len(v) for v in self.by_seat.values()),
                "seats_with_faces": len(self.by_seat)}


def collect_from_video(video: str | Path, *, frame_times: Sequence[float],
                       boxes_at: dict[float, list[tuple[int, Sequence[float]]]],
                       encoder: FaceEncoder | None = None,
                       every_seconds: float = 10.0,
                       time_tolerance: float = 0.26) -> SeatFaceEvidence:
    """Second decode pass, sparse, purely for faces. Optional, and off unless asked for.

    `boxes_at` maps an analysed frame's video-second to the (seat_id, person box) pairs
    already established by the pose pass, so this pass never re-detects people — it only
    looks inside boxes whose seat is already known. That ordering is what keeps face
    evidence *attached to a seat* rather than becoming an identity mechanism of its own.

    `every_seconds` is 10 by CHOICE, not for speed alone: consecutive frames of a seated
    pupil are near-duplicates, so 300 observations taken 0.5 s apart carry barely more
    independent information than 90 taken 10 s apart, while costing 3.3x the model time
    and biasing the mean toward whatever pose the pupil happened to hold longest.

    Sampled times are matched to the nearest analysed frame within `time_tolerance`
    (half a sampling interval at 2 fps) rather than assumed equal: the two passes derive
    their times from the same frame indices, and a float comparison that assumes so is a
    silent zero-face run waiting for a different sampling rate.
    """
    from classvision.video import decode

    encoder = encoder or FaceEncoder()
    evidence = SeatFaceEvidence()
    known = np.array(sorted(frame_times), dtype=np.float64)
    if known.size == 0:
        return evidence

    sample_fps = 1.0 / every_seconds if every_seconds > 0 else 1.0
    for sample in decode.samples(video, sample_fps,
                                 start_seconds=float(known[0]),
                                 end_seconds=float(known[-1])):
        position = int(np.argmin(np.abs(known - sample.video_seconds)))
        nearest = float(known[position])
        if abs(nearest - sample.video_seconds) > time_tolerance:
            continue
        entries = boxes_at.get(nearest)
        if not entries:
            continue
        evidence.frames_examined += 1
        for seat_id, box in entries:
            evidence.add(seat_id, encoder.embed_person(sample.image, box,
                                                       video_seconds=nearest))
    return evidence
