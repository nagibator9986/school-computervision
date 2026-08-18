"""Face recognition config. **Not about meals**, so it does not live in `canteen.py`.

A hall camera needs a `RecognitionPolicy`. It must not have to import a module about meal
sessions to get one -- that coupling is what made `worker/canteen.py` the only place in the
system that could recognise a face at all.

Every number here is recorded with the measurement beside it. The legacy shipped
`SAME_PERSON_SIMILARITY = 0.35`; against this school's data that constant would call 55
different pairs of children the same person. The measured band says 0.60. Guessing is what
cost the legacy eighteen thresholds and 1 816 NULL canteen records.
"""

from __future__ import annotations

from pydantic import Field

from qorgan.config.common import Base

# The worst GENUINE impostor in this school's own gallery. Measured, not chosen: all 142
# photographs embedded with the production model, the full 138x138 cosine matrix, the six
# duplicate enrolments removed -- 9 447 pairs of DIFFERENT children, of which the highest
# scores 0.472 and exactly one exceeds 0.45.
#
# A `min_score` at or below this knowingly accepts a confusion we have already measured.
#
# It is checked at CONFIG-LOAD time (`config/loader.py`), on the MERGED value, because that
# is the only layer that is true. It is NOT a field validator: a unit test may legitimately
# build `RecognitionPolicy(min_score=0.1)` to exercise the matching logic -- it is testing a
# decision function, not proposing to run a school on it -- and a validator on the leaf
# cannot tell a test fixture from a production camera. A check aimed at the wrong target is
# not a check.
#
# This is a property of THIS roster. `qorgan pupils gallery-report` recomputes it; if the
# school sends new photographs, re-derive it from a measurement rather than from memory.
MEASURED_IMPOSTOR_CEILING = 0.472


class FaceGate(Base):
    """Minimum face size before we will even attempt to bind a recognition to a track.

    **On the school's hall cameras, 0 of 14 970 measured faces clear this gate.** Not 2.2%,
    which is the figure this docstring carried until it was checked: that number came from
    the 2560x1440 HD evidence burst, which is *not* the stream the worker analyses.

    **There is no single analysis resolution, and this docstring will not name one.**
    It is `capture.frame_width x frame_height` on THAT camera's merged config. `base.yaml`
    holds a default; profiles override it; the merged value is the only one that runs.

    This paragraph used to enumerate which profiles override -- and the list was wrong, in a
    docstring whose whole subject is that you must not trust a quoted number. Any list of
    "the ones that override" is a second source of truth that goes stale the moment someone
    edits a YAML file, which is exactly the bug. So: no list. Read the camera's config, or
    run `qorgan identity camera-report`, which reads it for you.

    At the hall's real 1280x720: median face **11.5 px**, p90 22.5 px, largest in the whole
    corpus **50 px**. The strict 60 px gate needs a 120 px face in the clip, and no such
    face exists in 14 970.

    Two consequences, and the second is the reason this paragraph exists.

    First: face recognition on a hall camera is not poor, it is **arithmetically
    impossible**. It is a camera-placement fact, and no threshold is a substitute for one.
    An 11-px face upscaled to ArcFace's 112-px input is mush.

    Second, and this is the trap: a plausible-but-wrong "2.2%" invites the next engineer to
    reason "low but non-zero -- drop `min_width` to 40 and recover some". **It has been
    tried, on the numbers.** Lowering to the 38 px small-face gate lets **77 of 14 970**
    faces through -- and **not one of them is recognised**: the best score among all 77 is
    **0.350**, against a `min_score` of 0.45. So the recovery is not "small", it is
    **zero**, and the faces that are big enough are still too degraded to score. A number
    that is merely misleading is worse in a config file than no number, because the file
    exists to stop exactly that.

    So: run `qorgan identity camera-report` before you touch a number here. It answers, per
    stream and at the resolution that stream is really analysed at -- **read from the
    camera's own config, never assumed** -- whether a camera can recognise anybody at all.
    That is not a threshold question, and the legacy asked it eighteen times in the form of
    a threshold.
    """

    min_width: int = Field(default=60, ge=1)
    min_height: int = Field(default=70, ge=1)
    min_area: int = Field(default=4200, ge=1)

    def accepts(self, width: int, height: int) -> bool:
        return (
            width >= self.min_width
            and height >= self.min_height
            and width * height >= self.min_area
        )


class RecognitionPolicy(Base):
    """One decision rule: accept iff score >= min_score AND gap >= min_gap.

    **`min_score` is 0.50, and that is a MEASURED FLOOR, not a settled value.**

    Measured (spec §2.2): every one of the school's 142 photographs embedded with this
    model, then the full 138x138 cosine matrix -- 9 453 pairs. With the six duplicate
    enrolments removed, the genuine impostor distribution is p50 0.094, p90 0.214,
    p99 0.331, and **max 0.472**. Exactly one pair of 9 447 lands above 0.45.

    So the previous default of 0.45 sat BELOW the worst genuine impostor. Margin -0.022.
    It admitted a known confusion. 0.50 sits inside the band that is empty from 0.48 to
    0.77, above every impostor in the data.

    **The ceiling is UNMEASURED.** Those scores are gallery-photo against gallery-photo.
    In production the query is a CAMERA face -- blurred, off-angle, small -- so this probe
    gives a hard floor and says nothing about whether a real camera face can REACH 0.50 at
    all.

    It was probed against 14 970 real faces in 250 hall clips. Scores do reach 0.604 --
    but ONLY on the 2560x1440 HD burst, which is not the stream the worker analyses. At the
    hall's real analysis resolution (capture.frame_width x frame_height = **1280x720**;
    `hall.yaml` overrides `base.yaml`'s 960x540 default), **0 of those 14 970 faces clear
    the strict 60 px gate**, and of the 77 that clear the 38 px small-face gate **not one is
    accepted at any threshold** -- the best score among them is **0.350**. Zero recognitions
    in 14 970 faces. Face recognition on a hall camera is not poor, it is arithmetically
    impossible, and no threshold here can change that -- it is a question of optics
    (spec §2.4).

    So the hall says nothing either way about the CANTEEN, whose entry camera is
    close-range. The ceiling remains open, and this remains a floor.

    What closes it: footage from the CANTEEN ENTRY camera of pupils we can name. One
    volunteer walking through. Until then this is a floor, and it must not be written up
    as a settled number.

    And before trusting ANY camera with recognition, run `qorgan identity camera-report`:
    it answers, per stream and at the resolution that stream is really analysed at -- read
    from that camera's OWN config, because there is no fleet-wide analysis resolution to
    assume -- whether the camera can recognise anybody at all. That is the question the
    legacy asked eighteen times, in the form of a threshold. It is not a threshold question.

    On the EXIT camera specifically (`config/canteen.py::ExitSettings`) this floor is
    deliberately strict, and the cost is real: a session we cannot close force-closes as
    UNKNOWN rather than risk a false match. That cost is measured, not assumed --
    `qorgan pupils report` counts it as `forced_unknown` (sessions closed by
    `CloseReason.TIMEOUT`). If that number spikes, the floor is costing more than it is
    worth and we will SEE it, rather than guess.

    `gap` is top1 - top2, ranked by PERSON (see `faces.matching._rank`). The legacy ranked
    by photo, so top1 and top2 were two shots of the same child and the gap was
    structurally ~0 -- the gate rejected everybody, and 1 816 of 1 820 canteen records
    came out Unknown. No value of `min_gap` here can rescue a gap computed that way.

    Separately: legacy set gap to a huge sentinel when only one candidate existed, which
    disabled the check in the case where a single weak match is most dangerous. Here a
    lone candidate has no gap evidence, so `single_candidate_gap` is what it is actually
    worth -- 0.0 by default.
    """

    min_score: float = Field(default=0.50, gt=0.0, lt=1.0)
    min_gap: float = Field(default=0.05, ge=0.0, lt=1.0)
    single_candidate_gap: float = Field(default=0.0, ge=0.0, lt=1.0)
    # face_gate DELETED -- and this one is worth a sentence, because it is not like the
    # others. `SoftAccumulator.face_gate` IS live (`identity/service.py::_soften` calls
    # `.accepts`), so the name looked alive to every grep. This one was not: `faces.
    # matching.identify` reads min_score, min_gap and single_candidate_gap off the policy
    # and never the gate, so the STRICT recognition path applies NO face-size gate at
    # all -- while the canteen profiles set `recognition.face_gate.min_width: 52` and
    # believed it did.
    #
    # Deleting it changes no behaviour: nothing read it. But whether the strict path
    # SHOULD size-gate is a real question this task is not entitled to answer -- see
    # `.superpowers/sdd/task-6-report.md`. If the answer is yes, the gate goes back with
    # a consumer attached, and this test is what will keep it honest.


class SoftAccumulator(Base):
    """Accept a lower-scoring match if the SAME person comes top-1 repeatedly.

    This is the "small face" path, and it is real domain knowledge, not a hack:
    younger pupils' faces are systematically below the size gate, so a strict
    single-shot threshold simply never recognises the first-graders. Keep it.

    One model, reused for the entry small-face path and the exit soft path -- the
    legacy wrote this same logic out four times with four sets of key names.

    Measured (spec §2.7): this is also where merging duplicate enrolments bites. On the
    small-face path at **>=38 px AT HD**, merging takes accepts from 3 to 9 and gap-kills
    from 6 to 0. At the 60 px gate the footage is too sparse to show any effect at all.

    **Read that scope carefully: >=38 px at HD is >=19 px at the hall's real 1280x720 --
    BELOW this gate entirely.** So the accept-count above describes gallery faces on the HD
    burst, NOT production, and it says nothing about the hall, where nothing is recognised
    regardless (spec §2.4). What survives unqualified is the GAP COLLAPSE it comes from
    (0.001 -> 0.413 after merge): that is a property of the gallery -- one human enrolled
    twice sits in his own top-2 -- so it holds on any camera at any resolution. Merging
    therefore matters where faces are big enough to be recognised at all: the CANTEEN.
    """

    enabled: bool = False
    min_score: float = Field(default=0.34, gt=0.0, lt=1.0)
    min_gap: float = Field(default=0.12, ge=0.0, lt=1.0)
    min_hits: int = Field(default=2, ge=2)
    window_seconds: float = Field(default=6.0, gt=0)
    face_gate: FaceGate = FaceGate(min_width=38, min_height=48, min_area=1800)


class FaceModelSettings(Base):
    """One InsightFace instance per process. Legacy created up to five."""

    model_name: str = "buffalo_l"
    model_version: str = "1.0"
    det_size: int = Field(default=640, ge=160)
    embedding_dim: int = Field(default=512, ge=1)
    normalized: bool = True


class BindingSettings(Base):
    """Recognise once per TRACK, not once per frame.

    The old canteen worker embedded every face in every due frame, every 0.25 s. The
    expensive half of that is the 512-d ArcFace embedding. For five children queuing over
    ten seconds it cost roughly 200 embeddings; per-track binding costs **five**.

    So: watch a track, keep only the best face seen so far, and after `min_face_frames`
    observations OR `max_wait_seconds` -- whichever comes first, because a child who turns
    their head for the whole queue must still be recognised -- embed once and bind.

    Accepted => never recognised again. Rejected => retried up to `max_attempts` with
    `retry_backoff_seconds` between tries (this is where the small-face path lives).
    Track lost for `track_ttl_seconds` => the binding is evicted, because the next person
    to get that track id is a different child (spec §4.4).
    """

    min_face_frames: int = Field(default=3, ge=1)
    max_wait_seconds: float = Field(default=1.5, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    track_ttl_seconds: float = Field(default=3.0, gt=0)
