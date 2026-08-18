"""Classroom config: the thresholds behind the four lesson metrics.

**READ THIS BEFORE TRUSTING ANY DEFAULT IN THIS FILE.**

We hold no recording of a lesson. Not one. The `class/` folder is 140 photographs, and a
photograph cannot tell you how long a hand stays up. So unlike `config/bullying.py` --
whose numbers are at least the legacy's, converted, and whose worst offender was caught
by a MEASURED noise floor -- every default below was reasoned from body geometry and then
chosen. None has been checked against a child in a chair.

They are written as ratios of a person's own shoulder width and as counts of analysed
frames, so that they are at least the same threshold at the front and the back of the
room (see `classroom/posture.py`). That makes them transferable. It does not make them
right, and the difference is the whole of this docstring.

Each default below carries the reasoning that produced it, so that whoever finally has
footage can see what was assumed and check that specific assumption rather than
re-deriving the lot. Where a number is a plain guess it says GUESS, in those letters.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from qorgan.config.common import Base


class PoseSettings(Base):
    """The pose model, as a whole-classroom camera needs it.

    Separate from `bullying.SkeletonSettings` because the two ask the model different
    questions, not because there are two models. The bullying path feeds it a ~320 px
    crop of exactly two people; this feeds it the whole room. `min_frames`/`max_frames`
    would be meaningless here (there is no candidate window to sample), and a knob that
    means nothing on the camera whose YAML it appears in is what this project keeps
    finding and deleting. The MODEL and the extraction are shared -- one `PoseModel`
    per worker process, and `qorgan.models.pose.keypoints_from` converts for both.
    """

    model: str = "yolov8n-pose.pt"
    conf: float = Field(default=0.25, gt=0.0, lt=1.0)

    # The pose model letterboxes its input to this size. 320 (the bullying crop width) is
    # the wrong number here by a wide margin: that crop holds two people, this frame holds
    # a class. At 320 across a room of 25, a child is ~25 px wide and their shoulders are
    # ~12 px apart -- below `MIN_USABLE_SHOULDER_PX`, so every metric would return unknown
    # and the module would produce an empty report while appearing to work.
    #
    # 960 is a GUESS, and specifically a guess about a room we have never seen: it assumes
    # a class occupies most of the frame width and gives roughly 70 px per child across a
    # 7 m room. It is the one key here whose cost is GPU time rather than accuracy, so it
    # is the first to lower if the worker cannot keep up. Settle it by running the module
    # against the first lesson recording the school sends and comparing the count of
    # usable skeletons at 640, 960 and 1280.
    imgsz: int = Field(default=960, ge=320, le=1920)


class HandRaiseRules(Base):
    """«Сколько раз поднял руку» -- the first of the four facts promised in §8."""

    # How far above the shoulder line a wrist must sit, in shoulder widths.
    #
    # Reasoning: an adult head is about 0.6 shoulder widths tall and a child's rather
    # more; a hand raised to answer puts the wrist somewhere around head height. Half of
    # that leaves room for the tentative half-raise children actually do, while staying
    # clear of a wrist resting on the desk (below the line, negative) or lifted to write
    # (a little above zero). 0.35 is therefore a REASONED GUESS from proportions, not a
    # measurement, and it is the single number most likely to be wrong: the whole metric
    # is the count of times this line is crossed.
    above_shoulder_ratio: float = Field(default=0.35, gt=0.0, le=3.0)

    # A raise must persist for this many ANALYSED frames before it counts.
    #
    # Counted in analysed frames rather than seconds on purpose: it exists to reject
    # single-frame pose noise, and pose noise is per-frame, not per-second. At the
    # shipped `det_every: 1` and 15 fps this is a fifth of a second -- long enough to
    # discard a flicker, far too short to discard a stretch. GUESS: nothing measures how
    # noisy this model is on a seated child, because nothing has run it on one.
    min_hold_observations: int = Field(default=3, ge=1)

    # The wrist must be back DOWN for this many analysed frames before another raise can
    # be counted. Without it, a hand wavering either side of the line is counted dozens of
    # times and the child with the least steady arm tops the report. GUESS, same footing
    # as above; deliberately equal to `min_hold_observations`, because there is no
    # evidence on which to make it different.
    min_gap_observations: int = Field(default=3, ge=1)


class PlaceRules(Base):
    """«Сколько раз встал» and «сколько времени вне места» -- facts two and three.

    Both are measured against the track's OWN settled position, never against a seating
    plan (we have none) and never against other children. That is what §8 promised --
    «сравниваем ребёнка только с ним самим» -- and here it is also the only thing the
    geometry supports: see `classroom/posture.py` on why no single frame can say whether
    somebody is standing.
    """

    # How many analysed frames a track must be present before its seated baseline is
    # fixed. Everything about that track before this is measured against nothing and is
    # therefore not counted at all -- which is why the report carries `settled` per track:
    # a track that never settles produces zeroes that mean "not measured", and zeroes that
    # mean two things is the defect `migrations/0005` exists about.
    #
    # 30 frames is ~2 s at 15 fps. GUESS: long enough that a child still sitting down is
    # not frozen as the baseline, short enough not to lose the start of the lesson.
    settle_observations: int = Field(default=30, ge=1)

    # How far the shoulder line must RISE above the settled baseline to count as standing,
    # in shoulder widths.
    #
    # Reasoning: standing up from a chair raises the shoulders by roughly the seated
    # thigh-to-seat height, which for a child is comparable to their own shoulder width.
    # 0.8 sits just under that, so a genuine stand clears it and leaning back or sitting
    # up straight (a few centimetres) does not. GUESS from proportions.
    rise_ratio: float = Field(default=0.8, gt=0.0, le=5.0)

    # As `HandRaiseRules.min_hold_observations`, for standing, and for the same reason.
    min_hold_observations: int = Field(default=3, ge=1)

    # How far the track's anchor must move from its settled position to be «вне места»,
    # in shoulder widths. Two shoulder widths is roughly one seat over -- far enough that
    # leaning across a desk does not count, close enough that standing in the aisle does.
    # GUESS.
    away_ratio: float = Field(default=2.0, gt=0.0, le=20.0)

    # An excursion shorter than this is not reported as time away. Its job is to stop
    # tracker jitter and a half-second lean accumulating into minutes across a lesson.
    # GUESS: five seconds is about the shortest absence a teacher would call being away
    # from your place, and no measurement supports it over three or ten.
    min_away_seconds: float = Field(default=5.0, gt=0)


class LessonRules(Base):
    """The lesson as a record: how long it may run, and what counts as being at it."""

    # A lesson nobody closed is force-closed after this long, exactly as a meal session
    # is (`canteen/sessions.py`). Better to record honestly that we lost track of a lesson
    # than to leave it open for the rest of the year, silently absorbing tomorrow's
    # tracks into yesterday's report. 60 minutes for a 45-minute lesson plus the change.
    max_lesson_minutes: float = Field(default=60.0, gt=0)

    # A lesson ends when the room has held no tracked person for this long. This is the
    # ORDINARY ending, and it exists because there is no bell signal and the school has
    # given us no timetable -- so an empty room is the only evidence of an ending the
    # system actually has. `LessonCloseReason` therefore has no BELL member: an enum
    # member nothing can produce is a column that lies about the range of a record.
    #
    # 5 minutes is a GUESS, sized to be longer than a class filing out and shorter than
    # the gap between lessons. Too low and one long occlusion splits a lesson in two; too
    # high and consecutive lessons merge into one report.
    end_after_empty_minutes: float = Field(default=5.0, gt=0)

    # A track not seen for this long is finished: flushed and evicted (rule R8). It is NOT
    # a claim that the child left -- an occlusion this long ends a ByteTrack track too,
    # and the child who reappears gets a NEW id and a new row. That is the fragmentation
    # the report has to warn about, and this is the knob that sets its rate.
    track_idle_seconds: float = Field(default=5.0, gt=0)

    # Hard ceiling on the ledgers held in memory at once (rule R8). A lesson is 45 minutes
    # of frames and track ids only ever go up; the legacy leaked several dicts keyed on
    # them. 80 is roughly twice the largest class this school has, which leaves room for
    # fragmentation without leaving room for a leak.
    max_tracks: int = Field(default=80, ge=1)

    # How often the in-memory ledgers are written to the database. This is the whole of
    # what a mid-lesson worker restart costs: at 30 s, a crash at minute 40 of a 45-minute
    # lesson loses at most the last half-minute of counting, and never the lesson. The
    # legacy kept canteen sessions in a module global and lost every one of them on
    # restart, silently.
    flush_interval_seconds: float = Field(default=30.0, gt=0)

    # A track observed for less than this is a FRAGMENT, and the report counts it apart
    # from the rest instead of listing it as a child who was barely there. Stamped onto
    # the lesson row when the lesson opens, so that re-reading an old lesson cannot
    # silently restate it under today's YAML.
    #
    # 300 s of a 45-minute lesson. GUESS, and the one that most changes how the report
    # reads: raise it and more of the room is declared fragmentary; lower it and
    # fragments are presented as children.
    min_presence_seconds: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def _presence_fits_inside_a_lesson(self) -> LessonRules:
        if self.end_after_empty_minutes >= self.max_lesson_minutes:
            raise ValueError(
                f"classroom.lesson.end_after_empty_minutes ({self.end_after_empty_minutes} "
                f"min) is not shorter than max_lesson_minutes ({self.max_lesson_minutes} "
                "min): the room could never be empty long enough to end a lesson normally, "
                "so every lesson would be force-closed as TIMEOUT and the close reason "
                "would stop carrying any information"
            )
        if self.min_presence_seconds >= self.max_lesson_minutes * 60:
            raise ValueError(
                f"classroom.lesson.min_presence_seconds ({self.min_presence_seconds}s) is "
                f"not shorter than max_lesson_minutes ({self.max_lesson_minutes} min): no "
                "track could ever be long enough to count as present, so every lesson "
                "would report an empty room"
            )
        return self


class ClassroomConfig(Base):
    """Everything a classroom camera needs, and deliberately nothing else.

    There is no identity block and no face block, on purpose. §8 promised the school
    there would be no recognition in a classroom, and at this distance the measurement
    agrees: 14 970 corridor faces, median 11.5 px, zero recognised. A schema with nowhere
    to put a `min_score` is a promise the config file cannot break.
    """

    pose: PoseSettings = PoseSettings()
    hand_raise: HandRaiseRules = HandRaiseRules()
    place: PlaceRules = PlaceRules()
    lesson: LessonRules = LessonRules()
