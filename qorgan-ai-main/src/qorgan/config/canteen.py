"""Canteen config: **meal sessions**, and the three canteen camera roles.

The legacy system had 18 overlapping recognition thresholds, four near-identical
"soft accumulator" blocks, and six different minimum-face-size gates. That cascade
was the fossil record of trying to fix a broken recognition pipeline by tuning it:
1816 of its 1820 canteen records have student_id = NULL.

The recognition models themselves now live in `config/identity.py`, because face
recognition is not about meals and a hall camera must be able to import it without
dragging the canteen in with it. This module imports identity; identity never imports
this module.
"""

from __future__ import annotations

from pydantic import Field

from qorgan.config.common import Base
from qorgan.config.identity import (
    BindingSettings,
    FaceModelSettings,
    RecognitionPolicy,
    SoftAccumulator,
)


class SessionRules(Base):
    """The domain core. These rules were earned in a live canteen; port them exactly."""

    # Only the entry camera opens a session; only the exit camera closes it.
    # Inside cameras confirm presence and may late-bind an identity to an Unknown session.
    entry_cooldown_seconds: float = Field(default=60.0, ge=0)
    # exit_cooldown_seconds DELETED -- it was read only inside `_cooldown_block(entering=
    # False)`, a branch whose one caller always passes `entering=True`; `close()` never
    # called it, so the knob applied to nothing. The spec asks for a cooldown "on entry and
    # on exit", but a cooldown exists to stop a second RECORD for the same child, and only
    # `open` creates records: a person has at most one open session, a closed session never
    # reopens, and a close with no open session is a no-op, so an exit cannot be double-
    # counted for a cooldown to prevent. Worse, the dead branch filtered `opened_at`, so had
    # it run an "exit cooldown" would have blocked a CLOSE because a session was recently
    # OPENED -- the wrong column, a buggy duplicate of `exit_min_session_age_seconds` below.

    # The exit camera sees the back of someone who just walked in, so it refuses to close a
    # session this young. There is no escape hatch. The quick-return path that used to
    # bypass this guard (quick_return_enabled, quick_return_max_age_seconds) is DELETED: it
    # was reachable only through close(quick_return=True), and production
    # (worker/canteen.py::_on_exit) never passed it. Telling a genuine "walked in, saw the
    # queue, walked out" apart from the exit camera catching a just-entered child's face
    # needs a departure signal the pipeline does not produce; inventing a trigger would
    # corrupt records worse than not having the outcome. See enums.py::SessionOutcome.
    exit_min_session_age_seconds: float = Field(default=30.0, ge=0)

    # A session nobody ever exited is force-closed as unknown.
    max_session_minutes: float = Field(default=90.0, gt=0)

    # staff_presence_ttl_seconds DELETED -- nothing read it, so the "separate inside list
    # with a TTL" it described never existed. `SessionManager.open` takes `is_staff` and
    # simply opens no session (canteen/sessions.py); there is no list to expire.

    # A session may be opened for a face we could not identify.
    allow_unknown_sessions: bool = True


class MealOutcomeRules(Base):
    """ "Ate / did not eat" is decided purely by dwell time.

    In the legacy `ate_at_or_above_seconds` was hardcoded in the service while the
    not_eaten_seconds / eaten_minutes keys sat in the YAML doing nothing at all. Here it is
    real config.

    There was a third tier once: a `< left_immediately_below_seconds` "came in and left"
    band. It was unreachable and its knob is DELETED. The exit guard refuses any close
    younger than `exit_min_session_age_seconds` (30s), and the only bypass --
    close(quick_return=True) -- was never triggered in production, so classify() was only
    ever called with a dwell already past 30s and the band could not fire. See
    enums.py::SessionOutcome for the outcome and reason that went with it.
    """

    ate_at_or_above_seconds: float = Field(default=60.0, gt=0)

    def classify(self, dwell_seconds: float) -> str:
        if dwell_seconds < self.ate_at_or_above_seconds:
            return "not_ate"
        return "ate"


class EntrySettings(Base):
    """canteen_entry: the only camera that opens a session.

    `face_roi` used to live here. It is gone, with `max_faces_per_tick` and the exit's copy
    of these two keys: no code ever read it, and a declared key nothing reads is a lie the
    config file tells the next engineer to tune.
    """

    recognition: RecognitionPolicy = RecognitionPolicy()
    small_face: SoftAccumulator = SoftAccumulator(enabled=True)
    binding: BindingSettings = BindingSettings()

    # These two bound the UNKNOWN meal sessions, and only this camera has any (it is the
    # only one that opens a session at all). `SessionManager` dedups by person_id and an
    # Unknown session has none, so nothing downstream can tell two of them apart:
    #
    #   * a person box below `min_person_box_area` is a figure at the far end of the room,
    #     not a child at the door, and a meal record made from one is invented;
    #   * no second Unknown session within `person_cooldown_seconds` of the last from a
    #     track that is the SAME CHILD under a new track id -- which is what a long
    #     occlusion at a busy door makes ByteTrack produce.
    #
    # It is not a global cooldown: a second child in the queue keeps their session however
    # soon they step forward. See `qorgan.canteen.unknowns`, which states the rule and its
    # failure mode.
    person_cooldown_seconds: float = Field(default=5.0, ge=0)
    min_person_box_area: int = Field(default=3200, ge=1)


class ExitSettings(Base):
    """canteen_exit: the only camera that closes a session."""

    # min_score was 0.42 here, which is below the worst measured genuine impostor (0.472,
    # spec §2.2) -- it admitted a known confusion on the camera that CLOSES meal sessions.
    # The floor applies to every camera. min_gap stays wide because the exit camera looks
    # at the backs of heads and its faces are the worst in the building.
    #
    # Raising it means some sessions we cannot close get force-closed as unknown -- a
    # visible, countable hole. The alternative is a session closed on a false match, a
    # record that looks like real data and corrupts TWO children at once: the one wrongly
    # matched out, and the one who actually left and is left with an open session. We
    # choose the countable hole.
    #
    # `qorgan pupils report` counts these holes as `forced_unknown` (sessions closed by
    # CloseReason.TIMEOUT, i.e. the exit camera never recognised anybody). If that number
    # spikes, the threshold is too high, and we will SEE it rather than guess. 0.50 is a
    # measured floor, not a settled value -- see `config/identity.py::RecognitionPolicy` --
    # and it gets re-derived the day the school sends canteen footage of a named volunteer.
    recognition: RecognitionPolicy = RecognitionPolicy(min_score=0.50, min_gap=0.18)
    soft: SoftAccumulator = SoftAccumulator(enabled=True, window_seconds=8.0)
    binding: BindingSettings = BindingSettings()

    # How often to RE-ATTEMPT a close the session machine refused. `close()` is not
    # terminal: it refuses a session younger than `exit_min_session_age_seconds`, and that
    # refusal is "not yet", not "no". This is the cadence the old per-frame loop retried on,
    # and retrying is the difference between a session that closes and one that silently
    # force-closes as UNKNOWN 90 minutes later. It is NOT the tracking cadence -- ByteTrack
    # needs every frame, and gating it here is what split one child into two records.
    watch_interval_seconds: float = Field(default=0.25, gt=0)
    # watch_window_seconds DELETED -- never read. There is no bounded watch window: the
    # exit camera retries the close for as long as the track lives.
    # `max_faces_per_tick` used to live here, and so did `face_roi`, `person_cooldown_seconds`
    # and `min_person_box_area`. They are gone. The last two were copied from EntrySettings
    # and could never have meant anything here: they bound the creation of UNKNOWN meal
    # sessions, and the exit camera does not open sessions. A config key no code reads is
    # exactly what this project refuses to ship.


class InsideSettings(Base):
    """canteen_inside: confirms presence, may late-bind an identity. Never opens or closes."""

    recognition: RecognitionPolicy = RecognitionPolicy()
    binding: BindingSettings = BindingSettings()
    recognition_interval_seconds: float = Field(default=1.5, gt=0)
    # exit_missing_frames DELETED -- never read, and it could not have meant anything
    # here: an inside camera never closes a session. Track loss is ByteTrack's business
    # (`PairMetrics.tracker_max_lost`).


class CanteenConfig(Base):
    session: SessionRules = SessionRules()
    meal_outcome: MealOutcomeRules = MealOutcomeRules()
    face_model: FaceModelSettings = FaceModelSettings()

    # Exactly one of these is set, chosen by the camera's role. The camera model
    # enforces that; see config/camera.py.
    entry: EntrySettings | None = None
    exit: ExitSettings | None = None
    inside: InsideSettings | None = None
