# Identity Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the identity service described in `docs/superpowers/specs/2026-07-13-identity-service-design.md`. Identity stops being *inferred from a name* and starts being *read from the school's own IDs*. The name-based machinery (`generate_external_id`, `CONFUSABLES`, `Identity`, `Suspect`, `CollisionReport`, `find_namesakes`, `qorgan pupils check-namesakes`) is deleted. Recognition moves from once-per-face-per-frame to **once per person track**. The GPU guard stops being correct by accident. Six people who hold two school IDs each become visible, and mergeable.

**Architecture:**

```
src/qorgan/config/identity.py    FaceGate, RecognitionPolicy, SoftAccumulator,
                                 FaceModelSettings, BindingSettings  (moved out of canteen.py)
src/qorgan/config/canteen.py     SessionRules, MealOutcomeRules + the canteen camera ROLE
                                 blocks, which now COMPOSE the models above.
                                 canteen.py imports identity.py; identity.py imports nothing
                                 from canteen.py — that one-way edge is the whole point of
                                 the move (a hall camera can import identity without
                                 dragging in the canteen).

src/qorgan/identity/naming.py    pure: normalise_class, is_class_folder, display_name
src/qorgan/identity/roster.py    pure: RosterEntry, BadFilename, folder_role, external_id_for
src/qorgan/identity/report.py    pure: the cross-person matrix, duplicates, extrapolation
src/qorgan/identity/merge.py     impure: re-point photos/embeddings/sessions, deactivate
src/qorgan/identity/camera.py    the face-size diagnostic
src/qorgan/identity/tracks.py    pure: assign_faces_to_tracks(faces, person_boxes)
src/qorgan/identity/binding.py   pure: the bind / retry / evict state machine.
                                 No GPU, no clock, no DB.
src/qorgan/identity/service.py   the impure shell: recognizer + gallery + bindings
src/qorgan/identity/cli.py       `qorgan identity camera-report`

src/qorgan/planning/costs.py     pure: the VRAM cost model and the grouping
src/qorgan/planning/measure.py   impure: child processes, nvidia-smi
src/qorgan/planning/cli.py       `qorgan plan-workers`

src/qorgan/faces/identity.py     DELETED
scripts/vram_spike.py            DELETED (becomes `qorgan plan-workers`)
```

`FaceRecognizer` splits into `detect_faces()` (cheap: boxes, landmarks, det_score) and
`embed(image, face)` (expensive: the 512-d vector). `CanteenPipeline` gains a
`PersonDetector` (YOLOv8n + ByteTrack), assigns faces to person tracks, keeps only the best
face per track, and embeds **once per track**.

**Tech Stack:** Python 3.11, pydantic 2.13 (`extra="forbid"`), SQLAlchemy 2.0 + Alembic,
numpy 2.4, InsightFace `buffalo_l` on onnxruntime-gpu 1.26 (CUDA), Ultralytics YOLOv8n +
ByteTrack, pytest 8.3, ruff 0.9.6.

---

## Global Constraints

These hold for **every** task in this plan. A step that breaks one of them is not done.

- **no file >500 lines; no function >50 lines.** Enforced by `tests/test_code_limits.py`
  (rule R1). Split before you exceed it, not after.
- **no secret outside env vars.** Enforced by `tests/test_no_secrets.py` (rule R4).
- **no absolute path in the DB (use the `RelPath` column type).** Rule R6, enforced by
  `qorgan.db.types.RelPath`.
- **every web endpoint authenticated.** No new endpoint in this plan; if you add one, it
  carries the existing dependency.
- **`extra="forbid"` on all config models.** They all inherit `qorgan.config.common.Base`,
  which sets it. Never override `model_config`.
- **baseline is 757 tests passing and `ruff check .` clean and it must stay that way.**
  Verified before you start:
  `.venv/Scripts/python.exe -m pytest -q` → `757 passed`
  `.venv/Scripts/python.exe -m ruff check .` → `All checks passed!`
  Every task ends green. A task that leaves the suite red is not finished.
- **the venv python is `.venv/Scripts/python.exe`** (run pytest as
  `.venv/Scripts/python.exe -m pytest`).
- **NEVER `git add -A` and NEVER commit anything under `student_photos/`,
  `original_student_photos/`, or `eval/clips/` — always check `git status` first.**
  Those directories hold 142 photographs of real children and 663 clips of a real school.
  Every commit in this plan names its paths explicitly. Run `git status` before each one.

Two more, specific to this plan:

- **The filename prefix lies.** The `учитель` folder contains files named
  `student_469_….jpg`. Person type comes from the **FOLDER**, never from the filename.
- **No silent fallback, ever.** A filename that does not match
  `^(student|staff)_(\d+)_(\d+)\.(jpg|jpeg|png)$` is a **hard error naming the file**. It
  is never a guessed identity.

---

### Task 1: Harden the GPU guard

Spec §3. **Production works today.** `gpu.py` imports torch before onnxruntime, which loads
the CUDA DLLs into the process, and InsightFace then resolves them. This task is hardening
against an import reorder, not a rescue.

`ort.get_available_providers()` reports what onnxruntime was *compiled* with. It returns
`CUDAExecutionProvider` even in a process where the provider DLL cannot load and every
session silently falls back to the CPU. The guard must **build a real ONNX session and read
back `session.get_providers()[0]`**. That is true regardless of import order, and it is the
only check that cannot be fooled.

**Files:**
- Modify: `src/qorgan/gpu.py`
- Modify: `pyproject.toml` (declare `onnx`, which `gpu.py` now imports directly)
- Test: `tests/test_gpu.py`

**Interfaces:**

*Consumes:* nothing new.

*Produces:*
```python
# src/qorgan/gpu.py
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

@dataclass(frozen=True, slots=True)
class GpuReport:
    torch_version: str
    torch_cuda: bool
    device_name: str | None
    onnx_providers: tuple[str, ...]        # what ORT was COMPILED with. Diagnostic text only.
    onnx_session_provider: str | None      # what a REAL session actually ran on. The truth.

    @property
    def onnx_cuda(self) -> bool: ...       # == (onnx_session_provider == CUDA_PROVIDER)
    @property
    def ok(self) -> bool: ...
    def problems(self) -> list[str]: ...

def inspect_gpu() -> GpuReport: ...
def require_gpu() -> GpuReport: ...        # unchanged signature
```

Note the breaking change: `GpuReport` gains a **required** field `onnx_session_provider`.
Every construction site must pass it. There are two: `inspect_gpu()` and the `_report()`
helper in `tests/test_gpu.py`.

**Steps:**

- [ ] **Step 1: Write the failing tests.**

Replace the `_report` helper and add three tests in `tests/test_gpu.py`. Keep the existing
tests; only `_report` changes shape.

```python
def _report(
    *,
    torch_cuda: bool,
    providers: tuple[str, ...],
    session_provider: str | None = CUDA_PROVIDER,
) -> GpuReport:
    return GpuReport(
        torch_version="2.11.0+cu128",
        torch_cuda=torch_cuda,
        device_name="fake" if torch_cuda else None,
        onnx_providers=providers,
        onnx_session_provider=session_provider,
    )
```

and append, at the end of the file:

```python
# -- the guard must not believe get_available_providers() --------------------


def test_a_compiled_in_provider_that_no_session_can_use_is_not_ok() -> None:
    """THE hardening. `ort.get_available_providers()` reports what onnxruntime was
    COMPILED with, not what a session can actually run on. It returns
    CUDAExecutionProvider even in a process where the provider DLL cannot load and every
    session silently falls back to the CPU — I demonstrated exactly that (spec §3).

    A guard that reads that list says ok=True while the whole system runs 40x too slow.
    """
    report = _report(
        torch_cuda=True,
        providers=(CUDA_PROVIDER, CPU_PROVIDER),  # the lie
        session_provider=CPU_PROVIDER,  # the truth
    )

    assert not report.onnx_cuda, "the guard believed get_available_providers()"
    assert not report.ok
    assert any("fell back to" in problem for problem in report.problems())


def test_a_session_that_reaches_cuda_is_ok_whatever_the_compiled_list_says() -> None:
    report = _report(torch_cuda=True, providers=(), session_provider=CUDA_PROVIDER)

    assert report.onnx_cuda
    assert report.ok


def test_a_session_that_could_not_be_built_at_all_is_not_ok() -> None:
    """onnxruntime raising is a failure, not an unknown. Refuse."""
    report = _report(torch_cuda=True, providers=(CUDA_PROVIDER,), session_provider=None)

    assert not report.ok


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA device on this machine")
def test_a_real_onnx_session_on_this_machine_runs_on_cuda() -> None:
    """The guard builds a session and asserts session.get_providers()[0] is CUDA.
    get_available_providers() is not consulted, because it lies (spec §3, §6)."""
    report = inspect_gpu()

    assert report.onnx_session_provider == CUDA_PROVIDER, (
        "a real onnxruntime session did not land on CUDA. InsightFace would run on the "
        f"CPU, ~40x too slow. The session reported: {report.onnx_session_provider}"
    )
```

and extend the import line at the top of the file:

```python
from qorgan.gpu import CPU_PROVIDER, CUDA_PROVIDER, GpuReport, inspect_gpu, require_gpu
```

- [ ] **Step 2: Run it, watch it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_gpu.py -q
```

Expected failure — collection error, because neither `CPU_PROVIDER` nor the
`onnx_session_provider` field exists yet:

```
ImportError: cannot import name 'CPU_PROVIDER' from 'qorgan.gpu'
```

- [ ] **Step 3: Implement.**

Rewrite `src/qorgan/gpu.py` from the `CUDA_PROVIDER` constant downwards. The module
docstring stays; append a paragraph to it explaining the session probe.

Append to the module docstring, before the closing `"""`:

```
And the guard that checks all this must not itself be a lie. `get_available_providers()`
reports what onnxruntime was COMPILED with — it says CUDAExecutionProvider even in a
process where the provider DLL cannot load and every session falls back to the CPU. So we
build a real session and read back what it actually got. That is the only check that
cannot be fooled by an import reorder.
```

Then:

```python
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True, slots=True)
class GpuReport:
    torch_version: str
    torch_cuda: bool
    device_name: str | None
    # What onnxruntime was COMPILED with. Diagnostic text only — never a decision.
    onnx_providers: tuple[str, ...]
    # What a REAL session actually ran on. This is the one that cannot be fooled.
    onnx_session_provider: str | None

    @property
    def onnx_cuda(self) -> bool:
        """Did a real session land on CUDA?

        NOT `CUDA_PROVIDER in self.onnx_providers`. That list is what onnxruntime was
        built with, and it says CUDA even when every session silently runs on the CPU.
        """
        return self.onnx_session_provider == CUDA_PROVIDER

    @property
    def ok(self) -> bool:
        return self.torch_cuda and self.onnx_cuda

    def problems(self) -> list[str]:
        issues = []
        if not self.torch_cuda:
            issues.append(
                f"torch {self.torch_version} cannot see a CUDA device. Install the CUDA build: "
                'pip install -e ".[ai]" --extra-index-url https://download.pytorch.org/whl/cu128'
            )
        if not self.onnx_cuda:
            issues.append(
                f"onnxruntime built a session and it fell back to "
                f"{self.onnx_session_provider or 'nothing — the session would not build'}, "
                f"not {CUDA_PROVIDER}. It advertises: {', '.join(self.onnx_providers) or 'none'} "
                "— but advertising is not running. Usually the CPU-only `onnxruntime` package "
                "is installed alongside `onnxruntime-gpu` and shadows it, or the CUDA DLLs are "
                "not in the process (import torch BEFORE onnxruntime). Fix: "
                "pip uninstall -y onnxruntime && "
                "pip install --force-reinstall --no-deps onnxruntime-gpu==1.26.0"
            )
        return issues


def inspect_gpu() -> GpuReport:
    # This import order is load-bearing, not alphabetical: importing torch first
    # loads the CUDA runtime DLLs into the process, and on Windows that is what lets
    # onnxruntime find them afterwards. Sorting these two lines breaks the GPU --
    # and _probe_session() below is what NOTICES when someone does.
    import torch  # noqa: I001
    import onnxruntime as ort

    available = torch.cuda.is_available()
    return GpuReport(
        torch_version=torch.__version__,
        torch_cuda=available,
        device_name=torch.cuda.get_device_name(0) if available else None,
        onnx_providers=tuple(ort.get_available_providers()),
        onnx_session_provider=_probe_session(),
    )


def _probe_session() -> str | None:
    """Build a real ONNX session and report the provider it actually got.

    `get_available_providers()` cannot tell you this: it lists what the wheel was compiled
    with, and returns CUDAExecutionProvider from a process where the provider DLL will not
    load and every session falls back to the CPU. Only a session knows.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # the CPU fallback is a warning; we report it ourselves

    try:
        session = ort.InferenceSession(
            _probe_model_bytes(),
            options,
            providers=[CUDA_PROVIDER, CPU_PROVIDER],
        )
        providers = session.get_providers()
    except Exception:
        logger.exception("onnxruntime could not build a session at all")
        return None

    return providers[0] if providers else None


def _probe_model_bytes() -> bytes:
    """The smallest ONNX graph that is still a real one: relu on a single float."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "qorgan_gpu_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString()


def require_gpu() -> GpuReport:
    """Called by a worker before it loads a model. Refuses to run 40x too slow."""
    report = inspect_gpu()
    if not report.ok:
        for problem in report.problems():
            logger.error("GPU check failed: %s", problem)
        raise RuntimeError("GPU not usable; refusing to start. See the log for how to fix it.")

    logger.info(
        "GPU ready",
        extra={
            "device": report.device_name,
            "torch": report.torch_version,
            "onnx_session": report.onnx_session_provider,
        },
    )
    return report
```

Then update `src/qorgan/cli.py::_cmd_doctor` so `qorgan doctor` prints the truth rather
than the advertisement. Replace the body between `report = inspect_gpu()` and
`if report.ok:` with:

```python
    report = inspect_gpu()
    print(f"torch:            {report.torch_version}")
    print(f"CUDA device:      {report.device_name or 'NONE'}")
    print(f"onnx advertises:  {', '.join(report.onnx_providers) or 'none'}")
    # The line that matters. The one above is what the wheel was COMPILED with, and it
    # says CUDA even when every session silently runs on the CPU (spec §3).
    print(f"onnx SESSION ran: {report.onnx_session_provider or 'FAILED TO BUILD'}")
```

Finally, declare the dependency `gpu.py` now imports directly. In `pyproject.toml`, inside
the `ai = [` list, add after `"onnxruntime-gpu==1.26.0",`:

```toml
    # gpu.py builds a real ONNX session to prove CUDA is reachable (spec §3). insightface
    # already pulls this in; we import it, so we pin it.
    "onnx==1.22.0",
```

- [ ] **Step 4: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_gpu.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `tests/test_gpu.py` all pass (including the GPU-gated one on this machine);
the full suite is `760 passed` (757 + 3 new non-GPU tests; the 4th is GPU-gated and also
runs here, so on this box expect `761 passed`).

- [ ] **Step 5: Commit.**

```
git status
git add src/qorgan/gpu.py src/qorgan/cli.py pyproject.toml tests/test_gpu.py
git commit -m "Harden the GPU guard: build a real session, do not believe the advertisement

get_available_providers() reports what onnxruntime was COMPILED with. It returns
CUDAExecutionProvider from a process where the provider DLL cannot load and every session
silently falls back to the CPU -- demonstrated. So the guard was load-bearing on an
incidental import order and could not detect its own failure.

GpuReport.onnx_cuda now reads back session.get_providers()[0] from a real ONNX session.
That is true regardless of import order, and it is the only check that cannot be fooled.
Production was on the GPU all along; this closes the gap between 'the guard passes' and
'the GPU is actually being used'."
```

---

### Task 2: Move the face-recognition config into `config/identity.py`

Spec §5. `config/canteen.py` is about **meals**. Face recognition is not about meals — a
hall camera needs `RecognitionPolicy` and must not have to import the canteen to get it.
Move the five models out; add `BindingSettings` for the per-track binding of Task 9.

And set the measured floor: **`RecognitionPolicy.min_score = 0.50`** (spec §2.2, §2.3).
0.45 sits *below* the worst genuine impostor at 0.472 — margin **−0.022** — so today's
default admits a known confusion. 0.50 is inside the empty band. It ships as a **floor**
with the ceiling not yet settled, and the config says so in words.

**Files:**
- Create: `src/qorgan/config/identity.py`
- Modify: `src/qorgan/config/canteen.py`
- Modify: `src/qorgan/faces/recognizer.py`, `src/qorgan/faces/matching.py`,
  `src/qorgan/faces/accumulator.py`, `src/qorgan/faces/importer.py`,
  `src/qorgan/faces/cli.py`, `src/qorgan/worker/canteen.py` (the 6 src importers)
- Test: `tests/test_config_schema.py`, `tests/test_faces_matching.py`,
  `tests/test_faces_gallery.py`, `tests/test_faces_import.py` (the 4 test importers)

**Interfaces:**

*Consumes:* `qorgan.config.common.Base` (which sets `extra="forbid"`, `frozen=True`).

*Produces:*
```python
# src/qorgan/config/identity.py
class FaceGate(Base):
    min_width: int = 60
    min_height: int = 70
    min_area: int = 4200
    def accepts(self, width: int, height: int) -> bool: ...

class RecognitionPolicy(Base):
    min_score: float = 0.50          # MEASURED FLOOR. See the docstring.
    min_gap: float = 0.05
    single_candidate_gap: float = 0.0
    face_gate: FaceGate = FaceGate()

class SoftAccumulator(Base):
    enabled: bool = False
    min_score: float = 0.34
    min_gap: float = 0.12
    min_hits: int = 2
    window_seconds: float = 6.0
    face_gate: FaceGate = FaceGate(min_width=38, min_height=48, min_area=1800)

class FaceModelSettings(Base):
    model_name: str = "buffalo_l"
    model_version: str = "1.0"
    det_size: int = 640
    embedding_dim: int = 512
    normalized: bool = True

class BindingSettings(Base):        # NEW — consumed by Task 9
    min_face_frames: int = 3
    max_wait_seconds: float = 1.5
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    track_ttl_seconds: float = 3.0
```
`config/canteen.py` keeps `SessionRules`, `MealOutcomeRules`, and the canteen **camera role
blocks** (`EntrySettings`, `ExitSettings`, `InsideSettings`, `CanteenConfig`) — which now
*compose* the models above by importing them. The dependency edge runs
`canteen.py → identity.py` and never back, which is the whole point of the move.

**Steps:**

- [ ] **Step 1: Write the failing tests.**

In `tests/test_config_schema.py`, change the import line

```python
from qorgan.config.canteen import MealOutcomeRules, SoftAccumulator
```

to

```python
from qorgan.config.canteen import MealOutcomeRules
from qorgan.config.identity import BindingSettings, RecognitionPolicy, SoftAccumulator
```

and append these tests at the end of the file:

```python
# -- the identity config, and the measured floor under min_score ---------------


def test_the_default_min_score_sits_above_the_worst_measured_impostor() -> None:
    """138 real pupils, 9 447 genuine impostor pairs: the WORST scores 0.472.

    The old default was 0.45, which is BELOW that -- margin -0.022. It admitted a known
    confusion. 0.50 sits inside the band that is empty from 0.48 to 0.77 (spec §2.2).
    """
    worst_measured_impostor = 0.472

    assert RecognitionPolicy().min_score > worst_measured_impostor


def test_the_binding_settings_default_to_something_a_queue_can_live_with() -> None:
    """Five children queuing over ten seconds must cost 5 embeddings, not ~200."""
    binding = BindingSettings()

    assert binding.min_face_frames >= 2, "one frame is not corroboration, it is a glance"
    assert binding.max_wait_seconds > 0
    assert binding.max_attempts >= 1


def test_the_identity_config_does_not_drag_in_the_canteen() -> None:
    """THE point of the move. A hall camera needs RecognitionPolicy and must not have to
    import a module about meal sessions to get it (spec §5)."""
    import ast
    from pathlib import Path

    from tests.conftest import SRC_DIR

    source = (SRC_DIR / "qorgan" / "config" / "identity.py").read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "qorgan.config.canteen" not in imported
    assert isinstance(Path(SRC_DIR), Path)
```

- [ ] **Step 2: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_config_schema.py -q
```

Expected failure — collection error:

```
ModuleNotFoundError: No module named 'qorgan.config.identity'
```

- [ ] **Step 3: Implement — create `src/qorgan/config/identity.py`.**

```python
"""Face recognition config. **Not about meals**, so it does not live in `canteen.py`.

A hall camera needs a `RecognitionPolicy`. It must not have to import a module about meal
sessions to get one — that coupling is what made `worker/canteen.py` the only place in the
system that could recognise a face at all.

Every number here is recorded with the measurement beside it. The legacy shipped
`SAME_PERSON_SIMILARITY = 0.35`; against this school's data that constant would call 55
different pairs of children the same person. The measured band says 0.60. Guessing is what
cost the legacy eighteen thresholds and 1 816 NULL canteen records.
"""

from __future__ import annotations

from pydantic import Field

from qorgan.config.common import Base


class FaceGate(Base):
    """Minimum face size before we will even attempt to bind a recognition to a track.

    **0 of 14 970 measured hall faces clear this gate** (spec §2.4). NOT 2.2% -- that figure
    is measured on the 2560x1440 HD burst, which is not the stream the worker analyses.

    There is no single analysis resolution to quote: it is PER PROFILE. `hall.yaml` and
    `canteen_entry.yaml` override `capture.frame_width` to **1280x720**; everything else
    inherits 960x540 from `base.yaml`. At the hall's real 1280x720 the median face is
    **11.5 px** and the largest in the whole corpus is **50 px** -- the strict 60 px gate
    needs a 120 px face in the clip, and no such face exists.

    That is a **camera-placement fact**, and no threshold anywhere is a substitute for it.
    And beware the trap: a plausible-but-wrong "2.2%" invites "low but non-zero -- drop
    min_width to 40 and recover some". It recovers NOTHING. Lowering to the 38 px small-face
    gate lets **77 of 14 970** faces through -- and **not one of them is recognised**: the
    best score among all 77 is 0.350, against a min_score of 0.45. Run
    `qorgan identity camera-report` before you touch a number in this file: it reads the
    resolution from the CAMERA'S OWN config rather than assuming one.
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
    all. Probed against 14 970 real faces in 250 hall clips, scores reach 0.604 -- but only
    on the HD burst. At the hall's real analysis resolution (1280x720 -- `hall.yaml`
    overrides `base.yaml`'s 960x540 default), **0 of those 14 970 faces clear the strict
    60 px gate** and, of the 77 that clear the 38 px small-face gate, **none is accepted at
    any threshold**: the best score among them is 0.350. Recognition on a
    hall camera is arithmetically impossible, not merely poor (spec §2.4). The hall
    therefore says nothing either way about the CANTEEN, whose entry camera is close-range:
    it fails to prove success, it does not prove failure.

    What closes it: footage from the CANTEEN ENTRY camera of pupils we can name. One
    volunteer walking through. Until then this is a floor, and it must not be written up
    as a settled number.

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
    face_gate: FaceGate = FaceGate()


class SoftAccumulator(Base):
    """Accept a lower-scoring match if the SAME person comes top-1 repeatedly.

    This is the "small face" path, and it is real domain knowledge, not a hack:
    younger pupils' faces are systematically below the size gate, so a strict
    single-shot threshold simply never recognises the first-graders. Keep it.

    One model, reused for the entry small-face path and the exit soft path -- the
    legacy wrote this same logic out four times with four sets of key names.

    Measured (spec §2.5): this is also where merging duplicate enrolments bites. On the
    small-face path at >=38 px, merging takes accepts from 3 to 9 and gap-kills from 6 to
    0. At the 60 px gate the footage is too sparse to show any effect at all.
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
```

- [ ] **Step 4: Implement — gut `src/qorgan/config/canteen.py`.**

Replace the module docstring and delete `FaceGate`, `RecognitionPolicy`, `SoftAccumulator`
and `FaceModelSettings` from it, importing them instead. The file becomes, in full:

```python
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

from pydantic import Field, model_validator

from qorgan.config.common import Base, ZoneRect
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
    exit_cooldown_seconds: float = Field(default=60.0, ge=0)

    # The exit camera sees the back of someone who just walked in, so it refuses to
    # close a session this young -- except via the explicit quick-return path.
    exit_min_session_age_seconds: float = Field(default=30.0, ge=0)
    quick_return_enabled: bool = True
    quick_return_max_age_seconds: float = Field(default=30.0, ge=0)

    # A session nobody ever exited is force-closed as unknown.
    max_session_minutes: float = Field(default=90.0, gt=0)

    # Staff never create meal sessions; they go on a separate "inside" list with a TTL.
    staff_presence_ttl_seconds: float = Field(default=120.0, gt=0)

    # A session may be opened for a face we could not identify.
    allow_unknown_sessions: bool = True


class MealOutcomeRules(Base):
    """ "Ate / did not eat" is decided purely by dwell time.

    In the legacy these two numbers were hardcoded in the service while the
    not_eaten_seconds / eaten_minutes keys sat in the YAML doing nothing at all.
    Here they are real config -- and the ladder is validated.
    """

    left_immediately_below_seconds: float = Field(default=20.0, gt=0)
    ate_at_or_above_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> MealOutcomeRules:
        if self.left_immediately_below_seconds >= self.ate_at_or_above_seconds:
            raise ValueError(
                "meal_outcome: left_immediately_below_seconds must be < ate_at_or_above_seconds"
            )
        return self

    def classify(self, dwell_seconds: float) -> str:
        if dwell_seconds < self.left_immediately_below_seconds:
            return "left_immediately"
        if dwell_seconds < self.ate_at_or_above_seconds:
            return "not_ate"
        return "ate"


class EntrySettings(Base):
    """canteen_entry: the only camera that opens a session."""

    face_roi: ZoneRect = ZoneRect(x1=0.20, y1=0.15, x2=0.90, y2=0.92)
    recognition: RecognitionPolicy = RecognitionPolicy()
    small_face: SoftAccumulator = SoftAccumulator(enabled=True)
    binding: BindingSettings = BindingSettings()
    person_cooldown_seconds: float = Field(default=5.0, ge=0)
    min_person_box_area: int = Field(default=3200, ge=1)


class ExitSettings(Base):
    """canteen_exit: the only camera that closes a session."""

    face_roi: ZoneRect = ZoneRect(x1=0.03, y1=0.06, x2=0.98, y2=0.84)
    # min_score was 0.42 here, which is below the worst measured genuine impostor (0.472,
    # spec §2.2) -- it admitted a known confusion on the camera that CLOSES meal sessions.
    # The floor applies to every camera. min_gap stays wide because the exit camera looks
    # at the backs of heads and its faces are the worst in the building.
    recognition: RecognitionPolicy = RecognitionPolicy(min_score=0.50, min_gap=0.18)
    soft: SoftAccumulator = SoftAccumulator(enabled=True, window_seconds=8.0)
    binding: BindingSettings = BindingSettings()
    watch_interval_seconds: float = Field(default=0.25, gt=0)
    watch_window_seconds: float = Field(default=20.0, gt=0)
    max_faces_per_tick: int = Field(default=4, ge=1)
    person_cooldown_seconds: float = Field(default=5.0, ge=0)
    min_person_box_area: int = Field(default=3200, ge=1)


class InsideSettings(Base):
    """canteen_inside: confirms presence, may late-bind an identity. Never opens or closes."""

    recognition: RecognitionPolicy = RecognitionPolicy()
    binding: BindingSettings = BindingSettings()
    recognition_interval_seconds: float = Field(default=1.5, gt=0)
    exit_missing_frames: int = Field(default=20, ge=1)


class CanteenConfig(Base):
    session: SessionRules = SessionRules()
    meal_outcome: MealOutcomeRules = MealOutcomeRules()
    face_model: FaceModelSettings = FaceModelSettings()

    # Exactly one of these is set, chosen by the camera's role. The camera model
    # enforces that; see config/camera.py.
    entry: EntrySettings | None = None
    exit: ExitSettings | None = None
    inside: InsideSettings | None = None
```

- [ ] **Step 5: Implement — repoint the six src importers.**

Exactly six lines change. Each is a one-line import edit; no other code moves.

| file | old | new |
|---|---|---|
| `src/qorgan/faces/recognizer.py:20` | `from qorgan.config.canteen import FaceModelSettings` | `from qorgan.config.identity import FaceModelSettings` |
| `src/qorgan/faces/matching.py:42` | `from qorgan.config.canteen import RecognitionPolicy` | `from qorgan.config.identity import RecognitionPolicy` |
| `src/qorgan/faces/accumulator.py:24` | `from qorgan.config.canteen import SoftAccumulator as SoftConfig` | `from qorgan.config.identity import SoftAccumulator as SoftConfig` |
| `src/qorgan/faces/importer.py:32` | `from qorgan.config.canteen import FaceModelSettings` | `from qorgan.config.identity import FaceModelSettings` |
| `src/qorgan/faces/cli.py:9` | `from qorgan.config.canteen import FaceModelSettings` | `from qorgan.config.identity import FaceModelSettings` |
| `src/qorgan/worker/canteen.py:23` | `from qorgan.config.canteen import RecognitionPolicy, SoftAccumulator` | `from qorgan.config.identity import RecognitionPolicy, SoftAccumulator` |

`src/qorgan/config/camera.py` imports `CanteenConfig` from `config.canteen` — unchanged.

- [ ] **Step 6: Implement — repoint the four test importers.**

| file | old | new |
|---|---|---|
| `tests/test_faces_matching.py:8` | `from qorgan.config.canteen import FaceGate, RecognitionPolicy, SoftAccumulator` | `from qorgan.config.identity import FaceGate, RecognitionPolicy, SoftAccumulator` |
| `tests/test_faces_gallery.py:8` | `from qorgan.config.canteen import FaceModelSettings, RecognitionPolicy` | `from qorgan.config.identity import FaceModelSettings, RecognitionPolicy` |
| `tests/test_faces_import.py:12` | `from qorgan.config.canteen import FaceModelSettings` | `from qorgan.config.identity import FaceModelSettings` |
| `tests/test_config_schema.py:10` | `from qorgan.config.canteen import MealOutcomeRules, SoftAccumulator` | already done in Step 1 |

`tests/test_canteen_worker.py:17` and `tests/test_canteen_sessions.py:12` import
`MealOutcomeRules, SessionRules` — those stay in `config.canteen`. Do not touch them.

- [ ] **Step 7: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_config_schema.py tests/test_faces_matching.py tests/test_faces_gallery.py tests/test_faces_import.py tests/test_canteen_worker.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green. `tests/test_faces_matching.py` pins `min_score=0.45` explicitly in its own
`POLICY`, so the default change does not touch it; every other default-policy test matches
an exact gallery vector and scores ~1.0.

- [ ] **Step 8: Commit.**

```
git status
git add src/qorgan/config/identity.py src/qorgan/config/canteen.py src/qorgan/faces/recognizer.py src/qorgan/faces/matching.py src/qorgan/faces/accumulator.py src/qorgan/faces/importer.py src/qorgan/faces/cli.py src/qorgan/worker/canteen.py tests/test_config_schema.py tests/test_faces_matching.py tests/test_faces_gallery.py tests/test_faces_import.py
git commit -m "Face recognition config moves out of the canteen, and min_score gets its floor

A hall camera needs a RecognitionPolicy and should not have to import a module about meal
sessions to get one. FaceGate, RecognitionPolicy, SoftAccumulator and FaceModelSettings
move to config/identity.py, joined by a new BindingSettings. canteen.py imports identity;
identity never imports canteen.

min_score: 0.45 -> 0.50. Measured on the school's own 142 photographs: with the six
duplicate enrolments removed, the worst genuine impostor pair of 9 447 scores 0.472. The
old default sat BELOW that -- margin -0.022 -- and admitted a known confusion. 0.50 is
inside the band that is empty from 0.48 to 0.77.

It is a FLOOR. The ceiling -- whether a real camera face can reach 0.50 at all -- is not
yet measured, and the docstring says so in words rather than pretending otherwise."
```

---

### Task 3: Delete the name-based identity machinery

> **MERGED WITH TASK 4 — execute Tasks 3 and 4 as ONE task, with ONE commit.**
>
> As written, Task 3 deletes `faces/identity.py` and `ExternalIdSource.GENERATED` while the
> DB migration that matches those models lands only in Task 4 — so Task 3 alone leaves
> `test_the_migration_matches_the_models` RED. That contradicts this plan's own Global
> Constraint ("Every task ends green. A task that leaves the suite red is not finished").
>
> The Python deletion and the migration are one atomic change: you cannot review or reject
> one without the other. So do Task 3's steps AND Task 4's steps, run the full suite, and
> commit once, green. Ignore Task 3's "Step 10: the suite is RED by design" — it is not.


Spec §1. The school sent 142 photographs named `student_333_1778595343147.jpg`, and `333`
is a **real school ID**, unique across all 142. Identity is *given*, not inferred.

So everything built to infer an identity from a name goes: `generate_external_id`,
`CONFUSABLES`, `normalise_name`, `Identity`, `Suspect`, `CollisionReport`,
`SAME_PERSON_SIMILARITY`, `find_namesakes`, and `qorgan pupils check-namesakes`.
`normalise_class` stays — a class folder still comes in three spellings.

`ExternalIdSource.GENERATED` goes with them: after this task and Task 5, nothing in the
system produces a generated id. The DB side of that removal is Task 4's migration; this
task removes the Python member and repoints the model default.

Nothing tests `import_archive` / `_import_photo` / `_store` today (they need a real GPU
recognizer), so they are deleted here rather than half-migrated. Task 5 rebuilds the
import on top of the roster parser this task introduces.

**Files:**
- Create: `src/qorgan/identity/__init__.py`, `src/qorgan/identity/naming.py`,
  `src/qorgan/identity/roster.py`
- Delete: `src/qorgan/faces/identity.py`
- Modify: `src/qorgan/faces/importer.py` (gut it back to `safe_extract` + `ImportReport`),
  `src/qorgan/faces/cli.py` (drop `check-namesakes` and the ZIP `import` command),
  `src/qorgan/enums.py` (drop `ExternalIdSource.GENERATED`),
  `src/qorgan/db/models/person.py` (default `external_id_source` to `ROSTER`),
  `pyproject.toml` (the RUF001 justification points at a file that will not exist),
  `README.md`, `HANDOFF.md` (they document `check-namesakes`)
- Test: Create `tests/test_identity_naming.py`, `tests/test_identity_roster.py`;
  modify `tests/test_faces_import.py` (strip the identity and namesake tests)

**Interfaces:**

*Consumes:* `qorgan.enums.PersonType`.

*Produces:*
```python
# src/qorgan/identity/naming.py
def normalise_class(raw: str) -> str: ...
    # "5А" == "5-А" == "5 а" -> "5А". Used to RECOGNISE a class folder, never to store one.
def is_class_folder(name: str) -> bool: ...
    # True for "1-А" .. "11-Б"

# src/qorgan/identity/roster.py
FILENAME = re.compile(r"^(student|staff)_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

class BadFilename(ValueError): ...

@dataclass(frozen=True, slots=True)
class RosterEntry:
    external_id: str          # "student_333" -- the matched prefix + id, verbatim
    person_type: PersonType
    class_name: str | None    # "5-А", the FOLDER NAME verbatim (not normalised)
    position: str | None      # "учитель", or None

def folder_role(folder: str) -> tuple[PersonType, str | None, str | None]: ...
    # raises ValueError naming the folder
def external_id_for(filename: str) -> str: ...
    # raises BadFilename naming the file
def entry_for(folder: str, filename: str) -> RosterEntry: ...

# src/qorgan/faces/importer.py  (what SURVIVES this task)
IMAGE_SUFFIXES: set[str]
MIN_FACE_PX: int = 60

@dataclass(frozen=True, slots=True)
class Unenrollable:
    photo: str      # the file name
    reason: str
    faces: int

@dataclass
class ImportReport:
    people: int = 0
    photos: int = 0
    embeddings: int = 0
    unenrollable: list[Unenrollable] = field(default_factory=list)
    def cannot_enrol(self, photo: str, reason: str, faces: int) -> None: ...
    def summary(self) -> str: ...

def safe_extract(archive: Path, destination: Path) -> list[Path]: ...
```

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_identity_naming.py`.**

```python
"""Class names, and the fact that the school's ID is the identity."""

from __future__ import annotations

import pytest

from qorgan.identity.naming import is_class_folder, normalise_class


def test_class_spellings_are_unified() -> None:
    """`5А`, `5-А` and `5 а` are one class. The legacy stored all three."""
    assert normalise_class("5А") == normalise_class("5-А") == normalise_class("5 а")


def test_a_normalised_class_is_upper_case_and_has_no_separators() -> None:
    assert normalise_class(" 11 - б ") == "11Б"


@pytest.mark.parametrize("folder", ["1-А", "1-Б", "1-В", "5-А", "7-А", "11-А", "11-Б", "10-А"])
def test_every_class_folder_the_school_sent_is_recognised(folder: str) -> None:
    assert is_class_folder(folder)


@pytest.mark.parametrize("folder", ["staff", "учитель", "", "12-А", "0-А", "photos", "5"])
def test_a_folder_that_is_not_a_class_is_not_a_class(folder: str) -> None:
    assert not is_class_folder(folder)
```

- [ ] **Step 2: Write the failing tests — `tests/test_identity_roster.py`.**

```python
"""The roster: the FOLDER decides who someone is. The filename only carries their id.

Two traps in this data, both found by looking rather than assuming.
"""

from __future__ import annotations

import pytest

from qorgan.enums import PersonType
from qorgan.identity.roster import BadFilename, RosterEntry, entry_for, external_id_for, folder_role


# -- the folder decides -------------------------------------------------------


def test_a_class_folder_is_a_pupil_and_keeps_its_name_verbatim() -> None:
    """The class is stored as the school wrote it -- `5-А`, with the hyphen -- because it
    is shown to a human: `Ученик 333, 5-А`."""
    person_type, class_name, position = folder_role("5-А")

    assert person_type is PersonType.STUDENT
    assert class_name == "5-А"
    assert position is None


def test_the_staff_folder_is_staff() -> None:
    person_type, class_name, position = folder_role("staff")

    assert person_type is PersonType.STAFF
    assert class_name is None
    assert position is None


def test_the_teacher_folder_is_staff_with_a_position() -> None:
    person_type, class_name, position = folder_role("учитель")

    assert person_type is PersonType.STAFF
    assert class_name is None
    assert position == "учитель"


def test_an_unknown_folder_is_a_hard_error_naming_the_folder() -> None:
    """Never a guess. A folder we do not understand is a question for the school."""
    with pytest.raises(ValueError, match="кухня"):
        folder_role("кухня")


# -- THE trap: the filename prefix lies ---------------------------------------


def test_a_teacher_whose_photo_is_named_student_is_still_staff() -> None:
    """**The `учитель` folder contains files named `student_469_….jpg`.**

    A teacher's photo is named "student". The obvious pattern is wrong, and trusting it
    would have filed two teachers as pupils. Person type comes from the FOLDER, never
    from the filename (spec §1.1).
    """
    entry = entry_for("учитель", "student_469_1778954922.jpg")

    assert entry == RosterEntry(
        external_id="student_469",
        person_type=PersonType.STAFF,
        class_name=None,
        position="учитель",
    )


def test_a_pupil_in_a_class_folder_carries_the_class() -> None:
    entry = entry_for("5-А", "student_333_1778595343147.jpg")

    assert entry == RosterEntry(
        external_id="student_333",
        person_type=PersonType.STUDENT,
        class_name="5-А",
        position=None,
    )


def test_staff_keep_the_staff_prefix_in_their_external_id() -> None:
    """The external_id is the matched prefix + id, verbatim. `staff_334` is not
    `student_334`, and the two are different people until a human says otherwise."""
    assert entry_for("staff", "staff_334_1778595388766.jpg").external_id == "staff_334"


# -- no silent fallback, ever -------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "photo.jpg",
        "Иванов Иван.jpg",  # the legacy's whole world
        "student_333.jpg",  # no timestamp
        "student__1778595343147.jpg",  # no id
        "teacher_469_1778954922.jpg",  # not a prefix we know
        "student_469_1778954922.bmp",  # not an image we take
        "student_469_1778954922",  # no suffix
    ],
)
def test_a_filename_that_does_not_match_is_a_hard_error_naming_the_file(filename: str) -> None:
    """**The single most important rule in this spec.**

    The legacy's characteristic failure was not that it got identity wrong -- it was that
    it INVENTED an identity and carried on. A refusal is recoverable. A quiet guess is a
    child eating someone else's lunch (spec §1.2).
    """
    with pytest.raises(BadFilename) as caught:
        external_id_for(filename)

    assert filename in str(caught.value), "the error must name the file, or nobody can fix it"


def test_the_case_of_the_suffix_does_not_matter() -> None:
    assert external_id_for("student_333_1778595343147.JPG") == "student_333"
```

- [ ] **Step 3: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_naming.py tests/test_identity_roster.py -q
```

Expected failure — collection error on both:

```
ModuleNotFoundError: No module named 'qorgan.identity'
```

- [ ] **Step 4: Implement — the new `qorgan.identity` package.**

`src/qorgan/identity/__init__.py`:

```python
"""Who is this person? The school issued the answer; we do not invent one.

The legacy derived identity from a NAME parsed out of a photo filename, so two children
called Иванов Иван in the same class collapsed into one person: one of them ate every
lunch, the other never existed. This package does not have that failure mode available to
it, because it never derives an identity from anything. It reads one.
"""
```

`src/qorgan/identity/naming.py`:

```python
"""Class names and display names. Pure: no I/O, no config, no database.

`display_name` is the one place the fallback name is written. The school has not yet sent
an ID -> name table, so until it does the UI says `Ученик 333, 5-А`. Written once, used by
the web, the reports and the CLI alike -- the legacy wrote its equivalent three times and
the three disagreed.
"""

from __future__ import annotations

import re
import unicodedata

# `5А`, `5-А` and `5 а` are one class. Hyphens (of every flavour) and spaces are noise.
_CLASS_NOISE = re.compile(r"[\s\-‐-―]+")
_CLASS = re.compile(r"^(1[01]|[1-9])[А-ЯЁ]$")


def normalise_class(raw: str) -> str:
    """`5А`, `5-А` and `5 а` are one class. The legacy stored all three.

    Used to RECOGNISE a class folder. The class we STORE is the folder name verbatim,
    because it is shown to a human: `Ученик 333, 5-А`, hyphen and all.
    """
    text = unicodedata.normalize("NFKC", raw).strip()
    return _CLASS_NOISE.sub("", text).upper()


def is_class_folder(name: str) -> bool:
    """`1-А` .. `11-Б`. Anything else is staff, or a question for the school."""
    return bool(_CLASS.fullmatch(normalise_class(name)))
```

`src/qorgan/identity/roster.py`:

```python
"""Reading the school's roster off its own directory tree.

Two traps in this data, both found by looking rather than assuming.

**The filename prefix lies.** The `учитель` folder contains files named
`student_469_….jpg`. A teacher's photo is named "student". Person type comes from the
FOLDER, never from the filename. The obvious pattern is wrong, and trusting it would have
filed two teachers as pupils.

**No silent fallback, ever.** A filename that does not match the pattern is a hard error
naming the file. It is never a guessed identity. This is the single most important rule in
the spec: the legacy's characteristic failure was not that it got identity wrong -- it was
that it INVENTED an identity and carried on. A refusal is recoverable. A quiet guess is a
child eating someone else's lunch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from qorgan.enums import PersonType
from qorgan.identity.naming import is_class_folder

# student_333_1778595343147.jpg -- the 333 is a REAL school ID, unique across all 142.
FILENAME = re.compile(r"^(student|staff)_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

STAFF_FOLDER = "staff"
TEACHER_FOLDER = "учитель"


class BadFilename(ValueError):
    """A photo whose name we cannot read. Not an identity to guess at."""


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One person, as the school's own directory tree describes them."""

    external_id: str  # "student_333" -- the matched prefix + id, verbatim
    person_type: PersonType
    class_name: str | None  # "5-А", the folder name verbatim
    position: str | None  # "учитель", or None


def folder_role(folder: str) -> tuple[PersonType, str | None, str | None]:
    """(person_type, class_name, position). The folder decides, and only the folder."""
    if is_class_folder(folder):
        return PersonType.STUDENT, folder, None
    if folder == STAFF_FOLDER:
        return PersonType.STAFF, None, None
    if folder == TEACHER_FOLDER:
        return PersonType.STAFF, None, TEACHER_FOLDER
    raise ValueError(
        f"unknown roster folder {folder!r}: it is not a class (1-А .. 11-Б), "
        f"not {STAFF_FOLDER!r} and not {TEACHER_FOLDER!r}. Ask the school what it is "
        "rather than guessing who is inside it."
    )


def external_id_for(filename: str) -> str:
    """`student_333_1778595343147.jpg` -> `student_333`. Anything else raises."""
    match = FILENAME.match(filename)
    if match is None:
        raise BadFilename(
            f"{filename!r} does not match {FILENAME.pattern}. A photo whose name we cannot "
            "read is not an identity to guess at -- a quiet guess is a child eating "
            "someone else's lunch. Fix the name, or ask the school whose face this is."
        )
    prefix, number = match.group(1).lower(), match.group(2)
    return f"{prefix}_{number}"


def entry_for(folder: str, filename: str) -> RosterEntry:
    """The folder says WHO they are. The filename says only WHICH one."""
    person_type, class_name, position = folder_role(folder)
    return RosterEntry(
        external_id=external_id_for(filename),
        person_type=person_type,
        class_name=class_name,
        position=position,
    )
```

- [ ] **Step 5: Implement — delete the name machinery.**

Delete the file:

```
git rm src/qorgan/faces/identity.py
```

Then cut `src/qorgan/faces/importer.py` back to `safe_extract` + `ImportReport`. The whole
file becomes:

```python
r"""Photographs in, embeddings out. Task 5 puts the roster walk on top of this.

Two things the legacy got wrong, both fixed here:

  * **Zip Slip.** It called `extractall()` with no validation of the member names at all,
    so an archive containing `..\..\Windows\System32\...` would write there — and the
    endpoint that accepted archives had no authentication (audit H-03). Every member is
    checked to land inside the extraction directory.

  * **Files were deleted before the transaction committed**, so a crash mid-import left
    the database rolled back and the photographs already gone (M-22). Nothing is deleted
    here, and the database work commits per person.

What is gone: the namesake machinery. It answered a question this data does not ask. The
school issues the ids; we read them (spec §1).
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from qorgan.logging_setup import get_logger
from qorgan.paths import PathOutsideRoot, ensure_within

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# A photo whose face is smaller than this is not worth embedding: it will produce a
# vector that matches everybody a little and nobody enough.
MIN_FACE_PX = 60


@dataclass(frozen=True, slots=True)
class Unenrollable:
    """A photograph we could not turn into a face. It is REPORTED, never dropped.

    Four of the school's staff photographs contain no detectable face at all --
    staff_465, staff_466, staff_467, staff_468. They cannot be enrolled, and a person the
    system can never recognise is something the school has to know about (spec §1.1).
    """

    photo: str
    reason: str
    faces: int


@dataclass
class ImportReport:
    people: int = 0
    photos: int = 0
    embeddings: int = 0
    unenrollable: list[Unenrollable] = field(default_factory=list)

    def cannot_enrol(self, photo: str, reason: str, faces: int) -> None:
        self.unenrollable.append(Unenrollable(photo=photo, reason=reason, faces=faces))

    @property
    def has_unenrollable(self) -> bool:
        return bool(self.unenrollable)

    def summary(self) -> str:
        lines = [
            f"Imported {self.people} people, {self.photos} photos, {self.embeddings} embeddings.",
        ]
        if not self.unenrollable:
            return "\n".join(lines)

        lines.append("")
        lines.append(
            f"{len(self.unenrollable)} photo(s) could NOT be enrolled. These people exist "
            "in the roster and the system can never recognise them. They will appear on "
            "every 'did not eat' report, for ever, until the school sends another photo:"
        )
        for item in sorted(self.unenrollable, key=lambda u: u.photo):
            lines.append(f"  {item.photo} — {item.reason}")
        return "\n".join(lines)


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    """Unzip, refusing any member that would land outside the destination.

    The legacy called `extractall()` with no checks whatsoever, on a directory chosen by
    an unauthenticated HTTP caller. A crafted archive could write anywhere on the server.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            target = destination / member.filename
            try:
                ensure_within(destination, target)
            except PathOutsideRoot:
                # Not a warning. Somebody built this archive on purpose.
                raise ValueError(
                    f"{archive.name}: refusing member {member.filename!r} — it would "
                    "escape the extraction directory (Zip Slip)"
                ) from None

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(target)

    return extracted
```

- [ ] **Step 6: Implement — drop the CLI commands that no longer exist.**

In `src/qorgan/faces/cli.py`: delete the `import_cmd` and `check_cmd` blocks from
`add_parser`, and delete `cmd_import` and `cmd_check`. Delete the now-unused imports
`from qorgan.config.identity import FaceModelSettings` and
`from qorgan.enums import PersonType`. `add_parser` becomes:

```python
def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("pupils", help="pupils, staff and the canteen record")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    report_cmd = sub.add_parser("report", help="who ate, and who did not, on a given day")
    report_cmd.add_argument(
        "--day", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)"
    )
    report_cmd.add_argument("--csv", type=Path, help="write the full report to a CSV file")
    report_cmd.set_defaults(func=cmd_report)
```

Task 5 adds `import-roster` and `import` back; Task 6 adds `gallery-report`; Task 7 adds
`merge`.

- [ ] **Step 7: Implement — remove `ExternalIdSource.GENERATED`.**

Nothing produces a generated id any more. In `src/qorgan/enums.py`:

```python
class ExternalIdSource(StrEnum):
    """Where a person's external_id came from.

    There used to be a GENERATED member, for ids derived from a name at photo import.
    There is no such thing now: the school issues the ids and we read them. An id we
    invented is an id we can be wrong about, and being wrong about identity is how a child
    eats someone else's lunch.
    """

    ROSTER = "roster"  # authoritative, from the school's own directory tree
```

In `src/qorgan/db/models/person.py`, replace the `external_id_source` column and its
comment:

```python
    # The identity key. Unique, and never guessed from a filename.
    external_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # There is exactly one source, and it is the school. An id we invented is an id we can
    # be wrong about.
    external_id_source: Mapped[ExternalIdSource] = mapped_column(
        Enum(ExternalIdSource, native_enum=False), default=ExternalIdSource.ROSTER, nullable=False
    )
```

The database column narrows from `VARCHAR(9)` (`"GENERATED"`) to `VARCHAR(6)`
(`"ROSTER"`). That is a schema change, and `tests/test_migrations.py::
test_the_migration_matches_the_models` will fail until Task 4 writes the migration. **That
is expected and it is why Tasks 3 and 4 are one green step apart** — see Step 9.

- [ ] **Step 8: Implement — strip `tests/test_faces_import.py`.**

Delete everything except the Zip Slip block, and rewrite the report block. The whole file
becomes:

```python
"""Importing pupils: Zip Slip, and the photos that cannot be enrolled."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from qorgan.faces.importer import ImportReport, safe_extract

# -- Zip Slip ----------------------------------------------------------------


def test_a_normal_archive_extracts(tmp_path: Path) -> None:
    archive = tmp_path / "roster.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("5-А/student_333_1778595343147.jpg", b"pretend-jpeg")

    extracted = safe_extract(archive, tmp_path / "out")

    assert len(extracted) == 1
    assert extracted[0].read_bytes() == b"pretend-jpeg"


def test_an_archive_that_tries_to_escape_is_refused(tmp_path: Path) -> None:
    """The legacy called extractall() with NO validation of member names, on a directory
    chosen by an unauthenticated HTTP caller. A crafted archive could write anywhere on
    the server (audit H-03)."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../pwned.txt", b"gotcha")

    with pytest.raises(ValueError, match="Zip Slip"):
        safe_extract(archive, tmp_path / "out")

    assert not (tmp_path.parent / "pwned.txt").exists()


def test_an_absolute_member_path_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/etc/passwd", b"gotcha")

    # Either it is rejected, or it lands harmlessly inside the destination -- never
    # outside it. What matters is that nothing escapes.
    try:
        extracted = safe_extract(archive, tmp_path / "out")
    except ValueError:
        return
    for path in extracted:
        assert (tmp_path / "out") in path.parents


# -- the report --------------------------------------------------------------


def test_a_clean_import_reports_only_what_it_did() -> None:
    report = ImportReport(people=138, photos=138, embeddings=138)

    assert "Imported 138 people" in report.summary()
    assert not report.has_unenrollable


def test_photos_that_cannot_be_enrolled_are_itemised_not_dropped() -> None:
    """Four of the school's staff photographs contain no detectable face at all --
    staff_465 through staff_468. They cannot be enrolled, and a person the system can
    never recognise is something the school has to know about (spec §1.1)."""
    report = ImportReport(people=138, photos=142, embeddings=138)
    report.cannot_enrol("staff_465_1778595393105.jpg", "no face found", faces=0)
    report.cannot_enrol("групповое.jpg", "3 faces found; a roster photo must show one", faces=3)

    summary = report.summary()

    assert report.has_unenrollable
    assert "2 photo(s) could NOT be enrolled" in summary
    assert "staff_465_1778595393105.jpg" in summary
    assert "3 faces found" in summary
```

- [ ] **Step 9: Implement — the last references.**

`pyproject.toml`, the RUF001 justification block, points at a file that will not exist.
Replace lines 91-101 (`# RUF001/2/3 flag...` down to the closing of the comment) with:

```toml
    # RUF001/2/3 flag Cyrillic characters that look like Latin ones. This is a Russian-
    # language school: class names ("5-А"), the `учитель` roster folder, and every
    # operator-facing string are Cyrillic, so the rule fires on essentially every line of
    # domain text.
    #
    # The hazard the rule exists for -- a homoglyph making one person look like two --
    # cannot bite here any more. Identity is the school's own id, read from the filename
    # (`student_333`), which is ASCII. See qorgan/identity/roster.py.
```

`README.md` line 51 documents `qorgan pupils check-namesakes`. Replace that line with:

```
qorgan pupils import-roster student_photos/student_photos   # the folder decides who is who
qorgan pupils gallery-report            # who is enrolled twice, and can this gallery work?
```

`HANDOFF.md` line 58 refers to `qorgan pupils check-namesakes`. Replace the sentence with:

```
`qorgan pupils gallery-report`) and reports it, but it cannot *resolve* it. Only the
```

- [ ] **Step 10: Run the tests. The suite is RED on migrations, by design.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_naming.py tests/test_identity_roster.py tests/test_faces_import.py -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: those three files pass, ruff is clean.

```
.venv/Scripts/python.exe -m pytest -q
```

Expected: **one failure**, and only one:

```
FAILED tests/test_migrations.py::test_the_migration_matches_the_models -
    AssertionError: models and migrations disagree: [('modify_type', None, 'persons',
    'external_id_source', ...)]
```

`ExternalIdSource` lost a member, so the column narrows. Task 4 writes that migration. **Do
not commit here.** Go straight to Task 4 and commit the two together — Tasks 3 and 4 are
one green step. (If you are running these as separate subagents, the Task 3 agent hands the
Task 4 agent an uncommitted, one-test-red tree, and says so.)

---

### Task 4: `persons.full_name` becomes nullable, and `display_name` is written once

Spec §1, §5. The name is a **nullable display field**. Until the school sends an ID → name
table the UI says `Ученик 333, 5-А`. One pure `display_name(person)`, used by the web, the
reports and the CLI alike — so the fallback name is written once, not three times.

Same migration carries Task 3's `ExternalIdSource.GENERATED` removal.

**Files:**
- Create: `migrations/versions/0002_identity_is_given_not_inferred.py`
- Modify: `src/qorgan/identity/naming.py` (add `display_name`),
  `src/qorgan/db/models/person.py` (`full_name` nullable + a `display` property),
  `src/qorgan/faces/gallery.py` (`PersonInfo` gains `position`, `full_name` nullable,
  a `display` property), `src/qorgan/canteen/reports.py` (`Meal` gains `person_type`,
  `position`, a `display` property), `src/qorgan/web/routes/canteen.py`,
  `src/qorgan/web/templates/canteen.html`, `src/qorgan/faces/cli.py`
- Test: `tests/test_identity_naming.py` (add the display_name block)

**Interfaces:**

*Consumes:* `qorgan.enums.PersonType`.

*Produces:*
```python
# src/qorgan/identity/naming.py
class Named(Protocol):
    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None

def display_name(person: Named) -> str: ...
    # full_name if the school ever sends one; otherwise
    #   STUDENT           -> "Ученик 333, 5-А"   (or "Ученик 333" with no class)
    #   STAFF, no position-> "Сотрудник 334"
    #   STAFF + position  -> "Учитель 469"

# db/models/person.py
class Person:
    full_name: Mapped[str | None]     # was: Mapped[str], nullable=False
    @property
    def display(self) -> str: ...     # == display_name(self)

# faces/gallery.py
@dataclass(frozen=True, slots=True)
class PersonInfo:
    person_id: int
    external_id: str
    full_name: str | None             # was: str
    person_type: PersonType
    class_name: str | None
    position: str | None              # NEW
    @property
    def is_staff(self) -> bool: ...
    @property
    def display(self) -> str: ...

# canteen/reports.py
@dataclass(frozen=True, slots=True)
class Meal:
    person_id: int
    external_id: str
    full_name: str | None
    person_type: PersonType           # NEW
    class_name: str | None
    position: str | None              # NEW
    outcome: SessionOutcome | None
    dwell_seconds: float | None
    opened_at: datetime
    @property
    def display(self) -> str: ...
```
`Person`, `PersonInfo` and `Meal` all satisfy `Named`, so one function serves all three.

**Steps:**

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_identity_naming.py`:

```python
# -- the display name: written once, used everywhere ------------------------


from dataclasses import dataclass  # noqa: E402

from qorgan.enums import PersonType  # noqa: E402
from qorgan.identity.naming import display_name  # noqa: E402


@dataclass(frozen=True, slots=True)
class _Person:
    external_id: str
    full_name: str | None = None
    person_type: PersonType = PersonType.STUDENT
    class_name: str | None = None
    position: str | None = None


def test_a_pupil_with_no_name_is_shown_by_their_id_and_class() -> None:
    """There is no roster of NAMES. There is a roster of IDS. Until the school sends the
    ID -> name table, this is the honest thing to put on a screen (spec §1)."""
    pupil = _Person(external_id="student_333", class_name="5-А")

    assert display_name(pupil) == "Ученик 333, 5-А"


def test_a_pupil_with_no_class_still_has_an_id() -> None:
    assert display_name(_Person(external_id="student_333")) == "Ученик 333"


def test_staff_are_not_called_pupils() -> None:
    cook = _Person(external_id="staff_334", person_type=PersonType.STAFF)

    assert display_name(cook) == "Сотрудник 334"


def test_a_teacher_is_called_a_teacher() -> None:
    """`учитель/student_469_….jpg` -- the filename says student, the FOLDER says teacher,
    and the folder is right (spec §1.1)."""
    teacher = _Person(
        external_id="student_469", person_type=PersonType.STAFF, position="учитель"
    )

    assert display_name(teacher) == "Учитель 469"


def test_a_real_name_always_wins() -> None:
    """The day the school sends the ID -> name table, nothing else has to change."""
    pupil = _Person(external_id="student_333", full_name="Петрова Мария", class_name="5-А")

    assert display_name(pupil) == "Петрова Мария"
```

- [ ] **Step 2: Run it, watch it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_naming.py -q
```

Expected failure:

```
ImportError: cannot import name 'display_name' from 'qorgan.identity.naming'
```

- [ ] **Step 3: Implement — `display_name`.**

Append to `src/qorgan/identity/naming.py` (and add `from typing import Protocol` and
`from qorgan.enums import PersonType` to its imports):

```python
_NUMBER = re.compile(r"(\d+)")


class Named(Protocol):
    """Anything with an identity. `Person`, `PersonInfo` and `Meal` all satisfy it."""

    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None


def display_name(person: Named) -> str:
    """What a human sees. **The name is a display field, not an identity.**

    There is no roster of names. There is a roster of IDS -- `student_333` -- and the id
    is what the canteen record is keyed on. Until the school sends an ID -> name table,
    the honest thing to put on a screen is the id and the class.

    Written once. The legacy wrote its equivalent in three places and the three disagreed.
    """
    if person.full_name:
        return person.full_name

    number = _number(person.external_id)

    if person.person_type is PersonType.STAFF:
        if person.position:
            # `учитель` -> `Учитель 469`. The FOLDER said teacher; the filename lied.
            return f"{person.position.capitalize()} {number}"
        return f"Сотрудник {number}"

    if person.class_name:
        return f"Ученик {number}, {person.class_name}"
    return f"Ученик {number}"


def _number(external_id: str) -> str:
    """`student_333` -> `333`. The school's own id, which is the whole point."""
    match = _NUMBER.search(external_id)
    return match.group(1) if match else external_id
```

- [ ] **Step 4: Implement — the model, the gallery, the reports.**

`src/qorgan/db/models/person.py` — add the import, make `full_name` nullable, add the
property:

```python
from qorgan.identity.naming import display_name
```

```python
    # Nullable. There is no roster of NAMES -- only of ids. The name is a display field,
    # filled in the day the school sends an ID -> name table, and until then the UI says
    # `Ученик 333, 5-А` (spec §1).
    full_name: Mapped[str | None] = mapped_column(String(255))
```

and, after `__table_args__`:

```python
    @property
    def display(self) -> str:
        """One definition, used by the web, the reports and the CLI alike."""
        return display_name(self)
```

`src/qorgan/faces/gallery.py` — `PersonInfo`:

```python
@dataclass(frozen=True, slots=True)
class PersonInfo:
    """Enough to make a decision without going back to the database mid-frame."""

    person_id: int
    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None = None

    @property
    def is_staff(self) -> bool:
        """Staff never open a meal session. One definition, used everywhere.

        The legacy had this heuristic written out twice with different marker words, so
        the same person could be staff in the database and a pupil at runtime (M-28).
        """
        return self.person_type is PersonType.STAFF

    @property
    def display(self) -> str:
        return display_name(self)
```

with `from qorgan.identity.naming import display_name` added to its imports, `position=row.position`
added to the `PersonInfo(...)` construction in `load_gallery`, and `Person.position` added
to the `select(...)` in `_read_rows`.

`src/qorgan/canteen/reports.py` — `Meal`:

```python
@dataclass(frozen=True, slots=True)
class Meal:
    person_id: int
    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None
    outcome: SessionOutcome | None
    dwell_seconds: float | None
    opened_at: datetime

    @property
    def display(self) -> str:
        return display_name(self)
```

with `from qorgan.identity.naming import display_name` imported, `Person.person_type` and
`Person.position` added to the `select(...)` in `_meals_between`, and the `Meal(...)`
construction there gaining `person_type=row.person_type, position=row.position`.

- [ ] **Step 5: Implement — show the display name.**

`src/qorgan/web/templates/canteen.html`: replace `{{ person.full_name }}` with
`{{ person.display }}` (line 51), and both `{{ meal.full_name }}` with `{{ meal.display }}`
(lines 65 and 82).

`src/qorgan/web/routes/canteen.py`: replace `m.full_name` / `p.full_name` in the three
`sorted(..., key=...)` lambdas with `m.display` / `p.display`, and `{meal.full_name}` /
`{person.full_name}` in the three CSV f-strings with `{meal.display}` / `{person.display}`.

`src/qorgan/faces/cli.py::cmd_report` and `_write_csv`: replace every `.full_name` with
`.display` (six occurrences).

Telegram does not name people — `events/recorder.py::summarise` uses the *camera's*
display name — so there is nothing to change there.

- [ ] **Step 6: Implement — the migration.**

`migrations/versions/0002_identity_is_given_not_inferred.py`:

```python
"""identity is given, not inferred

`persons.full_name` becomes nullable: the school sent ids, not names, and the name is a
display field we do not have yet.

`ExternalIdSource.GENERATED` is gone: nothing derives an identity from a name any more, so
the column narrows. Rows that carry it were created by the deleted name-based import --
their `external_id` is a hash of a filename, which is exactly the invented identity the
spec forbids. They are deleted rather than relabelled, because calling a guess a roster
entry would be a lie in the database. `qorgan pupils import-roster` recreates all 142
correctly, from the school's own ids.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-13 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A person whose identity was invented from a filename is precisely what §1.2 forbids.
    # photos and embeddings cascade; canteen_sessions.person_id is SET NULL.
    op.execute(sa.text("DELETE FROM persons WHERE external_id_source = 'GENERATED'"))

    with op.batch_alter_table('persons', schema=None) as batch_op:
        batch_op.alter_column(
            'full_name',
            existing_type=sa.String(length=255),
            nullable=True,
        )
        batch_op.alter_column(
            'external_id_source',
            existing_type=sa.Enum(
                'GENERATED', 'ROSTER', name='externalidsource', native_enum=False
            ),
            type_=sa.Enum('ROSTER', name='externalidsource', native_enum=False),
            existing_nullable=False,
        )


def downgrade() -> None:
    # Going back needs a name in every row, and there are none. The id is what we have.
    op.execute(
        sa.text(
            "UPDATE persons SET full_name = 'Ученик ' || external_id "
            "WHERE full_name IS NULL"
        )
    )

    with op.batch_alter_table('persons', schema=None) as batch_op:
        batch_op.alter_column(
            'external_id_source',
            existing_type=sa.Enum('ROSTER', name='externalidsource', native_enum=False),
            type_=sa.Enum(
                'GENERATED', 'ROSTER', name='externalidsource', native_enum=False
            ),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'full_name',
            existing_type=sa.String(length=255),
            nullable=False,
        )
```

- [ ] **Step 7: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_naming.py tests/test_migrations.py tests/test_canteen_reports.py tests/test_web_pages.py tests/test_faces_gallery.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green. `test_the_migration_matches_the_models` — the one Task 3 left red — now
passes. Existing tests that set `full_name` still see it, because `display_name` returns a
real name whenever there is one, so nothing rendered changes for them.

- [ ] **Step 8: Commit — Tasks 3 and 4 together.**

```
git status
git add src/qorgan/identity/ src/qorgan/faces/importer.py src/qorgan/faces/cli.py src/qorgan/faces/gallery.py src/qorgan/enums.py src/qorgan/db/models/person.py src/qorgan/canteen/reports.py src/qorgan/web/routes/canteen.py src/qorgan/web/templates/canteen.html migrations/versions/0002_identity_is_given_not_inferred.py pyproject.toml README.md HANDOFF.md tests/test_identity_naming.py tests/test_identity_roster.py tests/test_faces_import.py
git add -u src/qorgan/faces/identity.py
git commit -m "Identity is given, not inferred: delete the name-based machinery

The school sent 142 photographs named student_333_1778595343147.jpg, and 333 is a REAL
school ID, unique across all 142. So identity is given. Everything built to infer one from
a name is answering a question this data does not ask: generate_external_id, the
Cyrillic/Latin CONFUSABLES table, normalise_name, Identity, Suspect, CollisionReport,
find_namesakes, SAME_PERSON_SIMILARITY, and \`qorgan pupils check-namesakes\`. Gone.

ExternalIdSource.GENERATED goes with them -- nothing invents an id any more -- and the
migration DELETES the rows that carried it, because a hash of a filename is exactly the
invented identity §1.2 forbids, and calling it a roster entry would be a lie in the
database.

persons.full_name becomes nullable. The name is a display field we do not have yet. One
pure display_name(person) -> 'Ученик 333, 5-А', used by the web, the reports and the CLI
alike, so the fallback is written once rather than three times.

In its place: identity/roster.py. The FOLDER decides who someone is; the filename only
says which one. And a filename that does not match the pattern is a hard error naming the
file -- never a guess."
```

---

### Task 5: `qorgan pupils import-roster <dir>`

> **DO NOT PRE-FILTER FILES BY EXTENSION. This is the whole task in one sentence.**
>
> `faces/importer.py` still defines `IMAGE_SUFFIXES`, and `identity/roster.py` defines
> `FILENAME`. Those are **two independent definitions of "a file we accept"**, and Task 3+4's
> review found the trap they set:
>
> If the roster walker filters by `path.suffix in IMAGE_SUFFIXES` before calling
> `roster.entry_for`, then a file whose extension is in one list and not the other is
> **silently skipped** instead of raising `BadFilename`. That is a silent fallback -- exactly
> the failure this whole spec exists to eliminate, rebuilt by accident, one task after
> deleting it.
>
> **So: call `entry_for` on EVERY file in a roster folder and let `BadFilename` be the single
> gate.** One definition of what we accept, and it raises. Then **delete `IMAGE_SUFFIXES`** --
> it is dead code (its only consumer, `import_archive`, was deleted in Task 3+4) and a dead
> constant that looks alive is how the second definition got here.
>
> A photo we cannot parse must be REFUSED BY NAME, never skipped. A refusal is recoverable;
> a quiet skip is a child who silently does not exist in the system.


Spec §4.3. Walk the directory tree. **The FOLDER decides person_type and class. The
filename prefix LIES.** A filename that does not match is a hard error naming the file.
`external_id_source = ROSTER`. `full_name = NULL`.

Photos with 0 or >1 faces are **itemised**, not dropped — four staff photos (`staff_465`
through `staff_468`) contain no detectable face at all. They still get a `Person` row and a
`PersonPhoto` row with a `quality_note`, and no `FaceEmbedding`. That is what makes them
visible to `gallery-report` (Task 6) for ever, rather than only in one run's stdout.

The existing ZIP path stays for the web upload, re-expressed as `safe_extract` +
`import_directory` so there is **one import, not two**.

**Files:**
- Modify: `src/qorgan/faces/importer.py` (add the walk and the store)
- Modify: `src/qorgan/faces/cli.py` (add `import-roster` and `import`)
- Test: `tests/test_faces_import.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.config.identity import FaceModelSettings          # Task 2
from qorgan.identity.roster import BadFilename, RosterEntry, entry_for  # Task 3
from qorgan.faces.recognizer import DetectedFace, FaceRecognizer
    # DetectedFace(box: Box, embedding: np.ndarray, detection_score: float)
    #   .width -> int, .height -> int
    # FaceRecognizer.detect(frame: np.ndarray) -> list[DetectedFace]
from qorgan.faces.importer import ImportReport, Unenrollable, safe_extract, MIN_FACE_PX
from qorgan.enums import ExternalIdSource, PersonType
from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
```

*Produces:*
```python
# src/qorgan/faces/importer.py
def import_directory(
    root: Path,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    *,
    report: ImportReport | None = None,
) -> ImportReport: ...
    # raises BadFilename on the first unreadable filename; raises ValueError on an
    # unknown folder. Neither is recoverable and neither is guessed at.

def import_archive(
    archive: Path,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    *,
    report: ImportReport | None = None,
) -> ImportReport: ...
    # safe_extract + import_directory. One import, not two.
```
The recognizer is duck-typed in the tests: anything with `.detect(np.ndarray) ->
list[DetectedFace]` works, so the whole import is testable without a GPU.

**Steps:**

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_faces_import.py`. Add the imports it needs at the top of the file:

```python
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings
from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
from qorgan.detection.geometry import Box
from qorgan.enums import ExternalIdSource, PersonType
from qorgan.faces.importer import ImportReport, import_directory, safe_extract
from qorgan.faces.recognizer import DetectedFace
from qorgan.identity.roster import BadFilename
from qorgan.settings import Settings

SETTINGS = FaceModelSettings()
```

and then:

```python
# -- the roster walk ---------------------------------------------------------


class FakeRecognizer:
    """Returns a scripted number of faces per photo. No model, no GPU."""

    def __init__(self, faces_per_photo: int = 1, size: int = 120) -> None:
        self.faces_per_photo = faces_per_photo
        self.size = size
        self.calls = 0

    def detect(self, _frame: np.ndarray) -> list[DetectedFace]:
        self.calls += 1
        return [
            DetectedFace(
                box=Box(0.0, 0.0, float(self.size), float(self.size)),
                embedding=_face(self.calls * 10 + index),
                detection_score=0.9,
            )
            for index in range(self.faces_per_photo)
        ]


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _photo(root: Path, folder: str, name: str) -> Path:
    """A real 200x200 JPEG on disk. cv2 has to be able to decode it."""
    import cv2

    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    buffer.tofile(str(path))
    return path


def test_a_pupil_is_imported_with_the_schools_own_id(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """`external_id = student_333`, `external_id_source = ROSTER`, `full_name = NULL`.
    Nothing is derived from anything (spec §4.3)."""
    _photo(tmp_path, "5-А", "student_333_1778595343147.jpg")

    report = import_directory(tmp_path, FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.external_id == "student_333"
    assert person.external_id_source is ExternalIdSource.ROSTER
    assert person.full_name is None
    assert person.person_type is PersonType.STUDENT
    assert person.class_name == "5-А"
    assert report.people == 1
    assert report.embeddings == 1


def test_person_type_comes_from_the_folder_never_from_the_filename(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """**THE trap.** The `учитель` folder contains files named `student_469_….jpg`. A
    teacher's photo is named "student". Trusting the obvious pattern would have filed two
    teachers as pupils (spec §1.1)."""
    _photo(tmp_path, "учитель", "student_469_1778954922.jpg")

    import_directory(tmp_path, FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.person_type is PersonType.STAFF, "a teacher was filed as a pupil"
    assert person.position == "учитель"
    assert person.class_name is None
    assert person.external_id == "student_469"  # the id is the id; only the TYPE was a lie
    assert person.display == "Учитель 469"


def test_staff_are_staff(settings: Settings, session: Session, tmp_path: Path) -> None:
    _photo(tmp_path, "staff", "staff_334_1778595388766.jpg")

    import_directory(tmp_path, FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.person_type is PersonType.STAFF
    assert person.position is None
    assert person.display == "Сотрудник 334"


def test_a_filename_that_does_not_match_stops_the_import_dead(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """**The single most important rule in this spec.** A refusal is recoverable. A quiet
    guess is a child eating someone else's lunch (spec §1.2)."""
    _photo(tmp_path, "5-А", "student_333_1778595343147.jpg")
    _photo(tmp_path, "5-А", "photo.jpg")

    with pytest.raises(BadFilename, match="photo.jpg"):
        import_directory(tmp_path, FakeRecognizer(), SETTINGS)


def test_an_unknown_folder_stops_the_import_dead(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    _photo(tmp_path, "кухня", "staff_465_1778595393105.jpg")

    with pytest.raises(ValueError, match="кухня"):
        import_directory(tmp_path, FakeRecognizer(), SETTINGS)


# -- the photos that cannot be enrolled --------------------------------------


def test_a_photo_with_no_face_is_itemised_and_the_person_still_exists(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Four staff photos contain no detectable face at all -- staff_465 through
    staff_468. They cannot be enrolled. They must be ITEMISED, not silently dropped: the
    school has four members of staff the system can never recognise, and it needs to know
    that (spec §1.1)."""
    _photo(tmp_path, "staff", "staff_465_1778595393105.jpg")

    report = import_directory(tmp_path, FakeRecognizer(faces_per_photo=0), SETTINGS)

    assert report.embeddings == 0
    assert [u.photo for u in report.unenrollable] == ["staff_465_1778595393105.jpg"]
    assert report.unenrollable[0].faces == 0

    session.expire_all()
    # The person exists -- they are on the roster -- and the DB records WHY they have no
    # face, so `gallery-report` can find them for ever, not just in this run's stdout.
    person = session.scalars(select(Person)).one()
    assert person.external_id == "staff_465"
    assert session.scalars(select(FaceEmbedding)).all() == []
    photo = session.scalars(select(PersonPhoto)).one()
    assert photo.person_id == person.id
    assert "no face" in photo.quality_note


def test_a_group_photo_is_itemised_not_guessed_at(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Which one is the pupil? A group photograph cannot say, and guessing here poisons
    the gallery for everybody."""
    _photo(tmp_path, "5-А", "student_333_1778595343147.jpg")

    report = import_directory(tmp_path, FakeRecognizer(faces_per_photo=3), SETTINGS)

    assert report.embeddings == 0
    assert report.unenrollable[0].faces == 3
    assert "3 faces" in report.unenrollable[0].reason

    session.expire_all()
    assert session.scalars(select(FaceEmbedding)).all() == []


def test_a_face_too_small_to_embed_is_itemised(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    _photo(tmp_path, "5-А", "student_333_1778595343147.jpg")

    report = import_directory(tmp_path, FakeRecognizer(size=30), SETTINGS)

    assert report.embeddings == 0
    assert "too small" in report.unenrollable[0].reason


# -- one import, not two ------------------------------------------------------


def test_importing_the_same_roster_twice_does_not_create_the_child_twice(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The id is the school's, and it is stable. Re-running the import is safe."""
    _photo(tmp_path, "5-А", "student_333_1778595343147.jpg")

    import_directory(tmp_path, FakeRecognizer(), SETTINGS)
    import_directory(tmp_path, FakeRecognizer(), SETTINGS)

    session.expire_all()
    assert len(session.scalars(select(Person)).all()) == 1


def test_the_zip_path_and_the_directory_path_are_the_same_import(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The web upload takes a ZIP. It must not be a second, divergent importer -- the
    legacy had `analyze_aggression` in three copies and they had already drifted apart."""
    from qorgan.faces.importer import import_archive

    photo = _photo(tmp_path / "src", "5-А", "student_333_1778595343147.jpg")
    archive = tmp_path / "roster.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(photo, arcname="5-А/student_333_1778595343147.jpg")

    report = import_archive(archive, FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.external_id == "student_333"
    assert person.class_name == "5-А"
    assert report.embeddings == 1
```

- [ ] **Step 2: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_faces_import.py -q
```

Expected failure — collection error:

```
ImportError: cannot import name 'import_directory' from 'qorgan.faces.importer'
```

- [ ] **Step 3: Implement — the walk and the store.**

Append to `src/qorgan/faces/importer.py`, and extend its imports with:

```python
import hashlib

import numpy as np

from qorgan.config.identity import FaceModelSettings
from qorgan.db.engine import session_scope
from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
from qorgan.enums import ExternalIdSource
from qorgan.faces.recognizer import DetectedFace, FaceRecognizer
from qorgan.identity.roster import RosterEntry, entry_for
from qorgan.settings import get_settings, resolve
```

```python
def import_directory(
    root: Path,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    *,
    report: ImportReport | None = None,
) -> ImportReport:
    """Walk the school's own directory tree. The FOLDER decides who someone is.

    A filename that does not match the pattern raises, naming the file. An unknown folder
    raises, naming the folder. Neither is guessed at: the legacy's characteristic failure
    was not getting identity wrong, it was INVENTING one and carrying on (spec §1.2).
    """
    result = report or ImportReport()

    for photo in sorted(p for p in root.rglob("*") if p.is_file()):
        if photo.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # The folder is the one directly containing the photo. Raises on both counts.
        entry = entry_for(photo.parent.name, photo.name)
        _import_photo(photo, entry, recognizer, settings, result)

    return result


def import_archive(
    archive: Path,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    *,
    report: ImportReport | None = None,
) -> ImportReport:
    """The web upload. `safe_extract` + `import_directory` -- one import, not two."""
    staging = resolve(get_settings().media_root) / ".import" / archive.stem
    try:
        safe_extract(archive, staging)
        return import_directory(staging, recognizer, settings, report=report)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _import_photo(
    photo: Path,
    entry: RosterEntry,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    report: ImportReport,
) -> None:
    import cv2

    # NOT cv2.imread: it returns None for any non-ASCII path on Windows, and the class
    # folders are Cyrillic.
    image = cv2.imdecode(np.fromfile(str(photo), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        _store(entry, photo, None, settings, report, note="could not be decoded", faces=0)
        return

    faces = recognizer.detect(image)
    face, note = _usable(faces)
    _store(entry, photo, face, settings, report, note=note, faces=len(faces))


def _usable(faces: list[DetectedFace]) -> tuple[DetectedFace | None, str | None]:
    """One face, big enough. Anything else is reported rather than guessed at."""
    if not faces:
        return None, "no face found"
    if len(faces) > 1:
        # Which one is the pupil? A group photograph cannot say, and guessing here poisons
        # the gallery for everybody.
        return None, f"{len(faces)} faces found; a roster photo must show one"

    face = faces[0]
    if min(face.width, face.height) < MIN_FACE_PX:
        return None, f"face too small ({face.width}x{face.height}, need {MIN_FACE_PX}px)"
    return face, None


def _store(
    entry: RosterEntry,
    photo: Path,
    face: DetectedFace | None,
    settings: FaceModelSettings,
    report: ImportReport,
    *,
    note: str | None,
    faces: int,
) -> None:
    """Write the person, the photo, and -- if the photo yielded a face -- the embedding.

    An unenrollable photo still gets a Person and a PersonPhoto carrying the reason. The
    person is on the roster whether or not we can recognise them, and recording WHY in the
    database is what lets `gallery-report` find them again next term rather than only in
    this run's stdout.

    Committed per person. The legacy did the whole import in one giant transaction while
    deleting files from disk BEFORE the commit, so a crash halfway through rolled the
    database back over photographs that were already gone (audit M-22).
    """
    from qorgan.paths import media_root, to_relative

    root = media_root()
    stored = _copy_into_media(photo, root, entry)

    if note is not None:
        report.cannot_enrol(photo.name, note, faces)

    with session_scope() as session:
        person = _get_or_create(session, entry, report)
        session.add(
            PersonPhoto(
                person_id=person.id,
                path=to_relative(stored, root),
                sha256=_digest(stored),
                width=face.width if face else None,
                height=face.height if face else None,
                quality_note=note,
            )
        )
        report.photos += 1

        if face is None:
            return

        session.add(
            FaceEmbedding(
                person_id=person.id,
                model_name=settings.model_name,
                model_version=settings.model_version,
                dim=settings.embedding_dim,
                normalized=settings.normalized,
                vector=face.embedding.astype(np.float32).tobytes(),
            )
        )
        report.embeddings += 1


def _copy_into_media(photo: Path, root: Path, entry: RosterEntry) -> Path:
    folder = root / "people" / (entry.class_name or entry.position or "staff")
    folder.mkdir(parents=True, exist_ok=True)
    stored = folder / photo.name
    shutil.copy2(photo, stored)
    return stored


def _get_or_create(session, entry: RosterEntry, report: ImportReport) -> Person:
    from sqlalchemy import select

    person = session.scalar(select(Person).where(Person.external_id == entry.external_id))
    if person is not None:
        return person

    person = Person(
        external_id=entry.external_id,
        # ROSTER. The school issued this id; we did not invent it.
        external_id_source=ExternalIdSource.ROSTER,
        # NULL. There is no roster of names, and `Ученик 333, 5-А` is the honest display.
        full_name=None,
        person_type=entry.person_type,
        class_name=entry.class_name,
        position=entry.position,
    )
    session.add(person)
    session.flush()
    report.people += 1
    return person


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

Watch the 500-line limit on `importer.py`: with Task 3's remnant (~110 lines) plus this,
it lands around 300. Fine.

- [ ] **Step 4: Implement — the CLI.**

In `src/qorgan/faces/cli.py`, restore the imports

```python
from qorgan.config.identity import FaceModelSettings
```

and add both commands to `add_parser`, above the `report_cmd` block:

```python
    roster_cmd = sub.add_parser(
        "import-roster",
        help="import the school's photo directory. The FOLDER decides who is who.",
        description=(
            "Walks the tree. A folder named 1-А .. 11-Б is a class of pupils; `staff` is "
            "staff; `учитель` is staff with the position учитель. The FILENAME only "
            "carries the school's id -- and it lies about the type: the учитель folder "
            "contains files named student_469_….jpg. A filename that does not match "
            "student|staff_<id>_<timestamp>.jpg is a HARD ERROR naming the file. It is "
            "never a guessed identity."
        ),
    )
    roster_cmd.add_argument("directory", type=Path, help="e.g. student_photos/student_photos")
    roster_cmd.set_defaults(func=cmd_import_roster)

    import_cmd = sub.add_parser(
        "import", help="import the same directory tree, zipped (this is the web upload path)"
    )
    import_cmd.add_argument("archives", type=Path, nargs="+")
    import_cmd.set_defaults(func=cmd_import)
```

and the two commands:

```python
def cmd_import_roster(args: argparse.Namespace) -> int:
    from qorgan.faces.importer import ImportReport, import_directory
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.identity.roster import BadFilename

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 1

    settings = FaceModelSettings()
    recognizer = FaceRecognizer.shared(settings)

    try:
        report = import_directory(args.directory, recognizer, settings, report=ImportReport())
    except (BadFilename, ValueError) as exc:
        # Loud, and it names the file. A refusal is recoverable; a quiet guess is a child
        # eating someone else's lunch.
        print(f"\nIMPORT REFUSED: {exc}", file=sys.stderr)
        return 1

    print()
    print(report.summary())
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from qorgan.faces.importer import ImportReport, import_archive
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.identity.roster import BadFilename

    settings = FaceModelSettings()
    recognizer = FaceRecognizer.shared(settings)
    report = ImportReport()

    for archive in args.archives:
        if not archive.is_file():
            print(f"  ! not a file, skipping: {archive}")
            continue
        print(f"  importing {archive.name} ...")
        try:
            import_archive(archive, recognizer, settings, report=report)
        except (BadFilename, ValueError) as exc:
            print(f"\nIMPORT REFUSED: {exc}", file=sys.stderr)
            return 1

    print()
    print(report.summary())
    return 0
```

`sys` is already imported in `cli.py`? It is not — add `import sys` to the module imports.

- [ ] **Step 5: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_faces_import.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green.

- [ ] **Step 6: Run it against the real roster — the acceptance test for this task.**

```
.venv/Scripts/python.exe -m qorgan pupils import-roster student_photos/student_photos
```

Expected: 142 photos, 138 embeddings, and **exactly four** unenrollable photos itemised —
`staff_465`, `staff_466`, `staff_467`, `staff_468` — with `no face found`. If any other
photo appears in that list, or if the import refuses a filename, **stop and read the
message**: something about this data is not what the spec says it is.

**Do not commit anything under `student_photos/`.** The import writes into `MEDIA_ROOT` and
the database, neither of which is in the repo. Check `git status` and confirm it is clean
except for the source files you changed.

- [ ] **Step 7: Commit.**

```
git status
git add src/qorgan/faces/importer.py src/qorgan/faces/cli.py tests/test_faces_import.py
git commit -m "qorgan pupils import-roster: the folder decides, and the filename lies

Walks the school's own directory tree. A folder 1-А..11-Б is a class of pupils; \`staff\`
is staff; \`учитель\` is staff with a position. Person type comes from the FOLDER, never
from the filename -- because the учитель folder contains files named student_469_….jpg,
and trusting the obvious pattern would have filed two teachers as pupils.

external_id = student_333, external_id_source = ROSTER, full_name = NULL. Nothing is
derived from anything. A filename that does not match the pattern is a hard error naming
the file: a refusal is recoverable, a quiet guess is a child eating someone else's lunch.

Photos with 0 or >1 faces are itemised, and the person and the reason are written to the
database rather than to one run's stdout -- four staff photographs contain no detectable
face at all, and the school has four members of staff the system can never recognise. It
needs to know that next term too, not just today.

The ZIP path is now safe_extract + import_directory. One import, not two."
```

---

### Task 6: `qorgan pupils gallery-report`

Spec §4.1. Not a diagnostic — a **shipped command**, because the failure it finds is live in
the data. It reports:

- the cross-person similarity histogram;
- **duplicate enrolments** at the measured threshold **0.60** — pairs of *different*
  `external_id`s whose faces are the same person. Six of them. It fires six times.
- the impostor ceiling (max 0.472) and the implied floor under `min_score`;
- photos that cannot be enrolled, itemised, read back out of the database;
- the extrapolation `1 − (1 − p)^(S−1)` for S in (142, 500, 800, 1200).

**Files:**
- Create: `src/qorgan/identity/report.py`
- Modify: `src/qorgan/faces/cli.py`
- Test: Create `tests/test_identity_report.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.faces.gallery import Gallery, PersonInfo, load_gallery
    # Gallery.matrix: np.ndarray (N, 512), L2-normalised
    # Gallery.person_ids: np.ndarray (N,) int64      -- one row per EMBEDDING
    # Gallery.people: dict[int, PersonInfo]
    # PersonInfo.external_id: str, .display -> str
from qorgan.config.identity import FaceModelSettings, RecognitionPolicy
```

*Produces:*
```python
# src/qorgan/identity/report.py
DUPLICATE_SIMILARITY = 0.60   # MEASURED. The band is empty from 0.48 to 0.77.
IMPOSTOR_GATE = 0.45          # the gate the extrapolation is quoted against
SCHOOL_SIZES = (142, 500, 800, 1200)

@dataclass(frozen=True, slots=True)
class Bucket:
    low: float
    high: float
    count: int

@dataclass(frozen=True, slots=True)
class DuplicatePair:
    person_a: int          # db id
    person_b: int
    external_a: str
    external_b: str
    display_a: str
    display_b: str
    similarity: float

@dataclass(frozen=True, slots=True)
class SchoolRisk:
    size: int
    risk_per_child: float      # 1 - (1-p)**(size-1)
    children_affected: float   # size * risk_per_child

@dataclass(frozen=True, slots=True)
class Unenrolled:
    external_id: str
    display: str
    photo: str
    reason: str

@dataclass(frozen=True, slots=True)
class GalleryReport:
    people: int
    pairs: int
    histogram: tuple[Bucket, ...]
    below_histogram: int
    duplicates: tuple[DuplicatePair, ...]
    impostor_pairs: int
    impostor_p50: float
    impostor_p90: float
    impostor_p99: float
    impostor_max: float
    impostor_above_gate: int
    gate: float
    extrapolation: tuple[SchoolRisk, ...]
    unenrolled: tuple[Unenrolled, ...]
    @property
    def impostor_probability(self) -> float: ...
    def summary(self) -> str: ...

def person_similarity(gallery: Gallery) -> tuple[np.ndarray, list[int]]: ...
    # (P, P) best-per-person cosine matrix and the person ids that index it. PURE.
def extrapolate(p: float, sizes: Sequence[int] = SCHOOL_SIZES) -> tuple[SchoolRisk, ...]: ...
def analyse(
    gallery: Gallery,
    unenrolled: Sequence[Unenrolled] = (),
    *,
    duplicate_similarity: float = DUPLICATE_SIMILARITY,
    gate: float = IMPOSTOR_GATE,
) -> GalleryReport: ...   # PURE
def read_unenrolled() -> tuple[Unenrolled, ...]: ...   # the one impure function: reads the DB
def gallery_report(settings: FaceModelSettings) -> GalleryReport: ...  # load + read + analyse
```

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_identity_report.py`.**

```python
"""The gallery report. Not a diagnostic -- a shipped command, because the failure it finds
is live in the school's data and fires six times."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
from qorgan.enums import PersonType
from qorgan.faces.gallery import Gallery, PersonInfo, load_gallery, normalise
from qorgan.identity.report import (
    DUPLICATE_SIMILARITY,
    analyse,
    extrapolate,
    person_similarity,
    read_unenrolled,
)
from qorgan.settings import Settings

MODEL_NAME, MODEL_VERSION = "buffalo_l", "1.0"


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _same_face(vector: np.ndarray, seed: int, strength: float) -> np.ndarray:
    """A second photo of the same person, at a controllable similarity."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(size=512).astype(np.float32)
    jitter = jitter / np.linalg.norm(jitter)
    mixed = vector * strength + jitter * (1.0 - strength)
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def _gallery(*rows: tuple[int, str, np.ndarray]) -> Gallery:
    return Gallery(
        matrix=normalise(np.stack([vector for _, _, vector in rows])),
        person_ids=np.array([pid for pid, _, _ in rows], dtype=np.int64),
        people={
            pid: PersonInfo(
                person_id=pid,
                external_id=external,
                full_name=None,
                person_type=PersonType.STUDENT,
                class_name="5-А",
                position=None,
            )
            for pid, external, _ in rows
        },
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


# -- the matrix ---------------------------------------------------------------


def test_the_matrix_is_per_person_not_per_photo() -> None:
    """A child with two photos is ONE row and ONE column. The gap rule is by person; so
    is this."""
    alice = _face(1)
    matrix, ids = person_similarity(
        _gallery(
            (10, "student_333", alice),
            (10, "student_333", _same_face(alice, 2, 0.98)),
            (20, "student_334", _face(3)),
        )
    )

    assert matrix.shape == (2, 2)
    assert ids == [10, 20]


# -- the duplicate enrolments: six people hold two school IDs each ------------


def test_two_ids_holding_one_face_are_flagged_as_a_duplicate_enrolment() -> None:
    """**Two different children do not score 0.999.**

    A second enrolment batch re-registered five people who already existed. Their meals
    split across both ids, so the school's existing canteen records are already wrong for
    them. And the duplicate sits in top-2 and kills the gap, so the system is not
    inaccurate about these six -- it is BLIND to them (spec §2.1, §2.5).
    """
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "staff_464", him),
            (20, "student_477", _same_face(him, 2, 0.995)),  # the same human, second batch
            (30, "student_333", _face(9)),
        )
    )

    assert len(report.duplicates) == 1
    pair = report.duplicates[0]
    assert {pair.external_a, pair.external_b} == {"staff_464", "student_477"}
    assert pair.similarity >= DUPLICATE_SIMILARITY


def test_two_different_children_are_not_a_duplicate() -> None:
    """The legacy carried SAME_PERSON_SIMILARITY = 0.35. Against this data that constant
    would call 55 pairs the same person. The measured band says 0.60 (spec §2.2)."""
    report = analyse(_gallery((10, "student_333", _face(1)), (20, "student_334", _face(2))))

    assert report.duplicates == ()


def test_a_pair_at_0_47_is_an_impostor_and_a_pair_at_0_78_is_a_duplicate() -> None:
    """The band is empty from 0.48 to 0.77. That emptiness IS the measurement: the pairs
    at the top are not a tail of the impostor distribution, they are a different
    population (spec §2)."""
    him = _face(1)
    near = _same_face(him, 5, 0.47)  # ~0.47: two different children who look alike
    twin = _same_face(him, 6, 0.78)  # ~0.78: one human, two ids

    report = analyse(
        _gallery((10, "student_333", him), (20, "student_334", near), (30, "student_472", twin))
    )

    flagged = {frozenset((p.external_a, p.external_b)) for p in report.duplicates}
    assert frozenset(("student_333", "student_472")) in flagged
    assert frozenset(("student_333", "student_334")) not in flagged


# -- the impostor ceiling, and the floor it implies --------------------------


def test_the_report_names_the_worst_impostor_and_the_floor_it_implies() -> None:
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "student_333", him),
            (20, "student_334", _same_face(him, 5, 0.40)),
            (30, "student_335", _face(7)),
        )
    )

    assert report.impostor_max == pytest.approx(0.40, abs=0.03)
    summary = report.summary()
    assert "impostor" in summary.lower()
    assert "floor" in summary.lower()


def test_duplicates_are_excluded_from_the_impostor_statistics() -> None:
    """A duplicate is not an impostor -- it is the same human. Leaving it in the impostor
    distribution would put its 0.999 at the top and make the ceiling look catastrophic."""
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "staff_464", him),
            (20, "student_477", _same_face(him, 2, 0.995)),
            (30, "student_333", _face(9)),
        )
    )

    assert report.impostor_max < DUPLICATE_SIMILARITY
    assert report.impostor_pairs == 2  # 3 pairs, minus the one duplicate


# -- the extrapolation --------------------------------------------------------


def test_the_risk_grows_with_the_school() -> None:
    """9 447 impostor pairs give P(two different children >= 0.45) = 1.06e-4. A school of
    S gives each child S-1 impostors, so P(a child has >=1 impostor above the gate) is
    1 - (1-p)^(S-1). At 800 pupils that is 8.1%: roughly one child in twelve. This is the
    argument for a SECOND photo per child (spec §4.1)."""
    risks = extrapolate(1.06e-4, sizes=(142, 500, 800, 1200))

    by_size = {r.size: r for r in risks}
    assert by_size[142].risk_per_child == pytest.approx(0.015, abs=0.002)
    assert by_size[800].risk_per_child == pytest.approx(0.081, abs=0.003)
    assert by_size[1200].risk_per_child == pytest.approx(0.119, abs=0.004)
    assert by_size[800].children_affected == pytest.approx(65, abs=3)


def test_a_gallery_with_no_impostors_above_the_gate_extrapolates_to_zero_risk() -> None:
    assert all(risk.risk_per_child == 0.0 for risk in extrapolate(0.0))


# -- the photos that could not be enrolled -----------------------------------


def test_the_report_reads_the_unenrollable_photos_back_out_of_the_database(
    settings: Settings, session: Session
) -> None:
    """Four staff photographs contain no detectable face. The import itemised them; the
    gallery report finds them again, next term, without re-running the import."""
    person = Person(
        external_id="staff_465",
        person_type=PersonType.STAFF,
        full_name=None,
    )
    session.add(person)
    session.flush()
    session.add(
        PersonPhoto(
            person_id=person.id,
            path="people/staff/staff_465_1778595393105.jpg",
            sha256="0" * 64,
            quality_note="no face found",
        )
    )
    session.commit()

    unenrolled = read_unenrolled()

    assert len(unenrolled) == 1
    assert unenrolled[0].external_id == "staff_465"
    assert unenrolled[0].reason == "no face found"
    assert unenrolled[0].display == "Сотрудник 465"


def test_an_empty_gallery_reports_honestly_rather_than_dividing_by_zero(
    settings: Settings, session: Session
) -> None:
    """The system must run end to end with zero pupils imported."""
    report = analyse(load_gallery(MODEL_NAME, MODEL_VERSION))

    assert report.people == 0
    assert report.pairs == 0
    assert report.impostor_probability == 0.0
    assert "0 people" in report.summary()


def test_the_embeddings_are_not_needed_to_prove_the_report_runs(
    settings: Settings, session: Session
) -> None:
    """A sanity check that FaceEmbedding is still the table we read."""
    assert FaceEmbedding.__tablename__ == "face_embeddings"
```

- [ ] **Step 2: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_report.py -q
```

Expected failure — collection error:

```
ModuleNotFoundError: No module named 'qorgan.identity.report'
```

- [ ] **Step 3: Implement — `src/qorgan/identity/report.py`.**

```python
"""Can this gallery work at all, and who is enrolled twice?

Not a diagnostic. A **shipped command**, because the failure it finds is live in the
school's data and it fires six times.

Measured before a line of the module was written, because if one photo per child cannot
separate 142 children it certainly cannot separate 800, and the module would be built on
sand. Every photo embedded with the production model, then the full 138x138 cosine matrix
(138, not 142: four staff photos have no face).

The band is EMPTY from 0.48 to 0.77. That emptiness is the measurement: the six pairs at
the top are not a tail of the impostor distribution -- they are a different population.
Two different children do not score 0.999. **Six people hold two school IDs each**, their
meals split across both, and the school's existing canteen records are already wrong for
them.

This is the exact mirror of the legacy's namesake bug. The legacy collapsed two children
into one identity; this data does the reverse. The machinery we deleted was aimed at the
wrong failure. This is aimed at the one that is actually present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qorgan.config.identity import FaceModelSettings, RecognitionPolicy
from qorgan.faces.gallery import Gallery, load_gallery

# MEASURED (spec §2.1). Above this, two different external_ids are one human. The legacy
# carried SAME_PERSON_SIMILARITY = 0.35, which against this data would have called 55
# pairs the same person.
DUPLICATE_SIMILARITY = 0.60

# The gate the extrapolation below is quoted against. It is the OLD min_score -- the whole
# point is to show what it would cost at a real school's size.
IMPOSTOR_GATE = 0.45

SCHOOL_SIZES = (142, 500, 800, 1200)

_HIST_LOW, _HIST_HIGH, _HIST_WIDTH = 0.30, 1.00, 0.05
_BAR = 60  # widest bar in the histogram, in characters


@dataclass(frozen=True, slots=True)
class Bucket:
    low: float
    high: float
    count: int


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two external_ids, one human. Detection is not resolution -- see `qorgan pupils
    merge`. Which id is canonical is a decision only the school can make."""

    person_a: int
    person_b: int
    external_a: str
    external_b: str
    display_a: str
    display_b: str
    similarity: float


@dataclass(frozen=True, slots=True)
class SchoolRisk:
    size: int
    risk_per_child: float
    children_affected: float


@dataclass(frozen=True, slots=True)
class Unenrolled:
    external_id: str
    display: str
    photo: str
    reason: str


@dataclass(frozen=True, slots=True)
class GalleryReport:
    people: int
    pairs: int
    histogram: tuple[Bucket, ...]
    below_histogram: int
    duplicates: tuple[DuplicatePair, ...]
    impostor_pairs: int
    impostor_p50: float
    impostor_p90: float
    impostor_p99: float
    impostor_max: float
    impostor_above_gate: int
    gate: float
    extrapolation: tuple[SchoolRisk, ...]
    unenrolled: tuple[Unenrolled, ...]

    @property
    def impostor_probability(self) -> float:
        """P(two different children score above the gate)."""
        if self.impostor_pairs == 0:
            return 0.0
        return self.impostor_above_gate / self.impostor_pairs

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.people} people, {self.pairs} distinct cross-person pair(s).",
                "",
                *_histogram_lines(self),
                "",
                *_impostor_lines(self),
                "",
                *_duplicate_lines(self),
                "",
                *_extrapolation_lines(self),
                "",
                *_unenrolled_lines(self),
            ]
        )


def person_similarity(gallery: Gallery) -> tuple[np.ndarray, list[int]]:
    """The best cross-person cosine, per PERSON pair. Pure.

    A child with two photos is one row and one column, not two. The gap rule is by person
    (see `faces.matching._rank`, and the 1 816 NULL records that not doing so cost); so is
    this.
    """
    if gallery.is_empty:
        return np.zeros((0, 0), dtype=np.float32), []

    ids = sorted({int(pid) for pid in gallery.person_ids.tolist()})
    index = {pid: row for row, pid in enumerate(ids)}

    scores = gallery.matrix @ gallery.matrix.T
    best = np.full((len(ids), len(ids)), -1.0, dtype=np.float32)

    rows = [index[int(pid)] for pid in gallery.person_ids.tolist()]
    for i, person_i in enumerate(rows):
        for j, person_j in enumerate(rows):
            value = float(scores[i, j])
            if value > best[person_i, person_j]:
                best[person_i, person_j] = value

    np.fill_diagonal(best, 1.0)
    return best, ids


def extrapolate(p: float, sizes: Sequence[int] = SCHOOL_SIZES) -> tuple[SchoolRisk, ...]:
    """A school of S gives each child S-1 impostors: 1 - (1-p)^(S-1).

    At 800 pupils and p = 1.06e-4 that is 8.1% -- roughly one child in twelve has an
    impostor above a 0.45 gate. This is the argument for a SECOND photo per child, and it
    belongs in the questions to the school (spec §4.1).
    """
    return tuple(
        SchoolRisk(
            size=size,
            risk_per_child=(risk := 1.0 - (1.0 - p) ** (size - 1)),
            children_affected=size * risk,
        )
        for size in sizes
    )


def analyse(
    gallery: Gallery,
    unenrolled: Sequence[Unenrolled] = (),
    *,
    duplicate_similarity: float = DUPLICATE_SIMILARITY,
    gate: float = IMPOSTOR_GATE,
) -> GalleryReport:
    """Pure: a Gallery in, the whole report out. No database, no GPU."""
    matrix, ids = person_similarity(gallery)
    upper = np.triu_indices(len(ids), k=1)
    scores = matrix[upper] if len(ids) > 1 else np.zeros(0, dtype=np.float32)

    duplicates = _duplicates(matrix, ids, gallery, duplicate_similarity)
    impostors = scores[scores < duplicate_similarity]
    above = int((impostors >= gate).sum())
    probability = float(above / len(impostors)) if len(impostors) else 0.0

    return GalleryReport(
        people=len(ids),
        pairs=int(scores.size),
        histogram=_histogram(scores),
        below_histogram=int((scores < _HIST_LOW).sum()),
        duplicates=duplicates,
        impostor_pairs=int(impostors.size),
        impostor_p50=_percentile(impostors, 50),
        impostor_p90=_percentile(impostors, 90),
        impostor_p99=_percentile(impostors, 99),
        impostor_max=_percentile(impostors, 100),
        impostor_above_gate=above,
        gate=gate,
        extrapolation=extrapolate(probability),
        unenrolled=tuple(unenrolled),
    )


def _duplicates(
    matrix: np.ndarray, ids: list[int], gallery: Gallery, threshold: float
) -> tuple[DuplicatePair, ...]:
    found = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(matrix[i, j])
            if score < threshold:
                continue
            a, b = gallery.people[ids[i]], gallery.people[ids[j]]
            found.append(
                DuplicatePair(
                    person_a=ids[i],
                    person_b=ids[j],
                    external_a=a.external_id,
                    external_b=b.external_id,
                    display_a=a.display,
                    display_b=b.display,
                    similarity=score,
                )
            )
    return tuple(sorted(found, key=lambda pair: -pair.similarity))


def _histogram(scores: np.ndarray) -> tuple[Bucket, ...]:
    buckets = []
    edge = _HIST_LOW
    while edge < _HIST_HIGH - 1e-9:
        high = edge + _HIST_WIDTH
        count = int(((scores >= edge) & (scores < high)).sum())
        buckets.append(Bucket(low=edge, high=high, count=count))
        edge = high
    return tuple(buckets)


def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p)) if values.size else 0.0


def read_unenrolled() -> tuple[Unenrolled, ...]:
    """The photos the import could not turn into a face, read back out of the database.

    Four staff photographs contain no detectable face at all. The people are on the roster
    and the system can never recognise them. That fact belongs in the database, not in one
    run's stdout (spec §1.1).
    """
    from sqlalchemy import select

    from qorgan.db.engine import session_scope
    from qorgan.db.models import Person, PersonPhoto
    from qorgan.identity.naming import display_name

    with session_scope() as session:
        rows = session.execute(
            select(
                Person.external_id,
                Person.full_name,
                Person.person_type,
                Person.class_name,
                Person.position,
                PersonPhoto.path,
                PersonPhoto.quality_note,
            )
            .join(Person, Person.id == PersonPhoto.person_id)
            .where(PersonPhoto.quality_note.is_not(None))
            .order_by(Person.external_id)
        ).all()

    return tuple(
        Unenrolled(
            external_id=row.external_id,
            display=display_name(row),
            photo=row.path,
            reason=row.quality_note,
        )
        for row in rows
    )


def gallery_report(settings: FaceModelSettings | None = None) -> GalleryReport:
    model = settings or FaceModelSettings()
    gallery = load_gallery(model.model_name, model.model_version)
    return analyse(gallery, read_unenrolled())


# -- rendering ----------------------------------------------------------------


def _histogram_lines(report: GalleryReport) -> list[str]:
    peak = max((bucket.count for bucket in report.histogram), default=0)
    lines = [
        f"cross-person similarity — {report.people} people, {report.pairs} distinct pairs",
        "",
        f"  below {_HIST_LOW:.2f}    {report.below_histogram:>5}",
    ]
    for bucket in report.histogram:
        bar = "#" * round(_BAR * bucket.count / peak) if peak else ""
        note = ""
        if bucket.count == 0 and bucket.low >= 0.50 and bucket.high <= 0.75:
            note = "   <-- and not one pair lands in this band"
        lines.append(
            f"  [{bucket.low:.2f},{bucket.high:.2f})  {bucket.count:>5}  {bar}{note}"
        )
    return lines


def _impostor_lines(report: GalleryReport) -> list[str]:
    policy = RecognitionPolicy()
    return [
        f"genuine impostors (pairs below {DUPLICATE_SIMILARITY:.2f}): {report.impostor_pairs}",
        f"  p50 {report.impostor_p50:.3f}   p90 {report.impostor_p90:.3f}   "
        f"p99 {report.impostor_p99:.3f}   max {report.impostor_max:.3f}",
        f"  pairs >= {report.gate:.2f}: {report.impostor_above_gate}",
        "",
        f"  MEASURED FLOOR under min_score: {report.impostor_max:.3f} — anything at or "
        "below that admits a known confusion.",
        f"  min_score is {policy.min_score:.2f}. The ceiling — whether a real camera face "
        "can reach it at all — is NOT measured. Get canteen-entry footage of a pupil we "
        "can name.",
    ]


def _duplicate_lines(report: GalleryReport) -> list[str]:
    if not report.duplicates:
        return ["duplicate enrolments: none. Every person holds one id."]

    lines = [
        f"DUPLICATE ENROLMENTS: {len(report.duplicates)} pair(s) of DIFFERENT ids whose "
        f"faces are the same person (>= {DUPLICATE_SIMILARITY:.2f}).",
        "",
        "  Their meals split across both ids, so the canteen record is already wrong for "
        "them — and the duplicate sits in top-2 and kills the gap, so the system is not "
        "inaccurate about these people. It is BLIND to them.",
        "",
        "  Detection is not resolution. Which id is canonical is a decision only the "
        "school can make, and adjacent ids in one class may be identical twins. Nothing "
        "is merged automatically. Run: qorgan pupils merge <keep_id> <drop_id>",
        "",
    ]
    for pair in report.duplicates:
        lines.append(
            f"  {pair.similarity:.3f}  {pair.external_a:<14} ({pair.display_a})"
            f"  <->  {pair.external_b:<14} ({pair.display_b})"
        )
    return lines


def _extrapolation_lines(report: GalleryReport) -> list[str]:
    lines = [
        f"P(two different children >= {report.gate:.2f}) = "
        f"{report.impostor_probability:.2e}. A school of S gives each child S-1 "
        "impostors: 1 - (1-p)^(S-1).",
        "",
        "  school   risk per child   children affected",
    ]
    for risk in report.extrapolation:
        lines.append(
            f"  {risk.size:>6}   {risk.risk_per_child * 100:>13.1f} %   "
            f"{risk.children_affected:>17.0f}"
        )
    lines.append("")
    lines.append(
        "  This is the argument for a SECOND photo per child. Put it in the questions to "
        "the school."
    )
    return lines


def _unenrolled_lines(report: GalleryReport) -> list[str]:
    if not report.unenrolled:
        return ["every roster photo produced exactly one face."]

    lines = [
        f"{len(report.unenrolled)} photo(s) could NOT be enrolled. These people are on the "
        "roster and the system can never recognise them:",
        "",
    ]
    for item in report.unenrolled:
        lines.append(f"  {item.external_id:<14} ({item.display:<20}) {item.reason} — {item.photo}")
    return lines
```

Watch the 500-line limit: this lands around 380. If it grows, split the rendering helpers
into `identity/report_text.py`.

- [ ] **Step 4: Implement — the CLI.**

In `src/qorgan/faces/cli.py`, add to `add_parser`:

```python
    gallery_cmd = sub.add_parser(
        "gallery-report",
        help="can this gallery work at all, and who is enrolled twice?",
        description=(
            "The cross-person similarity matrix, the duplicate enrolments (two ids, one "
            "human -- six of them in this school), the impostor ceiling and the floor it "
            "puts under min_score, the photos that could not be enrolled, and what the "
            "impostor rate extrapolates to at 500, 800 and 1200 pupils."
        ),
    )
    gallery_cmd.set_defaults(func=cmd_gallery_report)
```

and:

```python
def cmd_gallery_report(_args: argparse.Namespace) -> int:
    from qorgan.identity.report import gallery_report

    report = gallery_report(FaceModelSettings())
    print(report.summary())

    # A duplicate enrolment is not a failure to import; it is a question for the school.
    # It exits non-zero so a script does not sail past it.
    return 1 if report.duplicates else 0
```

- [ ] **Step 5: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_report.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green.

- [ ] **Step 6: Run it against the real gallery — the acceptance test for this task.**

With the roster imported (Task 5, Step 6):

```
.venv/Scripts/python.exe -m qorgan pupils gallery-report
```

Expected, and this is the whole point of the task: **six duplicate pairs**, matching
spec §2.1 —

```
0.999  staff_464    <->  student_477
0.984  student_470  <->  staff_334
0.810  student_371  <->  student_472
0.792  student_369  <->  student_471
0.781  student_402  <->  student_473
0.774  student_438  <->  student_439    (may be identical twins — arithmetic cannot settle it)
```

impostor max ≈ 0.472, one impostor pair of ~9 447 above 0.45, and four unenrollable staff
photos. Exit code 1, because six pairs are a question for the school.

- [ ] **Step 7: Commit.**

```
git status
git add src/qorgan/identity/report.py src/qorgan/faces/cli.py tests/test_identity_report.py
git commit -m "qorgan pupils gallery-report: six people hold two school IDs each

Not a diagnostic -- a shipped command, because the failure it finds is live in the data.

Measured: every photo embedded with the production model, then the full 138x138 cosine
matrix. The band is EMPTY from 0.48 to 0.77, and that emptiness is the measurement: the
six pairs at the top are not a tail of the impostor distribution, they are a different
population. Two different children do not score 0.999.

A second enrolment batch re-registered five people who already existed. Their meals split
across both ids, so the school's canteen records are already wrong for them -- and their
own duplicate sits in top-2 and kills the gap, so the system is not inaccurate about these
six. It is BLIND to them, and no threshold anywhere would have shown why.

This is the exact mirror of the legacy's namesake bug: it collapsed two children into one
identity; this data does the reverse. The machinery we deleted was aimed at the wrong
failure. This is aimed at the one that is present, and it fires six times.

Also reports the impostor ceiling (0.472, the floor under min_score), the photos that
could not be enrolled, and the extrapolation 1-(1-p)^(S-1): at 800 pupils roughly one
child in twelve has an impostor above a 0.45 gate. That is the argument for a second photo
per child."
```

---

### Task 7: `qorgan pupils merge <keep_id> <drop_id>`

Spec §4.2. **Detection is not resolution.** Six people hold two IDs; which id is canonical
is a decision only the school can make, and `7-А 438/439` may be identical twins. So
`gallery-report` *detects*; `merge` *executes a decision a human made*. **It never runs
automatically.**

**Files:**
- Create: `src/qorgan/identity/merge.py`
- Modify: `src/qorgan/faces/cli.py`
- Test: Create `tests/test_identity_merge.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.db.models import CanteenSession, FaceEmbedding, Person, PersonPhoto
from qorgan.db.engine import session_scope
```

*Produces:*
```python
# src/qorgan/identity/merge.py
@dataclass(frozen=True, slots=True)
class MergeResult:
    keep_id: int
    drop_id: int
    keep_external: str
    drop_external: str
    photos_moved: int
    embeddings_moved: int
    sessions_moved: int
    def summary(self) -> str: ...

def resolve_external(external_id: str) -> int: ...   # raises LookupError naming the id
def merge_persons(keep_id: int, drop_id: int) -> MergeResult: ...
    # raises ValueError if keep_id == drop_id
    # raises LookupError if either person does not exist
```

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_identity_merge.py`.**

```python
"""Merging two ids that are one human. Never automatic; always a decision a human made."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import RecognitionPolicy
from qorgan.db.models import Camera, CanteenSession, FaceEmbedding, Person, PersonPhoto
from qorgan.db.types import utcnow
from qorgan.enums import CameraRole, CameraType, PersonType, SessionState
from qorgan.faces.gallery import load_gallery
from qorgan.faces.matching import Reason, identify
from qorgan.identity.merge import merge_persons, resolve_external
from qorgan.settings import Settings

MODEL_NAME, MODEL_VERSION = "buffalo_l", "1.0"


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _same_face(vector: np.ndarray, seed: int, strength: float = 0.995) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jitter = rng.normal(size=512).astype(np.float32)
    jitter = jitter / np.linalg.norm(jitter)
    mixed = vector * strength + jitter * (1.0 - strength)
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def _person(session: Session, external_id: str, vector: np.ndarray) -> Person:
    person = Person(
        external_id=external_id,
        person_type=PersonType.STUDENT,
        class_name="11-А",
        full_name=None,
    )
    session.add(person)
    session.flush()
    session.add(
        PersonPhoto(
            person_id=person.id,
            path=f"people/11-А/{external_id}.jpg",
            sha256="0" * 64,
        )
    )
    session.add(
        FaceEmbedding(
            person_id=person.id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            dim=512,
            normalized=True,
            vector=vector.astype(np.float32).tobytes(),
        )
    )
    session.commit()
    return person


def _camera(session: Session) -> Camera:
    camera = Camera(
        name="canteen_entry",
        display_name="Вход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.flush()
    return camera


# -- the merge ---------------------------------------------------------------


def test_photos_embeddings_and_sessions_all_re_point(settings: Settings, session: Session) -> None:
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))

    camera = _camera(session)
    session.add(
        CanteenSession(
            person_id=drop.id,
            entry_camera_id=camera.id,
            state=SessionState.OPEN,
            opened_at=utcnow(),
        )
    )
    session.commit()

    result = merge_persons(keep.id, drop.id)

    assert result.photos_moved == 1
    assert result.embeddings_moved == 1
    assert result.sessions_moved == 1

    session.expire_all()
    assert session.scalars(select(PersonPhoto)).all()[0].person_id == keep.id
    assert {e.person_id for e in session.scalars(select(FaceEmbedding))} == {keep.id}
    assert session.scalars(select(CanteenSession)).one().person_id == keep.id


def test_the_dropped_id_is_deactivated_not_deleted(settings: Settings, session: Session) -> None:
    """The id existed. The school issued it, and a record that says so is worth keeping."""
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))

    merge_persons(keep.id, drop.id)

    session.expire_all()
    assert session.get(Person, drop.id) is not None
    assert session.get(Person, drop.id).is_active is False
    assert session.get(Person, keep.id).is_active is True


def test_merging_a_person_into_themselves_is_refused(settings: Settings, session: Session) -> None:
    keep = _person(session, "staff_334", _face(1))

    with pytest.raises(ValueError, match="itself"):
        merge_persons(keep.id, keep.id)


def test_merging_someone_who_does_not_exist_is_refused(
    settings: Settings, session: Session
) -> None:
    keep = _person(session, "staff_334", _face(1))

    with pytest.raises(LookupError, match="9999"):
        merge_persons(keep.id, 9999)


def test_an_external_id_resolves_to_a_person(settings: Settings, session: Session) -> None:
    keep = _person(session, "staff_334", _face(1))

    assert resolve_external("staff_334") == keep.id

    with pytest.raises(LookupError, match="student_999"):
        resolve_external("student_999")


# -- THE regression test: the gap collapse, pinned (spec §2.5, §6) ------------


def test_a_human_under_two_ids_is_ambiguous_and_a_merge_makes_him_visible(
    settings: Settings, session: Session
) -> None:
    """**This is §2.5 turned into a regression test, so the failure cannot return
    silently.**

    The top hall matches were id=334 and id=470 — the duplicate pair. Both are the same
    human, so top-1 and top-2 are BOTH him, and the gap collapses:

        40px  id=334  score 0.604   gap +0.001   ->  after merge  +0.413

    min_gap is 0.05. **He is rejected as AMBIGUOUS every time.** This is the 1816-NULL
    mechanism alive in this data — and note that ranking by PERSON (last session's fix)
    cannot help, because the two rows genuinely ARE different persons in the database.

    For those six children the system is not inaccurate. It is BLIND, and no threshold
    anywhere would have shown why.
    """
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2, strength=0.999))
    _person(session, "student_333", _face(9))  # somebody else entirely

    policy = RecognitionPolicy()  # min_score 0.50, min_gap 0.05

    before = load_gallery(MODEL_NAME, MODEL_VERSION)
    rejected = identify(him, before.matrix, before.person_ids, policy)

    assert not rejected.accepted, "the duplicate did not collapse the gap; the test is broken"
    assert rejected.reason is Reason.AMBIGUOUS
    assert rejected.gap < policy.min_gap
    assert {r.person_id for r in rejected.ranked[:2]} == {keep.id, drop.id}

    merge_persons(keep.id, drop.id)

    after = load_gallery(MODEL_NAME, MODEL_VERSION)
    accepted = identify(him, after.matrix, after.person_ids, policy)

    assert accepted.accepted, "merging did not make him visible again"
    assert accepted.person_id == keep.id
    assert accepted.reason is Reason.ACCEPTED
    assert accepted.gap > policy.min_gap
```

- [ ] **Step 2: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_merge.py -q
```

Expected failure — collection error:

```
ModuleNotFoundError: No module named 'qorgan.identity.merge'
```

- [ ] **Step 3: Implement — `src/qorgan/identity/merge.py`.**

```python
"""Two ids, one human. **Never automatic.**

`gallery-report` detects six duplicate enrolments. It does not resolve them, and it must
not: which id is canonical is a decision only the school can make, and `7-А 438/439` --
adjacent ids, same class, both in the first enrolment batch, the lowest score of the six --
may be identical twins. Arithmetic cannot settle that, and a system that guessed would be
making up an identity, which is the one thing this module exists to stop.

So this command executes a decision a human already made. It re-points the photos, the
embeddings and the canteen sessions from `drop_id` onto `keep_id`, and deactivates
`drop_id` -- it does not delete it, because the school issued that id and a record saying
so is worth keeping.

Measured effect (spec §2.7): the gap collapse is real -- 0.001 -> **0.413** after merge --
and it is a property of the GALLERY (one human enrolled twice sits in his own top-2), so it
holds on any camera at any resolution. The accompanying A/B (accepts 3 -> 9, gap-kills
6 -> 0) was measured on gallery faces >=38 px **at HD**, which is below the hall's
production gate; it does NOT describe the hall, where nothing is recognised regardless.
Merging bites where faces are big enough to be recognised at all -- the CANTEEN.

It does not improve the system. It makes six specific people VISIBLE to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, update

from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import CanteenSession, FaceEmbedding, Person, PersonPhoto
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MergeResult:
    keep_id: int
    drop_id: int
    keep_external: str
    drop_external: str
    photos_moved: int
    embeddings_moved: int
    sessions_moved: int

    def summary(self) -> str:
        return (
            f"Merged {self.drop_external} (id {self.drop_id}) into "
            f"{self.keep_external} (id {self.keep_id}).\n"
            f"  photos:     {self.photos_moved}\n"
            f"  embeddings: {self.embeddings_moved}\n"
            f"  sessions:   {self.sessions_moved}\n"
            f"\n{self.drop_external} is now inactive. It leaves the gallery, so it can no "
            "longer sit in top-2 and kill the gap on its own twin — which is what made "
            "this person invisible to the system rather than merely hard to recognise."
        )


def resolve_external(external_id: str) -> int:
    """`student_470` -> the database id. Raises, naming the id, if nobody holds it."""
    with session_scope() as session:
        person_id = session.scalar(
            select(Person.id).where(Person.external_id == external_id)
        )
    if person_id is None:
        raise LookupError(f"no person holds external_id {external_id!r}")
    return int(person_id)


def merge_persons(keep_id: int, drop_id: int) -> MergeResult:
    """Re-point everything from `drop_id` onto `keep_id` and deactivate `drop_id`."""
    if keep_id == drop_id:
        raise ValueError(f"cannot merge person {keep_id} into itself")

    def _merge() -> MergeResult:
        with session_scope() as session:
            keep = _require(session, keep_id)
            drop = _require(session, drop_id)

            photos = _repoint(session, PersonPhoto, keep_id, drop_id)
            embeddings = _repoint(session, FaceEmbedding, keep_id, drop_id)
            sessions = _repoint(session, CanteenSession, keep_id, drop_id)

            # Not deleted. The school issued that id, and a record saying it existed --
            # and that a human decided it was a duplicate -- is worth keeping.
            drop.is_active = False

            return MergeResult(
                keep_id=keep_id,
                drop_id=drop_id,
                keep_external=keep.external_id,
                drop_external=drop.external_id,
                photos_moved=photos,
                embeddings_moved=embeddings,
                sessions_moved=sessions,
            )

    result = with_retry(_merge)
    logger.warning(
        "persons merged — a human decided these two ids are one person",
        extra={
            "keep": result.keep_external,
            "drop": result.drop_external,
            "sessions_moved": result.sessions_moved,
        },
    )
    return result


def _require(session, person_id: int) -> Person:
    person = session.get(Person, person_id)
    if person is None:
        raise LookupError(f"no person with id {person_id}")
    return person


def _repoint(session, model, keep_id: int, drop_id: int) -> int:
    """Hand every row this person owns to the person we are keeping."""
    result = session.execute(
        update(model).where(model.person_id == drop_id).values(person_id=keep_id)
    )
    return int(result.rowcount or 0)
```

- [ ] **Step 4: Implement — the CLI.**

In `src/qorgan/faces/cli.py`, add to `add_parser`:

```python
    merge_cmd = sub.add_parser(
        "merge",
        help="two ids are one human: re-point everything onto one of them",
        description=(
            "Executes a decision a HUMAN made. `gallery-report` finds the duplicates; it "
            "does not resolve them, because which id is canonical is a decision only the "
            "school can make -- and 7-А 438/439 may be identical twins, which arithmetic "
            "cannot settle. This never runs automatically."
        ),
    )
    merge_cmd.add_argument("keep_id", help="the external_id to keep, e.g. staff_334")
    merge_cmd.add_argument("drop_id", help="the external_id to retire, e.g. student_470")
    merge_cmd.set_defaults(func=cmd_merge)
```

and:

```python
def cmd_merge(args: argparse.Namespace) -> int:
    from qorgan.identity.merge import merge_persons, resolve_external

    try:
        keep = resolve_external(args.keep_id)
        drop = resolve_external(args.drop_id)
        result = merge_persons(keep, drop)
    except (LookupError, ValueError) as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 1

    print(result.summary())
    return 0
```

- [ ] **Step 5: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_merge.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green. The regression test is the one that matters — it pins §2.5 so the gap
collapse cannot return silently.

- [ ] **Step 6: Commit.**

```
git status
git add src/qorgan/identity/merge.py src/qorgan/faces/cli.py tests/test_identity_merge.py
git commit -m "qorgan pupils merge: detection is not resolution

gallery-report finds six people holding two school IDs each. It does not resolve them, and
it must not: which id is canonical is a decision only the school can make, and 7-А 438/439
-- adjacent ids, same class, both in the first enrolment batch, the lowest score of the
six -- may be identical twins. Arithmetic cannot settle that.

So merge executes a decision a human already made. It re-points photos, embeddings and
canteen sessions from drop onto keep, and deactivates drop rather than deleting it. It
never runs automatically.

The regression test is the point: a gallery with one human under two ids yields
gap < min_gap -> AMBIGUOUS, and after the merge the same face is ACCEPTED. That is §2.5
pinned, so the failure cannot come back silently -- because when it does come back, it
looks like nothing at all. The system is not inaccurate about those six people. It is
blind to them."
```

---

### Task 8: `qorgan identity camera-report <camera|clip>`

Spec §2.4, §2.6, §4.6.

> **AMENDED after measurement — read this before writing any code.**
>
> An earlier draft of this task said "2.2% of hall faces clear the 60 px gate". **That
> figure is wrong, and the way it is wrong is the whole point of this command.** It was
> measured on the 2560×1440 HD burst, a stream production never analyses.
>
> **A second draft then corrected it to 960×540 — and that was wrong too.** 960×540 is
> `base.yaml`'s DEFAULT. **`hall.yaml` and `canteen_entry.yaml` override it to 1280×720.**
> There is no fleet-wide analysis resolution, and *that* is precisely why this command must
> read the resolution from **the camera's own merged config** instead of assuming a number.
>
> The hall's real scale from the 2560×1440 clip is **0.5**, not 0.375. Re-expressed in the
> pixels the hall worker actually sees (1280×720):
>
> ```
> the 60px gate at 1280x720      == a 120px face in the clip that was measured
> the 38px small-face gate       == a  76px face in that clip
> largest face in 14 970 hall faces:  100px at HD  ->  50px at 1280x720
>
> faces clearing the 60px gate:      0 of 14 970
> faces clearing the 38px gate:     77 of 14 970  (0.51%)
>     of those 77, accepted at min_score 0.45:  0
>     of those 77, accepted at min_score 0.50:  0
>     best score among all 77:  0.350
>
> => ZERO recognitions in 14 970 faces.
>
> hall face size at 1280x720:  p50 11.5px   p90 22.5px   max 50px
>                       (HD:   p50 23px     p90 45px     max 100px)
> ```
>
> **Not 2.2%. Zero recognitions.** The conclusion is unchanged and slightly stronger, but the
> MECHANISM is different: it is not that no face reaches the gate — 77 do. It is that **not
> one of them scores high enough to be recognised.** A number measured on the wrong stream,
> or at the wrong resolution, is worse than no number.
>
> Therefore this command **MUST report per STREAM, not per camera** — the analysis substream
> and the burst separately, each measured at the resolution *that stream is actually
> analysed at* (use `prepare_frame`-equivalent scaling; the analysis figure must be computed
> on frames resized to **that camera's** `capture.frame_width × frame_height`, never on the
> raw decode and never on an assumed default).
>
> It answers exactly one question, and it must be phrased as that question:
> **"Can this camera recognise anybody at the resolution the worker actually feeds it?"**
>
> And it **GATES**: a camera whose faces essentially never clear the gate on its analysis
> stream **must not be usable as an identity camera**. Emit a clear refusal naming the
> stream, the gate, and the measured fraction. This turns "this camera recognises nobody"
> from a discovery made after months of tuning into a fact asserted at startup.

So: sample N frames from a camera (or a clip), **per stream**, report the face-size
distribution and the fraction clearing the gate at that stream's real analysis resolution.
The answer is to **move the camera** — never to lower a threshold.

This introduces the top-level `qorgan identity` command group.

**Files:**
- Create: `src/qorgan/identity/camera.py`, `src/qorgan/identity/cli.py`
- Modify: `src/qorgan/cli.py` (register the group)
- Test: Create `tests/test_identity_camera.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.config.identity import FaceGate, FaceModelSettings
from qorgan.config.camera import CameraConfig
from qorgan.config.loader import load_cameras     # -> dict[str, CameraConfig]
from qorgan.capture.stream import open_rtsp       # (url: str) -> cv2.VideoCapture
from qorgan.rtsp import build_url                 # (camera_name, RtspSettings, *, burst=False) -> str
from qorgan.faces.recognizer import FaceBox, FaceRecognizer   # FaceBox arrives in Task 9
```

**Task 9 dependency, resolved now:** `camera-report` only needs boxes, never vectors, so it
calls `recognizer.detect_faces()`. That method is added in Task 9. To keep **this** task
green on its own, `measure_faces` is written against a `Detector` protocol —
`detect(frame) -> list[<anything with .width and .height>]` — and the CLI passes an adapter.
Task 9, Step 8 swaps the adapter for `detect_faces` and deletes it. The adapter is three
lines and it is named as temporary in a comment; do not leave it behind.

*Produces:*
```python
# src/qorgan/identity/camera.py
class Sized(Protocol):
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Sized]: ...

@dataclass(frozen=True, slots=True)
class FaceSizeReport:
    frames: int
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    gate: FaceGate
    source: str
    @property
    def faces(self) -> int: ...
    @property
    def clearing_gate(self) -> int: ...
    @property
    def fraction_clearing(self) -> float: ...
    def width_percentile(self, p: float) -> float: ...
    def summary(self) -> str: ...

def measure_faces(
    frames: Iterable[np.ndarray], detector: Detector, gate: FaceGate, source: str
) -> FaceSizeReport: ...          # PURE over its inputs -- no camera, no clock, no DB
```

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_identity_camera.py`.**

```python
"""Can this camera ever recognise anybody? A question about optics, not thresholds."""

from __future__ import annotations

import numpy as np

from qorgan.config.identity import FaceGate
from qorgan.detection.geometry import Box
from qorgan.identity.camera import FaceSizeReport, measure_faces


class _Face:
    def __init__(self, width: int, height: int) -> None:
        self.box = Box(0.0, 0.0, float(width), float(height))

    @property
    def width(self) -> int:
        return int(self.box.width)

    @property
    def height(self) -> int:
        return int(self.box.height)


class ScriptedDetector:
    """One list of face sizes per frame."""

    def __init__(self, per_frame: list[list[tuple[int, int]]]) -> None:
        self._per_frame = per_frame
        self._index = 0

    def detect(self, _frame: np.ndarray) -> list[_Face]:
        sizes = self._per_frame[self._index]
        self._index += 1
        return [_Face(width, height) for width, height in sizes]


def _frames(count: int) -> list[np.ndarray]:
    return [np.zeros((1440, 2560, 3), dtype=np.uint8) for _ in range(count)]


def test_the_report_counts_every_face_it_saw() -> None:
    detector = ScriptedDetector([[(20, 24), (80, 96)], [], [(45, 54)]])

    report = measure_faces(_frames(3), detector, FaceGate(), source="hall_left")

    assert report.frames == 3
    assert report.faces == 3


def test_a_camera_whose_faces_are_too_small_says_so_in_percent() -> None:
    """**The measurement that ends eighteen threshold-tuning attempts.**

    250 clips of the school hall, 14 970 faces. The numbers depend entirely on WHICH STREAM
    you measure and AT WHAT RESOLUTION -- which is the whole reason this command exists, and
    the reason it reads the resolution from the camera's own config instead of assuming one:

      on the 2560x1440 HD burst:      p50 23 px,   p90 45 px,   2.2% clear the 60 px gate
      at the hall's real 1280x720:    p50 11.5px,  max 50 px,   **0 of 14 970** clear the
                                      strict 60 px gate; 77 clear the 38 px small-face gate
                                      and **none of the 77 is recognised** (best score 0.350)

    (1280x720, not 960x540: `hall.yaml` overrides `base.yaml`'s default.)

    Production analyses the substream. So the honest number of recognitions is ZERO, and an
    11-pixel face upscaled to ArcFace's 112-pixel input is mush. No threshold recovers it --
    the 77 faces that ARE big enough still score nowhere near the floor. The answer is to
    MOVE THE CAMERA (spec §2.4).
    """
    tiny = [[(23, 28)] for _ in range(98)]
    big = [[(80, 96)] for _ in range(2)]
    detector = ScriptedDetector(tiny + big)

    report = measure_faces(_frames(100), detector, FaceGate(), source="hall_left")

    assert report.clearing_gate == 2
    assert report.fraction_clearing == 0.02
    assert report.width_percentile(50) < 60

    summary = report.summary()
    assert "2.0%" in summary
    assert "move the camera" in summary.lower()


def test_a_camera_that_can_actually_see_faces_is_not_told_to_move() -> None:
    detector = ScriptedDetector([[(120, 140)] for _ in range(20)])

    report = measure_faces(_frames(20), detector, FaceGate(), source="canteen_entry")

    assert report.fraction_clearing == 1.0
    assert "move the camera" not in report.summary().lower()


def test_a_camera_that_sees_nobody_does_not_divide_by_zero() -> None:
    detector = ScriptedDetector([[] for _ in range(5)])

    report = measure_faces(_frames(5), detector, FaceGate(), source="yard_entry")

    assert report.faces == 0
    assert report.fraction_clearing == 0.0
    assert "no faces" in report.summary().lower()


def test_the_gate_is_the_config_gate_not_a_number_in_this_module() -> None:
    """The legacy had SIX different minimum-face-size gates. There is one."""
    detector = ScriptedDetector([[(45, 54)]])

    strict = measure_faces(_frames(1), detector, FaceGate(), source="c")
    assert strict.clearing_gate == 0

    detector = ScriptedDetector([[(45, 54)]])
    small = measure_faces(
        _frames(1),
        detector,
        FaceGate(min_width=38, min_height=48, min_area=1800),
        source="c",
    )
    assert small.clearing_gate == 1
    assert isinstance(small, FaceSizeReport)
```

- [ ] **Step 2: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_camera.py -q
```

Expected failure:

```
ModuleNotFoundError: No module named 'qorgan.identity.camera'
```

- [ ] **Step 3: Implement — `src/qorgan/identity/camera.py`.**

```python
"""Can this camera ever recognise anybody? **A question about optics, not thresholds.**

**0 of 14 970** faces in 250 clips of the school hall clear the enrolment gate -- at the
resolution the hall worker actually analyses (**1280x720**; `hall.yaml` overrides
`base.yaml`'s 960x540 default), where the median face is **11.5 px** and the largest in the
entire corpus is **50 px**. Lower the bar to the 38 px small-face gate and 77 faces get
through -- and **not one of them is recognised**: the best score among all 77 is **0.350**,
against a min_score of 0.45. Upscaled to ArcFace's 112-pixel input those faces are mush, and
no value of any threshold recovers them.

(On the 2560x1440 HD burst the same faces give 2.2%. That number is true and useless: it
describes a stream the analysis loop never touches. Reporting it per-camera instead of
per-STREAM -- or at an assumed resolution instead of the camera's own -- is the bug this
command exists to prevent. It has been made twice.)

The legacy spent eighteen overlapping thresholds looking for a number that would fix a
problem no number could fix. This diagnostic costs almost nothing to build -- the probe
that produced the measurement above IS the implementation -- and it is the question the
legacy never asked in eighteen attempts at tuning.

If it says 2%, move the camera.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qorgan.config.identity import FaceGate

# Below this fraction, tuning is not the answer and never will be.
HOPELESS = 0.10


class Sized(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Sized]: ...


@dataclass(frozen=True, slots=True)
class FaceSizeReport:
    frames: int
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    gate: FaceGate
    source: str

    @property
    def faces(self) -> int:
        return len(self.widths)

    @property
    def clearing_gate(self) -> int:
        return sum(
            1
            for width, height in zip(self.widths, self.heights, strict=True)
            if self.gate.accepts(width, height)
        )

    @property
    def fraction_clearing(self) -> float:
        return self.clearing_gate / self.faces if self.faces else 0.0

    def width_percentile(self, p: float) -> float:
        return float(np.percentile(self.widths, p)) if self.widths else 0.0

    def summary(self) -> str:
        if not self.faces:
            return (
                f"{self.source}: {self.frames} frame(s), NO FACES AT ALL.\n"
                "Either nobody walked through, or this camera cannot see a face. Neither "
                "is fixed by a threshold."
            )

        lines = [
            f"{self.source}: {self.frames} frame(s), {self.faces} face(s).",
            "",
            "face width",
            f"  p50 {self.width_percentile(50):.0f}px   "
            f"p90 {self.width_percentile(90):.0f}px   "
            f"max {self.width_percentile(100):.0f}px",
            "",
            f"clearing the {self.gate.min_width}x{self.gate.min_height}px gate: "
            f"{self.clearing_gate} / {self.faces}  ({self.fraction_clearing * 100:.1f}%)",
        ]

        if self.fraction_clearing < HOPELESS:
            lines += [
                "",
                "This camera can never recognise anybody, and **no threshold will fix "
                "that**. A 23-pixel face upscaled to ArcFace's 112-pixel input is mush. "
                "The answer is to MOVE THE CAMERA — closer, or lower, or both. The legacy "
                "spent eighteen thresholds looking for a number that could not exist.",
            ]
        return "\n".join(lines)


def measure_faces(
    frames: Iterable[np.ndarray],
    detector: Detector,
    gate: FaceGate,
    source: str,
) -> FaceSizeReport:
    """Pure over its inputs: frames in, distribution out. No camera, no clock, no DB."""
    widths: list[int] = []
    heights: list[int] = []
    count = 0

    for frame in frames:
        count += 1
        for face in detector.detect(frame):
            widths.append(face.width)
            heights.append(face.height)

    return FaceSizeReport(
        frames=count,
        widths=tuple(widths),
        heights=tuple(heights),
        gate=gate,
        source=source,
    )
```

- [ ] **Step 4: Implement — `src/qorgan/identity/cli.py`.**

```python
"""`qorgan identity` — the questions that are about optics, not thresholds."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_FRAMES = 200


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("identity", help="face recognition diagnostics")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    camera_cmd = sub.add_parser(
        "camera-report",
        help="can this camera ever recognise anybody?",
        description=(
            "Samples frames from a camera and reports, PER STREAM and at the resolution "
            "that stream is really analysed at -- read from THAT CAMERA'S config, never "
            "assumed -- the face-size distribution and the fraction clearing the "
            "recognition gate. Measured on the school's hall (1280x720, because hall.yaml "
            "overrides base.yaml's 960x540): 0 of 14 970 faces clear the strict gate, the "
            "median face is 11.5px, and of the 77 that clear the small-face gate NONE is "
            "recognised. That is a CAMERA-PLACEMENT fact, and no amount of tuning "
            "is a substitute for it. If this says ~0%, move the camera -- do not lower "
            "the gate, because there is nothing under it to recover."
        ),
    )
    camera_cmd.add_argument("source", help="a camera name from config/cameras, or a path to a clip")
    camera_cmd.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    camera_cmd.add_argument(
        "--stride", type=int, default=5, help="take every Nth frame, so we span the clip"
    )
    camera_cmd.set_defaults(func=cmd_camera_report)


def cmd_camera_report(args: argparse.Namespace) -> int:
    from qorgan.config.identity import FaceGate, FaceModelSettings
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.identity.camera import HOPELESS, measure_faces

    try:
        capture = _open(args.source)
    except (LookupError, OSError) as exc:
        print(f"cannot read {args.source!r}: {exc}", file=sys.stderr)
        return 1

    recognizer = FaceRecognizer.shared(FaceModelSettings())
    try:
        report = measure_faces(
            _sample(capture, args.frames, args.stride),
            recognizer,
            FaceGate(),
            source=str(args.source),
        )
    finally:
        capture.release()

    print(report.summary())
    # Non-zero when the camera is hopeless, so a script cannot sail past it.
    return 1 if report.faces and report.fraction_clearing < HOPELESS else 0


def _open(source: str):
    """A clip on disk, or a camera in config/cameras."""
    import cv2

    from qorgan.capture.stream import open_rtsp
    from qorgan.config.loader import load_cameras
    from qorgan.rtsp import build_url

    path = Path(source)
    if path.is_file():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"opencv could not open the clip {path}")
        return capture

    cameras = load_cameras()
    camera = cameras.get(source)
    if camera is None:
        raise LookupError(
            f"no clip at {source!r} and no camera called that. Known cameras: "
            f"{', '.join(sorted(cameras))}"
        )
    return open_rtsp(build_url(camera.name, camera.rtsp))


def _sample(capture, frames: int, stride: int) -> Iterator[np.ndarray]:
    """Every `stride`-th frame, up to `frames` of them — so we span the source rather
    than measuring one second of it."""
    taken = 0
    index = 0
    while taken < frames:
        ok, frame = capture.read()
        if not ok:
            return
        if index % stride == 0:
            taken += 1
            yield frame
        index += 1
```

`recognizer` is passed where a `Detector` is expected. `FaceRecognizer.detect()` returns
`DetectedFace`, which has `.width`/`.height`, so it satisfies `Sized` today. **Task 9,
Step 8 changes this call to `detect_faces()`** — cheaper, because camera-report never needs
a vector.

- [ ] **Step 5: Implement — register the group.**

In `src/qorgan/cli.py::build_parser`, beside the existing deferred parser imports:

```python
    from qorgan.evaluation.cli import add_parser as add_eval_parser
    from qorgan.faces.cli import add_parser as add_pupils_parser
    from qorgan.identity.cli import add_parser as add_identity_parser

    add_eval_parser(subparsers)
    add_pupils_parser(subparsers)
    add_identity_parser(subparsers)
```

- [ ] **Step 6: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_camera.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green.

- [ ] **Step 7: Commit.**

```
git status
git add src/qorgan/identity/camera.py src/qorgan/identity/cli.py src/qorgan/cli.py tests/test_identity_camera.py
git commit -m "qorgan identity camera-report: the question the legacy never asked

0 of 14 970 faces in 250 clips of the school hall clear the enrolment gate -- at 1280x720,
the resolution the hall worker actually analyses (hall.yaml overrides base.yaml's 960x540
default), where the median face is 11.5px and the largest in the whole corpus is 50px. The
38px small-face gate is cleared by 77 of them, and NOT ONE is recognised: the best score
among all 77 is 0.350. Upscaled to ArcFace's 112-pixel input those faces are mush, and no
value of any threshold recovers them.

On the 2560x1440 HD burst the same faces give 2.2%. That number is true and useless: it
describes a stream the analysis loop never touches. Reporting per-CAMERA instead of
per-STREAM -- or at an assumed resolution instead of the camera's own -- is exactly the bug
this command exists to prevent, and it is the bug that put a misleading number into this
repo twice.

That is a CAMERA-PLACEMENT fact. The legacy spent eighteen overlapping thresholds looking
for a number that would fix a problem no number could fix. This command samples N frames per
stream, reports the face-size distribution and the fraction clearing the gate, and if it says
~0% it says in words: move the camera.

It cost almost nothing to build -- the probe that produced the measurement above IS the
implementation."
```

---

### Task 9: `IdentityService` — recognise once per track, not once per frame

> **CORRECTED. The original Step 9 rebuilt the 1816-NULL bug. Read this before the steps.**
>
> The first draft installed `if not found.newly_bound: continue` in `CanteenPipeline.on_frame`,
> commented "we already know who this track is". But `newly_bound=False` covers **two different
> states**: *already bound* AND *just embedded and rejected*. So every **unrecognised** child
> would be skipped past `_act` and `_record_attempt` entirely — leaving
> `allow_unknown_sessions=True` and `_on_entry`'s body intact and **unreachable**.
>
> Consequences, both fatal and both silent:
> 1. **The entry camera could never open an Unknown meal session again.** An unrecognised child
>    eats and is never recorded. `min_score`'s ceiling is UNMEASURED, so this could mean *zero*
>    sessions — the 1816-NULL failure, rebuilt inside the module written to prevent it.
> 2. **`RecognitionAttempt` becomes a table of successes only** (`accepted` permanently True),
>    destroying the one instrument that can measure that ceiling.
>
> And it would have looked *green*: the new test used the recognised pupil, so it passed while
> three existing tests died.
>
> **`newly_bound: bool` is the wrong shape — it cannot express three outcomes.** The two things
> being conflated have different lifetimes, and must be separated:
>
> **1. `_record_attempt` runs on EVERY embed** (1..`max_attempts` per track), never gated on
> binding. That table is the INSTRUMENT for the unmeasured ceiling; if it cannot record a
> failure it measures nothing. It still falls from ~200 rows per child to ~3, which is the
> whole point of the task.
>
> **2. `_act` runs EXACTLY ONCE per track, when the track is RESOLVED.** Resolved means:
> - **BOUND** — recognised. Act with the person.
> - **EXHAUSTED** — embedded `max_attempts` times, all rejected. Act as **UNKNOWN**.
> - **TRACK LOST while still unresolved** — a child who walks through in 2 s, whose track dies
>   while still `RETRYING`, is still **a child who walked in**. On eviction, if the track never
>   acted, act as **UNKNOWN**. (With the defaults — `max_attempts=3` × `backoff=1.0` against
>   `track_ttl=3.0` — a fast walker hits this. It is not an edge case.)
>
> `RETRYING` alone is NOT resolved: do not act, keep trying.
>
> **The invariant: every track that ever held a face acts exactly once, and a track that never
> got a good look still opens an Unknown session.** A hole we can count beats a child who
> silently never ate — the same asymmetry as the exit threshold in Task 12.
>
> Use an explicit state (`BOUND` / `RETRYING` / `EXHAUSTED`) plus a `should_act` flag, named so
> the next reader cannot confuse "we already acted" with "we failed to recognise".


Spec §4.4. `worker/canteen.py` calls `recognizer.detect()` — face detection **and** the
512-d ArcFace embedding — on every due frame, every 0.25 s, for every face in shot. The
expensive half is the embedding. For five children queuing over ten seconds that is roughly
**200 embeddings**. Per-track binding costs **five**.

The design (the client's doc §12.2): track → best face frame → recognise once → bind →
cache → re-run only when the track is lost.

> **SCOPE, amended after measurement (spec §2.4, §2.5) — read before writing code.**
>
> `IdentityService` is a **canteen** capability. It is built so that *any* camera can call
> it — that is the point of moving the config out of `config/canteen.py`, and it costs
> nothing beyond where the models live. **But the only cameras that will actually use it in
> this phase are the canteen cameras.**
>
> **Do NOT build a per-event / HD-burst identity binding for the bullying cameras.** It was
> considered and measured, and it does not work:
>
> - At the analysis resolution the hall worker really uses (**1280×720** — `hall.yaml`
>   overrides `base.yaml`'s 960×540 default), **0 of 14 970** hall faces clear the strict
>   gate, and of the 77 that clear the small-face gate **none is recognised** (best score
>   0.350). Identity on the hall analysis stream is arithmetically impossible, not merely
>   poor.
> - On the HD burst — where the pixels do exist — 332 of 14 970 faces clear the 60 px gate,
>   and **exactly one** is accepted at `min_score = 0.50`. Corrected for our 138-of-~800
>   gallery coverage, that is **~2% of events producing a name**.
>
> A path that fires once in fifty is not a capability, and building it would create an
> expectation the optics cannot honour. **Bullying events stay anonymous at the current
> camera placement**, and the system says so. Naming children in a bullying event needs a
> camera that can see a face — a chokepoint or face-height camera. That is the school's
> decision and it is optical, not a threshold. YAGNI: when such a camera exists, this
> architecture already accommodates it.

**Files:**
- Modify: `src/qorgan/faces/recognizer.py` (split `detect_faces()` / `embed()`)
- Create: `src/qorgan/identity/tracks.py`, `src/qorgan/identity/binding.py`,
  `src/qorgan/identity/service.py`
- Modify: `src/qorgan/worker/canteen.py` (bind per track, embed once),
  `src/qorgan/worker/entrypoint.py` (a `PersonDetector` per canteen camera; `require_gpu`),
  `src/qorgan/identity/cli.py` (use `detect_faces`, drop the temporary adapter)
- Test: Create `tests/test_identity_tracks.py`, `tests/test_identity_binding.py`,
  `tests/test_identity_service.py`; modify `tests/test_canteen_worker.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.detection.geometry import Box       # .width/.height/.area/.center (floats)
from qorgan.models.person import PersonDetector # .detect(frame) -> dict[int, Box]  (track_id -> box)
from qorgan.config.identity import BindingSettings, FaceGate, RecognitionPolicy, SoftAccumulator
from qorgan.faces.matching import Reason, Recognition, identify
from qorgan.faces.accumulator import TrackAccumulator, accept_small_face
from qorgan.faces.gallery import GalleryCache, PersonInfo
```

*Produces:*
```python
# src/qorgan/faces/recognizer.py
@dataclass(frozen=True, slots=True)
class FaceBox:
    """A face WITHOUT its vector. Cheap: this is what detection costs."""
    box: Box
    detection_score: float
    landmarks: np.ndarray          # (5, 2) float32 -- ArcFace needs these to align the crop
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def quality(self) -> float: ...    # area * detection_score

@dataclass(frozen=True, slots=True)
class DetectedFace:                    # unchanged
    box: Box
    embedding: np.ndarray
    detection_score: float

class FaceRecognizer:
    def detect_faces(self, frame: np.ndarray) -> list[FaceBox]: ...   # NEW: cheap
    def embed(self, frame: np.ndarray, face: FaceBox) -> np.ndarray: ...  # NEW: expensive
    def detect(self, frame: np.ndarray) -> list[DetectedFace]: ...    # detect_faces + embed
                                                                      # kept for the importer

# src/qorgan/identity/tracks.py
def assign_faces_to_tracks(
    faces: Sequence[FaceBox], person_boxes: Mapping[int, Box]
) -> dict[int, FaceBox]: ...
    # Best face per track, by containment. PURE. A face in nobody's box is dropped.

# src/qorgan/identity/binding.py
class BindState(StrEnum):
    OBSERVING = "observing"
    BOUND = "bound"
    RETRYING = "retrying"
    EXHAUSTED = "exhausted"

@dataclass(frozen=True, slots=True)
class Binding:
    track_id: int
    state: BindState
    person_id: int | None
    score: float
    attempts: int
    observations: int
    best: FaceBox | None
    first_seen: float
    last_seen: float
    next_attempt_at: float

class BindingTable:
    def __init__(self, config: BindingSettings) -> None: ...
    def observe(self, track_id: int, face: FaceBox | None, now: float) -> Binding: ...
    def should_embed(self, track_id: int, now: float) -> bool: ...
    def bind(self, track_id: int, recognition: Recognition, now: float) -> Binding: ...
    def get(self, track_id: int) -> Binding | None: ...
    def person_for(self, track_id: int) -> int | None: ...
    def evict(self, live: Collection[int], now: float) -> list[int]: ...
    def __len__(self) -> int: ...

# src/qorgan/identity/service.py
@dataclass(frozen=True, slots=True)
class Identified:
    track_id: int
    person_id: int | None
    person: PersonInfo | None
    recognition: Recognition
    face: FaceBox
    newly_bound: bool          # True on the frame the binding was made. False for ever after.

class IdentityService:
    def __init__(
        self,
        recognizer: FaceRecognizer,
        gallery: GalleryCache,
        policy: RecognitionPolicy,
        binding: BindingSettings,
        *,
        soft: SoftAccumulator | None = None,
    ) -> None: ...
    def on_frame(
        self, image: np.ndarray, person_boxes: dict[int, Box], now: float
    ) -> list[Identified]: ...
    def evict(self, live: Collection[int], now: float) -> list[int]: ...
```

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_identity_tracks.py`.**

```python
"""Faces to person tracks, by containment. A pure function, tested without a GPU."""

from __future__ import annotations

import numpy as np

from qorgan.detection.geometry import Box
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.tracks import assign_faces_to_tracks


def _face(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> FaceBox:
    return FaceBox(
        box=Box(x1, y1, x2, y2),
        detection_score=score,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )


def test_a_face_inside_a_person_belongs_to_that_person() -> None:
    faces = [_face(110, 110, 150, 160)]
    people = {7: Box(100, 100, 200, 400)}

    assert assign_faces_to_tracks(faces, people) == {7: faces[0]}


def test_a_face_in_nobodys_box_is_dropped() -> None:
    """A face with no person under it is a poster, a reflection, or a bug. It never gets a
    track, so it never gets a meal session."""
    assert assign_faces_to_tracks([_face(10, 10, 40, 50)], {7: Box(500, 500, 600, 800)}) == {}


def test_a_face_between_two_people_goes_to_the_tighter_box() -> None:
    """Two children stand close, so the face lands inside BOTH boxes. It belongs to the
    one whose box it fits most tightly — the person actually standing there, not the one
    behind them with the bigger box."""
    face = _face(110, 110, 150, 160)
    tight = Box(100, 100, 200, 400)
    loose = Box(50, 50, 400, 900)

    assert assign_faces_to_tracks([face], {7: loose, 9: tight}) == {9: face}


def test_one_track_keeps_only_its_BEST_face() -> None:
    """Rule R8: one object per track, not a list. Quality = area x detection score, so a
    big confident face beats a small hesitant one."""
    small = _face(110, 110, 130, 135, score=0.99)
    big = _face(110, 110, 170, 180, score=0.80)
    people = {7: Box(100, 100, 200, 400)}

    assert assign_faces_to_tracks([small, big], people) == {7: big}


def test_two_people_each_keep_their_own_face() -> None:
    left = _face(110, 110, 150, 160)
    right = _face(310, 110, 350, 160)
    people = {7: Box(100, 100, 200, 400), 9: Box(300, 100, 400, 400)}

    assert assign_faces_to_tracks([left, right], people) == {7: left, 9: right}


def test_no_people_means_no_assignments() -> None:
    assert assign_faces_to_tracks([_face(10, 10, 40, 50)], {}) == {}
```

- [ ] **Step 2: Write the failing tests — `tests/test_identity_binding.py`.**

```python
"""The bind / retry / evict state machine. Pure: a fake clock, no GPU, no DB."""

from __future__ import annotations

import numpy as np

from qorgan.config.identity import BindingSettings
from qorgan.detection.geometry import Box
from qorgan.faces.matching import Ranked, Reason, Recognition
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.binding import BindingTable, BindState

CONFIG = BindingSettings(
    min_face_frames=3,
    max_wait_seconds=1.5,
    max_attempts=2,
    retry_backoff_seconds=1.0,
    track_ttl_seconds=3.0,
)


def _face(score: float = 0.9, size: float = 100.0) -> FaceBox:
    return FaceBox(
        box=Box(0.0, 0.0, size, size * 1.2),
        detection_score=score,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )


def _accepted(person_id: int = 10) -> Recognition:
    return Recognition(person_id, 0.82, 0.31, Reason.ACCEPTED, (Ranked(person_id, 0.82),))


def _rejected() -> Recognition:
    return Recognition(None, 0.21, 0.02, Reason.LOW_SCORE, (Ranked(10, 0.21), Ranked(20, 0.19)))


# -- observing ----------------------------------------------------------------


def test_one_glance_is_not_enough_to_spend_an_embedding_on() -> None:
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)

    assert not table.should_embed(1, now=0.0)


def test_after_enough_frames_we_embed() -> None:
    table = BindingTable(CONFIG)
    for tick in range(CONFIG.min_face_frames):
        table.observe(1, _face(), now=float(tick) * 0.1)

    assert table.should_embed(1, now=0.3)


def test_a_child_who_keeps_turning_away_is_still_recognised_eventually() -> None:
    """`max_wait_seconds`. A track we have only seen once, but have been watching for a
    second and a half, is embedded anyway — otherwise the child who looks at the floor for
    the whole queue is never recognised at all."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)
    table.observe(1, None, now=1.0)  # no face this frame; the track is still there

    assert not table.should_embed(1, now=1.0)
    assert table.should_embed(1, now=1.6)


def test_a_track_with_no_face_at_all_is_never_embedded() -> None:
    """There is nothing to embed. `should_embed` must not promise a face we do not have."""
    table = BindingTable(CONFIG)
    for tick in range(5):
        table.observe(1, None, now=float(tick))

    assert not table.should_embed(1, now=10.0)


def test_the_best_face_seen_so_far_is_the_one_we_keep() -> None:
    """One object per track, not a list (rule R8)."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(score=0.99, size=40.0), now=0.0)
    binding = table.observe(1, _face(score=0.80, size=120.0), now=0.1)

    assert binding.best is not None
    assert binding.best.width == 120


# -- binding ------------------------------------------------------------------


def test_an_accepted_track_is_never_recognised_again() -> None:
    """**The whole point.** Five children queuing over ten seconds cost 5 embeddings, not
    200 (spec §4.4)."""
    table = BindingTable(CONFIG)
    for tick in range(3):
        table.observe(1, _face(), now=float(tick) * 0.1)
    table.bind(1, _accepted(person_id=42), now=0.3)

    assert table.person_for(1) == 42
    assert table.get(1).state is BindState.BOUND

    for tick in range(40):
        table.observe(1, _face(), now=1.0 + tick * 0.1)
        assert not table.should_embed(1, now=1.0 + tick * 0.1), "a bound track was re-embedded"


def test_a_rejected_track_is_retried_after_a_backoff() -> None:
    """This is where the small-face path lives: a weak look, then a better one."""
    table = BindingTable(CONFIG)
    for tick in range(3):
        table.observe(1, _face(), now=float(tick) * 0.1)
    table.bind(1, _rejected(), now=0.3)

    assert table.get(1).state is BindState.RETRYING
    assert not table.should_embed(1, now=0.5), "retried with no backoff at all"

    table.observe(1, _face(), now=1.4)
    assert table.should_embed(1, now=1.4)


def test_a_track_that_keeps_failing_gives_up_rather_than_burning_the_gpu_forever() -> None:
    table = BindingTable(CONFIG)
    for attempt in range(CONFIG.max_attempts):
        at = attempt * 2.0
        table.observe(1, _face(), now=at)
        table.bind(1, _rejected(), now=at)

    assert table.get(1).state is BindState.EXHAUSTED
    table.observe(1, _face(), now=100.0)
    assert not table.should_embed(1, now=100.0)


# -- eviction -----------------------------------------------------------------


def test_a_track_that_is_gone_is_evicted() -> None:
    """The next child to get this track id is a DIFFERENT child. A binding that outlives
    its track hands one pupil another pupil's identity."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)
    table.bind(1, _accepted(), now=0.0)

    assert table.evict(live=(1,), now=1.0) == []
    assert table.person_for(1) == 10

    assert table.evict(live=(), now=1.0) == []  # still inside the TTL; a flicker is not a loss
    assert table.evict(live=(), now=5.0) == [1]
    assert table.person_for(1) is None


def test_the_table_is_bounded() -> None:
    """Rule R8. Track ids only ever increase and a canteen runs all year."""
    table = BindingTable(CONFIG)

    for track_id in range(500):
        now = float(track_id)
        table.observe(track_id, _face(), now=now)
        table.evict(live=(track_id,), now=now)

    assert len(table) < 10
```

- [ ] **Step 3: Write the failing tests — `tests/test_identity_service.py`.**

```python
"""One embedding per track. This is the test the whole design exists for."""

from __future__ import annotations

import numpy as np

from qorgan.config.identity import BindingSettings, RecognitionPolicy
from qorgan.detection.geometry import Box
from qorgan.faces.gallery import Gallery, GalleryCache, PersonInfo, normalise
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.service import IdentityService

FACE_BOX = Box(110.0, 110.0, 170.0, 182.0)
PERSON_BOX = Box(100.0, 100.0, 220.0, 500.0)


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class SpyRecognizer:
    """Counts what it is asked to do. The embedding is the expensive half."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._embedding = embedding
        self.detect_calls = 0
        self.embed_calls = 0

    def detect_faces(self, _frame: np.ndarray) -> list[FaceBox]:
        self.detect_calls += 1
        return [
            FaceBox(
                box=FACE_BOX,
                detection_score=0.95,
                landmarks=np.zeros((5, 2), dtype=np.float32),
            )
        ]

    def embed(self, _frame: np.ndarray, _face: FaceBox) -> np.ndarray:
        self.embed_calls += 1
        return self._embedding


class FrozenGallery(GalleryCache):
    """A GalleryCache that never touches the database."""

    def __init__(self, gallery: Gallery) -> None:
        self._frozen = gallery

    def get(self) -> Gallery:
        return self._frozen

    def reload(self) -> Gallery:
        return self._frozen


def _gallery(*people: tuple[int, np.ndarray]) -> FrozenGallery:
    return FrozenGallery(
        Gallery(
            matrix=normalise(np.stack([vector for _, vector in people])),
            person_ids=np.array([pid for pid, _ in people], dtype=np.int64),
            people={
                pid: PersonInfo(
                    person_id=pid,
                    external_id=f"student_{pid}",
                    full_name=None,
                    person_type=__import__(
                        "qorgan.enums", fromlist=["PersonType"]
                    ).PersonType.STUDENT,
                    class_name="5-А",
                    position=None,
                )
                for pid, _ in people
            },
            model_name="buffalo_l",
            model_version="1.0",
        )
    )


def _service(recognizer: SpyRecognizer, gallery: FrozenGallery) -> IdentityService:
    return IdentityService(
        recognizer=recognizer,  # type: ignore[arg-type]
        gallery=gallery,
        policy=RecognitionPolicy(),
        binding=BindingSettings(min_face_frames=3, max_wait_seconds=1.5),
    )


def _frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# -- THE test ----------------------------------------------------------------


def test_one_track_costs_exactly_one_embedding_across_forty_frames() -> None:
    """**The whole design, in one assertion.**

    The old canteen worker called `detect()` -- detection AND the 512-d ArcFace embedding
    -- on every due frame, every 0.25 s, for every face in shot. The expensive half is the
    embedding. For five children queuing over ten seconds that is ~200 embeddings.

    Per track: watch, keep the best face, embed ONCE, bind, and never look again (§4.4).
    """
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    for tick in range(40):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1)

    assert recognizer.embed_calls == 1, (
        f"the track was embedded {recognizer.embed_calls} times. That is the bug this "
        "module exists to kill."
    )
    assert recognizer.detect_calls == 40  # detection is cheap; it runs every frame


def test_the_person_is_bound_to_the_track_and_reported_once_as_new() -> None:
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    new = []
    for tick in range(10):
        for found in service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1):
            if found.newly_bound:
                new.append(found)

    assert len(new) == 1, "a meal session would be opened more than once for one child"
    assert new[0].track_id == 7
    assert new[0].person_id == 10
    assert new[0].person is not None
    assert new[0].person.display == "Ученик 10, 5-А"


def test_five_children_queuing_cost_five_embeddings_not_two_hundred() -> None:
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    boxes = {
        track: Box(100.0 + track * 200, 100.0, 220.0 + track * 200, 500.0) for track in range(5)
    }
    # The spy always puts its face at FACE_BOX, so only track 0's box contains it; give
    # every track its own frame instead, which is the honest way to count.
    for track, box in boxes.items():
        for tick in range(40):
            service.on_frame(_frame(), {track: PERSON_BOX if track == 0 else box}, now=tick * 0.1)

    assert recognizer.embed_calls <= 5


def test_a_face_with_no_person_under_it_never_reaches_the_gpu() -> None:
    """A face with nobody under it is a poster or a reflection. It costs nothing."""
    recognizer = SpyRecognizer(_vector(1))
    service = _service(recognizer, _gallery((10, _vector(1))))

    for tick in range(20):
        service.on_frame(_frame(), {}, now=tick * 0.1)

    assert recognizer.embed_calls == 0


def test_a_lost_track_is_evicted_so_the_next_child_is_not_mistaken_for_this_one() -> None:
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(
        recognizer,
        _gallery((10, alice), (20, _vector(2))),
    )

    for tick in range(5):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1)
    assert recognizer.embed_calls == 1

    # The track is gone for longer than the TTL, then a NEW person gets the same id.
    for tick in range(50):
        service.on_frame(_frame(), {}, now=1.0 + tick * 0.1)
    for tick in range(5):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=10.0 + tick * 0.1)

    assert recognizer.embed_calls == 2, "a recycled track id inherited someone else's identity"
```

- [ ] **Step 4: Run all three, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_tracks.py tests/test_identity_binding.py tests/test_identity_service.py -q
```

Expected failure — collection errors:

```
ImportError: cannot import name 'FaceBox' from 'qorgan.faces.recognizer'
ModuleNotFoundError: No module named 'qorgan.identity.tracks'
```

- [ ] **Step 5: Implement — split the recognizer.**

Replace the body of `src/qorgan/faces/recognizer.py` below the imports (keep the module
docstring and append a paragraph to it):

Append to the docstring:

```
**Detection and embedding are two different prices.** Finding a face is cheap; turning it
into a 512-d ArcFace vector is not. The canteen worker used to pay both on every face in
every due frame. So they are two methods now, and the caller chooses -- which is what lets
`IdentityService` embed once per person rather than forty times per person.
```

```python
@dataclass(frozen=True, slots=True)
class FaceBox:
    """A face WITHOUT its vector. This is what detection costs, and it is the cheap half."""

    box: Box
    detection_score: float
    # ArcFace aligns the crop from these before it embeds. Carrying them here is what lets
    # detection and embedding be two separate calls.
    landmarks: np.ndarray  # (5, 2) float32

    @property
    def width(self) -> int:
        return int(self.box.width)

    @property
    def height(self) -> int:
        return int(self.box.height)

    @property
    def quality(self) -> float:
        """How good a look is this? Big and confident beats small and hesitant.

        One number, so "the best face of this track so far" is a comparison and not an
        argument.
        """
        return self.box.area * self.detection_score


@dataclass(frozen=True, slots=True)
class DetectedFace:
    box: Box
    embedding: np.ndarray  # L2-normalised, so a dot product is a cosine
    detection_score: float

    @property
    def width(self) -> int:
        return int(self.box.width)

    @property
    def height(self) -> int:
        return int(self.box.height)


class FaceRecognizer:
    """Finds faces and embeds them. Shared by every camera in a worker group."""

    _instance: FaceRecognizer | None = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: FaceModelSettings, device_id: int = 0) -> None:
        from insightface.app import FaceAnalysis

        from qorgan.gpu import CUDA_PROVIDER, inspect_gpu

        report = inspect_gpu()
        if not report.onnx_cuda:
            # InsightFace falls back to the CPU with a warning nobody reads, and runs
            # ~40x too slow. On a canteen door that is not slow: it is broken. The guard
            # builds a real session, so it cannot be fooled by an import order (gpu.py).
            raise RuntimeError(
                f"onnxruntime did not reach {CUDA_PROVIDER}; refusing to run face "
                f"recognition on the CPU. A real session got: "
                f"{report.onnx_session_provider}. Run `qorgan doctor`."
            )

        self.settings = settings
        self._app = FaceAnalysis(name=settings.model_name, providers=[CUDA_PROVIDER])
        self._app.prepare(ctx_id=device_id, det_size=(settings.det_size, settings.det_size))
        self._lock = threading.Lock()

        logger.info(
            "face recognizer loaded",
            extra={"model": settings.model_name, "det_size": settings.det_size},
        )

    @classmethod
    def shared(cls, settings: FaceModelSettings, device_id: int = 0) -> FaceRecognizer:
        """One per process. Never build a second one -- see the module docstring."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings, device_id)
            return cls._instance

    # -- the cheap half ------------------------------------------------------

    def detect_faces(self, frame: np.ndarray) -> list[FaceBox]:
        """Boxes, landmarks and detection scores. **No embeddings.**

        This runs on every frame. It must not cost what `embed` costs, which is why it
        calls the detection model directly rather than `FaceAnalysis.get()` -- that
        convenience runs recognition, gender/age and both landmark models on every face.
        """
        if frame.size == 0:
            return []

        with self._lock:
            boxes, landmarks = self._app.det_model.detect(frame, max_num=0, metric="default")

        if boxes is None or len(boxes) == 0:
            return []

        found = []
        for index in range(len(boxes)):
            x1, y1, x2, y2, score = (float(v) for v in boxes[index][:5])
            points = (
                np.asarray(landmarks[index], dtype=np.float32)
                if landmarks is not None
                else np.zeros((5, 2), dtype=np.float32)
            )
            found.append(
                FaceBox(box=Box(x1, y1, x2, y2), detection_score=score, landmarks=points)
            )
        return found

    # -- the expensive half --------------------------------------------------

    def embed(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """One face -> one 512-d L2-normalised vector. **This is the expensive call.**

        Called ONCE per person track, not once per frame. That difference is ~200
        embeddings against 5 for a queue of five children (spec §4.4).
        """
        from insightface.app.common import Face

        raw = Face(
            bbox=np.array(
                [face.box.x1, face.box.y1, face.box.x2, face.box.y2], dtype=np.float32
            ),
            kps=face.landmarks,
            det_score=face.detection_score,
        )
        with self._lock:
            vector = self._app.models["recognition"].get(frame, raw)

        return _normalise(np.asarray(vector, dtype=np.float32).ravel())

    # -- both, for the importer ----------------------------------------------

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Every face in the frame, embedded. Used by the roster import, where every
        photo has exactly one face and we want its vector immediately."""
        faces = self.detect_faces(frame)
        embedded = []

        for face in faces:
            vector = self.embed(frame, face)
            if vector.shape != (self.settings.embedding_dim,):
                logger.warning(
                    "discarding a face with an unexpected embedding shape",
                    extra={
                        "shape": list(vector.shape),
                        "expected": self.settings.embedding_dim,
                    },
                )
                continue
            embedded.append(
                DetectedFace(
                    box=face.box,
                    embedding=vector,
                    detection_score=face.detection_score,
                )
            )
        return embedded


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else (vector / norm).astype(np.float32)
```

- [ ] **Step 6: Implement — `src/qorgan/identity/tracks.py`.**

```python
"""Which face belongs to which person? **A pure function, and that is the point.**

The canteen worker used to see a list of faces and nothing else — no idea whether the face
in frame 40 was the same child as the face in frame 1. So it recognised every face, every
time, and the small-face accumulator corroborated hits across whole different children
(spec §4.5).

A person track answers that, and assigning a face to one is geometry. Geometry does not
need a GPU, so this is unit-testable with a handful of boxes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qorgan.detection.geometry import Box
from qorgan.faces.recognizer import FaceBox


def assign_faces_to_tracks(
    faces: Sequence[FaceBox], person_boxes: Mapping[int, Box]
) -> dict[int, FaceBox]:
    """The best face per person track. **One object per track, not a list** (rule R8).

    A face is assigned to the person box that CONTAINS its centre. Two children standing
    close means one face sits inside both boxes, so the tightest box wins — the person
    actually standing there, not the one behind them.

    A face inside nobody's box is DROPPED. It is a poster, a reflection, or a bug, and it
    never gets a track, so it never gets a meal session.
    """
    best: dict[int, FaceBox] = {}

    for face in faces:
        track_id = _owner(face, person_boxes)
        if track_id is None:
            continue
        current = best.get(track_id)
        if current is None or face.quality > current.quality:
            best[track_id] = face

    return best


def _owner(face: FaceBox, person_boxes: Mapping[int, Box]) -> int | None:
    """The tightest person box containing this face's centre."""
    cx, cy = face.box.center

    containing = [
        (box.area, track_id)
        for track_id, box in person_boxes.items()
        if box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2
    ]
    if not containing:
        return None
    return min(containing)[1]
```

- [ ] **Step 7: Implement — `src/qorgan/identity/binding.py`.**

```python
"""Bind, retry, evict. **A pure state machine: no GPU, no clock, no database.**

The clock is an argument. That is what makes "a child who turns away for the whole queue
is still recognised after max_wait_seconds" a unit test rather than a thing you find out
about in a canteen.

  * Watch a track. Keep only the **best face seen so far** — one object, not a list (R8).
  * After `min_face_frames` observations OR `max_wait_seconds`, whichever comes first,
    the caller may spend one embedding.
  * Accepted => BOUND. Never recognised again.
  * Rejected => RETRYING, after a backoff, up to `max_attempts`. (This is where the
    small-face path lives: several weak looks at one child are worth more than one.)
  * Track lost for `track_ttl_seconds` => evicted. **The next child to get that track id
    is a different child**, and a binding that outlives its track hands one pupil another
    pupil's identity.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from enum import StrEnum

from qorgan.config.identity import BindingSettings
from qorgan.faces.matching import Recognition
from qorgan.faces.recognizer import FaceBox


class BindState(StrEnum):
    OBSERVING = "observing"  # watching, not yet worth an embedding
    BOUND = "bound"  # we know who this is. Done.
    RETRYING = "retrying"  # a look that was not good enough. Try again shortly.
    EXHAUSTED = "exhausted"  # out of attempts. Stop burning the GPU on this one.


@dataclass(frozen=True, slots=True)
class Binding:
    track_id: int
    state: BindState
    person_id: int | None
    score: float
    attempts: int
    observations: int
    best: FaceBox | None
    first_seen: float
    last_seen: float
    next_attempt_at: float


class BindingTable:
    """One per camera. Not thread-safe, because one camera loop owns it."""

    def __init__(self, config: BindingSettings) -> None:
        self.config = config
        self._bindings: dict[int, Binding] = {}

    def __len__(self) -> int:
        return len(self._bindings)

    def get(self, track_id: int) -> Binding | None:
        return self._bindings.get(track_id)

    def person_for(self, track_id: int) -> int | None:
        binding = self._bindings.get(track_id)
        return binding.person_id if binding and binding.state is BindState.BOUND else None

    def observe(self, track_id: int, face: FaceBox | None, now: float) -> Binding:
        """This track is still here. Keep its best face; count the look."""
        current = self._bindings.get(track_id)
        if current is None:
            current = Binding(
                track_id=track_id,
                state=BindState.OBSERVING,
                person_id=None,
                score=0.0,
                attempts=0,
                observations=0,
                best=None,
                first_seen=now,
                last_seen=now,
                next_attempt_at=now,
            )

        best = current.best
        if face is not None and (best is None or face.quality > best.quality):
            best = face

        updated = replace(
            current,
            observations=current.observations + (1 if face is not None else 0),
            best=best,
            last_seen=now,
        )
        self._bindings[track_id] = updated
        return updated

    def should_embed(self, track_id: int, now: float) -> bool:
        """Is it worth spending one 512-d ArcFace embedding on this track right now?"""
        binding = self._bindings.get(track_id)
        if binding is None or binding.best is None:
            return False
        if binding.state in (BindState.BOUND, BindState.EXHAUSTED):
            return False
        if now < binding.next_attempt_at:
            return False

        enough_looks = binding.observations >= self.config.min_face_frames
        # A child who looks at the floor for the whole queue must still be recognised.
        waited_long_enough = (now - binding.first_seen) >= self.config.max_wait_seconds
        return enough_looks or waited_long_enough

    def bind(self, track_id: int, recognition: Recognition, now: float) -> Binding:
        """Apply the one recognition we spent an embedding on."""
        binding = self._bindings[track_id]
        attempts = binding.attempts + 1

        if recognition.accepted:
            updated = replace(
                binding,
                state=BindState.BOUND,
                person_id=recognition.person_id,
                score=recognition.score,
                attempts=attempts,
                last_seen=now,
            )
        else:
            exhausted = attempts >= self.config.max_attempts
            updated = replace(
                binding,
                state=BindState.EXHAUSTED if exhausted else BindState.RETRYING,
                person_id=None,
                score=recognition.score,
                attempts=attempts,
                # Start the next look from scratch: the face that failed is not the face
                # that will succeed.
                observations=0,
                best=None,
                next_attempt_at=now + self.config.retry_backoff_seconds,
                last_seen=now,
            )

        self._bindings[track_id] = updated
        return updated

    def evict(self, live: Collection[int], now: float) -> list[int]:
        """Forget tracks that have been gone longer than the TTL.

        A short absence is a flicker -- a head turns, YOLO loses a frame -- and evicting on
        it would re-embed the same child every second. A long one is a different child
        walking into a recycled track id, and NOT evicting on that is how one pupil is
        recorded eating another pupil's lunch.
        """
        cutoff = now - self.config.track_ttl_seconds
        gone = [
            track_id
            for track_id, binding in self._bindings.items()
            if track_id not in live and binding.last_seen < cutoff
        ]
        for track_id in gone:
            del self._bindings[track_id]
        return gone
```

- [ ] **Step 8: Implement — `src/qorgan/identity/service.py`.**

```python
"""The impure shell: recognizer + gallery + bindings. Everything it DECIDES is pure.

    detect_faces (cheap, every frame)
      -> assign_faces_to_tracks   (pure geometry)
      -> BindingTable             (pure state machine)
      -> embed + identify         (expensive, ONCE per track)

For five children queuing over ten seconds: **5 embeddings instead of ~200** (spec §4.4).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from qorgan.config.identity import BindingSettings, RecognitionPolicy, SoftAccumulator
from qorgan.detection.geometry import Box
from qorgan.faces.accumulator import TrackAccumulator, accept_small_face
from qorgan.faces.gallery import GalleryCache, PersonInfo
from qorgan.faces.matching import Recognition, identify
from qorgan.faces.recognizer import FaceBox, FaceRecognizer
from qorgan.identity.binding import BindingTable, BindState
from qorgan.identity.tracks import assign_faces_to_tracks
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Identified:
    """One track, this frame."""

    track_id: int
    person_id: int | None
    person: PersonInfo | None
    recognition: Recognition
    face: FaceBox
    # True on the ONE frame the binding was made, and never again. A caller that opens a
    # meal session on this cannot open forty of them.
    newly_bound: bool

    @property
    def is_staff(self) -> bool:
        return self.person is not None and self.person.is_staff


class IdentityService:
    """One per camera. Owns that camera's bindings; shares the recognizer and gallery."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        gallery: GalleryCache,
        policy: RecognitionPolicy,
        binding: BindingSettings,
        *,
        soft: SoftAccumulator | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._gallery = gallery
        self._policy = policy
        self._table = BindingTable(binding)
        self._soft = TrackAccumulator(soft) if soft else None

    def on_frame(
        self, image: np.ndarray, person_boxes: dict[int, Box], now: float
    ) -> list[Identified]:
        """Cheap detection every frame; one expensive embedding per track, ever."""
        faces = assign_faces_to_tracks(self._recognizer.detect_faces(image), person_boxes)

        for track_id in person_boxes:
            self._table.observe(track_id, faces.get(track_id), now)

        self.evict(person_boxes.keys(), now)

        return [
            found
            for track_id, face in faces.items()
            if (found := self._resolve(track_id, face, image, now)) is not None
        ]

    def _resolve(
        self, track_id: int, face: FaceBox, image: np.ndarray, now: float
    ) -> Identified | None:
        """Report a bound track; spend an embedding on one that is ready for it."""
        if not self._table.should_embed(track_id, now):
            return self._already_bound(track_id, face)

        # THE expensive call. Once per track.
        best = self._table.get(track_id).best or face
        gallery = self._gallery.get()
        recognition = identify(
            self._recognizer.embed(image, best), gallery.matrix, gallery.person_ids, self._policy
        )
        recognition = self._soften(recognition, best, now)

        binding = self._table.bind(track_id, recognition, now)
        if binding.state is not BindState.BOUND:
            return Identified(
                track_id=track_id,
                person_id=None,
                person=None,
                recognition=recognition,
                face=best,
                newly_bound=False,
            )

        logger.info(
            "track bound",
            extra={
                "track_id": track_id,
                "person_id": binding.person_id,
                "score": round(recognition.score, 3),
                "attempts": binding.attempts,
            },
        )
        return Identified(
            track_id=track_id,
            person_id=binding.person_id,
            person=gallery.info(binding.person_id),
            recognition=recognition,
            face=best,
            newly_bound=True,
        )

    def _already_bound(self, track_id: int, face: FaceBox) -> Identified | None:
        person_id = self._table.person_for(track_id)
        if person_id is None:
            return None

        binding = self._table.get(track_id)
        gallery = self._gallery.get()
        return Identified(
            track_id=track_id,
            person_id=person_id,
            person=gallery.info(person_id),
            recognition=Recognition(
                person_id, binding.score, 0.0, __import__(
                    "qorgan.faces.matching", fromlist=["Reason"]
                ).Reason.ACCEPTED
            ),
            face=face,
            newly_bound=False,
        )

    def _soften(self, recognition: Recognition, face: FaceBox, now: float) -> Recognition:
        """The small-face path. Younger pupils' faces are systematically below the size
        gate, so a strict single-shot threshold never recognises the first-graders at all."""
        if self._soft is None:
            return recognition
        if not self._soft.config.face_gate.accepts(face.width, face.height):
            return recognition
        return accept_small_face(recognition, self._soft, now)

    def evict(self, live: Collection[int], now: float) -> list[int]:
        gone = self._table.evict(live, now)
        if self._soft is not None:
            self._soft.prune(now)
        return gone
```

Note the `__import__` in `_already_bound` is ugly. Replace it: add `Reason` to the
`qorgan.faces.matching` import at the top of the file and write
`Recognition(person_id, binding.score, 0.0, Reason.ACCEPTED)`. Do the same in the test
file's `_gallery` helper (import `PersonType` from `qorgan.enums` at the top). Ruff's `F401`
and readability both want this; the `__import__` above is a mistake, not a style.

- [ ] **Step 9: Implement — rewire `worker/canteen.py`.**

`CanteenPipeline` gains a `PersonDetector` and delegates recognition to `IdentityService`.
The role logic (entry opens, exit closes, inside confirms) is untouched — that is the part
that was already right.

Replace the imports and the class down to `_decide`:

```python
from qorgan.canteen.sessions import SessionManager
from qorgan.capture import Frame
from qorgan.config.camera import CanteenCamera
from qorgan.config.identity import BindingSettings, RecognitionPolicy, SoftAccumulator
from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import RecognitionAttempt
from qorgan.db.types import utcnow
from qorgan.enums import CameraRole
from qorgan.identity.service import Identified, IdentityService
from qorgan.logging_setup import get_logger
from qorgan.models.person import PersonDetector

logger = get_logger(__name__)
```

Delete the `Decision` dataclass — `Identified` replaces it, and it carries `is_staff`
already. Then:

```python
class CanteenPipeline:
    """One camera's worth of canteen logic."""

    def __init__(
        self,
        camera: CanteenCamera,
        camera_id: int,
        person: PersonDetector,
        identity: IdentityService,
        sessions: SessionManager,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self._person = person
        self._identity = identity
        self._sessions = sessions
        # None, not 0.0. A monotonic clock starts near zero, so a 0.0 sentinel would make
        # the camera skip its own first frames -- exactly when a queue is forming at the
        # door.
        self._last_scan: float | None = None

    # -- the frame loop ------------------------------------------------------

    def on_frame(self, _camera: CanteenCamera, frame: Frame) -> str:
        """Track the people, recognise each ONCE, and act according to our role."""
        if not self._due(frame.captured_at):
            return "ok"

        people = self._person.detect(frame.image)
        acted = False

        for found in self._identity.on_frame(frame.image, people, frame.captured_at):
            if not found.newly_bound:
                # We already know who this track is. The whole point: no second embedding,
                # and no second meal session either.
                continue
            self._record_attempt(found)
            acted |= self._act(found)

        return "alert" if acted else "ok"

    def _due(self, now: float) -> bool:
        """Recognition is expensive. An inside camera does not need to run it ten times a
        second, and the legacy called detect_faces up to THIRTY times a second purely to
        decide whether to keep going (audit M-33)."""
        if self._last_scan is not None and now - self._last_scan < self._interval():
            return False
        self._last_scan = now
        return True

    # -- acting ---------------------------------------------------------------

    def _act(self, found: Identified) -> bool:
        role = self.camera.role
        if role is CameraRole.CANTEEN_ENTRY:
            return self._on_entry(found)
        if role is CameraRole.CANTEEN_EXIT:
            return self._on_exit(found)
        return self._on_inside(found)
```

then rename the parameter in `_on_entry`, `_on_exit`, `_on_inside` and `_record_attempt`
from `decision: Decision` to `found: Identified`, and inside each replace
`decision.recognition` with `found.recognition`, `decision.is_staff` with `found.is_staff`,
and `decision.face` with `found.face`. The bodies are otherwise unchanged — including
`_on_inside`'s long comment about why late-binding is deliberately not done, which is still
true and still load-bearing.

`_record_attempt` loses its `now` parameter (it never used it) and its
`decision.recognition.person_id` reads become `found.person_id`.

Finally, replace `_policy_for` with a builder that hands `IdentityService` everything it
needs:

```python
def build_identity(camera: CanteenCamera, recognizer, gallery) -> IdentityService:
    """One policy object per camera role.

    The legacy had EIGHTEEN overlapping thresholds, with "strong" gates bypassed by
    "soft" gates 0.02 apart — decorative rather than functional, and the fossil record of
    trying to fix a broken recognition pipeline by tuning it.
    """
    policy, soft, binding = _blocks_for(camera)
    return IdentityService(
        recognizer=recognizer,
        gallery=gallery,
        policy=policy,
        binding=binding,
        soft=soft,
    )


def _blocks_for(
    camera: CanteenCamera,
) -> tuple[RecognitionPolicy, SoftAccumulator | None, BindingSettings]:
    canteen = camera.canteen
    if canteen.entry is not None:
        return canteen.entry.recognition, canteen.entry.small_face, canteen.entry.binding
    if canteen.exit is not None:
        return canteen.exit.recognition, canteen.exit.soft, canteen.exit.binding
    if canteen.inside is not None:
        return canteen.inside.recognition, None, canteen.inside.binding
    raise ValueError(f"camera {camera.name!r} has no canteen role block")
```

`_max_faces()` is gone — the number of faces we look at is now the number of people YOLO is
tracking, which is the honest bound. Delete it and `max_faces_per_tick` stays in the config
as a dead key? **No.** Remove `max_faces_per_tick` from `ExitSettings` in
`config/canteen.py` — a config key no code reads is exactly what this project refuses to
ship (audit: dead config).

- [ ] **Step 10: Implement — a `PersonDetector` for the canteen cameras.**

In `src/qorgan/worker/entrypoint.py::_build_canteen`, replace the imports and the return:

```python
    from qorgan.canteen.sessions import SessionManager
    from qorgan.events.store import ensure_cameras
    from qorgan.faces.gallery import GalleryCache
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.gpu import require_gpu
    from qorgan.models.person import PersonDetector
    from qorgan.worker.canteen import CanteenPipeline, build_identity

    # Refuse to run 40x too slow on the CPU rather than doing it silently. The canteen
    # cameras now load YOLO as well as InsightFace, and both fail silently to the CPU.
    require_gpu()

    camera_ids = ensure_cameras(cameras)
    first = next(iter(canteen.values()))
    model = first.canteen.face_model

    recognizer = FaceRecognizer.shared(model)
    gallery = GalleryCache(model.model_name, model.model_version)

    entry_id = _role_id(cameras, camera_ids, CameraRole.CANTEEN_ENTRY)
    exit_id = _role_id(cameras, camera_ids, CameraRole.CANTEEN_EXIT)

    sessions = SessionManager(
        first.canteen.session,
        first.canteen.meal_outcome,
        entry_camera_id=entry_id or camera_ids[first.name],
        exit_camera_id=exit_id,
    )

    return {
        name: CanteenPipeline(
            camera=camera,
            camera_id=camera_ids[name],
            # One detector per CAMERA, never per group: Ultralytics keeps its tracker
            # state on the model object, so two cameras sharing one would hand each
            # other's children the same track ids -- and a track id is now an identity.
            person=PersonDetector(camera.yolo, name, device=group.device),
            identity=build_identity(camera, recognizer, gallery),
            sessions=sessions,
        )
        for name, camera in canteen.items()
    }
```

Update the `_build_canteen` docstring to say the canteen cameras now carry a YOLO too, and
that `qorgan plan-workers` (Task 11) is what sizes the fleet for it.

- [ ] **Step 11: Implement — `camera-report` uses the cheap call.**

In `src/qorgan/identity/cli.py::cmd_camera_report`, the `measure_faces(...)` call passes
`recognizer` where a `Detector` is expected. Change `identity/camera.py`'s `Detector`
protocol method from `detect` to `detect_faces`:

```python
class Detector(Protocol):
    def detect_faces(self, frame: np.ndarray) -> list[Sized]: ...
```

and in `measure_faces`, `for face in detector.detect_faces(frame):`. Then in
`tests/test_identity_camera.py`, rename `ScriptedDetector.detect` to `detect_faces`.
`camera-report` never needs a vector, and now it never pays for one.

- [ ] **Step 12: Implement — update `tests/test_canteen_worker.py`.**

`FakeRecognizer` gains the split API and a `FakePersonDetector` appears. Replace the
`FakeRecognizer` class and `_pipeline` helper:

```python
from qorgan.config.identity import BindingSettings
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.service import IdentityService
from qorgan.worker.canteen import CanteenPipeline

PERSON_BOX = Box(0.0, 0.0, 300.0, 700.0)


class FakeRecognizer:
    """Returns whatever face we tell it to. No model, no GPU."""

    def __init__(self) -> None:
        self.faces: list[FaceBox] = []
        self.embedding: np.ndarray = np.zeros(512, dtype=np.float32)
        self.embed_calls = 0

    def show(self, embedding: np.ndarray, width: int = 90, height: int = 110) -> None:
        self.embedding = embedding
        self.faces = [
            FaceBox(
                box=Box(10.0, 10.0, 10.0 + width, 10.0 + height),
                detection_score=0.9,
                landmarks=np.zeros((5, 2), dtype=np.float32),
            )
        ]

    def show_nobody(self) -> None:
        self.faces = []

    def detect_faces(self, _frame: np.ndarray) -> list[FaceBox]:
        return self.faces

    def embed(self, _frame: np.ndarray, _face: FaceBox) -> np.ndarray:
        self.embed_calls += 1
        return self.embedding


class FakePersonDetector:
    """One person, always in shot, always track 1 — until told otherwise."""

    def __init__(self) -> None:
        self.people: dict[int, Box] = {1: PERSON_BOX}

    def detect(self, _frame: np.ndarray) -> dict[int, Box]:
        return self.people


def _pipeline(
    role: CameraRole,
    rows: dict,
    recognizer: FakeRecognizer,
    person: FakePersonDetector | None = None,
) -> CanteenPipeline:
    sessions = SessionManager(SessionRules(), MealOutcomeRules(), rows["entry_id"], rows["exit_id"])
    camera_id = {
        CameraRole.CANTEEN_ENTRY: rows["entry_id"],
        CameraRole.CANTEEN_EXIT: rows["exit_id"],
        CameraRole.CANTEEN_INSIDE: rows["inside_id"],
    }[role]
    camera = _camera(role)

    identity = IdentityService(
        recognizer=recognizer,  # type: ignore[arg-type]
        gallery=GalleryCache(MODEL_NAME, MODEL_VERSION),
        policy=_blocks_for(camera)[0],
        binding=BindingSettings(min_face_frames=1, max_wait_seconds=0.0001),
        soft=None,
    )
    return CanteenPipeline(
        camera=camera,
        camera_id=camera_id,
        person=person or FakePersonDetector(),  # type: ignore[arg-type]
        identity=identity,
        sessions=sessions,
    )
```

with `from qorgan.worker.canteen import _blocks_for` imported. `min_face_frames=1` and a
tiny `max_wait_seconds` mean one `on_frame` call binds, which is what every existing test
in that file assumes. The existing tests then pass unchanged, except:

- `test_the_exit_camera_closes_a_session` drives two pipelines with one recognizer; each has
  its own `IdentityService` and its own bindings, so it still works.
- Add one new test at the end of that file:

```python
def test_a_child_at_the_door_is_embedded_once_not_once_per_frame(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The old worker called detect() -- detection AND the embedding -- on every due
    frame. For five children queuing over ten seconds that is ~200 embeddings (spec §4.4)."""
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])
    pipeline = _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer)

    for tick in range(40):
        pipeline.on_frame(None, _frame(at=tick * 0.5))

    assert recognizer.embed_calls == 1

    session.expire_all()
    assert len(session.scalars(select(CanteenSession)).all()) == 1
```

- [ ] **Step 13: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_identity_tracks.py tests/test_identity_binding.py tests/test_identity_service.py tests/test_canteen_worker.py tests/test_identity_camera.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green. If `tests/test_code_limits.py` fails on `worker/canteen.py` or
`identity/service.py`, split the file rather than shortening a comment.

- [ ] **Step 14: Commit.**

```
git status
git add src/qorgan/faces/recognizer.py src/qorgan/identity/tracks.py src/qorgan/identity/binding.py src/qorgan/identity/service.py src/qorgan/identity/camera.py src/qorgan/identity/cli.py src/qorgan/worker/canteen.py src/qorgan/worker/entrypoint.py src/qorgan/config/canteen.py tests/test_identity_tracks.py tests/test_identity_binding.py tests/test_identity_service.py tests/test_identity_camera.py tests/test_canteen_worker.py
git commit -m "IdentityService: recognise once per track, not once per frame

worker/canteen.py called recognizer.detect() -- face detection AND the 512-d ArcFace
embedding -- on every due frame, every 0.25s, for every face in shot. The expensive half
is the embedding. For five children queuing over ten seconds that is roughly 200 of them.

So FaceRecognizer splits: detect_faces() is cheap and runs every frame; embed() is
expensive and runs ONCE per person track. PersonDetector (YOLOv8n + ByteTrack, which the
hall cameras already run) comes to the canteen cameras and gives real track ids that
survive a head-turn. Faces are assigned to tracks by containment -- a pure function, unit
tested with no GPU. Per track we keep only the best face seen so far: one object, not a
list (R8). After min_face_frames observations or max_wait_seconds -- whichever comes
first, because the child who looks at the floor for the whole queue must still be
recognised -- we embed once, identify once, and bind.

Accepted => never recognised again. Rejected => retried with backoff, which is where the
small-face path lives. Track lost => evicted, because the next child to get that track id
is a different child.

Five children queuing: 5 embeddings instead of ~200. The test that pins it counts embed()
calls across 40 frames of one track and asserts exactly 1.

max_faces_per_tick is deleted rather than left as a key nobody reads: the bound on how many
faces we look at is now the number of people YOLO is tracking, which is the honest one."
```

---

### Task 10: The accumulator is keyed by the wrong thing

Spec §4.5. `TrackAccumulator.hits` is keyed by **`person_id`**
(`faces/accumulator.py:41`), and one accumulator is shared across the whole camera. So weak
top-1 hits from **different children** currently corroborate each other: a crowd of unknowns
can vote a stranger into being pupil X — **and that closes a meal session.**

The class is *called* `TrackAccumulator` and has never known what a track is. Task 9 gave it
one. Keying by `(track_id, person_id)` falls straight out of per-track binding.

**Files:**
- Modify: `src/qorgan/faces/accumulator.py`
- Modify: `src/qorgan/identity/service.py` (pass the track id through)
- Test: `tests/test_faces_matching.py`

**Interfaces:**

*Consumes:* `qorgan.config.identity.SoftAccumulator`, `qorgan.faces.matching.Recognition`.

*Produces:* the same names, one argument wider. **Every call site must pass a track id.**
```python
# src/qorgan/faces/accumulator.py
Key = tuple[int, int]     # (track_id, person_id)

@dataclass(frozen=True, slots=True)
class Hit:
    track_id: int         # NEW
    person_id: int
    score: float
    gap: float
    at: float

@dataclass
class TrackAccumulator:
    config: SoftConfig
    hits: dict[Key, deque[Hit]] = field(default_factory=dict)   # was: dict[int, deque[Hit]]

    def observe(self, recognition: Recognition, now: float, track_id: int) -> int | None: ...
    def evidence(self, track_id: int, person_id: int) -> int: ...
    def clear(self, track_id: int | None = None) -> None: ...   # a whole track, or everything
    def prune(self, now: float) -> None: ...

def accept_small_face(
    recognition: Recognition, accumulator: TrackAccumulator, now: float, track_id: int
) -> Recognition: ...
```

**Steps:**

- [ ] **Step 1: Write the failing test.**

In `tests/test_faces_matching.py`, replace the small-face block from
`def test_one_weak_look_is_not_enough` to the end of the file. Every call gains a
`track_id`, and one new test proves the bug is dead:

```python
def test_one_weak_look_is_not_enough() -> None:
    accumulator = _soft()
    result = accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)

    assert not result.accepted


def test_the_same_child_seen_twice_on_ONE_track_is_recognised() -> None:
    """The first-graders. Their faces are systematically below the size gate, so a strict
    single-shot threshold never recognises them AT ALL."""
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(10), accumulator, now=1.0, track_id=1)

    assert result.accepted
    assert result.person_id == 10
    assert result.reason is Reason.ACCEPTED


def test_two_different_TRACKS_do_not_corroborate_each_other() -> None:
    """**THE bug (spec §4.5).**

    `hits` was keyed by person_id, and one accumulator was shared across the whole camera.
    So a weak, wrong top-1 from child A and a weak, wrong top-1 from child B -- two
    entirely different children, both matching pupil X badly -- ADDED UP, and voted a
    stranger into being pupil X.

    That closes a meal session. A crowd of unknowns at a canteen door is not corroboration;
    it is a crowd.
    """
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(10), accumulator, now=1.0, track_id=2)

    assert not result.accepted, (
        "two different children corroborated each other into being one pupil, and that "
        "pupil is now recorded as having eaten"
    )
    assert accumulator.evidence(track_id=1, person_id=10) == 1
    assert accumulator.evidence(track_id=2, person_id=10) == 1


def test_two_different_children_on_one_track_do_not_corroborate_either() -> None:
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(20), accumulator, now=1.0, track_id=1)

    assert not result.accepted


def test_looks_that_are_too_far_apart_in_time_do_not_corroborate() -> None:
    """Two glances a minute apart are not corroboration; they are two separate children
    walking past the same door."""
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(10), accumulator, now=60.0, track_id=1)

    assert not result.accepted


def test_an_ambiguous_face_is_never_rescued_by_repetition() -> None:
    """Seeing two children who look alike five times running does not tell you which one
    it was. The accumulator cannot cure ambiguity; only a better look can."""
    from qorgan.faces.matching import Ranked, Recognition

    accumulator = _soft()
    ambiguous = Recognition(
        None, 0.60, 0.01, Reason.AMBIGUOUS, (Ranked(10, 0.60), Ranked(20, 0.59))
    )

    for tick in range(5):
        result = accept_small_face(ambiguous, accumulator, now=float(tick), track_id=1)
        assert not result.accepted


def test_a_face_that_matched_nobody_never_accumulates() -> None:
    """Otherwise the accumulator eventually 'recognises' a child out of pure noise."""
    accumulator = _soft()
    noise = _weak_look(10, score=0.05, gap=0.001)

    for tick in range(10):
        assert not accept_small_face(noise, accumulator, now=float(tick), track_id=1).accepted


def test_an_accepted_face_is_left_alone() -> None:
    from qorgan.faces.matching import Ranked, Recognition

    accumulator = _soft()
    already = Recognition(10, 0.9, 0.4, Reason.ACCEPTED, (Ranked(10, 0.9),))

    assert accept_small_face(already, accumulator, now=0.0, track_id=1) is already


def test_a_finished_track_can_be_forgotten_in_one_call() -> None:
    accumulator = _soft()
    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    accept_small_face(_weak_look(20), accumulator, now=0.0, track_id=1)
    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=2)

    accumulator.clear(track_id=1)

    assert accumulator.evidence(track_id=1, person_id=10) == 0
    assert accumulator.evidence(track_id=2, person_id=10) == 1


def test_the_accumulator_is_bounded() -> None:
    """Rule R8. A canteen runs all year, and track ids only ever increase."""
    accumulator = _soft()

    for index in range(500):
        accept_small_face(_weak_look(index), accumulator, now=float(index), track_id=index)
        accumulator.prune(now=float(index))

    assert len(accumulator.hits) < 20
```

- [ ] **Step 2: Run it, watch it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_faces_matching.py -q
```

Expected failure — every small-face test errors on the new keyword, and the new one names
the bug:

```
TypeError: accept_small_face() got an unexpected keyword argument 'track_id'
```

- [ ] **Step 3: Implement — rekey the accumulator.**

In `src/qorgan/faces/accumulator.py`, append to the module docstring:

```
**And the bug this design kills on the way past.** `hits` used to be keyed by `person_id`,
with one accumulator shared across the whole camera. So weak top-1 hits from DIFFERENT
children corroborated each other: a crowd of unknowns at the door could vote a stranger
into being pupil X — and that closes a meal session. The class was called
`TrackAccumulator` and had never known what a track was. Now it does, and the key is
`(track_id, person_id)` (spec §4.5).
```

then:

```python
Key = tuple[int, int]  # (track_id, person_id)


@dataclass(frozen=True, slots=True)
class Hit:
    track_id: int
    person_id: int
    score: float
    gap: float
    at: float


@dataclass
class TrackAccumulator:
    """Repeated looks at ONE track. Two tracks are two children, and they do not vote
    together."""

    config: SoftConfig
    hits: dict[Key, deque[Hit]] = field(default_factory=dict)

    def observe(self, recognition: Recognition, now: float, track_id: int) -> int | None:
        """Record a look at ONE track. Returns a person id once we have seen enough.

        Only the top-1 counts, and only if it is at least *plausible* — a face that
        matched nobody tells us nothing, and letting it accumulate would eventually
        "recognise" a child out of pure noise.
        """
        if not self.config.enabled:
            return None

        top = recognition.top1
        if top is None:
            return None
        if top.score < self.config.min_score or recognition.gap < self.config.min_gap:
            return None

        window = self.hits.setdefault((track_id, top.person_id), deque())
        window.append(Hit(track_id, top.person_id, top.score, recognition.gap, now))
        self._expire(window, now)

        if len(window) >= self.config.min_hits:
            return top.person_id
        return None

    def _expire(self, window: deque[Hit], now: float) -> None:
        """Hits older than the window are gone. Two glances a minute apart are not
        corroboration; they are two separate children walking past."""
        cutoff = now - self.config.window_seconds
        while window and window[0].at < cutoff:
            window.popleft()

    def evidence(self, track_id: int, person_id: int) -> int:
        return len(self.hits.get((track_id, person_id), ()))

    def clear(self, track_id: int | None = None) -> None:
        """Forget one track, or everything. A track that has ended has no more to say."""
        if track_id is None:
            self.hits.clear()
            return
        for key in [k for k in self.hits if k[0] == track_id]:
            del self.hits[key]

    def prune(self, now: float) -> None:
        """Bounded (rule R8). Track ids only ever increase and a canteen runs all year."""
        for key in list(self.hits):
            window = self.hits[key]
            self._expire(window, now)
            if not window:
                del self.hits[key]


def accept_small_face(
    recognition: Recognition,
    accumulator: TrackAccumulator,
    now: float,
    track_id: int,
) -> Recognition:
    """Give a rejected recognition a second chance via accumulated evidence — **from this
    track, and only this track.**

    Only ever *upgrades* a rejection, and never touches an acceptance — a face the strict
    policy already accepted does not need help, and a face it rejected as AMBIGUOUS is
    rejected for a reason the accumulator cannot cure: seeing two children who look alike
    five times running does not tell you which one it was.
    """
    if recognition.accepted:
        return recognition
    if recognition.reason is Reason.AMBIGUOUS:
        return recognition

    person_id = accumulator.observe(recognition, now, track_id)
    if person_id is None:
        return recognition

    top = recognition.top1
    return Recognition(
        person_id=person_id,
        score=top.score if top else 0.0,
        gap=recognition.gap,
        reason=Reason.ACCEPTED,
        ranked=recognition.ranked,
    )
```

- [ ] **Step 4: Implement — pass the track id through the service.**

In `src/qorgan/identity/service.py`, `_soften` gains the track id:

```python
    def _soften(
        self, recognition: Recognition, face: FaceBox, now: float, track_id: int
    ) -> Recognition:
        """The small-face path. Younger pupils' faces are systematically below the size
        gate, so a strict single-shot threshold never recognises the first-graders at all.

        Keyed by TRACK: several weak looks at ONE child are worth more than one weak look.
        Several weak looks at a CROWD are worth nothing at all, and used to be worth a
        meal session (spec §4.5).
        """
        if self._soft is None:
            return recognition
        if not self._soft.config.face_gate.accepts(face.width, face.height):
            return recognition
        return accept_small_face(recognition, self._soft, now, track_id)
```

and its caller in `_resolve` becomes
`recognition = self._soften(recognition, best, now, track_id)`.

In `BindingTable.evict` the caller already prunes; add the accumulator eviction to
`IdentityService.evict` so a dead track's hits die with it:

```python
    def evict(self, live: Collection[int], now: float) -> list[int]:
        gone = self._table.evict(live, now)
        if self._soft is not None:
            for track_id in gone:
                self._soft.clear(track_id)
            self._soft.prune(now)
        return gone
```

- [ ] **Step 5: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_faces_matching.py tests/test_identity_service.py tests/test_canteen_worker.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green.

- [ ] **Step 6: Commit.**

```
git status
git add src/qorgan/faces/accumulator.py src/qorgan/identity/service.py tests/test_faces_matching.py
git commit -m "The small-face accumulator was letting a crowd vote a stranger into being a pupil

TrackAccumulator.hits was keyed by person_id, and ONE accumulator was shared across the
whole camera. So a weak, wrong top-1 from child A and a weak, wrong top-1 from child B --
two entirely different children, both matching pupil X badly -- added up. Two glances at a
crowd of unknowns cleared min_hits, and pupil X was 'recognised'.

That opens a meal session. Pupil X is recorded as having eaten, and if they never came,
they are also missing from the 'did not eat' report -- the one report the school actually
asked for.

The class is called TrackAccumulator and had never known what a track was. It does now, so
the key is (track_id, person_id), and a dead track's evidence dies with it. Several weak
looks at ONE child are worth more than one weak look. Several weak looks at a crowd are
worth nothing."
```

---

### Task 11: `qorgan plan-workers`

Spec §4.7. `config/workers.yaml` was measured on a 4 GB RTX 3050 and under-uses the
school's 4070. The numbers are sound — `vram_spike.py` imports torch at module level, so
InsightFace really was on the GPU when they were taken (§3) — but they are numbers **for the
wrong GPU**, and we cannot measure a GPU we do not have, and we will not guess.

So `scripts/vram_spike.py` becomes `qorgan plan-workers`: it runs **on the machine it will
run on**, measures the real per-process cost, and writes `config/workers.yaml`. The current
grouped config ships as the fallback.

**And it must account for the YOLO the canteen cameras gained in Task 9** — which is not one
YOLO per canteen *group* but one per canteen *camera*, because `PersonDetector` is per
camera (Ultralytics keeps its tracker state on the model object).

**Files:**
- Create: `src/qorgan/planning/__init__.py`, `src/qorgan/planning/costs.py`,
  `src/qorgan/planning/measure.py`, `src/qorgan/planning/cli.py`
- Delete: `scripts/vram_spike.py`
- Modify: `src/qorgan/cli.py` (register `plan-workers`)
- Test: Create `tests/test_worker_planner.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.config.camera import BullyingCamera, CameraConfig, CanteenCamera
from qorgan.config.loader import load_cameras          # -> dict[str, CameraConfig]
from qorgan.config.workers import WorkerGroup, WorkersConfig
```

*Produces:*
```python
# src/qorgan/planning/costs.py   -- PURE. No GPU, no nvidia-smi, no subprocess.
@dataclass(frozen=True, slots=True)
class Costs:
    context_mb: float        # a bare CUDA context in a fresh process
    yolo_mb: float           # ONE YOLOv8n + its ByteTrack state. Per CAMERA, not per group.
    pose_mb: float           # YOLOv8n-pose. Per GROUP: it has no per-camera state.
    insightface_mb: float    # buffalo_l. Per GROUP: one instance per process, always.

@dataclass(frozen=True, slots=True)
class Plan:
    groups: tuple[WorkerGroup, ...]
    costs: Costs
    total_mb: float
    headroom: float
    def estimated_mb(self, cameras: Mapping[str, CameraConfig]) -> float: ...
    def to_yaml(self, cameras: Mapping[str, CameraConfig], device_name: str) -> str: ...

def group_cost(costs: Costs, cameras: Sequence[CameraConfig]) -> float: ...
def plan_groups(
    cameras: Mapping[str, CameraConfig],
    costs: Costs,
    total_mb: float,
    *,
    headroom: float = 0.35,
) -> Plan: ...

# src/qorgan/planning/measure.py  -- IMPURE. Spawns children; reads nvidia-smi.
def gpu_total_mb() -> float: ...
def gpu_used_mb() -> float: ...
def measure_costs() -> Costs: ...      # raises RuntimeError with no CUDA device
```

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_worker_planner.py`.**

```python
"""Sizing the fleet. The measurement needs a GPU; the PLAN does not, so it is tested here."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qorgan.config.camera import CAMERA_ADAPTER, CameraConfig
from qorgan.config.workers import WorkersConfig
from qorgan.planning.costs import Costs, group_cost, plan_groups

# Measured on the RTX 3050 Laptop. Real numbers, so the test exercises real arithmetic.
COSTS = Costs(context_mb=140.0, yolo_mb=15.0, pose_mb=20.0, insightface_mb=700.0)


def _bullying(name: str) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": "main_hall",
            "name": name,
            "display_name": name,
            "rtsp": {"host": "10.0.0.1"},
        }
    )


def _canteen(name: str, role: str, block: str) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "canteen",
            "role": role,
            "name": name,
            "display_name": name,
            "rtsp": {"host": "10.0.0.2"},
            "canteen": {block: {}},
        }
    )


def _fleet() -> dict[str, CameraConfig]:
    return {
        "hall_left": _bullying("hall_left"),
        "hall_right": _bullying("hall_right"),
        "canteen_entry": _canteen("canteen_entry", "canteen_entry", "entry"),
        "canteen_exit": _canteen("canteen_exit", "canteen_exit", "exit"),
        "canteen_inside_left": _canteen("canteen_inside_left", "canteen_inside", "inside"),
    }


# -- the cost model -----------------------------------------------------------


def test_a_canteen_group_pays_for_one_yolo_PER_CAMERA() -> None:
    """**The thing §4.7 says must not be missed.**

    The canteen cameras gained a PersonDetector in §4.4, and Ultralytics keeps its tracker
    state on the model object -- so two cameras sharing one would hand each other's
    children the same track ids. One YOLO per CAMERA. InsightFace is one per PROCESS.
    """
    fleet = _fleet()
    one = group_cost(COSTS, [fleet["canteen_entry"]])
    two = group_cost(COSTS, [fleet["canteen_entry"], fleet["canteen_exit"]])

    assert one == pytest.approx(140.0 + 15.0 + 700.0)
    assert two == pytest.approx(140.0 + 30.0 + 700.0), "the second camera's YOLO was free"


def test_a_bullying_group_pays_for_one_pose_model_and_no_insightface() -> None:
    """The pose model has no per-camera state, so the group shares one. InsightFace is not
    loaded at all -- a hall camera does not recognise faces."""
    fleet = _fleet()
    cost = group_cost(COSTS, [fleet["hall_left"], fleet["hall_right"]])

    assert cost == pytest.approx(140.0 + 30.0 + 20.0)


def test_a_group_with_no_cameras_costs_nothing() -> None:
    assert group_cost(COSTS, []) == 0.0


# -- the plan -----------------------------------------------------------------


def test_a_big_gpu_gets_one_camera_per_process() -> None:
    """The spec asks for one OS process per camera. On a GPU that fits it, that is what it
    gets, and nothing in the code changes."""
    plan = plan_groups(_fleet(), COSTS, total_mb=12288.0)

    assert len(plan.groups) == 5
    assert sorted(c for g in plan.groups for c in g.cameras) == sorted(_fleet())


def test_a_small_gpu_groups_the_canteen_cameras_because_insightface_is_the_wall() -> None:
    """The expensive thing is NOT the CUDA context (~140 MB, models included). It is
    InsightFace buffalo_l, at ~700 MB per instance. Grouping the CANTEEN cameras is what
    makes the fleet fit."""
    plan = plan_groups(_fleet(), COSTS, total_mb=4096.0)

    canteen_groups = [
        g for g in plan.groups if any("canteen" in name for name in g.cameras)
    ]
    assert len(canteen_groups) < 3, "each canteen camera got its own 700 MB InsightFace"
    assert plan.estimated_mb(_fleet()) <= 4096.0 * (1 - plan.headroom)


def test_every_camera_lands_in_exactly_one_group() -> None:
    """A camera nobody runs is a camera nobody is watching."""
    fleet = _fleet()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    assigned = [camera for group in plan.groups for camera in group.cameras]
    assert sorted(assigned) == sorted(fleet)
    assert len(assigned) == len(set(assigned))


def test_a_gpu_too_small_for_even_one_canteen_camera_refuses_to_pretend() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        plan_groups(_fleet(), COSTS, total_mb=512.0)


# -- the file it writes -------------------------------------------------------


def test_the_yaml_it_writes_is_a_valid_workers_config() -> None:
    """It must load through the same schema the supervisor loads. A planner that writes a
    file the system cannot read is worse than no planner."""
    import yaml

    fleet = _fleet()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    parsed = yaml.safe_load(plan.to_yaml(fleet, device_name="NVIDIA GeForce RTX 4070"))
    config = WorkersConfig.model_validate(parsed)

    assert config.assigned_cameras == set(fleet)


def test_the_yaml_records_what_was_measured_and_on_what() -> None:
    """The current file's numbers are sound but they are for the WRONG GPU. A file that
    does not say which GPU it was measured on is how that happens."""
    fleet = _fleet()
    yaml_text = plan_groups(fleet, COSTS, total_mb=4096.0).to_yaml(
        fleet, device_name="NVIDIA GeForce RTX 4070"
    )

    assert "RTX 4070" in yaml_text
    assert "700" in yaml_text  # the InsightFace cost, which is the whole story
    assert "4096" in yaml_text


def test_the_planner_never_writes_a_config_two_groups_could_share_a_camera_in() -> None:
    fleet = _fleet()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    # WorkersConfig raises on a camera in two groups. Prove it would have caught us.
    with pytest.raises(ValidationError, match="more than one group"):
        WorkersConfig(
            groups=[
                {"name": "a", "cameras": ["hall_left"]},
                {"name": "b", "cameras": ["hall_left"]},
            ]
        )
    assert WorkersConfig(groups=[g.model_dump() for g in plan.groups])
```

- [ ] **Step 2: Run them, watch them fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_worker_planner.py -q
```

Expected failure:

```
ModuleNotFoundError: No module named 'qorgan.planning'
```

- [ ] **Step 3: Implement — `src/qorgan/planning/__init__.py`.**

```python
"""Sizing the worker fleet by MEASURING it, on the machine it will run on."""
```

- [ ] **Step 4: Implement — `src/qorgan/planning/costs.py`.**

```python
"""What a worker process costs, and how many cameras fit in one. **Pure.**

`config/workers.yaml` was measured on a 4 GB RTX 3050 and under-uses the school's 4070. The
numbers themselves are sound. But they are numbers for the WRONG GPU, and we cannot measure
a GPU we do not have, and we will not guess -- guessing at a number is what this whole
rewrite exists to stop.

So the arithmetic lives here, pure and tested, and `measure.py` supplies the four numbers
it needs by running real processes on the real card.

**The expensive thing is not the CUDA context.** Measured: ~140 MB, models included. It is
**InsightFace buffalo_l, at roughly 700 MB per instance** -- which is the audit's H-12
finding ("up to 5 InsightFace instances in one process") turned into a hard wall: one
instance per PROCESS is just as fatal on a small card. Grouping the canteen cameras is
therefore the whole game.

**And since §4.4 the canteen cameras carry a YOLO too** -- one per CAMERA, not one per
group, because Ultralytics keeps its tracker state on the model object and two cameras
sharing one would hand each other's children the same track ids. A track id is now an
identity, so that is not an optimisation we can take.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qorgan.config.camera import BullyingCamera, CameraConfig, CanteenCamera
from qorgan.config.workers import WorkerGroup

# How much of the card we refuse to plan into. Activations under real load are not in the
# static measurement, and a worker that OOMs at lunchtime is a worker that is not watching.
DEFAULT_HEADROOM = 0.35


@dataclass(frozen=True, slots=True)
class Costs:
    context_mb: float
    yolo_mb: float  # per CAMERA
    pose_mb: float  # per bullying GROUP
    insightface_mb: float  # per canteen GROUP


@dataclass(frozen=True, slots=True)
class Plan:
    groups: tuple[WorkerGroup, ...]
    costs: Costs
    total_mb: float
    headroom: float

    def estimated_mb(self, cameras: Mapping[str, CameraConfig]) -> float:
        return sum(
            group_cost(self.costs, [cameras[name] for name in group.cameras])
            for group in self.groups
        )

    def to_yaml(self, cameras: Mapping[str, CameraConfig], device_name: str) -> str:
        return _render(self, cameras, device_name)


def group_cost(costs: Costs, cameras: Sequence[CameraConfig]) -> float:
    """One process, N cameras. What does it hold?"""
    if not cameras:
        return 0.0

    total = costs.context_mb
    # One YOLO per camera. Not negotiable: the tracker state lives on the model.
    total += costs.yolo_mb * len(cameras)

    if any(isinstance(camera, BullyingCamera) for camera in cameras):
        # No per-camera state, so the group shares one.
        total += costs.pose_mb
    if any(isinstance(camera, CanteenCamera) for camera in cameras):
        # One InsightFace per PROCESS. Ever. This is the wall.
        total += costs.insightface_mb

    return total


def plan_groups(
    cameras: Mapping[str, CameraConfig],
    costs: Costs,
    total_mb: float,
    *,
    headroom: float = DEFAULT_HEADROOM,
) -> Plan:
    """One process per camera if the card can take it; otherwise group, canteen first."""
    budget = total_mb * (1.0 - headroom)

    bullying = [n for n, c in cameras.items() if isinstance(c, BullyingCamera)]
    canteen = [n for n, c in cameras.items() if isinstance(c, CanteenCamera)]

    for canteen_groups in range(len(canteen), 0, -1):
        for bullying_groups in range(len(bullying), 0, -1):
            groups = _lay_out(cameras, bullying, bullying_groups, canteen, canteen_groups)
            if _cost_of(groups, cameras, costs) <= budget:
                return Plan(
                    groups=tuple(groups), costs=costs, total_mb=total_mb, headroom=headroom
                )

    smallest = _lay_out(cameras, bullying, 1, canteen, 1)
    raise ValueError(
        f"this fleet does not fit: even one process per KIND needs "
        f"{_cost_of(smallest, cameras, costs):.0f} MB, and the budget is {budget:.0f} MB "
        f"({total_mb:.0f} MB less {headroom:.0%} headroom). InsightFace alone is "
        f"{costs.insightface_mb:.0f} MB. Buy a bigger card or run fewer cameras — do not "
        "raise the headroom and hope."
    )


def _lay_out(
    cameras: Mapping[str, CameraConfig],
    bullying: list[str],
    bullying_groups: int,
    canteen: list[str],
    canteen_groups: int,
) -> list[WorkerGroup]:
    groups = [
        WorkerGroup(name=f"bullying_{index + 1}", cameras=chunk)
        for index, chunk in enumerate(_chunks(bullying, bullying_groups))
    ]
    groups += [
        WorkerGroup(name=f"canteen_{index + 1}", cameras=chunk)
        for index, chunk in enumerate(_chunks(canteen, canteen_groups))
    ]
    return groups


def _chunks(names: list[str], count: int) -> list[list[str]]:
    """Split as evenly as possible. Empty groups are never emitted."""
    if not names:
        return []
    count = min(count, len(names))
    size, extra = divmod(len(names), count)
    chunks = []
    start = 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        chunks.append(names[start:end])
        start = end
    return chunks


def _cost_of(
    groups: list[WorkerGroup], cameras: Mapping[str, CameraConfig], costs: Costs
) -> float:
    return sum(
        group_cost(costs, [cameras[name] for name in group.cameras]) for group in groups
    )


def _render(plan: Plan, cameras: Mapping[str, CameraConfig], device_name: str) -> str:
    estimated = plan.estimated_mb(cameras)
    lines = [
        "# Which cameras run in which worker process.",
        "#",
        f"# WRITTEN BY `qorgan plan-workers`. MEASURED ON: {device_name} "
        f"({plan.total_mb:.0f} MB).",
        "#",
        "# Do not hand-edit the numbers below without re-measuring. The previous version of",
        "# this file was measured on an RTX 3050 and shipped to a school with a 4070: the",
        "# numbers were correct and they were for the wrong GPU.",
        "#",
        "#   CUDA context + process       ~%.0f MB" % plan.costs.context_mb,
        "#   YOLOv8n + ByteTrack          ~%.0f MB   PER CAMERA (the tracker state lives"
        % plan.costs.yolo_mb,
        "#                                            on the model; two cameras sharing one",
        "#                                            would swap their children's track ids)",
        "#   YOLOv8n-pose                 ~%.0f MB   per bullying group (no per-camera state)"
        % plan.costs.pose_mb,
        "#   InsightFace buffalo_l        ~%.0f MB   per canteen group. THIS IS THE WALL."
        % plan.costs.insightface_mb,
        "#",
        f"# This layout: ~{estimated:.0f} MB of {plan.total_mb:.0f} MB, "
        f"{plan.headroom:.0%} held back for activations under load.",
        "#",
        "# What this file does NOT change:",
        "#   - Coverage comes from here, never from which browser tab the operator has open",
        "#     (R3). Legacy analysed the stairs only while somebody was looking at them.",
        "#   - The supervisor restarts a group that dies, and kills one that wedges (R7).",
        "#",
        "# Every enabled camera appears in exactly one group, or startup fails. A camera",
        "# nobody runs is a camera nobody is watching.",
        "",
        "groups:",
    ]
    for group in plan.groups:
        cost = group_cost(plan.costs, [cameras[name] for name in group.cameras])
        lines.append(f"  - name: {group.name}")
        lines.append(f"    cameras: [{', '.join(group.cameras)}]")
        lines.append(f'    device: "{group.device}"')
        lines.append(f"    # ~{cost:.0f} MB")
        lines.append("")

    lines += [
        "heartbeat_interval_seconds: 5.0",
        "restart_backoff_seconds: 2.0",
        "restart_backoff_max_seconds: 60.0",
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 5: Implement — `src/qorgan/planning/measure.py`.**

This is `scripts/vram_spike.py`, moved and corrected: the canteen child now runs
`model.track(..., persist=True)` rather than `predict`, because ByteTrack allocates and
because that is what production does.

```python
r"""Measure real GPU memory per worker process, on the machine it will run on.

    .venv\Scripts\python.exe -m qorgan plan-workers

Each child loads exactly what its kind of worker loads in production, runs one inference to
force the real allocation, and holds it while the next child loads. Nothing here is
guessed, because a guessed VRAM number is a worker that dies at lunchtime.

Note the import order at the bottom of `_load`: torch first. That is what puts the CUDA
runtime DLLs in the process so onnxruntime can find them (see gpu.py §3), and it is why
this script's original numbers were real GPU numbers all along.
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import time

from qorgan.logging_setup import get_logger
from qorgan.planning.costs import Costs

logger = get_logger(__name__)

# What each child loads. The names are the keys of the Costs it produces.
BARE = "bare"  # a CUDA context and nothing else
YOLO = "yolo"  # + one YOLOv8n with its tracker
POSE = "pose"  # + YOLOv8n-pose
FACES = "faces"  # + InsightFace buffalo_l

_SETTLE_SECONDS = 1.0
_LOAD_TIMEOUT = 300.0


def _smi(query: str) -> list[str]:
    output = subprocess.run(  # noqa: S603
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in output.stdout.splitlines() if line.strip()]


def gpu_used_mb() -> float:
    """Device-wide VRAM in use, across every process.

    NOT torch.cuda.mem_get_info(): on Windows WDDM that reports memory from the calling
    context's point of view and simply cannot see a child process's allocation, which
    makes it useless for exactly this measurement.
    """
    return float(_smi("gpu=memory.used")[0])


def gpu_total_mb() -> float:
    return float(_smi("gpu=memory.total")[0])


def device_name() -> str:
    return _smi("gpu=name")[0]


def _load(kind: str, ready, done) -> None:
    """One child, loading exactly what its kind loads in production."""
    import torch  # noqa: F401, I001 -- FIRST. It puts the CUDA DLLs in the process.
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    torch.zeros(1, device="cuda:0")  # force the context

    if kind in (YOLO, POSE, FACES):
        from ultralytics import YOLO as Yolo

        detector = Yolo("yolov8n.pt")
        # track(), not predict(): ByteTrack allocates, and tracking is what production runs.
        detector.track(frame, imgsz=768, device="cuda:0", persist=True, verbose=False)

    if kind == POSE:
        from ultralytics import YOLO as Yolo

        pose = Yolo("yolov8n-pose.pt")
        pose.predict(frame, imgsz=768, device="cuda:0", verbose=False)

    if kind == FACES:
        from insightface.app import FaceAnalysis

        from qorgan.gpu import CUDA_PROVIDER, require_gpu

        # Refuse to report a number that is really a CPU measurement. The guard builds a
        # real ONNX session, so it cannot be fooled by an import order (gpu.py).
        require_gpu()
        faces = FaceAnalysis(name="buffalo_l", providers=[CUDA_PROVIDER])
        faces.prepare(ctx_id=0, det_size=(640, 640))
        faces.get(frame)

    ready.set()
    done.wait(timeout=240)


def _cost_of(kind: str) -> float:
    """Device-wide VRAM held by one child of this kind, in MB."""
    context = mp.get_context("spawn")
    ready, done = context.Event(), context.Event()

    before = gpu_used_mb()
    child = context.Process(target=_load, args=(kind, ready, done))
    child.start()
    try:
        if not ready.wait(timeout=_LOAD_TIMEOUT):
            raise RuntimeError(f"the {kind!r} probe never finished loading. Out of VRAM?")
        time.sleep(_SETTLE_SECONDS)  # let the driver settle before reading
        return gpu_used_mb() - before
    finally:
        done.set()
        child.join(timeout=20)
        if child.is_alive():
            child.terminate()


def measure_costs() -> Costs:
    """Four processes, one at a time. Each one's marginal cost is one of the numbers."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device. There is nothing to measure, and a guessed number is a "
            "worker that dies at lunchtime. config/workers.yaml stands as the fallback."
        )

    bare = _cost_of(BARE)
    yolo = _cost_of(YOLO)
    pose = _cost_of(POSE)
    faces = _cost_of(FACES)

    logger.info(
        "measured worker costs",
        extra={"bare": bare, "yolo": yolo, "pose": pose, "faces": faces},
    )
    return Costs(
        context_mb=bare,
        yolo_mb=max(yolo - bare, 1.0),
        pose_mb=max(pose - yolo, 1.0),
        insightface_mb=max(faces - yolo, 1.0),
    )
```

- [ ] **Step 6: Implement — `src/qorgan/planning/cli.py`.**

```python
"""`qorgan plan-workers` — measure this GPU, then write config/workers.yaml for it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKERS_PATH = Path("config/workers.yaml")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "plan-workers",
        help="measure this GPU and write config/workers.yaml for it",
        description=(
            "config/workers.yaml was measured on a 4 GB RTX 3050 and under-uses the "
            "school's 4070. The numbers are sound and they are for the WRONG GPU. We "
            "cannot measure a GPU we do not have, and we will not guess -- so this runs "
            "on the machine it will run on, loads exactly what each kind of worker loads "
            "in production, and writes the file. The current config ships as the fallback."
        ),
    )
    parser.add_argument("--out", type=Path, default=WORKERS_PATH)
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.35,
        help="fraction of the card held back for activations under load",
    )
    parser.add_argument("--dry-run", action="store_true", help="print it; do not write it")
    parser.set_defaults(func=cmd_plan_workers)


def cmd_plan_workers(args: argparse.Namespace) -> int:
    from qorgan.config.loader import load_cameras
    from qorgan.planning.costs import plan_groups
    from qorgan.planning.measure import device_name, gpu_total_mb, measure_costs

    cameras = {name: camera for name, camera in load_cameras().items() if camera.enabled}
    if not cameras:
        print("no enabled cameras; nothing to plan", file=sys.stderr)
        return 1

    try:
        costs = measure_costs()
        total = gpu_total_mb()
        device = device_name()
    except (RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"cannot measure this machine: {exc}", file=sys.stderr)
        print(f"{args.out} stands unchanged. It is the fallback, and it is honest about "
              "which GPU it was measured on.", file=sys.stderr)
        return 1

    print(f"device: {device}   total VRAM: {total:.0f} MB")
    print(f"  CUDA context      ~{costs.context_mb:.0f} MB")
    print(f"  YOLOv8n + track   ~{costs.yolo_mb:.0f} MB   per CAMERA")
    print(f"  YOLOv8n-pose      ~{costs.pose_mb:.0f} MB   per bullying group")
    print(f"  InsightFace       ~{costs.insightface_mb:.0f} MB   per canteen group\n")

    try:
        plan = plan_groups(cameras, costs, total, headroom=args.headroom)
    except ValueError as exc:
        print(f"cannot plan a fleet that fits: {exc}", file=sys.stderr)
        return 1

    text = plan.to_yaml(cameras, device)
    if args.dry_run:
        print(text)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}: {len(plan.groups)} group(s), "
          f"~{plan.estimated_mb(cameras):.0f} / {total:.0f} MB")
    return 0
```

- [ ] **Step 7: Implement — register it, and delete the script.**

In `src/qorgan/cli.py::build_parser`, beside the other deferred parser imports:

```python
    from qorgan.planning.cli import add_parser as add_planning_parser

    add_planning_parser(subparsers)
```

and:

```
git rm scripts/vram_spike.py
```

- [ ] **Step 8: Run the tests, watch them pass.**

```
.venv/Scripts/python.exe -m pytest tests/test_worker_planner.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: green. `tests/test_config_files.py` loads the checked-in
`config/workers.yaml` — do **not** overwrite it in this task; the file is regenerated on the
school's machine, not on this one.

- [ ] **Step 9: Run it here, dry, as the acceptance check.**

```
.venv/Scripts/python.exe -m qorgan plan-workers --dry-run
```

Expected on the RTX 3050 in this box: four numbers, InsightFace around 700 MB, and a
layout that groups the canteen cameras. It must print a YAML that parses. **Do not commit
the output** — it is a measurement of the wrong GPU, which is the exact mistake this
command exists to end.

- [ ] **Step 10: Commit.**

```
git status
git add src/qorgan/planning/ src/qorgan/cli.py tests/test_worker_planner.py
git add -u scripts/vram_spike.py
git commit -m "qorgan plan-workers: measure the GPU you have, not the one we had

config/workers.yaml was measured on a 4 GB RTX 3050 and under-uses the school's 4070. The
numbers themselves are sound -- vram_spike.py imports torch at module level, so InsightFace
really was on the GPU when they were taken. But they are numbers for the WRONG GPU, and we
cannot measure a GPU we do not have, and we will not guess.

So scripts/vram_spike.py becomes a command: it runs on the machine it will run on, loads
exactly what each kind of worker loads, measures the real marginal cost, and writes the
file. The current grouped config ships as the fallback, and the file it writes now says
which GPU it was measured on -- because not saying so is how it went wrong the first time.

It accounts for the YOLO the canteen cameras gained in §4.4, and it accounts for it per
CAMERA rather than per group: Ultralytics keeps its tracker state on the model object, and
a track id is now an identity, so two cameras cannot share one detector.

The arithmetic is pure and tested without a GPU. The measurement is the only part that
needs one."
```

---

## Self-review

**(a) Every spec section maps to a task.**

| spec | task |
|---|---|
| §1 identity is given, not inferred | 3, 4, 5 |
| §1.1 the filename prefix lies; four faceless staff photos | 5 (both tests are named) |
| §1.2 no silent fallback, ever | 3 (the parser), 5 (the import refuses) |
| §2 the measurement, the empty band | 6 (`gallery-report` reproduces the histogram) |
| §2.1 six people hold two school IDs | 6 (detect), 7 (resolve) |
| §2.2 `min_score` floor 0.50 | 2 |
| §2.3 the ceiling is unmeasured, and the config says so | 2 (in words, in the docstring) |
| §2.4 camera placement, and the diagnostic that should have existed | 8 |
| §2.5 the duplicates collapse the gap | 7 (the regression test) |
| §3 harden the GPU guard | 1 |
| §4.1 `gallery-report` | 6 |
| §4.2 `merge` | 7 |
| §4.3 `import-roster`; one import, not two | 5 |
| §4.4 `IdentityService`, once per track | 9 |
| §4.5 the accumulator bug | 10 |
| §4.6 `camera-report` | 8 |
| §4.7 `plan-workers` | 11 |
| §5 shape; `full_name` nullable; `display_name` | 2 (config), 3 (package), 4 (DB + naming) |
| §6 testing table | every row is a named test — see the map below |
| §7 what this spec does not do | honoured: `min_score` ships as a floor and says so; nothing is merged automatically; no file under `detection/`, `evaluation/` or `worker/camera_loop.py` is touched |

Spec §6's testing table, row by row:

| row | test |
|---|---|
| `assign_faces_to_tracks` | `tests/test_identity_tracks.py` (Task 9) |
| binding state machine | `tests/test_identity_binding.py` (Task 9) |
| import refuses a bad filename | `test_a_filename_that_does_not_match_stops_the_import_dead` (Task 5) |
| person type comes from the folder | `test_person_type_comes_from_the_folder_never_from_the_filename` (Task 5) |
| unenrollable photos are itemised | `test_a_photo_with_no_face_is_itemised_and_the_person_still_exists` (Task 5) |
| duplicate detection at 0.78 / not at 0.47 | `test_a_pair_at_0_47_is_an_impostor_and_a_pair_at_0_78_is_a_duplicate` (Task 6) |
| `merge` re-points and deactivates | `test_photos_embeddings_and_sessions_all_re_point`, `test_the_dropped_id_is_deactivated_not_deleted` (Task 7) |
| accumulator keyed by track | `test_two_different_TRACKS_do_not_corroborate_each_other` (Task 10) |
| one embedding per track | `test_one_track_costs_exactly_one_embedding_across_forty_frames` (Task 9) |
| the gap collapse is pinned | `test_a_human_under_two_ids_is_ambiguous_and_a_merge_makes_him_visible` (Task 7) |
| onnx really is on CUDA | `test_a_real_onnx_session_on_this_machine_runs_on_cuda` (Task 1) |

**(b) No placeholders.** Every code block is complete and runnable. Three places where I
knowingly leave a small thing for a later step, all named at the point they occur, none of
them a "TBD":

1. Task 8's `Detector` protocol uses `detect`; Task 9, Step 11 changes it to `detect_faces`
   and updates the one test. This is written down in both tasks.
2. Task 3 leaves `tests/test_migrations.py::test_the_migration_matches_the_models` red;
   Task 4 makes it green, and they commit together. Stated in Task 3, Step 10.
3. Task 9's `service.py` contains one `__import__(...)` I explicitly tell the implementer to
   replace with a top-level import (Step 8's closing note). It is an error in the draft,
   caught here, corrected in place rather than left to be found at runtime.

**(c) Type and name consistency across tasks.**

- `FaceBox` (Task 9) is produced by `FaceRecognizer.detect_faces`, consumed by
  `assign_faces_to_tracks`, `BindingTable.observe`, `IdentityService`, and — via the `Sized`
  protocol — `measure_faces` (Task 8). It carries `landmarks`, without which `embed()`
  cannot align the crop; that is why detection and embedding can be split at all.
- `DetectedFace` (box + embedding + det_score) survives unchanged and is what the **importer**
  consumes, so Task 5 does not have to change when Task 9 lands.
- `RosterEntry` (Task 3) → `importer._store` (Task 5). `external_id` is the matched
  prefix + id verbatim (`student_469` for the teacher), `class_name` is the folder name
  verbatim (`5-А`, with the hyphen) — because `display_name` prints it.
- `display_name(person: Named)` (Task 4) is satisfied by `Person` (ORM), `PersonInfo`
  (gallery) and `Meal` (reports). All three gain `position`; `full_name` becomes
  `str | None` on all three. One function, three callers, no third spelling.
- `PersonInfo.display` is asserted in Task 5 (`"Учитель 469"`), Task 6 (`"Сотрудник 465"`)
  and Task 9 (`"Ученик 10, 5-А"`) — the same function, three different shapes of person.
- `accept_small_face(recognition, accumulator, now, track_id)` (Task 10) has exactly two
  call sites after Task 9: `IdentityService._soften` and the tests. Task 9 calls it with the
  **old** three-argument signature deliberately, so the small-face path is never dead between
  the two tasks; Task 10 widens both in one commit.
- `BindingSettings` is defined in Task 2, unused until Task 9, and reachable from every
  canteen role block (`entry.binding`, `exit.binding`, `inside.binding`) so
  `worker/canteen._blocks_for` can hand it to `IdentityService`.
- `Costs` (Task 11) names its fields `yolo_mb` *per camera* and `insightface_mb` *per
  group*, and `group_cost` is the only place that knows the difference. The test asserts
  the second canteen camera's YOLO is not free, which is the §4.7 requirement in one line.

**One deviation from the letter of the brief, recorded rather than hidden.** The brief says
`config/canteen.py` keeps *only* `SessionRules` and `MealOutcomeRules`. It also keeps
`EntrySettings`, `ExitSettings`, `InsideSettings` and `CanteenConfig` — the canteen **camera
role blocks**. The spec gives them no new home, and they are not identity models: they carry
`face_roi`, `person_cooldown_seconds`, `min_person_box_area` and the meal-window wiring, all
of which are about a canteen camera and not about recognising a face. Moving them into
`config/identity.py` would put the canteen back inside identity, which is the exact coupling
§5 exists to break. So they stay, and they *compose* the identity models by importing them.
The functional requirement — **`identity.py` must not import `canteen.py`** — is tested
directly (`test_the_identity_config_does_not_drag_in_the_canteen`, Task 2).

---

### Task 12: Instrument the cost of raising the exit threshold to 0.50

`ExitSettings.recognition.min_score` was **0.42** — *below* the measured worst impostor
(0.472, spec §2.2) — on the camera that **closes meal sessions**. Task 2 raises it to 0.50.

**That choice has a price, and the price must be visible rather than assumed.**

The reasoning for paying it is an asymmetry:

- A session that is **never closed** force-closes as `UNKNOWN`. That is a **hole**. It is
  visible, it is countable, and it is recoverable.
- A session **wrongly closed** writes a false record that *looks like data*. And it corrupts
  **two** children at once: the impostor gets a meal they did not eat, and the child who
  actually left still has an open session that will itself time out.

A hole is recoverable; a confident lie is not. This is exactly §1.2 — refuse rather than
guess — applied to a threshold instead of a filename.

So we choose the hole. But **0.50 is the impostor floor, not a tuned value**, and it gets
re-derived the moment we have canteen footage of a named volunteer (§2.3). Until then we
count what it costs us.

**Files:**
- Modify: `src/qorgan/config/identity.py` (the comment beside `min_score`)
- Modify: `src/qorgan/canteen/reports.py` (`day_report` gains `forced_unknown`)
- Modify: `src/qorgan/faces/cli.py` (`pupils report` prints it)
- Test: Modify `tests/test_canteen_reports.py`

**Interfaces:**

*Consumes:*
```python
from qorgan.enums import CloseReason, SessionOutcome   # TIMEOUT / UNKNOWN already exist
from qorgan.db.models import CanteenSession
# canteen/sessions.py:257  force_close_stale(now) -> int   already writes
#   SessionOutcome.UNKNOWN + CloseReason.TIMEOUT   (sessions.py:277)
```

*Produces:*
```python
# canteen/reports.py -- DayReport gains one field:
#     forced_unknown: int   # sessions closed by TIMEOUT, i.e. no exit was ever recognised
```

- [ ] **Step 1: Write the failing test**

```python
def test_day_report_counts_sessions_forced_closed_as_unknown(db):
    """The price of a strict exit threshold is a session nobody closed.

    We would rather have a hole we can count than a false meal record we cannot
    detect -- but a price we do not measure is a price we are guessing at.
    """
    day = date(2026, 7, 13)
    _closed_normally(db, external_id="student_333", day=day)
    _forced_unknown(db, external_id="student_398", day=day)
    _forced_unknown(db, external_id="student_399", day=day)

    report = day_report(day)

    assert report.forced_unknown == 2
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canteen_reports.py::test_day_report_counts_sessions_forced_closed_as_unknown -v`
Expected: `AttributeError: 'DayReport' object has no attribute 'forced_unknown'`

- [ ] **Step 3: Count them**

In `canteen/reports.py`, add the field to `DayReport` and populate it in `day_report` with a
`select(func.count())` over `CanteenSession` for that day where
`close_reason == CloseReason.TIMEOUT`. Keep `day_report` under 50 lines; if it grows past
that, lift the count into a `_forced_unknown(session, day) -> int` helper.

- [ ] **Step 4: Run it and watch it pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canteen_reports.py -v`
Expected: PASS, and every existing report test still green.

- [ ] **Step 5: Say what the number means, where it is read**

In `src/qorgan/config/identity.py`, beside `min_score`:

```python
    # 0.50 is the IMPOSTOR FLOOR, not a tuned value. Measured: the worst genuine impostor
    # in the school's own 138-photo gallery scores 0.472 (spec A §2.2), so anything below
    # 0.50 admits a known confusion. The CEILING -- whether a real canteen-camera face can
    # reach 0.50 against a studio portrait -- is UNMEASURED (§2.3), and with one photo per
    # child it is tight.
    #
    # On the EXIT camera this is deliberately strict, and the cost is real: a session we
    # cannot close force-closes as UNKNOWN. We choose that, because a hole is recoverable
    # and a false meal record is not -- and a wrong close corrupts TWO children, since the
    # one who actually left is still sitting in an open session.
    #
    # `qorgan pupils report` counts the holes (`forced_unknown`). If that number spikes,
    # the threshold is too high and we will SEE it rather than guess.
    #
    # Re-derive this the day the school sends canteen footage of a named volunteer.
    min_score: float = Field(default=0.50, gt=0.0, lt=1.0)
```

- [ ] **Step 6: Surface it in `pupils report`**

Print `forced_unknown` alongside `never_came` / `came_but_did_not_eat` / `unknown_sessions`,
labelled so a human knows it is the exit threshold's bill:
`"Сессий закрыто по таймауту (выход не распознан): N"`.

- [ ] **Step 7: Full suite + lint**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`
Expected: all green, ruff clean.

- [ ] **Step 8: Commit**

```bash
git status                      # NEVER git add -A
git add src/qorgan/config/identity.py src/qorgan/canteen/reports.py \
        src/qorgan/faces/cli.py tests/test_canteen_reports.py
git commit -m "The exit threshold has a price, so count it

0.50 is the impostor floor, not a tuned value. On the exit camera it is
deliberately strict: a session we cannot close force-closes as unknown. We
choose the hole over the false record, because a wrong close corrupts two
children -- the impostor gets a meal, and the child who actually left keeps an
open session. day_report now counts the holes, so if the price spikes we see it
instead of guessing. Re-derive the moment we have canteen footage of a named
volunteer."
```
