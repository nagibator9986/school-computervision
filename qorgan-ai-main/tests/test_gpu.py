"""The GPU must not be silently absent.

Both inference stacks in this project degrade to the CPU without failing:

- A bare `pip install torch` gives a CPU build that works and is ~40x too slow.
- `insightface` depends on plain `onnxruntime`, which shadows `onnxruntime-gpu`
  and takes CUDAExecutionProvider away with it.

Both were live in this venv until they were measured. A 40x slowdown that announces
itself only as a warning nobody reads is how the legacy ended up running its face
recognition on the CPU (audit H-14).
"""

from __future__ import annotations

import pytest

from qorgan.gpu import CPU_PROVIDER, CUDA_PROVIDER, GpuReport, inspect_gpu, require_gpu

pytest.importorskip("torch", reason="the AI extra is not installed")

REPORT = inspect_gpu()
HAS_GPU = REPORT.torch_cuda


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


def test_a_healthy_stack_is_ok() -> None:
    assert _report(torch_cuda=True, providers=(CUDA_PROVIDER, "CPUExecutionProvider")).ok


def test_cpu_only_torch_is_reported() -> None:
    report = _report(torch_cuda=False, providers=(CUDA_PROVIDER,), session_provider=None)
    assert not report.ok
    assert any("CUDA build" in problem for problem in report.problems())


def test_onnxruntime_without_the_cuda_provider_is_reported() -> None:
    """The exact failure that made the first VRAM measurement a lie: InsightFace ran
    on the CPU, so the canteen worker looked like it cost 142 MB instead of 850 MB."""
    report = _report(
        torch_cuda=True,
        providers=("AzureExecutionProvider", "CPUExecutionProvider"),
        session_provider=CPU_PROVIDER,
    )
    assert not report.ok
    assert any("onnxruntime" in problem for problem in report.problems())
    # The message must say how to fix it, not merely that it is broken.
    assert any("onnxruntime-gpu" in problem for problem in report.problems())


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA device on this machine")
def test_this_machine_can_actually_use_its_gpu() -> None:
    """Runs on the real box. If someone reinstalls a dependency and quietly loses the
    CUDA execution provider, this is what tells them."""
    report = require_gpu()
    assert report.torch_cuda
    assert report.onnx_cuda, (
        "onnxruntime lost CUDAExecutionProvider. The AI stack would run on the CPU, "
        f"silently and ~40x too slow. Providers seen: {report.onnx_providers}"
    )


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
