"""Is the GPU actually being used? Ask loudly, at startup, before a worker runs.

Both of this project's inference stacks fail *silently* to the CPU:

- `pip install torch` without the CUDA wheel index gives you a CPU build that works
  perfectly and is ~40x too slow. The legacy requirements.txt did exactly this and
  did not pin onnxruntime at all (audit H-14).
- `onnxruntime-gpu` exposes CUDAExecutionProvider only if the CUDA/cuDNN libraries
  resolve. If they do not, InsightFace prints a warning nobody reads and runs on the
  CPU. Worse, `insightface` depends on plain `onnxruntime`, which if installed
  shadows the GPU build entirely.

Both were live in this venv until measured. A silent 40x slowdown on a safety system
is not acceptable, so we check and we refuse.

And the guard that checks all this must not itself be a lie. `get_available_providers()`
reports what onnxruntime was COMPILED with — it says CUDAExecutionProvider even in a
process where the provider DLL cannot load and every session falls back to the CPU. So we
build a real session and read back what it actually got. That is the only check that
cannot be fooled by an import reorder.
"""

from __future__ import annotations

from dataclasses import dataclass

from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

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
