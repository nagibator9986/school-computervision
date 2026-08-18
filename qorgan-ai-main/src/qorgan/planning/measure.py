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
    import torch  # noqa: I001 -- FIRST. It puts the CUDA DLLs in the process.
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
