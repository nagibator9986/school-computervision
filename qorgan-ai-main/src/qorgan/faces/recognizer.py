"""InsightFace: pixels in, face embeddings out. **One instance per process.**

That sentence is the whole design. The legacy created up to five `FaceAnalysis` objects
in a single process — three inside the recognition service (det 640, det 1024, relaxed
1024) plus another that `StudentService` built for itself, going round the singleton
(audit H-12). Measured on this machine, each one costs roughly 700 MB of VRAM, which on
a 4 GB card is not an inefficiency: it is the difference between running and not running.

Everything that DECIDES anything lives in `qorgan.faces.matching`, which is pure. This
module produces vectors and nothing else.

**Detection and embedding are two different prices.** Finding a face is cheap; turning it
into a 512-d ArcFace vector is not. The canteen worker used to pay both on every face in
every due frame. So they are two methods now, and the caller chooses -- which is what lets
`IdentityService` embed once per person rather than forty times per person.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from qorgan.config.identity import FaceModelSettings
from qorgan.detection.geometry import Box
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


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
            # ~40x too slow. On a canteen door that is not slow: it is broken. This is
            # the exact defect that made our first VRAM measurement a lie.
            raise RuntimeError(
                f"onnxruntime has no {CUDA_PROVIDER}; refusing to run face recognition "
                f"on the CPU. Providers seen: {report.onnx_providers}. Run `qorgan doctor`."
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
            found.append(FaceBox(box=Box(x1, y1, x2, y2), detection_score=score, landmarks=points))
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
