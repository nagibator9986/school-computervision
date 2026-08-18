"""All ORM models. Importing this module is what makes Base.metadata complete,
which is what Alembic autogenerate needs."""

from qorgan.db.models.auth import User
from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.models.camera import Camera
from qorgan.db.models.canteen import CanteenSession, MealWindow, RecognitionAttempt
from qorgan.db.models.classroom import Lesson, LessonTrack
from qorgan.db.models.classvision import (ClassvisionAttestation, ClassvisionFrame,
                                          ClassvisionLesson, ClassvisionPlace,
                                          ClassvisionPlaceLesson, ClassvisionReading,
                                          ClassvisionRun, ClassvisionTeacherLesson)
from qorgan.db.models.event import Event, Notification
from qorgan.db.models.person import FaceEmbedding, Person, PersonPhoto
from qorgan.db.models.psychologist import PsychologistNote
from qorgan.db.models.school import School, UndecidedSchool, sole_school_id
from qorgan.db.models.system import AppSetting, ModeLog, WorkerHeartbeat

__all__ = [
    "AppSetting",
    "Base",
    "Camera",
    "CanteenSession",
    "ClassvisionAttestation",
    "ClassvisionFrame",
    "ClassvisionLesson",
    "ClassvisionPlace",
    "ClassvisionPlaceLesson",
    "ClassvisionReading",
    "ClassvisionRun",
    "ClassvisionTeacherLesson",
    "Event",
    "FaceEmbedding",
    "Lesson",
    "LessonTrack",
    "MealWindow",
    "ModeLog",
    "Notification",
    "Person",
    "PersonPhoto",
    "PsychologistNote",
    "RecognitionAttempt",
    "School",
    "TimestampMixin",
    "UndecidedSchool",
    "User",
    "WorkerHeartbeat",
    "sole_school_id",
]
