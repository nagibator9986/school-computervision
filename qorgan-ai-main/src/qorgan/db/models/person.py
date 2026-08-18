"""People: pupils and staff, their photos, and their face embeddings.

Identity is keyed on `external_id`, not on a name. Legacy keyed it on
"Surname Firstname" + class parsed out of a photo *filename*, so two children with
the same name in the same class collapsed into one person -- and `person_type` was
re-derived on every boot from 24 substring patterns, silently turning any pupil whose
surname contained "охран" into staff and reverting manual corrections (audit H-02).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.models.school import school_key
from qorgan.db.types import RelPath
from qorgan.enums import ExternalIdSource, PersonType
from qorgan.identity.naming import display_name


class Person(Base, TimestampMixin):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = school_key(nullable=False)

    # The identity key, never guessed from a filename -- and unique WITHIN A SCHOOL,
    # not globally. The id belongs to the school that issued it: two schools may both
    # have a pupil numbered 7, and under a global UNIQUE the second school to import
    # its roster would find half of it rejected as duplicates of children it has never
    # met. That is the constraint multi-school breaks quietly, so it is named here.
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # There is exactly one source, and it is the school. An id we invented is an id we can
    # be wrong about.
    external_id_source: Mapped[ExternalIdSource] = mapped_column(
        Enum(ExternalIdSource, native_enum=False),
        default=ExternalIdSource.ROSTER,
        nullable=False,
    )

    # Nullable, and it stays nullable. The name is a DISPLAY field: identity is the id.
    # The school's ID -> name table arrived with the 2026 delivery, so this is now filled
    # for the 141 pupils (`qorgan.identity.names`, joined on external_id). It is still NULL
    # for anyone that table does not cover -- the staff carried over from 2025 -- and those
    # people show as `Ученик 333, 5-А` (spec §1). No migration was needed: the column has
    # existed since 0002, waiting for exactly this.
    full_name: Mapped[str | None] = mapped_column(String(255))
    # Set once, at import. Never re-derived at runtime.
    person_type: Mapped[PersonType] = mapped_column(
        Enum(PersonType, native_enum=False), nullable=False
    )
    class_name: Mapped[str | None] = mapped_column(String(32))  # pupils
    position: Mapped[str | None] = mapped_column(String(128))  # staff
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # WHY this person is inactive, for the one reason the system creates itself.
    #
    # `is_active=False` answered two different questions with one "no": "this id was
    # merged into another" and "this person left the school". They need opposite
    # treatment. `pupils merge --reactivate` exists to undo a merge that went the wrong
    # way -- across the pupil/staff line it decides whether a child appears in the meal
    # record at all -- and with nothing recording the difference it had to take the
    # operator's word for which case it was, on exactly the operation where being wrong
    # either revives an expelled pupil or refuses a child their history back.
    #
    # This is the project's THIRD boolean answering "no" to two questions. The first cost
    # it face recognition: `newly_bound=False` meant both "already bound" and "not
    # recognised", so the canteen could not tell a pupil it already knew from a stranger.
    #
    # NULL means "not retired by a merge, or retired before this column existed" -- an
    # unknown, never a claim that they left. Nothing is back-filled: a reason that was
    # never recorded is not recoverable, and guessing it is what `apply_runtime_migrations`
    # did to `person_type` on every boot (audit H-02).
    merged_into_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), index=True
    )

    photos: Mapped[list[PersonPhoto]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list[FaceEmbedding]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("school_id", "external_id", name="uq_persons_school_external_id"),
        Index("ix_persons_person_type", "person_type"),
        Index("ix_persons_class_name", "class_name"),
    )

    @property
    def display(self) -> str:
        """One definition, used by the web, the reports and the CLI alike."""
        return display_name(self)


class PersonPhoto(Base, TimestampMixin):
    __tablename__ = "person_photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Relative to MEDIA_ROOT. The column type rejects an absolute path.
    path: Mapped[str] = mapped_column(RelPath(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    quality_note: Mapped[str | None] = mapped_column(Text)

    person: Mapped[Person] = relationship(back_populates="photos")

    __table_args__ = (Index("ix_person_photos_sha256", "sha256"),)


class FaceEmbedding(Base, TimestampMixin):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed. Legacy had no index on the join column of its hottest query, and then
    # re-read every embedding blob in the table for every face in every frame.
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("person_photos.id", ondelete="SET NULL")
    )

    # Which model produced this vector. Legacy shipped a dead DeepFace/Facenet512
    # rebuild script that wrote 512-d vectors from a *different* model into the same
    # column; running it would have silently corrupted the whole InsightFace gallery.
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    person: Mapped[Person] = relationship(back_populates="embeddings")

    __table_args__ = (Index("ix_face_embeddings_model", "model_name", "model_version"),)
