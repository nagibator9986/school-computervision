"""Liveness. The only endpoint that is public and returns data."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    # Deliberately says nothing about the school, the cameras, or the pupils.
    return {"status": "ok"}
