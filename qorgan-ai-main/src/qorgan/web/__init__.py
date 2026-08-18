"""The web process. It never imports a worker module (rule R3)."""

from qorgan.web.app import create_app

__all__ = ["create_app"]
