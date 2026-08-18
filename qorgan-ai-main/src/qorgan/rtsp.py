"""Build RTSP URLs. One implementation, used by every worker.

Legacy had build_rtsp_url copy-pasted into two workers with the password baked into
ten YAML files, and then printed the resulting URL -- credentials and all -- into log
files and onto debug JPEGs that an unauthenticated web UI served (audit C-02).

Credentials come from the environment. The URL never reaches a log line un-redacted;
`safe_url()` is what you print.
"""

from __future__ import annotations

from urllib.parse import quote

from qorgan.config.common import RtspSettings
from qorgan.settings import RtspCredentials, get_settings


def build_url(camera_name: str, rtsp: RtspSettings, *, burst: bool = False) -> str:
    """The real URL, with credentials. Never log this -- log safe_url() instead."""
    path = rtsp.burst_path if burst else rtsp.analysis_path
    if burst and not path:
        raise ValueError(f"camera {camera_name!r}: burst requested but no rtsp.burst_path is set")

    credentials: RtspCredentials = get_settings().credentials_for(camera_name)
    user = quote(credentials.user, safe="")
    password = quote(credentials.password.get_secret_value(), safe="")
    userinfo = f"{user}:{password}@" if user or password else ""
    return f"rtsp://{userinfo}{rtsp.host}:{rtsp.port}{path}"


def safe_url(rtsp: RtspSettings, *, burst: bool = False) -> str:
    """The same URL with the credentials removed. This is the one you may print."""
    path = rtsp.burst_path if burst else rtsp.analysis_path
    return f"rtsp://{rtsp.host}:{rtsp.port}{path or ''}"
