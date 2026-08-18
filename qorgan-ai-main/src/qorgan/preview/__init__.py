"""The preview bus: workers PUB JPEG frames, the web process SUBs.

This is the seam that keeps rule R3 true. The web process never imports a worker
module, and a worker never imports the web layer. They share a socket and a schema,
nothing else.
"""

from qorgan.preview.protocol import PreviewHeader
from qorgan.preview.publisher import PreviewPublisher, encode_preview
from qorgan.preview.subscriber import Preview, PreviewSubscriber

__all__ = [
    "Preview",
    "PreviewHeader",
    "PreviewPublisher",
    "PreviewSubscriber",
    "encode_preview",
]
