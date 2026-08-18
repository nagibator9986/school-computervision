"""Notifications. An alert about a child being hurt must not be lost."""

from qorgan.notify.queue import NotificationWorker, enqueue
from qorgan.notify.telegram import SendResult, TelegramClient

__all__ = ["NotificationWorker", "SendResult", "TelegramClient", "enqueue"]
