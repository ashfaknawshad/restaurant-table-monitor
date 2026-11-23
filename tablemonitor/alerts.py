"""
Unattended-table notifications.

Every alert is saved as a snapshot under `alerts.snapshot_dir`. When `alerts.telegram`
is true, it is also posted to a Telegram chat via the Bot API, reading the bot token
and chat ID from environment variables (never from the config file, which is committed).

Setup: message @BotFather -> /newbot -> copy the token; add the bot to your group,
send a message there, then open https://api.telegram.org/bot<TOKEN>/getUpdates to
find the group's chat id (a negative number).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import AlertConfig
from .tables import Event, EventKind

ALERT_KINDS = {EventKind.UNATTENDED, EventKind.REMINDER, EventKind.RESOLVED}


def format_message(cfg: AlertConfig, e: Event) -> str:
    mins, secs = divmod(int(e.wait_seconds), 60)
    where = f"{cfg.branch} · {e.table_name} ({e.table_id})"
    if e.kind is EventKind.UNATTENDED:
        return (f"⚠️ Table unattended\n{where}\n{e.customers} guest(s) waiting {mins}m {secs:02d}s "
                f"with no staff visit.\nCamera: {cfg.camera}")
    if e.kind is EventKind.REMINDER:
        return f"⏰ Still unattended\n{where}\nWaiting {mins}m {secs:02d}s.\nCamera: {cfg.camera}"
    return f"✅ Resolved\n{where}\nStaff arrived after {mins}m {secs:02d}s."


class Notifier:
    def __init__(self, cfg: AlertConfig, enabled_telegram: bool | None = None):
        self.cfg = cfg
        self.snapshot_dir = Path(cfg.snapshot_dir)
        self.telegram = cfg.telegram if enabled_telegram is None else enabled_telegram
        self.token = os.environ.get(cfg.telegram_token_env, "")
        self.chat_id = os.environ.get(cfg.telegram_chat_env, "")
        if self.telegram and not (self.token and self.chat_id):
            print(f"[Alerts] Telegram enabled but {cfg.telegram_token_env} / "
                  f"{cfg.telegram_chat_env} are not set; alerts will only be saved locally.")
            self.telegram = False
        self.sent: list[dict] = []
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def handle(self, events: list[Event], snapshot: np.ndarray | None):
        for e in events:
            if e.kind not in ALERT_KINDS:
                continue
            text = format_message(self.cfg, e)
            image_path = None
            if snapshot is not None and e.kind is not EventKind.RESOLVED:
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                image_path = self.snapshot_dir / f"{stamp}_{e.table_id}_{e.kind.value}.jpg"
                cv2.imwrite(str(image_path), snapshot)
            print(f"[Alerts] {text.replace(chr(10), ' | ')}")
            self.sent.append({**e.to_dict(), "message": text,
                              "snapshot": str(image_path) if image_path else None})
            if self.telegram:
                self._queue.put((text, image_path))

    def close(self, timeout: float = 15.0):
        """Wait (briefly) for queued Telegram messages to go out."""
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.1)

    # Background sender so a slow network never stalls video processing.
    def _run(self):
        import requests

        base = f"https://api.telegram.org/bot{self.token}"
        while True:
            text, image_path = self._queue.get()
            for attempt in range(3):
                try:
                    if image_path:
                        with open(image_path, "rb") as f:
                            r = requests.post(f"{base}/sendPhoto", timeout=20,
                                              data={"chat_id": self.chat_id, "caption": text},
                                              files={"photo": f})
                    else:
                        r = requests.post(f"{base}/sendMessage", timeout=20,
                                          data={"chat_id": self.chat_id, "text": text})
                    if r.ok:
                        break
                    print(f"[Alerts] Telegram error {r.status_code}: {r.text[:200]}")
                except requests.RequestException as exc:
                    print(f"[Alerts] Telegram request failed: {exc}")
                time.sleep(2 * (attempt + 1))
            self._queue.task_done()
