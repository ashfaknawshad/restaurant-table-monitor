"""Drawing table states, people and the status panel onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from .people import Person, Role
from .tables import Table, TableMonitor, TableState

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREY = (170, 170, 170)
GREEN = (80, 200, 80)
AMBER = (0, 190, 255)
RED = (40, 40, 230)
STAFF_COLOR = (255, 140, 0)
CUSTOMER_COLOR = (80, 200, 80)
UNKNOWN_COLOR = (200, 200, 200)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(max(seconds, 0)), 60)
    return f"{m}:{s:02d}"


def table_color(tb: Table, t: float, warn_seconds: float) -> tuple[int, int, int]:
    if tb.state is TableState.EMPTY:
        return GREY
    if tb.alert_active:
        return RED
    return AMBER if tb.wait(t) >= warn_seconds else GREEN


def draw(frame: np.ndarray, monitor: TableMonitor, people: dict[int, Person], boxes: dict,
         t: float, staff_areas=None, highlight: str | None = None) -> np.ndarray:
    out = frame.copy()
    limit = monitor.rules.unattended_minutes * 60
    overlay = out.copy()
    for tid, tb in monitor.tables.items():
        color = table_color(tb, t, limit * 0.6)
        cv2.fillPoly(overlay, [tb.shape.points], color)
    for area in (staff_areas or {}).values():
        cv2.fillPoly(overlay, [area.points], STAFF_COLOR)
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

    for area in (staff_areas or {}).values():
        cv2.polylines(out, [area.points], True, STAFF_COLOR, 1, cv2.LINE_AA)

    for tid, tb in monitor.tables.items():
        color = table_color(tb, t, limit * 0.6)
        thick = 4 if (tid == highlight or tb.alert_active) else 2
        cv2.polylines(out, [tb.shape.points], True, color, thick, cv2.LINE_AA)
        cx, cy = tb.shape.points.mean(axis=0).astype(int)
        label = tid if tb.state is TableState.EMPTY else f"{tid} {_fmt(tb.wait(t))}"
        _tag(out, label, (cx, cy), color, center=True)

    for tid_, p in people.items():
        if not p.visible(t) or tid_ not in boxes:
            continue
        x1, y1, x2, y2 = (int(v) for v in boxes[tid_])
        color = {Role.STAFF: STAFF_COLOR, Role.CUSTOMER: CUSTOMER_COLOR}.get(p.role, UNKNOWN_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if p.role is Role.UNKNOWN else 2)
        label = {Role.STAFF: "S", Role.CUSTOMER: "C"}.get(p.role, "?") + str(p.track_id)
        if p.seated_table:
            label += f"@{p.seated_table}"
        _tag(out, label, (x1, y1), color, small=True)
        ax, ay = (int(v) for v in p.anchor)
        cv2.circle(out, (ax, ay), 3, color, -1, cv2.LINE_AA)

    _panel(out, monitor, t)
    return out


def _tag(img, text, org, color, center=False, small=False):
    scale = 0.4 if small else 0.55
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    if center:
        x -= w // 2
    y = max(y, h + 4)
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 1), FONT, scale, color, 1, cv2.LINE_AA)


def _panel(img, monitor: TableMonitor, t: float):
    limit = monitor.rules.unattended_minutes * 60
    lines = [(f"t={_fmt(t)}   table  state     wait  visits", (230, 230, 230))]
    for tid, tb in monitor.tables.items():
        state = "ALERT" if tb.alert_active else tb.state.value
        wait = _fmt(tb.wait(t)) if tb.state is TableState.OCCUPIED else "  -"
        lines.append((f"{tid:>12}  {state:<9} {wait:>5}  {tb.visits:>3}",
                      table_color(tb, t, limit * 0.6)))
    lh, x, y = 16, 8, 8
    w = 270
    h = lh * len(lines) + 8
    sub = img[y:y + h, x:x + w]
    cv2.addWeighted(sub, 0.35, np.zeros_like(sub), 0.65, 0, sub)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x + 6, y + lh * (i + 1)), FONT, 0.4, color, 1, cv2.LINE_AA)
