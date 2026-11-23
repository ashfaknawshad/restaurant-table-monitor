"""
Per-person state: where each tracked person is, whether they are seated at a table,
and whether they are a customer or staff.

A person only becomes a CUSTOMER by sitting: staying inside a table zone, nearly
still and with a sitting (not standing) box shape, for `seat_seconds`. Walking past
or standing at a table never makes someone a customer.

A person becomes STAFF from any of:
  * wearing the uniform colour (when configured for the brand),
  * spending time inside a staff-only area (behind the bar, kitchen door, POS),
  * stopping at several different tables without ever sitting down,
  * a manual track-ID override.
STAFF is sticky for the life of the track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import RulesConfig, StaffConfig
from .geometry import ZoneShape

# How long someone may step out of a zone (or be mis-placed by jitter) without
# resetting their dwell timer.
ZONE_GAP_SECONDS = 2.0
# Seconds a person must linger near a table for it to count as "stopping" there.
APPROACH_SECONDS = 2.0
# Sitting people's boxes are squat; standing people's are tall (height / width).
SEATED_MAX_ASPECT = 2.4
# Forget a person this long after their track was last seen.
FORGET_SECONDS = 60.0


class Role(Enum):
    UNKNOWN = "unknown"
    CUSTOMER = "customer"
    STAFF = "staff"


@dataclass
class Observation:
    track_id: int
    anchor: tuple[float, float]
    box_height: float
    aspect: float                      # box height / width
    uniform_fraction: float | None = None


@dataclass
class Person:
    track_id: int
    first_seen: float
    last_seen: float
    anchor: tuple[float, float] = (0.0, 0.0)
    box_height: float = 1.0
    speed: float = 0.0                 # box-heights per second (smoothed)
    role: Role = Role.UNKNOWN
    staff_reason: str = ""

    # Dwell inside a table zone
    zone: str | None = None
    zone_since: float = 0.0
    zone_last_in: float = 0.0
    zone_speed_sum: float = 0.0
    zone_aspect_sum: float = 0.0
    zone_samples: int = 0

    # Seated state
    seated_table: str | None = None
    seated_since: float = 0.0
    seated_last_in: float = 0.0

    # Staff evidence
    uniform_hits: int = 0
    uniform_samples: int = 0
    staff_area_time: float = 0.0
    near_table: str | None = None
    near_since: float = 0.0
    stopped_at: dict[str, float] = field(default_factory=dict)  # table -> last time

    @property
    def is_staff(self) -> bool:
        return self.role is Role.STAFF

    def visible(self, t: float) -> bool:
        return self.last_seen == t


def uniform_fraction(frame: np.ndarray, xyxy, hsv_ranges) -> float | None:
    """Fraction of the torso area matching any of the uniform HSV ranges."""
    if not hsv_ranges:
        return None
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = x1 + int(w * 0.25), x2 - int(w * 0.25)
    ty1, ty2 = y1 + int(h * 0.2), y1 + int(h * 0.5)
    H, W = frame.shape[:2]
    tx1, tx2, ty1, ty2 = max(tx1, 0), min(tx2, W), max(ty1, 0), min(ty2, H)
    if tx2 - tx1 < 3 or ty2 - ty1 < 3:
        return None
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    return float(mask.mean())


class PeopleState:
    def __init__(self, rules: RulesConfig, staff: StaffConfig,
                 tables: dict[str, ZoneShape], staff_areas: dict[str, ZoneShape]):
        self.rules = rules
        self.staff_cfg = staff
        self.tables = tables
        self.staff_areas = staff_areas
        self.people: dict[int, Person] = {}

    # ------------------------------------------------------------------

    def update(self, observations: list[Observation], t: float) -> dict[int, Person]:
        for obs in observations:
            p = self.people.get(obs.track_id)
            if p is None:
                p = Person(obs.track_id, first_seen=t, last_seen=t, anchor=obs.anchor,
                           box_height=obs.box_height)
                if obs.track_id in self.staff_cfg.track_ids:
                    self._make_staff(p, "manual")
                self.people[obs.track_id] = p
            self._observe(p, obs, t)

        for tid in list(self.people):
            p = self.people[tid]
            if p.last_seen == t:
                continue
            if p.seated_table and t - p.seated_last_in > self.rules.leave_seconds:
                p.seated_table = None
            if t - p.last_seen > FORGET_SECONDS:
                del self.people[tid]
        return self.people

    # ------------------------------------------------------------------

    def _observe(self, p: Person, obs: Observation, t: float):
        dt = t - p.last_seen
        if dt > 0:
            dx = obs.anchor[0] - p.anchor[0]
            dy = obs.anchor[1] - p.anchor[1]
            inst = (dx * dx + dy * dy) ** 0.5 / max(obs.box_height, 1.0) / dt
            p.speed = 0.6 * p.speed + 0.4 * inst
        p.anchor, p.box_height, p.last_seen = obs.anchor, obs.box_height, t
        x, y = obs.anchor
        step = min(dt, 1.0)

        # --- staff evidence -------------------------------------------------
        if obs.uniform_fraction is not None:
            p.uniform_samples += 1
            p.uniform_hits += obs.uniform_fraction >= self.staff_cfg.uniform_min_fraction
            if (p.uniform_samples >= self.staff_cfg.uniform_min_samples
                    and p.uniform_hits / p.uniform_samples >= 0.6):
                self._make_staff(p, "uniform")

        if any(a.contains(x, y) for a in self.staff_areas.values()):
            p.staff_area_time += step
            if p.staff_area_time >= self.staff_cfg.staff_area_seconds:
                self._make_staff(p, "staff area")

        # --- which table zone is the person in ----------------------------
        current = None
        if p.seated_table and self.tables[p.seated_table].contains(x, y):
            current = p.seated_table
        else:
            current = next((tid for tid, z in self.tables.items() if z.contains(x, y)), None)

        if current is not None and current == p.zone:
            p.zone_last_in = t
        elif current is None and p.zone is not None and t - p.zone_last_in <= ZONE_GAP_SECONDS:
            pass  # brief step out / jitter: keep the dwell running
        else:
            p.zone, p.zone_since, p.zone_last_in = current, t, t
            p.zone_speed_sum = p.zone_aspect_sum = 0.0
            p.zone_samples = 0
        if current is not None and current == p.zone:
            p.zone_speed_sum += p.speed
            p.zone_aspect_sum += obs.aspect
            p.zone_samples += 1

        # --- seated? ------------------------------------------------------
        if p.seated_table is not None:
            if current == p.seated_table:
                p.seated_last_in = t
            elif t - p.seated_last_in > self.rules.leave_seconds:
                p.seated_table = None
        elif (not p.is_staff and p.zone is not None and current == p.zone
              and t - p.zone_since >= self.rules.seat_seconds and p.zone_samples > 0
              and p.zone_speed_sum / p.zone_samples <= self.rules.seated_max_speed
              and p.zone_aspect_sum / p.zone_samples <= SEATED_MAX_ASPECT):
            p.seated_table, p.seated_since, p.seated_last_in = p.zone, p.zone_since, t
            if p.role is Role.UNKNOWN:
                p.role = Role.CUSTOMER

        # --- stopping at several tables without sitting -> staff ---------
        if p.role is Role.UNKNOWN and p.seated_table is None:
            near = next((tid for tid, z in self.tables.items() if z.near(x, y)), None)
            if near != p.near_table:
                p.near_table, p.near_since = near, t
            elif near is not None and t - p.near_since >= APPROACH_SECONDS:
                p.stopped_at[near] = t
            window = self.staff_cfg.visit_window_seconds
            recent = [k for k, v in p.stopped_at.items() if t - v <= window]
            if len(recent) >= self.staff_cfg.tables_visited:
                self._make_staff(p, f"stopped at {len(recent)} tables")

    @staticmethod
    def _make_staff(p: Person, reason: str):
        if p.role is not Role.STAFF:
            p.role, p.staff_reason = Role.STAFF, reason
            p.seated_table = None
