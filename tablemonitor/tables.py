"""
Per-table state machine and the unattended-table timer.

    EMPTY ──(a customer is seated)──► OCCUPIED ──(no seated customer for empty_seconds)──► EMPTY

While OCCUPIED, `last_service` starts at the moment customers sat down and is moved
forward whenever a staff member stays in the table's service area for
`service_seconds`. If `unattended_minutes` pass without service, an UNATTENDED
event fires, then a REMINDER every `reminder_minutes`, and RESOLVED once staff arrive.

Everything is decided at table level, so a customer's tracking ID changing mid-meal
(common under occlusion) does not reset the table or its timer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import RulesConfig, Zone
from .geometry import ZoneShape
from .people import Person


class TableState(Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"


class EventKind(Enum):
    OCCUPIED = "occupied"
    SERVED = "served"
    UNATTENDED = "unattended"
    REMINDER = "reminder"
    RESOLVED = "resolved"
    EMPTY = "empty"


@dataclass
class Event:
    kind: EventKind
    table_id: str
    table_name: str
    t: float
    wait_seconds: float = 0.0
    customers: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value, "table": self.table_id, "name": self.table_name,
            "t": round(self.t, 1), "wait_seconds": round(self.wait_seconds, 1),
            "customers": self.customers,
        }


@dataclass
class Table:
    table_id: str
    name: str
    shape: ZoneShape
    state: TableState = TableState.EMPTY
    occupied_since: float = 0.0
    last_customer_seen: float = 0.0
    last_service: float = 0.0
    customers: int = 0
    alert_active: bool = False
    last_alert: float = 0.0
    # Totals
    occupied_seconds: float = 0.0
    visits: int = 0
    alerts: int = 0
    longest_wait: float = 0.0
    # staff track id -> (time they arrived in the service area, already counted?)
    staff_near: dict[int, list] = field(default_factory=dict)

    def wait(self, t: float) -> float:
        return t - self.last_service if self.state is TableState.OCCUPIED else 0.0

    def occupied_total(self, t: float) -> float:
        extra = t - self.occupied_since if self.state is TableState.OCCUPIED else 0.0
        return self.occupied_seconds + extra


class TableMonitor:
    def __init__(self, zones: dict[str, Zone], rules: RulesConfig):
        self.rules = rules
        self.tables = {
            tid: Table(tid, z.name, ZoneShape(z.polygon, rules.service_margin_px))
            for tid, z in zones.items()
        }

    def shapes(self) -> dict[str, ZoneShape]:
        return {tid: tb.shape for tid, tb in self.tables.items()}

    def update(self, people: dict[int, Person], t: float) -> list[Event]:
        events: list[Event] = []
        r = self.rules
        for tb in self.tables.values():
            seated = [p for p in people.values() if p.seated_table == tb.table_id and not p.is_staff]
            tb.customers = len(seated)

            def ev(kind, wait=0.0):
                events.append(Event(kind, tb.table_id, tb.name, t, wait, tb.customers))

            # --- occupancy ---------------------------------------------
            if seated:
                tb.last_customer_seen = max(p.last_seen for p in seated)
                if tb.state is TableState.EMPTY:
                    tb.state = TableState.OCCUPIED
                    tb.occupied_since = min(p.seated_since for p in seated)
                    tb.last_service = tb.occupied_since
                    ev(EventKind.OCCUPIED)
            elif tb.state is TableState.OCCUPIED and t - tb.last_customer_seen >= r.empty_seconds:
                tb.occupied_seconds += tb.last_customer_seen - tb.occupied_since
                tb.state = TableState.EMPTY
                tb.alert_active = False
                tb.staff_near.clear()
                ev(EventKind.EMPTY)
                continue

            # --- staff service -------------------------------------------
            present = set()
            for p in people.values():
                if p.is_staff and p.visible(t) and tb.shape.near(*p.anchor):
                    present.add(p.track_id)
                    entry = tb.staff_near.setdefault(p.track_id, [t, False])
                    if tb.state is TableState.OCCUPIED and t - entry[0] >= r.service_seconds:
                        if not entry[1]:
                            entry[1] = True
                            tb.visits += 1
                            if tb.alert_active:
                                tb.alert_active = False
                                ev(EventKind.RESOLVED, tb.wait(t))
                            ev(EventKind.SERVED, tb.wait(t))
                        tb.last_service = t   # attending right now
            for sid in list(tb.staff_near):
                if sid not in present:
                    del tb.staff_near[sid]

            # --- unattended timer ----------------------------------------
            if tb.state is TableState.OCCUPIED:
                wait = tb.wait(t)
                tb.longest_wait = max(tb.longest_wait, wait)
                if not tb.alert_active and wait >= r.unattended_minutes * 60:
                    tb.alert_active, tb.last_alert = True, t
                    tb.alerts += 1
                    ev(EventKind.UNATTENDED, wait)
                elif tb.alert_active and t - tb.last_alert >= r.reminder_minutes * 60:
                    tb.last_alert = t
                    ev(EventKind.REMINDER, wait)
        return events

    def summary(self, t: float) -> dict:
        return {
            tid: {
                "name": tb.name,
                "state": tb.state.value,
                "occupied_seconds": round(tb.occupied_total(t), 1),
                "staff_visits": tb.visits,
                "unattended_alerts": tb.alerts,
                "longest_wait_seconds": round(tb.longest_wait, 1),
            }
            for tid, tb in self.tables.items()
        }
