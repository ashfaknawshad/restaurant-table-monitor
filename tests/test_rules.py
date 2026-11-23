"""Rule tests with synthetic people: no model or video needed."""

from tablemonitor.config import RulesConfig, StaffConfig, Zone
from tablemonitor.geometry import ZoneShape
from tablemonitor.people import Observation, PeopleState, Role
from tablemonitor.tables import EventKind, TableMonitor, TableState

DT = 0.2  # 5 fps

TABLES = {
    "T1": Zone("Window", [[0, 0], [100, 0], [100, 100], [0, 100]]),
    "T2": Zone("Middle", [[200, 0], [300, 0], [300, 100], [200, 100]]),
    "T3": Zone("Corner", [[400, 0], [500, 0], [500, 100], [400, 100]]),
}
BAR = {"S1": Zone("Bar", [[0, 300], [100, 300], [100, 400], [0, 400]])}


def seated(tid, x, y):
    return Observation(tid, (x, y), box_height=80, aspect=1.5)


def standing(tid, x, y):
    return Observation(tid, (x, y), box_height=160, aspect=3.0)


class Sim:
    def __init__(self, **rules):
        self.rules = RulesConfig(**{"unattended_minutes": 1.0, "reminder_minutes": 1.0, **rules})
        self.monitor = TableMonitor(TABLES, self.rules)
        self.people = PeopleState(self.rules, StaffConfig(), self.monitor.shapes(),
                                  {k: ZoneShape(z.polygon) for k, z in BAR.items()})
        self.t = 0.0
        self.events = []

    def run(self, seconds, observe):
        """observe(t) -> list of observations for that instant."""
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + DT, 6)
            people = self.people.update(observe(self.t), self.t)
            self.events += self.monitor.update(people, self.t)
        return self

    def kinds(self, table=None):
        return [e.kind for e in self.events if table is None or e.table_id == table]

    def table(self, tid):
        return self.monitor.tables[tid]


def test_sitting_customer_occupies_table():
    sim = Sim().run(25, lambda t: [seated(1, 50, 50)])
    assert sim.table("T1").state is TableState.OCCUPIED
    assert sim.people.people[1].role is Role.CUSTOMER
    assert EventKind.OCCUPIED in sim.kinds("T1")


def test_walking_through_a_zone_does_not_occupy():
    # Crosses T1 over 30 s at ~1 box-height per second.
    sim = Sim().run(30, lambda t: [standing(1, -20 + t * 5, 50)])
    assert sim.table("T1").state is TableState.EMPTY
    assert sim.people.people[1].role is not Role.CUSTOMER


def test_standing_still_at_a_table_does_not_occupy():
    # e.g. a waiter taking a long order: still, but upright.
    sim = Sim().run(40, lambda t: [standing(1, 50, 50)])
    assert sim.table("T1").state is TableState.EMPTY


def test_unattended_alert_then_resolved_by_staff():
    sim = Sim()
    sim.run(25, lambda t: [seated(1, 50, 50)])
    sim.run(60, lambda t: [seated(1, 50, 50)])
    assert EventKind.UNATTENDED in sim.kinds("T1")
    assert sim.table("T1").alert_active

    # Staff (identified via the bar) walks over and stands at the table edge.
    sim.run(5, lambda t: [seated(1, 50, 50), standing(9, 50, 350)])
    assert sim.people.people[9].role is Role.STAFF
    sim.run(5, lambda t: [seated(1, 50, 50), standing(9, 110, 50)])
    assert EventKind.RESOLVED in sim.kinds("T1")
    assert not sim.table("T1").alert_active
    assert sim.table("T1").visits == 1


def test_staff_visit_resets_timer_so_no_alert():
    sim = Sim()
    sim.run(5, lambda t: [standing(9, 50, 350)])           # becomes staff at the bar
    sim.run(25, lambda t: [seated(1, 50, 50)])
    sim.run(25, lambda t: [seated(1, 50, 50)])        # ~45 s since sitting down
    sim.run(5, lambda t: [seated(1, 50, 50), standing(9, 110, 50)])
    sim.run(40, lambda t: [seated(1, 50, 50)])
    assert EventKind.UNATTENDED not in sim.kinds("T1")
    assert EventKind.SERVED in sim.kinds("T1")


def test_tracking_id_change_does_not_reset_table():
    sim = Sim()
    sim.run(25, lambda t: [seated(1, 50, 50)])
    since = sim.table("T1").occupied_since
    # Track 1 is lost; the same guest comes back as track 2.
    sim.run(30, lambda t: [seated(2, 52, 50)])
    assert sim.table("T1").state is TableState.OCCUPIED
    assert sim.table("T1").occupied_since == since
    assert sim.kinds("T1").count(EventKind.OCCUPIED) == 1


def test_table_empties_after_guests_leave():
    sim = Sim(empty_seconds=30)
    sim.run(25, lambda t: [seated(1, 50, 50)])
    sim.run(40, lambda t: [])
    assert sim.table("T1").state is TableState.EMPTY
    assert EventKind.EMPTY in sim.kinds("T1")
    assert 20 <= sim.table("T1").occupied_seconds <= 30


def test_person_stopping_at_many_tables_becomes_staff():
    def path(t):
        # 4 s at each table in turn, never sitting.
        x = [50, 250, 450][min(int(t // 4), 2)]
        return [standing(5, x, 110)]
    sim = Sim().run(12, path)
    assert sim.people.people[5].role is Role.STAFF


def test_staff_sitting_down_does_not_occupy():
    sim = Sim()
    sim.run(5, lambda t: [standing(9, 50, 350)])
    sim.run(30, lambda t: [seated(9, 50, 50)])
    assert sim.table("T1").state is TableState.EMPTY


def test_reminder_repeats_while_waiting():
    sim = Sim()
    sim.run(25, lambda t: [seated(1, 50, 50)])
    sim.run(130, lambda t: [seated(1, 50, 50)])
    assert sim.kinds("T1").count(EventKind.UNATTENDED) == 1
    assert sim.kinds("T1").count(EventKind.REMINDER) >= 1
