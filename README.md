# Restaurant Table Monitor

[![tests](https://github.com/ashfaknawshad/restaurant-table-monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/ashfaknawshad/restaurant-table-monitor/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Turns a restaurant's existing CCTV into a service monitor.** It works out which tables
have seated guests, tells guests apart from staff, and messages the manager on Telegram when
a table has waited too long without anyone attending it.

No new hardware, no smart tables, no staff wearables: one camera stream and a config file.

---

## The problem

In a busy dining room, a table that was seated and then forgotten is easy to miss and costly:
slow service is one of the most common complaints in restaurant reviews. Managers can't watch
every table, and existing CCTV is only looked at after something has gone wrong.

## What it does

- **Knows who is a guest.** A person only counts as a customer once they actually *sit*:
  inside a table zone, nearly still, with a seated body shape, for 20 seconds. Walking past a
  table or standing at one (a waiter taking an order) never counts.
- **Knows who is staff.** Staff are recognised from their uniform colour (configurable per
  brand), from time spent in staff-only areas such as behind the bar, or from stopping at
  several tables without ever sitting down.
- **Times every table.** When guests have waited 5 minutes (configurable) with no staff visit,
  it sends a Telegram alert with a snapshot of the table, follows up with reminders, and posts
  a "resolved" message when someone arrives.
- **Logs service metrics.** Occupied time, staff visits, alerts and longest wait per table,
  as an event stream (`events.jsonl`) and a summary (`summary.json`).
- **Easy to set up per camera.** An interactive editor lets you draw and drag table zones on
  the camera's own footage, play the video underneath, and run the full pipeline from any
  frame before saving.

## How it works

```mermaid
flowchart LR
    A[Video file / RTSP / webcam] --> B[YOLO11 person detection<br/>1280 px, optional tiling]
    B --> C[ByteTrack<br/>persistent IDs]
    C --> D[People state<br/>seated? staff?]
    D --> E[Table state machines<br/>occupancy + service timer]
    E --> F[Telegram alert<br/>with snapshot]
    E --> G[events.jsonl<br/>summary.json<br/>annotated video]
```

Frames are analysed at 5 fps: sitting and serving happen over seconds, so this cuts compute by
80% compared with every frame without losing anything that matters.

### Guest or staff?

Each tracked person starts as **unknown** and is promoted by evidence:

| Becomes | When |
|---|---|
| **Customer** | Inside a table zone for `seat_seconds`, average speed below `seated_max_speed` box-heights/s, box shape squat enough to be sitting |
| **Staff** | Torso matches the uniform colour in most samples, *or* spends `staff_area_seconds` in a staff-only area, *or* stops at `tables_visited` different tables without sitting |

Staff status is sticky for the life of a track, and a staff member who sits down never makes
a table "occupied".

### Per-table state machine

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Occupied: guest seated
    Occupied --> Occupied: staff at table ≥ 3 s → timer resets
    Occupied --> Unattended: no staff visit for 5 min
    Unattended --> Unattended: reminder every 5 min
    Unattended --> Occupied: staff arrives → "resolved"
    Occupied --> Empty: no seated guest for 60 s
    Unattended --> Empty: no seated guest for 60 s
```

## Design decisions

- **Timers belong to tables, not people.** Trackers lose people behind passing waiters and give
  them new IDs. If timers were per person, every ID switch would reset a guest's wait. Here a
  table stays occupied as long as *any* seated guest is present, and an empty table needs 60 s
  with nobody seated, which covers the 20 s it takes a re-identified guest to count as seated
  again.
- **"Seated" means sitting, not standing nearby.** Speed is normalised by box height, so the
  same threshold works for guests near and far from the camera, and box shape separates a
  seated guest from a waiter standing still at the table edge.
- **Zone test point near the seat.** From a high camera, the centre of a seated person's box
  often lands on the floor or the neighbouring table. The point used is 70% of the way down
  the box, which falls on the chair.
- **Model choice measured, not assumed.** On a test frame with about 25 people, `yolov8n` at
  640 px found 5; `yolo11m` at 1280 px finds 19–21. Tiled inference over the far half of the
  room found only about 2 more at 3.5× the cost, so it is available but off by default.
- **Alerts never stall the video loop.** Telegram calls run on a background thread with retries.
  Bot tokens come from environment variables, never from the committed config.
- **Rules are tested without a model.** The people and table logic take plain observations,
  so the tests drive them with synthetic tracks and run in under a second.

## Results so far

On a 59-second clip of a dining room (7 tables, about 25 people, alert threshold lowered to
30 s for the demo):

- Guest counts at the four nearest occupied tables (2, 3, 4 and 1) closely matched a visual
  count of the frames.
- Bar staff were classified as staff through the staff-area rule, and people walking through
  the room stayed unclassified rather than becoming customers.
- Unattended alerts fired for the five occupied tables, which is correct for that clip, since no
  waiter visited them within 30 s.
- Throughput: about 0.5 s per analysed frame for `yolo11m` at 1280 px on a laptop CPU, so live
  use at 5 fps needs a GPU (for example a Jetson Orin or a mini PC with an RTX card).

Longer footage from a partner restaurant is the next step, to measure alert precision and
recall against a hand-labelled log.

## Quick start

```bash
git clone https://github.com/ashfaknawshad/restaurant-table-monitor
cd restaurant-table-monitor
python -m venv venv
venv\Scripts\activate             # macOS/Linux: source venv/bin/activate
pip install -e ".[dev]"
```

Model weights download automatically on first run. Footage is not included: use your own
video or a camera stream, and set `source:` in the config or pass `--source`.

**1. Draw the zones** for your camera. Table zones should cover the table *and its chairs*;
add staff-only areas (bar, kitchen door, POS) with **B**.

```bash
tablemonitor-zones --config configs/sample.yaml --source path/to/video.mp4
```

| Mouse / key | Action |
|---|---|
| Drag a corner / drag inside / click an edge | Move the corner / move the zone / add a corner |
| Right-click a corner | Delete it |
| N / B | New table zone / new staff-only area |
| Space, `,` `.`, slider | Play, step and scrub the video |
| Arrows, `[` `]` | Nudge, shrink/grow the selected zone |
| T | Run the full pipeline from the current frame with the unsaved zones |
| U or Ctrl+Z / S / Q | Undo / save / quit |

**2. Run the monitor**

```bash
tablemonitor --source path/to/video.mp4 --show
tablemonitor --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101" --telegram
tablemonitor --source path/to/video.mp4 --unattended-minutes 0.5   # demo alerts on a short clip
```

**3. Telegram alerts.** Create a bot with [@BotFather](https://t.me/BotFather), add it to the
managers' group, and set the token and the group's chat ID as environment variables:

```bash
set TELEGRAM_BOT_TOKEN=123456:ABC...        # macOS/Linux: export TELEGRAM_BOT_TOKEN=...
set TELEGRAM_CHAT_ID=-1001234567890
```

Then pass `--telegram` or set `alerts.telegram: true` in the config.

## Configuration

One YAML file per camera (see [`configs/sample.yaml`](configs/sample.yaml)). The main settings:

| Setting | Default | Meaning |
|---|---|---|
| `rules.seat_seconds` | 20 | Time seated in a zone before someone counts as a guest |
| `rules.unattended_minutes` | 5 | Wait without a staff visit before alerting |
| `rules.reminder_minutes` | 5 | Repeat interval while still unattended |
| `rules.service_seconds` | 3 | Time staff must stay at a table for it to count as a visit |
| `rules.service_margin_px` | 30 | How far around a table zone staff can stand while serving |
| `rules.empty_seconds` | 60 | No seated guest for this long → the table is empty |
| `staff.uniform_hsv_ranges` | `[]` | HSV ranges of the uniform, e.g. black shirts `[[[0, 0, 0], [180, 255, 60]]]` |
| `detector.model`, `detector.imgsz` | `yolo11m.pt`, 1280 | Larger model and resolution find distant guests |
| `detector.tiles` | `[]` | Extra zoomed crops for far tables (one extra model run each) |
| `process_fps` | 5 | Frames analysed per second |

## Project structure

```
tablemonitor/
  perception.py   YOLO detection (with optional tiling) and ByteTrack tracking
  people.py       per-person dwell, speed and posture; guest/staff decisions
  tables.py       per-table state machine, service visits, unattended timer
  alerts.py       snapshots and Telegram delivery on a background thread
  render.py       overlays: table states, roles, wait timers, status panel
  pipeline.py     main loop and CLI (`tablemonitor`)
  editor.py       interactive zone editor (`tablemonitor-zones`)
  config.py       typed YAML config
  geometry.py     zone polygons and service areas
configs/          one YAML file per camera
tests/            rule tests with synthetic tracks
```

## Tests

```bash
pytest
```

The tests cover sitting versus walking past versus standing at a table, unattended alerts and
their resolution, staff visits resetting the timer, tracking-ID switches, tables emptying,
staff detection from behaviour, and reminders. They run on every push via GitHub Actions.

## Limitations and roadmap

- **Uniform detection is a colour rule.** It needs calibrating per brand and can be confused by
  guests wearing the same colour. Next: a small staff/guest classifier trained on each brand's
  footage.
- **Heavily hidden guests are missed.** People fully behind others aren't detected. Fine-tuning
  the detector on the restaurant's own camera angle is the fix.
- **Posture is a box-shape heuristic.** Pose estimation would make sitting detection more robust.
- **Single camera.** Branches with several cameras need a table to be tracked across views and a
  central dashboard for all branches.
- **Table clearing.** Detecting "guests left, table not yet cleared" from an empty-table
  reference image.

## Privacy

The system works on people's positions and clothing colour only: no face recognition and no
identity. It is designed to run on hardware inside the restaurant, so footage never leaves the
premises; only alert snapshots go to the managers' group. Anyone deploying it should follow local CCTV and data protection rules,
including signage.

## License

[MIT](LICENSE)
