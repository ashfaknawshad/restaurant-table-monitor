"""
Main loop: read a video file or live stream, detect and track people, update
people/table state, raise alerts, and write an annotated video plus an event log.

    python -m tablemonitor.pipeline --config configs/sample.yaml --show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .alerts import Notifier
from .config import Config, load_config
from .geometry import ZoneShape, anchor_points
from .people import Observation, PeopleState, uniform_fraction
from .perception import PersonDetector, PersonTracker
from .render import draw
from .tables import EventKind, TableMonitor

WINDOW = "Table Monitor"


def _open(source: str):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    live = source.isdigit() or "://" in source
    return cap, live


def run(cfg: Config, source: str | None = None, output_dir: str = "output",
        show: bool = False, start_frame: int = 0, max_seconds: float | None = None,
        telegram: bool | None = None, detector: PersonDetector | None = None) -> dict:
    source = source or cfg.source
    cap, live = _open(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / cfg.process_fps))
    if start_frame and not live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None if live else cv2.VideoWriter(
        str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    events_file = open(out_dir / "events.jsonl", "w", encoding="utf-8")

    detector = detector or PersonDetector(cfg.detector)
    tracker = PersonTracker(cfg.tracker, fps / step)
    monitor = TableMonitor(cfg.tables, cfg.rules)
    staff_areas = {k: ZoneShape(z.polygon) for k, z in cfg.staff_areas.items()}
    people_state = PeopleState(cfg.rules, cfg.staff, monitor.shapes(), staff_areas)
    notifier = Notifier(cfg.alerts, telegram)

    print(f"[Monitor] {source}  {width}x{height} @ {fps:.1f} fps, analysing {fps / step:.1f} fps, "
          f"{len(cfg.tables)} tables, unattended after {cfg.rules.unattended_minutes:g} min")

    frame_idx = start_frame
    t0 = time.time()
    last_analysed = -1e9
    boxes: dict[int, tuple] = {}
    people: dict = {}
    t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (time.time() - t0) if live else frame_idx / fps
            if max_seconds is not None and t - start_frame / fps > max_seconds:
                break

            analyse = (t - last_analysed >= 1.0 / cfg.process_fps) if live else frame_idx % step == 0
            if analyse:
                last_analysed = t
                tracks = tracker(detector(frame))
                xyxy = np.array([tr.xyxy for tr in tracks], dtype=np.float32).reshape(-1, 4)
                anchors = anchor_points(xyxy, cfg.rules.anchor_y)
                observations = []
                for tr, (ax, ay) in zip(tracks, anchors):
                    x1, y1, x2, y2 = tr.xyxy
                    h, w = max(y2 - y1, 1.0), max(x2 - x1, 1.0)
                    observations.append(Observation(
                        tr.track_id, (float(ax), float(ay)), float(h), float(h / w),
                        uniform_fraction(frame, tr.xyxy, cfg.staff.uniform_hsv_ranges)))
                boxes = {tr.track_id: tuple(tr.xyxy) for tr in tracks}
                people = people_state.update(observations, t)
                events = monitor.update(people, t)
                for e in events:
                    events_file.write(json.dumps(e.to_dict()) + "\n")
                    if e.kind in (EventKind.OCCUPIED, EventKind.EMPTY, EventKind.SERVED):
                        print(f"  [{t:7.1f}s] {e.table_id}: {e.kind.value}"
                              + (f" (waited {e.wait_seconds:.0f}s)" if e.kind is EventKind.SERVED else ""))
                for e in events:
                    if e.kind in (EventKind.UNATTENDED, EventKind.REMINDER, EventKind.RESOLVED):
                        snap = draw(frame, monitor, people, boxes, t, staff_areas,
                                    highlight=e.table_id)
                        notifier.handle([e], snap)

            annotated = draw(frame, monitor, people, boxes, t, staff_areas)
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow(WINDOW, annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_idx += 1
            if analyse and not live and frame_idx % int(fps * 10) < step:
                print(f"  ... {t:6.1f}s processed ({time.time() - t0:.0f}s wall)")
    finally:
        cap.release()
        if writer:
            writer.release()
        events_file.close()
        if show:
            cv2.destroyWindow(WINDOW)
        notifier.close()

    summary = {
        "source": source,
        "duration_seconds": round(t, 1),
        "tables": monitor.summary(t),
        "alerts": notifier.sent,
        "staff_track_ids": sorted(p.track_id for p in people.values() if p.is_staff),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary, out_dir)
    return summary


def _print_summary(summary: dict, out_dir: Path):
    print("\n" + "=" * 64)
    print(f"{'table':<8}{'occupied':>10}{'visits':>8}{'alerts':>8}{'longest wait':>14}")
    for tid, s in summary["tables"].items():
        occ = s["occupied_seconds"]
        wait = s["longest_wait_seconds"]
        print(f"{tid:<8}{int(occ) // 60:>6}m{int(occ) % 60:02d}s{s['staff_visits']:>8}"
              f"{s['unattended_alerts']:>8}{int(wait) // 60:>10}m{int(wait) % 60:02d}s")
    print(f"\nOutputs in {out_dir}/: annotated.mp4, events.jsonl, summary.json")
    print("=" * 64)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Restaurant table monitoring")
    ap.add_argument("--config", "-c", default="configs/sample.yaml")
    ap.add_argument("--source", "-s", help="Video file, RTSP URL or webcam index (default: config)")
    ap.add_argument("--output", "-o", default="output")
    ap.add_argument("--show", action="store_true", help="Show the annotated video live")
    ap.add_argument("--start", type=float, default=0.0, help="Start at this many seconds (files)")
    ap.add_argument("--max-seconds", type=float, help="Stop after this much video")
    ap.add_argument("--unattended-minutes", type=float,
                    help="Override the alert threshold (e.g. 0.5 to demo on a short clip)")
    ap.add_argument("--telegram", action="store_true", help="Send alerts to Telegram")
    ap.add_argument("--device", help="cpu, cuda or mps (default: config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.unattended_minutes is not None:
        cfg.rules.unattended_minutes = args.unattended_minutes
        cfg.rules.reminder_minutes = max(args.unattended_minutes, 0.5)
    if args.device:
        cfg.detector.device = args.device
    if not cfg.tables:
        sys.exit("No tables in the config. Draw them first: python -m tablemonitor.editor")

    source = args.source or cfg.source
    start_frame = 0
    if args.start:
        cap, live = _open(source)
        start_frame = 0 if live else int(args.start * (cap.get(cv2.CAP_PROP_FPS) or 25.0))
        cap.release()
    run(cfg, source, args.output, args.show, start_frame, args.max_seconds,
        telegram=True if args.telegram else None)


if __name__ == "__main__":
    main()
