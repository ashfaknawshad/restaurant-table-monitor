"""
Interactive zone editor: draw table zones and staff-only areas on the camera's video,
play the video underneath while editing, and test the full pipeline before saving.

    python -m tablemonitor.editor --config configs/sample.yaml

MOUSE
  Drag a corner ............ move it
  Drag inside a zone ....... move the whole zone
  Click a zone's edge ...... insert a corner there
  Right-click a corner ..... delete it

KEYS
  Space / , . ... play-pause / step one frame (or use the Frame slider)
  Tab / 1-9 ..... select a zone
  Arrows / [ ] .. nudge 1 px / shrink-grow the selected zone
  N ............. new table zone      B ... new staff-only area (bar, kitchen door, POS)
  Enter ......... finish drawing      Esc . cancel drawing
  Delete / X .... delete the selected zone
  U / Ctrl+Z .... undo
  T ............. test the pipeline from this frame with the zones as they are now
  S ............. save to the config file
  H ............. toggle help         Q ... quit
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from .config import Config, Zone, load_config, save_config

WINDOW = "Zone Editor"
VERTEX_HIT = 8
EDGE_HIT = 6
STAFF_COLOR = [255, 140, 0]
TABLE_COLORS = [
    [0, 255, 128], [0, 200, 255], [255, 128, 0], [200, 0, 255], [0, 128, 255],
    [128, 255, 0], [255, 0, 128], [255, 255, 0], [0, 255, 255], [128, 128, 255],
]

# cv2.waitKeyEx codes: Windows, then Linux/GTK, then macOS
KEY_LEFT, KEY_UP = {2424832, 65361, 63234}, {2490368, 65362, 63232}
KEY_RIGHT, KEY_DOWN = {2555904, 65363, 63235}, {2621440, 65364, 63233}
KEY_DELETE = {3014656, 65535, 63272}
KEY_ENTER, KEY_ESC, KEY_TAB, KEY_BACKSPACE, KEY_CTRL_Z = {13, 10}, 27, 9, 8, 26


def _dist_to_segment(p, a, b):
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    c = a + t * ab
    return float(np.linalg.norm(p - c)), (float(c[0]), float(c[1]))


def _inside(points, x, y) -> bool:
    if len(points) < 3:
        return False
    cnt = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0


class VideoFrames:
    def __init__(self, source: str):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            sys.exit(f"[Editor] Cannot open video: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.index, self.frame = -1, None
        self.seek(0)
        if self.frame is None:
            sys.exit(f"[Editor] Cannot read frames from: {source}")

    @property
    def size(self):
        h, w = self.frame.shape[:2]
        return w, h

    def seek(self, idx: int):
        idx = int(np.clip(idx, 0, self.total - 1))
        if idx != self.index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.index, self.frame = idx, frame

    def next(self) -> bool:
        if self.index + 1 >= self.total:
            return False
        self.seek(self.index + 1)
        return True


class ZoneEditor:
    def __init__(self, cfg: Config, video: VideoFrames, scale: float):
        self.cfg = cfg
        self.video = video
        self.scale = scale
        w, h = video.size
        self.bounds = (w - 1, h - 1)
        self.selected = next((("table", k) for k in cfg.tables), None)
        self.history: list[tuple] = []
        self.dirty = False
        self.mouse = (0.0, 0.0)
        self.drag = None
        self.drawing: list | None = None
        self.drawing_kind = "table"
        self.playing = False
        self.last_tick = time.time()
        self.show_help = True
        self.message, self.message_until = "", 0.0
        self.quit_armed = False
        self._detector = None   # loaded on first test, reused afterwards

    # ---------- zone access ----------

    def group(self, kind):
        return self.cfg.tables if kind == "table" else self.cfg.staff_areas

    def zone(self, key) -> Zone:
        return self.group(key[0])[key[1]]

    def keys(self):
        return [("table", k) for k in self.cfg.tables] + [("staff", k) for k in self.cfg.staff_areas]

    def ordered_keys(self):
        ks = self.keys()
        if self.selected in ks:
            ks.remove(self.selected)
            ks.insert(0, self.selected)
        return ks

    # ---------- helpers ----------

    def flash(self, text, seconds=2.5):
        self.message, self.message_until = text, time.time() + seconds
        print(f"[Editor] {text}")

    def snapshot(self):
        return deepcopy(self.cfg.tables), deepcopy(self.cfg.staff_areas)

    def push_undo(self):
        self.history.append(self.snapshot())
        del self.history[:-200]
        self.dirty, self.quit_armed = True, False

    def undo(self):
        if not self.history:
            return self.flash("Nothing to undo")
        self.cfg.tables, self.cfg.staff_areas = self.history.pop()
        if self.selected not in self.keys():
            self.selected = None
        self.dirty = True
        self.flash("Undo")

    def clamp(self, x, y):
        bx, by = self.bounds
        return [int(np.clip(round(x), 0, bx)), int(np.clip(round(y), 0, by))]

    def hit_vertex(self, x, y):
        tol = VERTEX_HIT / min(self.scale, 1.0)
        for key in self.ordered_keys():
            for i, (px, py) in enumerate(self.zone(key).polygon):
                if (px - x) ** 2 + (py - y) ** 2 <= tol ** 2:
                    return key, i
        return None

    def hit_edge(self, x, y):
        if self.selected not in self.keys():
            return None
        pts = self.zone(self.selected).polygon
        best = None
        for i in range(len(pts)):
            d, c = _dist_to_segment((x, y), pts[i], pts[(i + 1) % len(pts)])
            if d <= EDGE_HIT and (best is None or d < best[0]):
                best = (d, i + 1, c)
        return None if best is None else (self.selected, best[1], best[2])

    def hit_zone(self, x, y):
        return next((k for k in self.ordered_keys() if _inside(self.zone(k).polygon, x, y)), None)

    def new_id(self, kind):
        if kind == "table":
            n = 1
            while f"T{n}" in self.cfg.tables:
                n += 1
            return f"T{n}"
        n = 1
        while f"S{n}" in self.cfg.staff_areas:
            n += 1
        return f"S{n}"

    # ---------- mouse ----------

    def on_mouse(self, event, mx, my, flags, _):
        x, y = mx / self.scale, my / self.scale
        self.mouse = (x, y)
        if self.drawing is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                pts = self.drawing
                if len(pts) >= 3 and (pts[0][0] - x) ** 2 + (pts[0][1] - y) ** 2 <= VERTEX_HIT ** 2:
                    self.finish_drawing()
                else:
                    pts.append(self.clamp(x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing.pop()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hv = self.hit_vertex(x, y)
            if hv:
                self.push_undo()
                self.selected = hv[0]
                self.drag = {"kind": "vertex", "key": hv[0], "i": hv[1]}
                return
            he = self.hit_edge(x, y)
            if he:
                key, i, (cx, cy) = he
                self.push_undo()
                self.zone(key).polygon.insert(i, self.clamp(cx, cy))
                self.drag = {"kind": "vertex", "key": key, "i": i}
                return
            hz = self.hit_zone(x, y)
            if hz:
                self.push_undo()
                self.selected = hz
                self.drag = {"kind": "zone", "key": hz, "start": (x, y),
                             "orig": deepcopy(self.zone(hz).polygon)}
                return
            self.selected = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            z = self.zone(self.drag["key"])
            if self.drag["kind"] == "vertex":
                z.polygon[self.drag["i"]] = self.clamp(x, y)
            else:
                orig = np.array(self.drag["orig"], dtype=float)
                bx, by = self.bounds
                dx = float(np.clip(x - self.drag["start"][0], -orig[:, 0].min(), bx - orig[:, 0].max()))
                dy = float(np.clip(y - self.drag["start"][1], -orig[:, 1].min(), by - orig[:, 1].max()))
                z.polygon = [self.clamp(px + dx, py + dy) for px, py in orig]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag and self.drag["kind"] == "zone" and \
                    self.drag["orig"] == self.zone(self.drag["key"]).polygon:
                self.history.pop()   # just a click to select
            self.drag = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            hv = self.hit_vertex(x, y)
            if hv:
                key, i = hv
                if len(self.zone(key).polygon) <= 3:
                    return self.flash("A zone needs at least 3 corners (Delete removes the zone)")
                self.push_undo()
                self.zone(key).polygon.pop(i)
                self.selected = key
            else:
                self.selected = self.hit_zone(x, y)

    def finish_drawing(self):
        if self.drawing is None:
            return
        if len(self.drawing) < 3:
            return self.flash("Need at least 3 corners - keep clicking, or Esc to cancel")
        self.push_undo()
        kind = self.drawing_kind
        zid = self.new_id(kind)
        if kind == "table":
            color = TABLE_COLORS[len(self.cfg.tables) % len(TABLE_COLORS)]
            self.cfg.tables[zid] = Zone(f"Table {zid[1:]}", self.drawing, list(color))
        else:
            self.cfg.staff_areas[zid] = Zone(f"Staff area {zid[1:]}", self.drawing, list(STAFF_COLOR))
        self.selected = (kind, zid)
        self.drawing = None
        self.flash(f"Added {zid} (rename it in the config file after saving)")

    # ---------- keyboard ----------

    def on_key(self, key) -> bool:
        if key == -1:
            return True
        ch = chr(key & 0xFF).lower() if key < 256 else ""

        if self.drawing is not None:
            if key in KEY_ENTER:
                self.finish_drawing()
            elif key == KEY_ESC:
                self.drawing = None
                self.flash("Cancelled")
            elif key == KEY_BACKSPACE and self.drawing:
                self.drawing.pop()
            return True

        if ch == "q" or key == KEY_ESC:
            if self.dirty and not self.quit_armed:
                self.quit_armed = True
                self.flash("Unsaved changes! Press Q again to quit without saving, or S to save", 5)
                return True
            return False
        self.quit_armed = False

        if ch == " ":
            self.playing = not self.playing
            self.last_tick = time.time()
        elif ch in ",.":
            self.playing = False
            self.set_frame(self.video.index + (1 if ch == "." else -1))
        elif key == KEY_TAB:
            ks = self.keys()
            if ks:
                i = ks.index(self.selected) + 1 if self.selected in ks else 0
                self.selected = ks[i % len(ks)]
        elif ch.isdigit() and ch != "0":
            if f"T{ch}" in self.cfg.tables:
                self.selected = ("table", f"T{ch}")
        elif key in KEY_LEFT | KEY_RIGHT | KEY_UP | KEY_DOWN:
            dx = -1 if key in KEY_LEFT else 1 if key in KEY_RIGHT else 0
            dy = -1 if key in KEY_UP else 1 if key in KEY_DOWN else 0
            self.nudge(dx, dy)
        elif ch in "[]":
            self.resize(0.95 if ch == "[" else 1.05)
        elif ch in "nb":
            self.playing = False
            self.drawing, self.drawing_kind = [], "table" if ch == "n" else "staff"
            what = "table zone (cover the table and its chairs)" if ch == "n" else \
                "staff-only area (bar, kitchen door, POS)"
            self.flash(f"New {what}: click corners, Enter to finish", 4)
        elif ch == "x" or key in KEY_DELETE:
            if self.selected in self.keys():
                self.push_undo()
                del self.group(self.selected[0])[self.selected[1]]
                self.flash(f"Deleted {self.selected[1]}")
                self.selected = None
        elif ch == "u" or key == KEY_CTRL_Z:
            self.undo()
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "s":
            self.save()
        elif ch == "t":
            self.run_test()
        return True

    def nudge(self, dx, dy):
        if self.selected not in self.keys():
            return
        z = self.zone(self.selected)
        pts = np.array(z.polygon)
        bx, by = self.bounds
        dx = int(np.clip(dx, -pts[:, 0].min(), bx - pts[:, 0].max()))
        dy = int(np.clip(dy, -pts[:, 1].min(), by - pts[:, 1].max()))
        if dx or dy:
            self.push_undo()
            z.polygon = [[int(x + dx), int(y + dy)] for x, y in pts]

    def resize(self, factor):
        if self.selected not in self.keys():
            return
        z = self.zone(self.selected)
        pts = np.array(z.polygon, dtype=float)
        c = pts.mean(axis=0)
        self.push_undo()
        z.polygon = [self.clamp(*(c + (p - c) * factor)) for p in pts]

    def save(self):
        try:
            path = save_config(self.cfg)
        except Exception as exc:
            return self.flash(f"Save failed: {exc}", 5)
        self.dirty = False
        self.flash(f"Saved to {path}")

    # ---------- video / test ----------

    def set_frame(self, idx):
        self.video.seek(idx)
        cv2.setTrackbarPos("Frame", WINDOW, self.video.index)

    def on_trackbar(self, pos):
        if pos != self.video.index:
            self.video.seek(pos)

    def tick(self):
        if not self.playing:
            return
        steps = int((time.time() - self.last_tick) * self.video.fps)
        if steps <= 0:
            return
        self.last_tick += steps / self.video.fps
        for _ in range(min(steps, 5)):
            if not self.video.next():
                self.playing = False
                break
        cv2.setTrackbarPos("Frame", WINDOW, self.video.index)

    def run_test(self):
        if not self.cfg.tables:
            return self.flash("Add at least one table zone before testing")
        from .perception import PersonDetector
        from .pipeline import run

        self.playing = False
        start = self.video.index
        print(f"\n[Editor] Testing from frame {start}; press Q in the monitor window to come back.")
        try:
            cv2.destroyWindow(WINDOW)
            if self._detector is None:
                self._detector = PersonDetector(self.cfg.detector)
            run(self.cfg.copy(), self.video.source, "output/zone_test", show=True,
                start_frame=start, telegram=False, detector=self._detector)
            self.flash("Test finished - summary in the terminal, video in output/zone_test/", 5)
        except Exception as exc:
            self.flash(f"Test failed: {exc}", 6)
        finally:
            self.open_window()
            self.set_frame(start)

    # ---------- drawing ----------

    def open_window(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        cv2.createTrackbar("Frame", WINDOW, max(self.video.index, 0), self.video.total - 1,
                           self.on_trackbar)

    def render(self):
        s = self.scale
        frame = cv2.resize(self.video.frame, None, fx=s, fy=s) if s != 1 else self.video.frame.copy()

        def P(pts):
            return (np.array(pts, dtype=float) * s).round().astype(np.int32)

        idle = self.drawing is None and not self.drag
        hv = self.hit_vertex(*self.mouse) if idle else None
        he = self.hit_edge(*self.mouse) if idle and not hv else None
        hz = self.hit_zone(*self.mouse) if idle else None

        overlay = frame.copy()
        for key in self.keys():
            z = self.zone(key)
            if len(z.polygon) >= 3:
                cv2.fillPoly(overlay, [P(z.polygon)], tuple(z.color))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        for key in self.keys():
            z = self.zone(key)
            pts, color = P(z.polygon), tuple(z.color)
            sel = key == self.selected
            if sel:
                cv2.polylines(frame, [pts], True, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, color, 3 if sel or key == hz else 2, cv2.LINE_AA)
            cx, cy = pts.mean(axis=0).astype(int)
            label = key[1] if key[0] == "table" else f"{key[1]} (staff)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (cx - lw // 2 - 4, cy - lh - 4), (cx + lw // 2 + 4, cy + 6), (0, 0, 0), -1)
            cv2.putText(frame, label, (cx - lw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            for i, p in enumerate(pts):
                if sel:
                    big = hv == (key, i) or (self.drag and self.drag.get("key") == key
                                             and self.drag.get("i") == i)
                    r = 8 if big else 5
                    cv2.circle(frame, tuple(p), r, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, tuple(p), r, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, tuple(p), 3, color, -1, cv2.LINE_AA)

        if hv and hv[0] != self.selected:
            p = P([self.zone(hv[0]).polygon[hv[1]]])[0]
            cv2.circle(frame, tuple(p), 8, (255, 255, 255), 2, cv2.LINE_AA)
        if he:
            c = P([he[2]])[0]
            cv2.circle(frame, tuple(c), 5, (255, 255, 255), 1, cv2.LINE_AA)

        if self.drawing is not None:
            color = (255, 255, 255) if self.drawing_kind == "table" else tuple(STAFF_COLOR)
            if self.drawing:
                cv2.polylines(frame, [P(self.drawing + [list(self.mouse)])], False, color, 2, cv2.LINE_AA)
                for p in P(self.drawing):
                    cv2.circle(frame, tuple(p), 5, color, -1, cv2.LINE_AA)
            if len(self.drawing) >= 3:
                cv2.circle(frame, tuple(P(self.drawing[:1])[0]), 10, (0, 255, 0), 2, cv2.LINE_AA)

        self._hud(frame)
        return frame

    def _hud(self, frame):
        H, W = frame.shape[:2]

        def panel(lines, y, scale=0.5):
            lh = int(22 * scale / 0.5)
            w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t in lines) + 16
            h = lh * len(lines) + 10
            y = y if y >= 0 else H + y - h
            sub = frame[y:y + h, 8:8 + w]
            cv2.addWeighted(sub, 0.3, np.zeros_like(sub), 0.7, 0, sub)
            for i, t in enumerate(lines):
                cv2.putText(frame, t, (16, y + lh * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (235, 235, 235), 1, cv2.LINE_AA)

        if self.drawing is not None:
            status = f"DRAWING new {self.drawing_kind} zone - {len(self.drawing)} corners (Enter: finish, Esc: cancel)"
        elif self.selected in self.keys():
            z = self.zone(self.selected)
            status = f"Selected: {self.selected[1]} - {z.name} ({len(z.polygon)} corners)"
        else:
            status = "Click a zone to select it"
        t = self.video.index / self.video.fps
        panel([status, f"Frame {self.video.index}/{self.video.total - 1}  t={t:5.1f}s  "
                       f"{'PLAYING' if self.playing else 'paused'}"
                       f"{'   * unsaved' if self.dirty else ''}"], 8)
        if self.show_help:
            panel([
                "Drag corner: move | Drag inside: move zone | Click edge: add corner | Right-click corner: delete",
                "N: new table | B: new staff area | Del/X: delete | U/Ctrl+Z: undo | Arrows: nudge | [ ]: size",
                "Space: play | , .: step | T: TEST from this frame | S: save | H: hide help | Q: quit",
            ], -8, 0.46)
        if time.time() < self.message_until:
            (tw, th), _ = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x, y = max((W - tw) // 2, 4), H // 2 - 40
            cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
            cv2.putText(frame, self.message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 255), 2, cv2.LINE_AA)

    def run(self):
        self.open_window()
        while True:
            self.tick()
            cv2.imshow(WINDOW, self.render())
            if not self.on_key(cv2.waitKeyEx(15)):
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    if self.dirty:
                        print("[Editor] Window closed; unsaved changes discarded.")
                    break
            except cv2.error:
                break
        cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw table zones and staff areas")
    ap.add_argument("--config", "-c", default="configs/sample.yaml")
    ap.add_argument("--source", "-s", help="Video to draw on (default: config source)")
    ap.add_argument("--scale", type=float, help="Display zoom (default: fit ~1600x900)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    video = VideoFrames(args.source or cfg.source)
    w, h = video.size
    scale = args.scale or max(0.5, min(2.0, 1600 / w, 900 / h))
    print(f"[Editor] {video.source} ({w}x{h}), zoom {scale:.2f}. Press H for help.")
    ZoneEditor(cfg, video, scale).run()


if __name__ == "__main__":
    main()
