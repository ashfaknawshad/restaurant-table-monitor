"""Person detection (YOLO) and multi-object tracking (ByteTrack)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from .config import DetectorConfig, TrackerConfig


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray        # [x1, y1, x2, y2]
    confidence: float


class PersonDetector:
    def __init__(self, cfg: DetectorConfig):
        from ultralytics import YOLO  # imported lazily: slow to load

        self.cfg = cfg
        self.model = YOLO(cfg.model)

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        dets = self._predict(frame, self.cfg.imgsz)
        if self.cfg.tiles:
            h, w = frame.shape[:2]
            parts = [dets]
            for fx0, fy0, fx1, fy1 in self.cfg.tiles:
                x0, y0, x1, y1 = int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)
                tile = self._predict(frame[y0:y1, x0:x1], self.cfg.tile_imgsz)
                if len(tile):
                    tile.xyxy = tile.xyxy + np.array([x0, y0, x0, y0], dtype=np.float32)
                    parts.append(tile)
            dets = merge_detections(sv.Detections.merge(parts), self.cfg.iou)
        return dets

    def _predict(self, image: np.ndarray, imgsz: int) -> sv.Detections:
        result = self.model.predict(
            image, imgsz=imgsz, conf=self.cfg.confidence, iou=self.cfg.iou,
            classes=[0], device=self.cfg.device, verbose=False,
        )[0]
        dets = sv.Detections.from_ultralytics(result)
        return dets[dets.class_id == 0] if len(dets) else dets


def merge_detections(dets: sv.Detections, iou: float, containment: float = 0.8) -> sv.Detections:
    """NMS across overlapping sources, then drop boxes mostly inside a stronger box
    (tile borders often cut a person into a head-only fragment)."""
    if len(dets) < 2:
        return dets
    dets = dets.with_nms(threshold=iou, class_agnostic=True)
    keep: list[int] = []
    for i in np.argsort(-dets.confidence):
        b = dets.xyxy[i]
        area = max(float((b[2] - b[0]) * (b[3] - b[1])), 1e-6)
        inside = False
        for j in keep:
            k = dets.xyxy[j]
            iw = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            ih = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            if iw * ih / area > containment:
                inside = True
                break
        if not inside:
            keep.append(int(i))
    return dets[np.array(sorted(keep))]


class PersonTracker:
    def __init__(self, cfg: TrackerConfig, process_fps: float):
        fps = max(1, round(process_fps))
        # ByteTrack measures its lost-track buffer in 1/30 s units regardless of frame rate.
        self.bytetrack = sv.ByteTrack(
            track_activation_threshold=cfg.activation,
            lost_track_buffer=int(cfg.lost_seconds * 30),
            minimum_matching_threshold=cfg.match,
            frame_rate=fps,
        )

    def __call__(self, dets: sv.Detections) -> list[Track]:
        tracked = self.bytetrack.update_with_detections(dets)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return []
        conf = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
        return [
            Track(int(tid), tracked.xyxy[i].astype(np.float32), float(conf[i]))
            for i, tid in enumerate(tracked.tracker_id)
        ]
