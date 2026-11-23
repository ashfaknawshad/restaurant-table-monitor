"""Per-camera configuration, loaded from and saved to YAML."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    model: str = "yolo11m.pt"
    imgsz: int = 1280
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    # Optional crops (x0, y0, x1, y1 as fractions of the frame) re-checked at higher zoom.
    tiles: list[list[float]] = field(default_factory=list)
    tile_imgsz: int = 960


@dataclass
class TrackerConfig:
    lost_seconds: float = 3.0      # keep a hidden person's ID this long
    activation: float = 0.3        # confidence needed to start a new track
    match: float = 0.8


@dataclass
class RulesConfig:
    anchor_y: float = 0.7              # point tested against zones, as a fraction of box height
    seat_seconds: float = 20.0         # time in a table zone, nearly still, before counting as seated
    seated_max_speed: float = 0.25     # box-heights per second; seated people barely move
    leave_seconds: float = 8.0         # out of the seat this long -> no longer seated
    empty_seconds: float = 60.0        # no seated customer this long -> table empty
    service_margin_px: float = 30.0    # service area = table zone grown by this much
    service_seconds: float = 3.0       # staff must stay in the service area this long
    unattended_minutes: float = 5.0    # alert when an occupied table waits this long
    reminder_minutes: float = 5.0      # repeat the alert this often while still waiting


@dataclass
class StaffConfig:
    track_ids: list[int] = field(default_factory=list)  # manual overrides (debugging)
    # Uniform colour rule: HSV ranges matched against the torso. Off until calibrated
    # for the brand, e.g. black shirts: [[0, 0, 0], [180, 255, 60]].
    uniform_hsv_ranges: list[list[list[int]]] = field(default_factory=list)
    uniform_min_fraction: float = 0.45
    uniform_min_samples: int = 5
    staff_area_seconds: float = 4.0    # time inside a staff-only area to be staff
    tables_visited: int = 3            # distinct tables approached without sitting
    visit_window_seconds: float = 300.0


@dataclass
class AlertConfig:
    branch: str = "Branch"
    camera: str = "Camera"
    telegram: bool = False
    telegram_token_env: str = "TELEGRAM_BOT_TOKEN"
    telegram_chat_env: str = "TELEGRAM_CHAT_ID"
    snapshot_dir: str = "output/alerts"


@dataclass
class Zone:
    name: str
    polygon: list[list[int]]
    color: list[int] = field(default_factory=lambda: [0, 200, 255])


@dataclass
class Config:
    source: str = ""
    process_fps: float = 5.0
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    staff: StaffConfig = field(default_factory=StaffConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    tables: dict[str, Zone] = field(default_factory=dict)
    staff_areas: dict[str, Zone] = field(default_factory=dict)
    path: Path | None = None

    def copy(self) -> "Config":
        return deepcopy(self)


def _section(cls, data: dict | None):
    data = data or {}
    known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**known)


def _zones(data: dict | None) -> dict[str, Zone]:
    return {zid: _section(Zone, z) for zid, z in (data or {}).items()}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(
        source=raw.get("source", ""),
        process_fps=float(raw.get("process_fps", 5.0)),
        detector=_section(DetectorConfig, raw.get("detector")),
        tracker=_section(TrackerConfig, raw.get("tracker")),
        rules=_section(RulesConfig, raw.get("rules")),
        staff=_section(StaffConfig, raw.get("staff")),
        alerts=_section(AlertConfig, raw.get("alerts")),
        tables=_zones(raw.get("tables")),
        staff_areas=_zones(raw.get("staff_areas")),
        path=path,
    )
    return cfg


class _Dumper(yaml.SafeDumper):
    """Block style for mappings; lists of plain numbers (points, colours) inline."""


def _represent_list(dumper, data):
    flow = all(isinstance(v, (int, float)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def save_config(cfg: Config, path: str | Path | None = None) -> Path:
    path = Path(path or cfg.path)
    data = {
        "source": cfg.source,
        "process_fps": cfg.process_fps,
        "detector": asdict(cfg.detector),
        "tracker": asdict(cfg.tracker),
        "rules": asdict(cfg.rules),
        "staff": asdict(cfg.staff),
        "alerts": asdict(cfg.alerts),
        "tables": {k: asdict(v) for k, v in cfg.tables.items()},
        "staff_areas": {k: asdict(v) for k, v in cfg.staff_areas.items()},
    }
    text = yaml.dump(data, Dumper=_Dumper, sort_keys=False, width=100, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path
