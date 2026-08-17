"""Periodic save and restore of the simulation state -- DESIGN.md §4.6.

A three-hour-old field is materially different from a fresh one: the agent
network has found its junctions, the reaction has organised into structure, and
the climate has drifted somewhere particular. None of that can be recovered by
running for a few seconds, so a crash, a reboot, or simply closing the window
would otherwise throw away exactly the thing the user came back for.

What is saved is the state that *accumulates*: the field textures at their
current ping-pong parity, the agent array, the climate and psi (OU) state, the
homeostat integrators, and the counters. What is left out is left out on purpose,
because at 1440p the whole world is ~230 MB and this is written every five
minutes for days on end:

* **Derived fields.** ``velocity`` is rewritten by the flow pass and
  ``reaction_prev`` by a texture copy, both unconditionally at the start of every
  tick and before anything reads them, so neither carries state between ticks.
  Skipping the two of them is a third of the payload. That they really are
  derived is asserted rather than assumed -- ``test_checkpoint.py`` compares them
  across a resume along with everything else, so a change that made either
  stateful would fail rather than silently corrupt a restore.
* **The deposit accumulator.** The trail pass drains it with ``atomicExchange``
  every tick, so between ticks it is already zero.
* **The output history** the safety stage uses as its slew-limiter reference.
  Left empty, the limiter brings the image up from black over a couple of
  seconds -- which is the fade-in wanted at startup anyway, and restoring it is
  the one way resuming could produce a hard cut.
* **The event scheduler's RNG stream.** Arrivals are exponential and therefore
  memoryless, so a fresh stream is statistically indistinguishable from the
  saved one. The *in-flight* events are restored, because those are mid-envelope
  and dropping them would be a step.

Geometry is a compatibility key rather than something to adapt to: resolution,
layer count, agent counts and climate size all follow from the window size and
the config, so a checkpoint whose shapes do not match the engine being restored
into is refused with a log line and the run starts from seeds. Every failure
here degrades to "start fresh" -- a mismatched, truncated or foreign file must
never be able to stop the application from opening.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import gpu_params

log = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Five minutes: long enough that the readback cost is negligible, short enough
# that a crash costs less field maturity than it takes to notice one.
DEFAULT_INTERVAL_SECONDS = 300.0

# Ping-pong fields, saved at their current parity and restored into both slots
# so the next flip cannot read stale data. Every field carrying state between
# ticks is in this list; see the module docstring for what is not and why.
PAIR_FIELDS = ("trail", "reaction", "pigment", "climate_a", "climate_b", "psi")

# Rewritten from scratch every tick before anything reads them, so they are not
# saved. Named here so the tests can check that this stays true.
DERIVED_FIELDS = ("reaction_prev", "velocity")


# --------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """One captured simulation state: JSON-able metadata plus raw arrays."""

    meta: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def tick_count(self) -> int:
        return int((self.meta.get("engine") or {}).get("tick_count", 0))

    @property
    def sim_seconds(self) -> float:
        sim_hz = float(self.meta.get("sim_hz") or 0.0)
        return self.tick_count / sim_hz if sim_hz > 0.0 else 0.0

    def describe(self) -> str:
        seconds = self.sim_seconds
        hours, rest = divmod(int(seconds), 3600)
        minutes = rest // 60
        age = max(time.time() - float(self.meta.get("created") or 0.0), 0.0)
        return (
            f"{hours}h {minutes:02d}m of simulation ({self.tick_count:,} ticks), "
            f"saved {describe_age(age)}"
        )


def describe_age(seconds: float) -> str:
    """Coarse "how long ago", at the resolution a human actually wants."""
    if seconds < 90.0:
        return f"{int(seconds)}s ago"
    if seconds < 5400.0:
        return f"{int(seconds / 60)}m ago"
    return f"{seconds / 3600:.1f}h ago"


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


def _read_texture(device, texture) -> np.ndarray:
    """Read one rgba16float texture back as ``(h, w, 4)`` float16.

    ``read_texture`` handles the 256-byte row alignment itself, so no padding
    arithmetic is needed here even for the small climate grids.
    """
    width, height = texture.width, texture.height
    raw = device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, 1),
    )
    return (
        np.frombuffer(raw, dtype=np.float16)[: width * height * 4]
        .reshape(height, width, 4)
        .copy()
    )


def _read_buffer(device, buffer) -> np.ndarray:
    """Read a storage buffer back as opaque bytes.

    Kept opaque on purpose: the agent record layout lives in ``engine.py`` and
    the statistics layout in ``gpu_params.py``, and duplicating either here
    would be a second place to get it wrong.
    """
    return np.frombuffer(device.queue.read_buffer(buffer), dtype=np.uint8).copy()


def capture(engine, scheduler=None, sim_hz: float = 0.0) -> Checkpoint:
    """Read the whole simulation state back off the GPU.

    Call this between ticks: the deposit accumulator is only guaranteed empty
    once the trail pass has consumed it.
    """
    device = engine.device
    arrays: dict[str, np.ndarray] = {}
    layers_meta: list[dict[str, Any]] = []

    for layer in engine.layers:
        index = layer.spec.index
        for name in PAIR_FIELDS:
            pair = getattr(layer, name)
            arrays[f"layer{index}.{name}"] = _read_texture(
                device, pair.textures[pair.index]
            )
        arrays[f"layer{index}.agents"] = _read_buffer(device, layer.agents_buf)

        climate = layer.climate_a.textures[0]
        layers_meta.append({
            "index": index,
            "width": layer.spec.width,
            "height": layer.spec.height,
            "agent_count": layer.spec.agent_count,
            "psi_width": layer.psi_dims[0],
            "psi_height": layer.psi_dims[1],
            "climate_width": climate.width,
            "climate_height": climate.height,
        })

    arrays["stats"] = _read_buffer(device, engine.stats_buf)

    meta: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "created": time.time(),
        "sim_hz": float(sim_hz),
        "engine": {
            "width": engine.width,
            "height": engine.height,
            "tick_count": int(engine.tick_count),
            "frame_count": int(engine.frame_count),
            "seed": int(engine.seed),
            "hue_phase": float(engine.hue_phase),
            "parallax": [
                [float(x), float(y)] for x, y in (engine._parallax or [])
            ],
        },
        "layers": layers_meta,
        "events": scheduler.state() if scheduler is not None else {},
    }
    return Checkpoint(meta=meta, arrays=arrays)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save(path: str | Path, checkpoint: Checkpoint) -> None:
    """Write a checkpoint, replacing any previous one atomically.

    Uncompressed: the payload is tens of megabytes of half-float field data,
    which barely compresses, and the point of the temp-file-plus-rename is that
    a crash mid-write leaves the *previous* checkpoint intact rather than a
    half-written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    payload = dict(checkpoint.arrays)
    payload["meta"] = np.array(json.dumps(checkpoint.meta))

    started = time.perf_counter()
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    log.info(
        "checkpointed %.1f MB to %s in %.2fs",
        path.stat().st_size / 1e6, path, time.perf_counter() - started,
    )


def load(path: str | Path) -> Checkpoint | None:
    """Read a checkpoint, or return ``None`` if there is nothing usable.

    Never raises: a truncated, foreign or half-deleted file must degrade to
    "start from a fresh field", not to a traceback at startup. ``allow_pickle``
    stays off, so nothing on disk can execute anything on load.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            if "meta" not in keys:
                log.warning("checkpoint %s has no metadata; ignoring it", path)
                return None
            meta = json.loads(str(data["meta"]))
            arrays = {key: data[key] for key in keys if key != "meta"}
    except Exception as exc:
        log.warning("could not read the checkpoint at %s (%s); ignoring it", path, exc)
        return None
    if not isinstance(meta, dict):
        log.warning("checkpoint %s has malformed metadata; ignoring it", path)
        return None
    return Checkpoint(meta=meta, arrays=arrays)


def discard(path: str | Path) -> None:
    """Delete the checkpoint and any interrupted write beside it."""
    path = Path(path)
    for candidate in (path, path.with_name(path.name + ".tmp")):
        try:
            candidate.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove %s: %s", candidate, exc)


def default_checkpoint_path() -> Path:
    """``$XDG_STATE_HOME/anastomosis/checkpoint.npz``.

    State rather than config or cache: it is neither hand-editable nor cheap to
    regenerate.
    """
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "anastomosis" / "checkpoint.npz"


class BackgroundSaver:
    """Serialises checkpoints on a worker thread.

    The GPU readback has to happen on the thread that owns the device, but
    writing tens of megabytes to disk does not, and doing that inline would hold
    a frame for long enough to notice. One writer at a time: if a previous write
    is somehow still running the save is skipped rather than queued, since the
    next one is only minutes away.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, path: str | Path, checkpoint: Checkpoint) -> bool:
        if self.busy:
            log.warning("previous checkpoint write still running; skipping this one")
            return False
        self._thread = threading.Thread(
            target=self._write,
            args=(Path(path), checkpoint),
            name="anastomosis-checkpoint",
            daemon=True,
        )
        self._thread.start()
        return True

    @staticmethod
    def _write(path: Path, checkpoint: Checkpoint) -> None:
        try:
            save(path, checkpoint)
        except Exception as exc:  # pragma: no cover - disk full, permissions
            log.error("could not write the checkpoint to %s: %s", path, exc)

    def join(self, timeout: float = 30.0) -> None:
        """Wait for any in-flight write. Called before exit and before a reset."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():  # pragma: no cover - pathologically slow disk
                log.warning("checkpoint write did not finish within %.0fs", timeout)
            self._thread = None


# --------------------------------------------------------------------------
# Restore
# --------------------------------------------------------------------------


def _expected_arrays(engine) -> dict[str, tuple[int, ...]]:
    """Array name -> required shape, derived from the live GPU resources."""
    expected: dict[str, tuple[int, ...]] = {
        "stats": (gpu_params.STATS_DTYPE.itemsize,),
    }
    for layer in engine.layers:
        index = layer.spec.index
        for name in PAIR_FIELDS:
            texture = getattr(layer, name).textures[0]
            expected[f"layer{index}.{name}"] = (texture.height, texture.width, 4)
        expected[f"layer{index}.agents"] = (layer.agents_buf.size,)
    return expected


def compatibility_problems(engine, checkpoint: Checkpoint) -> list[str]:
    """Human-readable reasons this checkpoint cannot be restored, if any.

    The metadata comparison exists for the log message; the shape comparison
    against the live resources is what actually makes the upload safe.
    """
    version = checkpoint.meta.get("version")
    if version != FORMAT_VERSION:
        return [f"format version {version!r}, expected {FORMAT_VERSION}"]

    problems: list[str] = []
    saved_engine = checkpoint.meta.get("engine") or {}
    saved_size = (saved_engine.get("width"), saved_engine.get("height"))
    if saved_size != (engine.width, engine.height):
        problems.append(
            f"saved at {saved_size[0]}x{saved_size[1]}, "
            f"now {engine.width}x{engine.height}"
        )

    saved_layers = checkpoint.meta.get("layers") or []
    if len(saved_layers) != len(engine.layers):
        problems.append(
            f"{len(saved_layers)} layers saved, {len(engine.layers)} now"
        )
    else:
        for layer, row in zip(engine.layers, saved_layers):
            if not isinstance(row, dict):
                problems.append("malformed layer metadata")
                break
            climate = layer.climate_a.textures[0]
            for label, saved, live in (
                ("size", (row.get("width"), row.get("height")),
                 (layer.spec.width, layer.spec.height)),
                ("agent count", row.get("agent_count"), layer.spec.agent_count),
                ("psi size", (row.get("psi_width"), row.get("psi_height")),
                 layer.psi_dims),
                ("climate size",
                 (row.get("climate_width"), row.get("climate_height")),
                 (climate.width, climate.height)),
            ):
                if saved != live:
                    problems.append(
                        f"layer {layer.spec.index} {label} {saved} != {live}"
                    )

    for name, shape in _expected_arrays(engine).items():
        array = checkpoint.arrays.get(name)
        if array is None:
            problems.append(f"missing {name}")
        elif tuple(array.shape) != shape:
            problems.append(f"{name} is {tuple(array.shape)}, expected {shape}")

    return problems


def _write_texture(device, texture, data: np.ndarray) -> None:
    payload = np.ascontiguousarray(data, dtype=np.float16)
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        payload,
        {"offset": 0, "bytes_per_row": payload.shape[1] * 8,
         "rows_per_image": payload.shape[0]},
        (payload.shape[1], payload.shape[0], 1),
    )


def restore(engine, checkpoint: Checkpoint, scheduler=None) -> bool:
    """Load a checkpoint into a freshly built engine.

    Returns False and changes nothing if the checkpoint does not match: every
    shape is validated before the first upload, so a refusal leaves the seeded
    field intact and the run simply starts new.
    """
    problems = compatibility_problems(engine, checkpoint)
    if problems:
        log.info(
            "saved state does not fit this session, starting from a fresh "
            "field (%s)", "; ".join(problems),
        )
        return False

    device = engine.device
    for layer in engine.layers:
        index = layer.spec.index
        for name in PAIR_FIELDS:
            pair = getattr(layer, name)
            data = checkpoint.arrays[f"layer{index}.{name}"]
            # Both slots, so the parity the next tick happens to start on cannot
            # resurrect the seeded state.
            for texture in pair.textures:
                _write_texture(device, texture, data)
            pair.index = 0
        device.queue.write_buffer(
            layer.agents_buf, 0, checkpoint.arrays[f"layer{index}.agents"].tobytes()
        )

    device.queue.write_buffer(engine.stats_buf, 0, checkpoint.arrays["stats"].tobytes())

    saved = checkpoint.meta.get("engine") or {}
    engine.tick_count = int(saved.get("tick_count", 0))
    engine.frame_count = int(saved.get("frame_count", 0))
    engine.seed = int(saved.get("seed", engine.seed)) & 0xFFFFFFFF
    hue_phase = float(saved.get("hue_phase", 0.0))
    engine.hue_phase = hue_phase % (2.0 * math.pi) if math.isfinite(hue_phase) else 0.0
    parallax = saved.get("parallax") or []
    if len(parallax) == len(engine.layers):
        engine._parallax = [[float(x), float(y)] for x, y in parallax]

    if scheduler is not None:
        scheduler.load_state(checkpoint.meta.get("events") or {})

    log.info("resumed from %s", checkpoint.describe())
    return True
