"""Periodic save and restore of the simulation state -- DESIGN.md §4.6.

A three-hour-old field is materially different from a fresh one: the agent
network has found its junctions, the reaction has organised into structure, and
the climate has drifted somewhere particular. None of that can be recovered by
running for a few seconds, so a crash, a reboot, or simply closing the window
would otherwise throw away exactly the thing the user came back for.

What is saved is the state that *accumulates*: the field textures at their
current ping-pong parity, the agent array, the climate and psi (OU) state, the
homeostat integrators, the global feature-size walk, and the counters. The rule
for what belongs here is not "is it large" but "does the next tick read
something the previous tick left behind" -- anything answering yes to that is
state a resume has to carry, wherever it happens to live. What is left out is
left out on purpose, because at 1440p the whole world is ~230 MB and this is
written every five minutes for days on end:

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
  and dropping them would be a step. The feature-size walk's noise stream *is*
  saved, for the opposite reason to a payload argument: it is a hundred bytes,
  and carrying it is what lets a resume be checked for being identical rather
  than merely similar.

Geometry is saved rather than required. Resolution, layer count, agent counts
and climate size all follow from the window size and the config, so treating
them as a compatibility key meant that resizing a window -- or editing any
config value that touched them -- silently threw away a field that had taken
hours to grow. What the file records instead is the geometry it was captured
at, and :func:`required_geometry` reads it back so the launch can *build* an
engine in that shape (``app.Application._start_engine``) before loading the
field into it. The window is presentation and follows itself; the config's
structural values take effect the next time a field is grown from seeds.

What is still refused is a file this build cannot use at all: a foreign format
version, missing or wrongly-shaped arrays, or a geometry no engine could be
built at. Every failure here degrades to "start fresh" -- a mismatched,
truncated or foreign file must never be able to stop the application opening.
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

from . import engine as engine_module
from . import gpu_params

log = logging.getLogger(__name__)

FORMAT_VERSION = 3
# Version 1 recorded the *window* size where version 2 records the simulation's
# own, which are the same number unless that session was resized after starting.
# Version 3 adds the morphology climate pair and the feature-size walk, both of
# which postdate version 2 and neither of which an older file can carry.
# Reading them costs a few lines and saves anyone upgrading their mature field.
OLDEST_READABLE_VERSION = 1

# Five minutes: long enough that the readback cost is negligible, short enough
# that a crash costs less field maturity than it takes to notice one.
DEFAULT_INTERVAL_SECONDS = 300.0

# Ping-pong fields, saved at their current parity and restored into both slots
# so the next flip cannot read stale data. Every field carrying state between
# ticks is in this list; see the module docstring for what is not and why.
PAIR_FIELDS = (
    "trail", "reaction", "pigment",
    "climate_a", "climate_b", "climate_c", "psi",
)

# Fields younger than the format itself: the version each first appeared in. A
# file written before that is not broken, it simply has nothing to say about
# them, so the engine keeps the field it seeded and loses that one channel's
# history rather than the whole field.
FIELDS_SINCE = {"climate_c": 3}

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

    for layer in engine.layers:
        index = layer.spec.index
        for name in PAIR_FIELDS:
            pair = getattr(layer, name)
            arrays[f"layer{index}.{name}"] = _read_texture(
                device, pair.textures[pair.index]
            )
        arrays[f"layer{index}.agents"] = _read_buffer(device, layer.agents_buf)

    arrays["stats"] = _read_buffer(device, engine.stats_buf)

    meta: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "created": time.time(),
        "sim_hz": float(sim_hz),
        # The shape the next launch has to build itself in to be able to load
        # this, rather than the shape it has to happen to already be in.
        "geometry": _geometry_meta(engine.geometry),
        "engine": {
            "tick_count": int(engine.tick_count),
            "frame_count": int(engine.frame_count),
            "seed": int(engine.seed),
            "hue_phase": float(engine.hue_phase),
            # The global feature-size walk: its value, and the position of the
            # stream that drives it (see `_walk_rng_state`).
            "du_walk": float(engine._du_walk),
            "walk_rng": _walk_rng_state(engine),
            "parallax": [
                [float(x), float(y)] for x, y in (engine._parallax or [])
            ],
        },
        "events": scheduler.state() if scheduler is not None else {},
    }
    return Checkpoint(meta=meta, arrays=arrays)


def _walk_rng_state(engine) -> dict[str, Any]:
    """The feature-size walk's noise stream, as plain JSON-able data.

    A hundred bytes, unlike every other omission in this module, so the argument
    that settled the event scheduler's stream -- statistically identical is good
    enough -- buys nothing here. Saving it instead makes a resume *identical*
    rather than merely similar, which is the property the suite can actually
    check, and checking it is what catches the next piece of state someone adds
    to the engine and forgets to add here.
    """
    try:
        state = engine._walk_rng.bit_generator.state
        return json.loads(json.dumps(state))
    except (AttributeError, TypeError, ValueError):  # pragma: no cover
        return {}


def _geometry_meta(geometry) -> dict[str, Any]:
    """A geometry as plain JSON-able data."""
    return {
        "sim_width": int(geometry.sim_width),
        "sim_height": int(geometry.sim_height),
        "layers": [
            {
                "index": int(layer.index),
                "width": int(layer.width),
                "height": int(layer.height),
                "agent_count": int(layer.agent_count),
                "psi_width": int(layer.psi_width),
                "psi_height": int(layer.psi_height),
                "climate_width": int(layer.climate_width),
                "climate_height": int(layer.climate_height),
            }
            for layer in geometry.layers
        ],
    }


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


def _field_shape(layer, name: str) -> tuple[int, int, int]:
    """The shape one saved field has, given the layer geometry it belongs to."""
    if name.startswith("climate"):
        width, height = layer.climate_width, layer.climate_height
    elif name == "psi":
        width, height = layer.psi_width, layer.psi_height
    else:
        width, height = layer.width, layer.height
    return (height, width, 4)


def _expected_arrays(
    geometry, version: int = FORMAT_VERSION
) -> dict[str, tuple[int, ...]]:
    """Array name -> required shape, for a given simulation geometry.

    Version-aware, because a field that did not exist when the file was written
    cannot be in it and must not be demanded of it.
    """
    expected: dict[str, tuple[int, ...]] = {
        "stats": (gpu_params.STATS_DTYPE.itemsize,),
    }
    for layer in geometry.layers:
        for name in PAIR_FIELDS:
            if version < FIELDS_SINCE.get(name, 0):
                continue
            expected[f"layer{layer.index}.{name}"] = _field_shape(layer, name)
        expected[f"layer{layer.index}.agents"] = (
            max(layer.agent_count, 1) * engine_module.AGENT_STRIDE,
        )
    return expected


def _read_geometry(meta: dict[str, Any]) -> engine_module.Geometry | None:
    """The geometry described by a checkpoint's metadata, or ``None``.

    Nothing in here is trusted: the values are coerced to ``int`` and the result
    is bounds-checked by the caller before an engine is built from it.
    """
    block = meta.get("geometry")
    if not isinstance(block, dict):
        # Format version 1, which kept the sizes in two other places and
        # recorded the window size rather than the simulation's.
        saved = meta.get("engine") or {}
        block = {
            "sim_width": saved.get("width"),
            "sim_height": saved.get("height"),
            "layers": meta.get("layers"),
        }

    rows = block.get("layers")
    if not isinstance(rows, list) or not rows:
        return None
    try:
        layers = tuple(
            engine_module.LayerGeometry(
                index=int(row["index"]),
                width=int(row["width"]),
                height=int(row["height"]),
                agent_count=int(row["agent_count"]),
                psi_width=int(row["psi_width"]),
                psi_height=int(row["psi_height"]),
                climate_width=int(row["climate_width"]),
                climate_height=int(row["climate_height"]),
            )
            for row in rows
        )
        return engine_module.Geometry(
            sim_width=int(block["sim_width"]),
            sim_height=int(block["sim_height"]),
            layers=layers,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _usable_geometry(
    checkpoint: Checkpoint,
) -> tuple[engine_module.Geometry | None, list[str]]:
    """``(geometry, problems)``: what this file needs, or why it is unusable."""
    version = checkpoint.meta.get("version")
    if (
        not isinstance(version, int)
        or not OLDEST_READABLE_VERSION <= version <= FORMAT_VERSION
    ):
        return None, [
            f"format version {version!r}, expected "
            f"{OLDEST_READABLE_VERSION} to {FORMAT_VERSION}"
        ]

    geometry = _read_geometry(checkpoint.meta)
    if geometry is None:
        return None, ["the metadata describes no usable geometry"]
    problems = geometry.problems()
    if problems:
        return None, problems

    # The file must actually hold what its own metadata claims, since that is
    # what the arrays are uploaded against.
    for name, shape in _expected_arrays(geometry, version).items():
        array = checkpoint.arrays.get(name)
        if array is None:
            problems.append(f"missing {name}")
        elif tuple(array.shape) != shape:
            problems.append(f"{name} is {tuple(array.shape)}, expected {shape}")
    if problems:
        return None, problems
    return geometry, []


def required_geometry(checkpoint: Checkpoint) -> engine_module.Geometry | None:
    """The geometry an engine must be built at to be able to load this, or ``None``.

    This is what stops the window size from mattering: instead of asking whether
    a saved field fits the session that is starting, the session asks what shape
    the field needs and starts in it. ``None`` means the file cannot be used at
    all -- foreign version, corrupt metadata, missing arrays, or a geometry no
    engine could be built at -- and the caller grows a new field instead.
    """
    geometry, problems = _usable_geometry(checkpoint)
    if problems:
        log.info(
            "the saved state cannot be used, starting from a fresh field (%s)",
            "; ".join(problems),
        )
    return geometry


def compatibility_problems(engine, checkpoint: Checkpoint) -> list[str]:
    """Human-readable reasons this checkpoint cannot be restored into *this* engine.

    Normally empty by construction: the launch builds its engine at
    :func:`required_geometry`. A non-empty list means either the file is unusable
    or the restore is being attempted into an engine built for something else,
    and either way it is what keeps the upload from writing mismatched shapes to
    the GPU.
    """
    geometry, problems = _usable_geometry(checkpoint)
    if geometry is None:
        return problems
    return geometry.differences(engine.geometry)


def _write_texture(device, texture, data: np.ndarray) -> None:
    payload = np.ascontiguousarray(data, dtype=np.float16)
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        payload,
        {"offset": 0, "bytes_per_row": payload.shape[1] * 8,
         "rows_per_image": payload.shape[0]},
        (payload.shape[1], payload.shape[0], 1),
    )


def _restore_walk(engine, saved: dict[str, Any]) -> None:
    """Put the global feature-size walk back where it was.

    Its value is accumulated state like any other: dropping it would restart the
    reaction's diffusion rate at the middle of its band and drift away from the
    saved field's feature size over the following minutes. The stream position
    goes back with it so the walk continues rather than merely resembling itself.

    Both halves are untrusted input, and neither is worth failing a restore over:
    a value that is not a finite number falls back to the walk's own centre, and
    a stream state this build cannot use leaves the fresh stream in place.
    """
    try:
        walk = float(saved.get("du_walk") or 0.0)
    except (TypeError, ValueError):
        walk = 0.0
    engine._du_walk = max(-2.0, min(2.0, walk)) if math.isfinite(walk) else 0.0

    state = saved.get("walk_rng")
    if not isinstance(state, dict) or not state:
        return
    try:
        engine._walk_rng.bit_generator.state = state
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        log.info("could not resume the feature-size walk's noise stream (%s)", exc)


def restore(engine, checkpoint: Checkpoint, scheduler=None) -> bool:
    """Load a checkpoint into a freshly built engine.

    Returns False and changes nothing if the checkpoint does not fit the engine
    it was handed: every shape is validated before the first upload, so a refusal
    leaves the seeded field intact and the run simply starts new. Building the
    engine so that it *does* fit is the caller's job -- see
    :func:`required_geometry`.
    """
    problems = compatibility_problems(engine, checkpoint)
    if problems:
        log.info(
            "saved state does not fit this engine, starting from a fresh "
            "field (%s)", "; ".join(problems),
        )
        return False

    device = engine.device
    for layer in engine.layers:
        index = layer.spec.index
        for name in PAIR_FIELDS:
            data = checkpoint.arrays.get(f"layer{index}.{name}")
            if data is None:
                # Only ever a field this file predates -- the validation above
                # refuses anything else missing -- so the seeded one stays.
                continue
            pair = getattr(layer, name)
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
    _restore_walk(engine, saved)

    if scheduler is not None:
        scheduler.load_state(checkpoint.meta.get("events") or {})

    log.info("resumed from %s", checkpoint.describe())
    return True
