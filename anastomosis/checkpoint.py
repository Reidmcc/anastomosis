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
written every fifteen minutes for days on end:

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

**One file per backend.** The layered stack, the volumetric slab (DESIGN.md
§5.1) and the rhizotron's soil column (§15) hold different state in different
shapes, and no amount of resampling turns one into another, so a checkpoint
records which backend wrote it and :func:`default_checkpoint_path` gives each
its own file. Switching backend therefore does not destroy the field you
switched away from: switch back and it is still there, older but intact. What
they share is everything about *how* a checkpoint behaves -- the same version
gating, the same build-to-fit-the-file launch, the same degrade-to-fresh on
anything unusable -- which is why the difference between them is one small
object per backend (:data:`LAYOUTS`) rather than more copies of this module.
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

from . import config as config_module
from . import engine as engine_module
from . import gpu_params
from . import rhizotron as rhizotron_module
from . import things as things_module
from . import volume as volume_module

log = logging.getLogger(__name__)

FORMAT_VERSION = 6
# Version 1 recorded the *window* size where version 2 records the simulation's
# own, which are the same number unless that session was resized after starting.
# Version 3 adds the morphology climate pair and the feature-size walk, both of
# which postdate version 2 and neither of which an older file can carry.
# Version 4 names the backend that wrote the file; anything older predates the
# volumetric slab and can only have come from the layered one.
# Version 5 widens the stats buffer for the feature-size loop (DESIGN.md 4.7
# step 5) and renames the walk it drives; an older file's shorter stats blob is
# zero-extended, which starts the loop's reference unseeded and so takes it
# from the first live measurement, exactly as a fresh field does.
# Version 6 widens it again for the deposit capacity's return, and changes what
# the trail texture's `.a` channel means -- it used to be written as a constant
# 1.0 and now carries the withheld EMA. An older file's 1.0 washes out of the
# EMA within a hundred ticks; the transient return it produces is bounded by
# the clamp and slewed below anything visible.
# Reading them costs a few lines and saves anyone upgrading their mature field.
OLDEST_READABLE_VERSION = 1

# Fifteen minutes. It was five, on the reasoning that a crash should cost less
# field maturity than it takes to notice one -- which is still the trade, just
# priced against a laptop as well as a desktop (DESIGN.md §8.3).
#
# What a save costs is a readback of every field plus an uncompressed write,
# and both scale with the simulation, not with the interval. A 1600p stack is
# roughly 130 MB of field and 35 MB of agents: at five minutes that is about
# 2 GB an hour of disk writes, 48 GB a day, on a drive whose endurance is a
# consumable -- for a program whose whole proposition is that it is left
# running for days. At fifteen it is a third of that.
#
# What it buys back is bounded and cheap: the field is not *lost* on a crash,
# only rolled back, and it is rolled back into a simulation explicitly built
# to keep growing. Fifteen minutes of a multi-day field is not a loss anybody
# can see. The one place the number is felt is device-loss recovery, where the
# readback is impossible by definition and the last save on disk is all there
# is -- `Application._rebuild_device` says so in the log when it happens.
DEFAULT_INTERVAL_SECONDS = 900.0

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

# The version the backend key first appeared in. A file without it is layered,
# because nothing else existed yet.
BACKEND_SINCE = 4

# The stats buffer is a raw byte blob rather than a named array per field, so
# its *size* is the only thing a reader can check it against -- and that size
# changes as controllers grow state, which makes it the one array whose
# expected shape is a function of the version. Versions 1-4 wrote the 20-field
# block that predates the feature-size loop; version 5 the 25-field block that
# predates the deposit capacity's return. Recorded as numbers rather than
# derived, because the point of each entry is to describe a layout this build
# no longer has.
STATS_BYTES_BY_VERSION = {1: 80, 2: 80, 3: 80, 4: 80, 5: 100}

# The stats block's field order at each historical width, so a resume can carry
# every quantity across a widening *by name*. Zero-extending the raw bytes is
# not enough: both widenings inserted fields mid-struct rather than appending,
# so under the new layout an old file's tail lands in the wrong fields -- a
# version-4 exposure would resume as a feature-size reference. Fields all being
# f32 is what keeps this a list of names rather than a schema.
_STATS_COMMON = (
    "sum_v", "sum_v2", "sum_activity", "count", "mean_v", "var_v",
    "mean_activity", "alive_frac", "corr_feed", "corr_kill", "corr_deposit",
    "corr_decay", "int_mass", "int_var", "int_activity", "prune_return",
)
_STATS_IMAGE = ("img_sum_l", "img_max_l", "img_count", "exposure")
STATS_FIELDS_BY_VERSION = {
    4: _STATS_COMMON + _STATS_IMAGE,
    5: _STATS_COMMON
        + ("mean_grad_v", "ell", "ell_ref", "corr_du", "ell_samples")
        + _STATS_IMAGE,
}
for _v in (1, 2, 3):
    STATS_FIELDS_BY_VERSION[_v] = STATS_FIELDS_BY_VERSION[4]

# Rewritten from scratch every tick before anything reads them, so they are not
# saved. Named here so the tests can check that this stays true.
#
# The slab has three more of them and they are omitted on the same grounds: the
# assembled flow potential and the blur scratch are rewritten every tick before
# anything reads them, and the interpolated pigment every *frame*. At the
# default slab those three are about 170 MB that would otherwise be written to
# disk on every save to no purpose.
DERIVED_FIELDS = ("reaction_prev", "velocity")
VOLUME_DERIVED_FIELDS = ("reaction_prev", "velocity", "potential", "scratch", "interp")


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


def _read_texture3(device, texture) -> np.ndarray:
    """Read one rgba16float 3D texture back as ``(d, h, w, 4)`` float16."""
    width, height = texture.width, texture.height
    depth = texture.depth_or_array_layers
    raw = device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, depth),
    )
    return (
        np.frombuffer(raw, dtype=np.float16)[: width * height * depth * 4]
        .reshape(depth, height, width, 4)
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
    layout = layout_for(engine.name)
    arrays = layout.capture(engine)
    arrays["stats"] = _read_buffer(engine.device, engine.stats_buf)

    meta: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "created": time.time(),
        "sim_hz": float(sim_hz),
        "backend": layout.name,
        # The shape the next launch has to build itself in to be able to load
        # this, rather than the shape it has to happen to already be in.
        "geometry": layout.geometry_meta(engine.geometry),
        "engine": {
            "tick_count": int(engine.tick_count),
            "frame_count": int(engine.frame_count),
            "seed": int(engine.seed),
            "hue_phase": float(engine.hue_phase),
            # The feature-size setpoint walk: its value, and the position of
            # the stream that drives it (see `_walk_rng_state`). Named
            # `du_walk` before version 5, when it drove the diffusion rate
            # directly rather than the setpoint a controller drives it to.
            "ell_walk": float(engine._ell_walk),
            "walk_rng": _walk_rng_state(engine),
            "parallax": [
                [float(x), float(y)] for x, y in (engine._parallax or [])
            ],
        },
        "events": scheduler.state() if scheduler is not None else {},
    }
    # Backend-specific counters -- the rhizotron's descent -- land inside the
    # same engine block, so the test that resumes bit-identically covers them
    # with no per-backend special case.
    meta["engine"].update(layout.engine_meta(engine))
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


def default_checkpoint_path(backend: str | None = None) -> Path:
    """``$XDG_STATE_HOME/anastomosis/checkpoint[-<backend>].npz``.

    State rather than config or cache: it is neither hand-editable nor cheap to
    regenerate.

    One file per backend, because the two hold incompatible state and no
    amount of resampling turns a stack of sheets into a slab. Separate files
    mean switching backend costs nothing permanent: switch back and the field
    you left is still there. The layered backend keeps the unsuffixed name it
    has always had, so an existing mature field is found unchanged.
    """
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    name = str(backend or config_module.DEFAULT_BACKEND)
    suffix = "" if name == config_module.DEFAULT_BACKEND else f"-{name}"
    return root / "anastomosis" / f"checkpoint{suffix}.npz"


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


class _Layout:
    """How one backend's state maps to and from a checkpoint's arrays.

    Four questions, and every one of them has a different answer per backend
    while the machinery around them has exactly one: what shape the geometry
    metadata takes, how to read it back safely, which arrays a file must hold
    to be usable, and how to move them on and off the GPU. Everything else in
    this module -- the version gating, the bounds checking, the degrade-to-fresh
    behaviour, the counters and walks in the metadata -- is written once.
    """

    name = "abstract"

    def geometry_meta(self, geometry) -> dict[str, Any]:
        raise NotImplementedError

    def read_geometry(self, meta: dict[str, Any]):
        raise NotImplementedError

    def expected_arrays(
        self, geometry, version: int = FORMAT_VERSION
    ) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError

    def capture(self, engine) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def restore_arrays(self, engine, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def engine_meta(self, engine) -> dict[str, Any]:
        """Backend-specific accumulated state that lives outside the arrays.

        Merged into the shared ``engine`` metadata block by :func:`capture`.
        Empty for the fungal backends -- everything host-side they accumulate
        (counters, walks, the drift) is state every backend has and the shared
        code saves. The rhizotron's descent is the first counter one backend
        has and the others do not.
        """
        return {}

    def restore_engine_meta(self, engine, saved: dict[str, Any]) -> None:
        """The restore half of :meth:`engine_meta`. Untrusted input."""


def _field_shape(layer, name: str) -> tuple[int, int, int]:
    """The shape one saved field has, given the layer geometry it belongs to."""
    if name.startswith("climate"):
        width, height = layer.climate_width, layer.climate_height
    elif name == "psi":
        width, height = layer.psi_width, layer.psi_height
    else:
        width, height = layer.width, layer.height
    return (height, width, 4)


class _LayeredLayout(_Layout):
    """The 2.5D stack: one set of fields and one agent array per layer."""

    name = "layered"

    def geometry_meta(self, geometry) -> dict[str, Any]:
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

    def read_geometry(self, meta: dict[str, Any]):
        """The geometry described by a checkpoint's metadata, or ``None``.

        Nothing in here is trusted: the values are coerced to ``int`` and the
        result is bounds-checked by the caller before an engine is built from
        it.
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

    def expected_arrays(
        self, geometry, version: int = FORMAT_VERSION
    ) -> dict[str, tuple[int, ...]]:
        expected: dict[str, tuple[int, ...]] = {}
        for layer in geometry.layers:
            for name in PAIR_FIELDS:
                if version < FIELDS_SINCE.get(name, 0):
                    continue
                expected[f"layer{layer.index}.{name}"] = _field_shape(layer, name)
            expected[f"layer{layer.index}.agents"] = (
                max(layer.agent_count, 1) * engine_module.AGENT_STRIDE,
            )
        return expected

    def capture(self, engine) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for layer in engine.layers:
            index = layer.spec.index
            for name in PAIR_FIELDS:
                pair = getattr(layer, name)
                arrays[f"layer{index}.{name}"] = _read_texture(
                    engine.device, pair.textures[pair.index]
                )
            arrays[f"layer{index}.agents"] = _read_buffer(
                engine.device, layer.agents_buf)
        return arrays

    def restore_arrays(self, engine, arrays: dict[str, np.ndarray]) -> None:
        device = engine.device
        for layer in engine.layers:
            index = layer.spec.index
            for name in PAIR_FIELDS:
                data = arrays.get(f"layer{index}.{name}")
                if data is None:
                    # Only ever a field this file predates -- the validation
                    # refuses anything else missing -- so the seeded one stays.
                    continue
                pair = getattr(layer, name)
                # Both slots, so the parity the next tick happens to start on
                # cannot resurrect the seeded state.
                for texture in pair.textures:
                    _write_texture(device, texture, data)
                pair.index = 0
            device.queue.write_buffer(
                layer.agents_buf, 0, arrays[f"layer{index}.agents"].tobytes())


class _VolumeLayout(_Layout):
    """The volumetric slab: one set of 3D fields and one agent array.

    The field *names* are the same as the layered backend's, because they are
    the same fields -- trail, reaction, pigment, three climate pairs and the
    weather potential. What differs is that there is one of each rather than
    one per layer, and that each is a volume.
    """

    name = "volumetric"

    def geometry_meta(self, geometry) -> dict[str, Any]:
        return {
            "width": int(geometry.width),
            "height": int(geometry.height),
            "depth": int(geometry.depth),
            "agent_count": int(geometry.agent_count),
            "psi_width": int(geometry.psi_width),
            "psi_height": int(geometry.psi_height),
            "psi_depth": int(geometry.psi_depth),
            "climate_width": int(geometry.climate_width),
            "climate_height": int(geometry.climate_height),
            "climate_depth": int(geometry.climate_depth),
        }

    def read_geometry(self, meta: dict[str, Any]):
        block = meta.get("geometry")
        if not isinstance(block, dict):
            return None
        try:
            return volume_module.VolumeGeometry(
                width=int(block["width"]),
                height=int(block["height"]),
                depth=int(block["depth"]),
                agent_count=int(block["agent_count"]),
                psi_width=int(block["psi_width"]),
                psi_height=int(block["psi_height"]),
                psi_depth=int(block["psi_depth"]),
                climate_width=int(block["climate_width"]),
                climate_height=int(block["climate_height"]),
                climate_depth=int(block["climate_depth"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _shape(geometry, name: str) -> tuple[int, int, int, int]:
        if name.startswith("climate"):
            w, h, d = geometry.climate_dims
        elif name == "psi":
            w, h, d = geometry.psi_dims
        else:
            w, h, d = geometry.dims
        return (d, h, w, 4)

    def expected_arrays(
        self, geometry, version: int = FORMAT_VERSION
    ) -> dict[str, tuple[int, ...]]:
        expected: dict[str, tuple[int, ...]] = {
            f"slab.{name}": self._shape(geometry, name) for name in PAIR_FIELDS
        }
        expected["slab.agents"] = (
            max(geometry.agent_count, 1) * volume_module.AGENT_STRIDE,
        )
        return expected

    def capture(self, engine) -> dict[str, np.ndarray]:
        slab = engine.slab
        arrays: dict[str, np.ndarray] = {}
        for name in PAIR_FIELDS:
            pair = getattr(slab, name)
            arrays[f"slab.{name}"] = _read_texture3(
                engine.device, pair.textures[pair.index])
        arrays["slab.agents"] = _read_buffer(engine.device, slab.agents_buf)
        return arrays

    def restore_arrays(self, engine, arrays: dict[str, np.ndarray]) -> None:
        device = engine.device
        slab = engine.slab
        for name in PAIR_FIELDS:
            data = arrays.get(f"slab.{name}")
            if data is None:
                continue
            pair = getattr(slab, name)
            for texture in pair.textures:
                _write_texture3(device, texture, data)
            pair.index = 0
        device.queue.write_buffer(
            slab.agents_buf, 0, arrays["slab.agents"].tobytes())


class _RhizotronLayout(_Layout):
    """The soil column: moisture, the root map, the tips, and the descent.

    Still the smallest layout, and that smallness is a design fact worth
    keeping visible: the rhizotron's soil is a pure function of (seed, world
    row) rather than a stored field (DESIGN.md §15.3), so what accumulates is
    the water, what the roots have built, and the plant's own few thousand
    tips -- plus a handful of numbers of descent state. The deposit
    accumulator is drained by `atomicExchange` every tick and so is empty
    between ticks, exactly like the fungal one.
    """

    name = "rhizotron"

    def geometry_meta(self, geometry) -> dict[str, Any]:
        return {
            "width": int(geometry.width),
            "height": int(geometry.height),
            "view_rows": int(geometry.view_rows),
            "max_axes": int(geometry.max_axes),
            "laterals_per_axis": int(geometry.laterals_per_axis),
            "fines_per_lateral": int(geometry.fines_per_lateral),
        }

    def read_geometry(self, meta: dict[str, Any]):
        block = meta.get("geometry")
        if not isinstance(block, dict):
            return None
        try:
            return rhizotron_module.RhizotronGeometry(
                width=int(block["width"]),
                height=int(block["height"]),
                view_rows=int(block["view_rows"]),
                max_axes=int(block["max_axes"]),
                laterals_per_axis=int(block["laterals_per_axis"]),
                fines_per_lateral=int(block["fines_per_lateral"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def expected_arrays(
        self, geometry, version: int = FORMAT_VERSION
    ) -> dict[str, tuple[int, ...]]:
        return {
            "column.moisture": (geometry.height, geometry.width, 4),
            "column.structure": (geometry.height, geometry.width, 4),
            "column.record": (geometry.height, geometry.width, 4),
            "column.tips": (
                max(geometry.tips_total, 1) * rhizotron_module.TIP_STRIDE,
            ),
        }

    def capture(self, engine) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name in ("moisture", "structure", "record"):
            pair = getattr(engine, name)
            arrays[f"column.{name}"] = _read_texture(
                engine.device, pair.textures[pair.index])
        arrays["column.tips"] = _read_buffer(engine.device, engine.tips.cur)
        return arrays

    def restore_arrays(self, engine, arrays: dict[str, np.ndarray]) -> None:
        for name in ("moisture", "structure", "record"):
            data = arrays.get(f"column.{name}")
            if data is None:
                continue
            pair = getattr(engine, name)
            for texture in pair.textures:
                _write_texture(engine.device, texture, data)
            pair.index = 0
        tips = arrays.get("column.tips")
        if tips is not None:
            for buffer in engine.tips.buffers:
                engine.device.queue.write_buffer(buffer, 0, tips.tobytes())
            engine.tips.index = 0

    def engine_meta(self, engine) -> dict[str, Any]:
        return {
            "descent": engine.descent_state(),
            "season": engine.season_state(),
        }

    def restore_engine_meta(self, engine, saved: dict[str, Any]) -> None:
        descent = saved.get("descent")
        if isinstance(descent, dict):
            engine.restore_descent(descent)
        season = saved.get("season")
        if isinstance(season, dict):
            engine.restore_season(season)


class _ThingsLayout(_Layout):
    """The Small Strange Things: the canvas, the population, the breath.

    The smallest layout of all, and the point of the whole port (DESIGN.md
    §18.2): one canvas texture (image and trail, one object) and one
    population buffer whose slots are identities -- nothing dies, so a
    saved village is the same village, its ages in ticks, its friendships
    intact. The deposit accumulator is drained by ``atomicExchange`` every
    tick and so is empty between ticks, exactly like the others.
    """

    name = "things"

    def geometry_meta(self, geometry) -> dict[str, Any]:
        return {
            "width": int(geometry.width),
            "height": int(geometry.height),
            "capacity": int(geometry.capacity),
        }

    def read_geometry(self, meta: dict[str, Any]):
        block = meta.get("geometry")
        if not isinstance(block, dict):
            return None
        try:
            return things_module.ThingsGeometry(
                width=int(block["width"]),
                height=int(block["height"]),
                capacity=int(block["capacity"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def expected_arrays(
        self, geometry, version: int = FORMAT_VERSION
    ) -> dict[str, tuple[int, ...]]:
        return {
            "world.canvas": (geometry.height, geometry.width, 4),
            "world.things": (
                max(geometry.capacity, 1) * things_module.THING_STRIDE,
            ),
        }

    def capture(self, engine) -> dict[str, np.ndarray]:
        pair = engine.canvas
        return {
            "world.canvas": _read_texture(
                engine.device, pair.textures[pair.index]),
            "world.things": _read_buffer(engine.device, engine.things.cur),
        }

    def restore_arrays(self, engine, arrays: dict[str, np.ndarray]) -> None:
        canvas = arrays.get("world.canvas")
        if canvas is not None:
            for texture in engine.canvas.textures:
                _write_texture(engine.device, texture, canvas)
            engine.canvas.index = 0
        things = arrays.get("world.things")
        if things is not None:
            for buffer in engine.things.buffers:
                engine.device.queue.write_buffer(buffer, 0, things.tobytes())
            engine.things.index = 0

    def engine_meta(self, engine) -> dict[str, Any]:
        return {"pulse": engine.pulse_state()}

    def restore_engine_meta(self, engine, saved: dict[str, Any]) -> None:
        pulse = saved.get("pulse")
        if isinstance(pulse, dict):
            engine.restore_pulse(pulse)


LAYOUTS: dict[str, _Layout] = {
    layout.name: layout
    for layout in (
        _LayeredLayout(), _VolumeLayout(), _RhizotronLayout(), _ThingsLayout()
    )
}


def layout_for(backend: str) -> _Layout:
    """The layout for a backend name, defaulting to the layered one.

    Untrusted, like everything else that arrives from a file: an unknown name
    resolves to the layered layout, whose validation then rejects the arrays
    for not being the shape it expects. That is the right failure -- one
    "cannot use this, starting fresh" rather than a special case per way of
    being wrong.
    """
    return LAYOUTS.get(str(backend or ""), LAYOUTS[config_module.DEFAULT_BACKEND])


def checkpoint_backend(checkpoint: Checkpoint) -> str:
    """Which backend wrote this file.

    A file without the key predates the volumetric slab, so it can only have
    come from the layered backend.
    """
    return str(checkpoint.meta.get("backend") or config_module.DEFAULT_BACKEND)


def stats_bytes_for(version: int) -> int:
    """How many bytes of stats block a file of this version is expected to hold.

    The stats buffer is the one array saved as a raw blob rather than as a named
    field per quantity, so its size is the only handle a reader has on it -- and
    unlike every texture here, that size is a function of the build rather than
    of the geometry.
    """
    return STATS_BYTES_BY_VERSION.get(version, gpu_params.STATS_DTYPE.itemsize)


def _expected_arrays(
    geometry, version: int = FORMAT_VERSION, backend: str | None = None
) -> dict[str, tuple[int, ...]]:
    """Array name -> required shape, for a given simulation geometry.

    Version-aware, because a field that did not exist when the file was written
    cannot be in it and must not be demanded of it. The stats blob is not here;
    it is checked against a minimum rather than an exact size, so it has its own
    handling in :func:`_usable_geometry`.
    """
    layout = layout_for(backend or config_module.DEFAULT_BACKEND)
    expected: dict[str, tuple[int, ...]] = {}
    expected.update(layout.expected_arrays(geometry, version))
    return expected


def _usable_geometry(
    checkpoint: Checkpoint, backend: str | None = None
) -> tuple[Any, list[str]]:
    """``(geometry, problems)``: what this file needs, or why it is unusable.

    ``backend`` is what the caller intends to *run*. Passing it turns a file
    written by the other backend into an ordinary "cannot use this" rather than
    an attempt to read a slab's metadata as a stack of layers. Left out, the
    file is taken at its word, which is what a restore into an already-built
    engine wants.
    """
    version = checkpoint.meta.get("version")
    if (
        not isinstance(version, int)
        or not OLDEST_READABLE_VERSION <= version <= FORMAT_VERSION
    ):
        return None, [
            f"format version {version!r}, expected "
            f"{OLDEST_READABLE_VERSION} to {FORMAT_VERSION}"
        ]

    saved_backend = checkpoint_backend(checkpoint)
    if backend is not None and saved_backend != backend:
        return None, [
            f"written by the {saved_backend} backend, this session runs "
            f"{backend}"
        ]

    layout = layout_for(saved_backend)
    geometry = layout.read_geometry(checkpoint.meta)
    if geometry is None:
        return None, ["the metadata describes no usable geometry"]
    problems = geometry.problems()
    if problems:
        return None, problems

    # The file must actually hold what its own metadata claims, since that is
    # what the arrays are uploaded against.
    #
    # The stats blob is checked against a floor rather than an exact size. Short
    # of what its own version wrote it is truncated and unusable; longer than
    # that it is a file from a build that had more to say, and `_fit_stats`
    # takes the part this one understands. Requiring an exact match would mean
    # every widening of the block threw away every field written before it, for
    # a few bytes of controller state that a fresh field re-derives in a minute.
    stats = checkpoint.arrays.get("stats")
    required = stats_bytes_for(version)
    if stats is None:
        problems.append("missing stats")
    elif stats.size < required:
        problems.append(f"stats is {stats.size} bytes, expected at least {required}")

    for name, shape in _expected_arrays(geometry, version, saved_backend).items():
        array = checkpoint.arrays.get(name)
        if array is None:
            problems.append(f"missing {name}")
        elif tuple(array.shape) != shape:
            problems.append(f"{name} is {tuple(array.shape)}, expected {shape}")
    if problems:
        return None, problems
    return geometry, []


def required_geometry(checkpoint: Checkpoint, backend: str | None = None):
    """The geometry an engine must be built at to be able to load this, or ``None``.

    This is what stops the window size from mattering: instead of asking whether
    a saved field fits the session that is starting, the session asks what shape
    the field needs and starts in it. ``None`` means the file cannot be used at
    all -- foreign version, corrupt metadata, missing arrays, a geometry no
    engine could be built at, or the wrong backend -- and the caller grows a new
    field instead.
    """
    geometry, problems = _usable_geometry(checkpoint, backend)
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
    geometry, problems = _usable_geometry(checkpoint, engine.name)
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


def _write_texture3(device, texture, data: np.ndarray) -> None:
    payload = np.ascontiguousarray(data, dtype=np.float16)
    depth, height, width = payload.shape[0], payload.shape[1], payload.shape[2]
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        payload,
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, depth),
    )


def _fit_stats(saved: np.ndarray, version: int) -> bytes:
    """The saved stats blob under this build's layout, migrated field by field.

    A file of the current version is passed through byte-for-byte. An older one
    is parsed under the layout its version wrote (`STATS_FIELDS_BY_VERSION`)
    and each field is copied into a fresh current-layout record by *name*;
    fields the old build did not have start at zero, which for every widening
    so far means the new controller re-derives its state from the restored
    field within a few time constants. Copying bytes instead would misfile
    everything after the insertion point -- the version-4 exposure multiplier,
    for instance, would resume as the feature-size reference.
    """
    payload = np.asarray(saved, dtype=np.uint8).reshape(-1)
    names = STATS_FIELDS_BY_VERSION.get(version)
    if names is None:  # current layout, or a future one already refused
        width = gpu_params.STATS_DTYPE.itemsize
        return payload[:width].tobytes()

    old_dtype = np.dtype([(name, np.float32) for name in names])
    record = np.frombuffer(
        payload[: old_dtype.itemsize].tobytes(), dtype=old_dtype)[0]
    current = np.zeros(1, dtype=gpu_params.STATS_DTYPE)
    for name in names:
        if name in gpu_params.STATS_DTYPE.names:
            current[name] = record[name]
    return current.tobytes()


def _restore_walk(engine, saved: dict[str, Any]) -> None:
    """Put the feature-size setpoint walk back where it was.

    Its value is accumulated state like any other: dropping it would restart the
    setpoint at the middle of its band and drift away from the saved field's
    feature size over the following minutes. The stream position goes back with
    it so the walk continues rather than merely resembling itself.

    A file written before version 5 carries the same walk under its old name,
    when it scaled the diffusion rate directly rather than the setpoint the
    controller now drives it to. The units are identical -- a bounded,
    unit-variance OU state -- so the old value is still the right place to
    resume from; what it multiplies is `ell_walk` rather than `du_walk` now.

    Both halves are untrusted input, and neither is worth failing a restore over:
    a value that is not a finite number falls back to the walk's own centre, and
    a stream state this build cannot use leaves the fresh stream in place.
    """
    raw = saved.get("ell_walk")
    if raw is None:
        raw = saved.get("du_walk")
    try:
        walk = float(raw or 0.0)
    except (TypeError, ValueError):
        walk = 0.0
    engine._ell_walk = max(-2.0, min(2.0, walk)) if math.isfinite(walk) else 0.0

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
    layout_for(engine.name).restore_arrays(engine, checkpoint.arrays)

    # Migrated by field name if it came from a build with a different layout.
    # Fields a narrower file did not carry start at zero, which is the right
    # value for every one so far: the controllers they belong to re-derive
    # their state from the restored field within a few time constants.
    device.queue.write_buffer(
        engine.stats_buf, 0,
        _fit_stats(
            checkpoint.arrays["stats"],
            int(checkpoint.meta.get("version") or FORMAT_VERSION)))

    saved = checkpoint.meta.get("engine") or {}
    engine.tick_count = int(saved.get("tick_count", 0))
    engine.frame_count = int(saved.get("frame_count", 0))
    engine.seed = int(saved.get("seed", engine.seed)) & 0xFFFFFFFF
    hue_phase = float(saved.get("hue_phase", 0.0))
    engine.hue_phase = hue_phase % (2.0 * math.pi) if math.isfinite(hue_phase) else 0.0
    # The camera drift. How many entries there are is the backend's business
    # -- one per layer under the layered stack, one camera under the slab -- so
    # this restores whatever was saved and lets `_update_parallax` discard it if
    # the count no longer matches. Losing it costs a slow re-drift from centre,
    # not a step.
    parallax = saved.get("parallax") or []
    try:
        engine._parallax = [
            [float(x), float(y)] for x, y in parallax
        ] or None
    except (TypeError, ValueError):
        engine._parallax = None
    _restore_walk(engine, saved)
    layout_for(engine.name).restore_engine_meta(engine, saved)

    if scheduler is not None:
        scheduler.load_state(checkpoint.meta.get("events") or {})

    log.info("resumed from %s", checkpoint.describe())
    return True
