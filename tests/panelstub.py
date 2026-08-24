"""A stand-in application for the control panel to talk to.

The panel is a Qt widget over one object -- the running :class:`Application` --
and the tests that exercise it care about a button, a slider or a shortcut
rather than about a live GPU. So they hand it this instead.

It lives in one place because the alternative was tried and did not hold: each
module that stood the panel up kept its own copy, and every control added to
the panel since -- the backend selector, then the slab's thickness -- broke all
of them at once, in a way that only showed up on a machine with PySide6
installed. One stub is one place to add the next such attribute.

Subclass it for whatever the test is actually watching; the counters and lists
belong to the test, not here.
"""

from __future__ import annotations

import dataclasses

from anastomosis import audio as audio_module
from anastomosis import config as config_module
from anastomosis import events as events_module
from anastomosis import gpu_params
from anastomosis import volume as volume_module

# The window the panel believes it is showing. Only its shape matters, and only
# to the thickness knob, whose ceiling is the shorter lateral axis of the slab
# this size implies.
SIZE = (2560, 1440)


def stats(**overrides) -> dict[str, float]:
    """What a stand-in engine's ``read_stats`` returns.

    Every field of the real block, not the two the status line happened to
    read when the stub was written. The real
    :meth:`~anastomosis.backend.Backend.read_stats` builds its dict from
    ``STATS_DTYPE.names``, so a status line may read any field in it and be
    right; a stub that lists a couple by hand is a `KeyError` waiting for the
    next quantity anyone decides to show. Deriving it from the same dtype is
    the same argument this module's docstring makes about panel attributes,
    one layer down.
    """
    values = {name: 0.0 for name in gpu_params.STATS_DTYPE.names}
    # Plausible rather than zero for the few the panel and the log line
    # actually render, so a formatted status string looks like a real one.
    values.update(mean_v=0.12, mean_activity=0.0012, var_v=0.009, ell=2.4,
                  exposure=1.0, count=1.0)
    unknown = set(overrides) - set(values)
    if unknown:
        raise KeyError(f"not fields of the stats block: {sorted(unknown)}")
    values.update(overrides)
    return values


class PanelApp:
    """Just enough of Application for the panel to talk to."""

    def __init__(
        self, backend: str = "layered", volume_detail: str = "standard"
    ) -> None:
        self.config = config_module.Config(
            backend=backend, volume_detail=volume_detail)
        self.params = self.config.resolve()
        self.backend = backend
        self.volume_detail = config_module.normalise_volume_detail(volume_detail)
        self.scheduler = events_module.EventScheduler(seed=1)
        self.engine = None
        # Never started: the panel only ever reads `describe()` from it, and
        # a stub that opened a real capture stream would make these tests
        # hostage to the machine's sound hardware.
        self.audio = audio_module.AudioDrive()
        self._frame_times = [0.01]
        self._sim_hz_scale = 1.0
        self._size = SIZE

    # -- parameters ---------------------------------------------------------

    def apply_macros(self, macros) -> None:
        self.config.macros = macros

    def set_mode(self, name: str) -> bool:
        wanted = config_module.normalise_mode(name)
        if wanted == config_module.normalise_mode(self.config.mode):
            return False
        self.config.mode = wanted
        return True

    def set_filaments(self, on: bool) -> bool:
        on = bool(on)
        if self.config.filaments == on:
            return False
        self.config.filaments = on
        return True

    def save_config(self) -> None:
        pass

    # -- the field ----------------------------------------------------------

    def checkpoint_status(self) -> str:
        return "off"

    def reset_simulation(self) -> None:
        pass

    def trigger_event(self, kind: str) -> bool:
        return self.scheduler.trigger(kind, self.params.events) is not None

    # -- structural choices -------------------------------------------------

    def switch_backend(self, name: str) -> bool:
        if name == self.backend:
            return False
        self.backend = name
        self.config.backend = name
        return True

    def volume_slab(self, depth: int | None = None):
        params = self.params
        if depth is not None and int(depth) != params.volume.depth:
            params = dataclasses.replace(
                params,
                volume=dataclasses.replace(params.volume, depth=int(depth)),
            )
        return volume_module.VolumeGeometry.derive(*self._size, params)

    def volume_depth_limits(self) -> tuple[int, int]:
        return volume_module.depth_limits(*self._size, self.params)

    def set_volume_depth(self, depth: int) -> int:
        self.params.volume.depth = int(depth)
        self.config.overrides["volume.depth"] = int(depth)
        return int(depth)

    def set_volume_detail(self, name: str) -> bool:
        wanted = config_module.normalise_volume_detail(name)
        if wanted == self.volume_detail:
            return False
        self.volume_detail = wanted
        self.config.volume_detail = wanted
        # The width feeds the slab the thickness row prices, so the stub has to
        # carry it into the parameters the way the application does -- a stub
        # that only recorded the name would let a width/thickness interaction
        # pass a test it should fail.
        self.params.volume.width = self.config.resolve().volume.width
        return True

    # -- the window ---------------------------------------------------------

    def toggle_fullscreen(self) -> bool:
        return False
