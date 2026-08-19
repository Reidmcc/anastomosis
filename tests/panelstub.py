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

from anastomosis import config as config_module
from anastomosis import events as events_module
from anastomosis import volume as volume_module

# The window the panel believes it is showing. Only its shape matters, and only
# to the thickness knob, whose ceiling is the shorter lateral axis of the slab
# this size implies.
SIZE = (2560, 1440)


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
        self._frame_times = [0.01]
        self._sim_hz_scale = 1.0
        self._size = SIZE

    # -- parameters ---------------------------------------------------------

    def apply_macros(self, macros) -> None:
        self.config.macros = macros

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
