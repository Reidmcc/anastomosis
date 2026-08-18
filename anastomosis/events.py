"""Poisson-arrival slow events -- DESIGN.md §4.3.

Statistical stationarity is its own kind of predictability. A perfectly
homeostatic system stops being surprising after an hour even though it never
repeats, because the *texture* of change becomes known. Events break that.

Every constraint here exists to keep an event from reading as punctuation:

* raised-cosine envelopes with attack and release measured in tens of seconds,
  so nothing ever steps;
* spatial localisation with a smooth radial falloff, capped at 25% of the
  screen -- which is also the WCAG flash-area threshold, so even a
  hypothetical instantaneous event could not meet the flash criterion on area
  alone;
* applied to the *climate* field, never to pigment or luminance, so the effect
  reaches the image only after several stages of diffusion and temporal
  lowpass.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
from dataclasses import dataclass
from typing import NamedTuple

from .config import EventParams
from . import gpu_params

log = logging.getLogger(__name__)


class Channels(NamedTuple):
    """What an event does to the climate field, per channel.

    Defaulted, so a kind only names the channels it actually uses and adding a
    channel does not touch the kinds that ignore it. The names must match the
    ``chan_*`` fields of :data:`anastomosis.gpu_params.EVENT_FIELDS`, which is
    asserted in the tests.
    """

    feed: float = 0.0
    kill: float = 0.0
    flow: float = 0.0
    hue: float = 0.0
    # Severance. `decay` and `prune` raise the rate at which the trail field
    # forgets, globally in the region and specifically on strands traffic has
    # abandoned; `repel` pushes the agents' junction commitment past the point
    # where it changes sign, so they veer away from filaments instead of
    # committing to them. See DESIGN.md 4.7 step 4.
    decay: float = 0.0
    prune: float = 0.0
    repel: float = 0.0

    @classmethod
    def coerce(cls, value: object) -> "Channels":
        """Rebuild the named tuple from whatever a checkpoint round-trip left.

        JSON has no named tuples, so `state` writes the channels positionally
        and they come back as a plain list -- which behaves like `Channels` for
        arithmetic and comparison and then fails in `pack`, which addresses the
        channels *by name*. Normalising on the way in is the only way that
        failure cannot come back; see `ActiveEvent.__post_init__`.

        Length is treated as a version marker, because that is what it is: a
        checkpoint written before a channel existed is short, so it takes the
        default (zero, i.e. the channel does nothing), and one written by a
        later build carrying channels this one has never heard of is long, so
        the surplus is dropped. Either way the event survives the restart,
        which is the whole point of saving it mid-envelope.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**{
                name: float(value[name]) for name in cls._fields if name in value
            })
        return cls(*(float(c) for c in tuple(value)[:len(cls._fields)]))


@dataclass
class ActiveEvent:
    """One in-flight perturbation."""

    x: float
    y: float
    radius: float
    peak: float
    channels: Channels
    attack: float
    hold: float
    release: float
    elapsed: float = 0.0
    kind: str = "bloom"

    def __post_init__(self) -> None:
        # One choke point for every way an event can be built -- spawned here,
        # restored from disk, or assembled by a test -- so `channels` is a
        # `Channels` by the time anything reads it.
        self.channels = Channels.coerce(self.channels)

    @property
    def duration(self) -> float:
        return self.attack + self.hold + self.release

    @property
    def finished(self) -> bool:
        return self.elapsed >= self.duration

    def envelope(self) -> float:
        """Raised cosine in, flat hold, raised cosine out. C1 at every join."""
        t = self.elapsed
        if t < self.attack:
            return 0.5 - 0.5 * math.cos(math.pi * t / max(self.attack, 1e-6))
        if t < self.attack + self.hold:
            return 1.0
        remaining = t - self.attack - self.hold
        phase = min(remaining / max(self.release, 1e-6), 1.0)
        return 0.5 + 0.5 * math.cos(math.pi * phase)

    def strength(self) -> float:
        return self.peak * self.envelope()


# Each kind perturbs a different combination of climate channels, so events are
# recognisably different in character rather than being one effect at varying
# amplitude.
EVENT_KINDS: dict[str, Channels] = {
    # A productive zone: more feed, less kill -- structure densifies.
    "bloom": Channels(feed=1.0, kill=-0.55, flow=0.15, hue=0.35),
    # The opposite: material thins out and dissolves.
    "dieback": Channels(feed=-0.85, kill=0.7, flow=0.1, hue=-0.25),
    # Mostly flow, so structure is carried and stretched rather than changed.
    "current": Channels(feed=0.1, flow=1.0, hue=0.15),
    # A regional hue shift with little structural effect.
    "tint": Channels(feed=0.15, kill=-0.1, flow=0.2, hue=1.0),
    # A rift: the network comes apart across the region rather than merely
    # thinning. `dieback` reaches only feed and kill, which removes material
    # while leaving the topology exactly as it was -- the same mesh, fainter.
    # Severance needs the three channels this kind adds: decay, so the trail
    # forgets faster; prune, so it forgets abandoned strands fastest (inert
    # while `agents.prune_gain` is zero); and repel, so the agents stop
    # knitting the gap closed behind it.
    #
    # Those three and nothing else, which is a measured decision rather than a
    # tidy one. Carrying a *small* feed and kill alongside them looked harmless
    # and was not: an event adds its amplitude to the climate every tick
    # against a mean reversion of 0.0016, so any channel a kind names is pinned
    # at the clamp for the length of the envelope and the amplitude only shapes
    # the ramp. Measured, the version carrying feed and kill took the reaction
    # inside the disc down by a third and the whole field's mass by 16% -- a
    # dieback wearing a rift's clothes, and a slow global brightness swing
    # through the exposure governor. Without them the effect on the network is
    # identical and the reaction is left alone, which is what makes this a
    # structural event: it changes how the material is connected, not how much
    # of it there is (DESIGN.md 4.7).
    "rift": Channels(decay=1.0, prune=1.0, repel=1.0),
}


class EventScheduler:
    """Poisson arrivals with a cap on concurrency."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.active: list[ActiveEvent] = []
        self._time_to_next: float | None = None
        self.spawned = 0

    def _sample_interval(self, rate_per_hour: float) -> float:
        rate = max(rate_per_hour, 1e-6) / 3600.0
        # Exponential inter-arrival: memoryless, so the next event is never
        # predictable from how long it has been since the last one.
        return self._rng.expovariate(rate)

    def update(self, dt: float, params: EventParams) -> list[ActiveEvent]:
        for event in self.active:
            event.elapsed += dt
        self.active = [e for e in self.active if not e.finished]

        if not params.enabled or params.max_concurrent <= 0:
            self._time_to_next = None
            return self.active

        if self._time_to_next is None:
            self._time_to_next = self._sample_interval(params.rate_per_hour)

        self._time_to_next -= dt
        while self._time_to_next <= 0.0:
            if len(self.active) < params.max_concurrent:
                self.active.append(self._spawn(params))
            self._time_to_next += self._sample_interval(params.rate_per_hour)

        return self.active

    def _spawn(self, params: EventParams) -> ActiveEvent:
        kind = self._rng.choice(list(EVENT_KINDS))
        base = EVENT_KINDS[kind]

        # Radius is capped well below the flash-area threshold, and jittered so
        # events are not all the same size.
        radius = params.max_radius_frac * self._rng.uniform(0.45, 1.0)
        jitter = lambda: self._rng.uniform(0.75, 1.25)  # noqa: E731

        event = ActiveEvent(
            x=self._rng.random(),
            y=self._rng.random(),
            radius=radius,
            peak=params.strength * self._rng.uniform(0.6, 1.0),
            channels=Channels(*(c * jitter() for c in base)),
            attack=params.attack_seconds * jitter(),
            hold=params.hold_seconds * jitter(),
            release=params.release_seconds * jitter(),
            kind=kind,
        )
        self.spawned += 1
        log.debug(
            "event %s at (%.2f, %.2f) r=%.3f for %.0fs",
            kind, event.x, event.y, event.radius, event.duration,
        )
        return event

    def pack(self, max_events: int) -> tuple[list[dict[str, float]], int]:
        """Return GPU records for the currently active events."""
        rows: list[dict[str, float]] = []
        for event in self.active[:max_events]:
            strength = event.strength()
            if strength <= 1e-5:
                continue
            row = {
                "pos_x": event.x,
                "pos_y": event.y,
                "radius": event.radius,
                "strength": strength,
            }
            row.update(
                (f"chan_{name}", value)
                for name, value in event.channels._asdict().items()
            )
            rows.append(row)
        return rows, len(rows)

    # -- checkpointing ------------------------------------------------------

    def state(self) -> dict:
        """Serialisable scheduler state.

        The RNG is deliberately absent. Inter-arrival times are exponential and
        therefore memoryless, so a fresh stream after a restart is
        indistinguishable from the saved one -- but the events currently
        *in flight* are mid-envelope, and losing those would be exactly the step
        the envelopes exist to prevent.
        """
        return {
            "active": [dataclasses.asdict(event) for event in self.active],
            "time_to_next": self._time_to_next,
            "spawned": self.spawned,
        }

    def load_state(self, state: dict) -> None:
        """Restore from :meth:`state`, ignoring anything unreadable.

        Tolerant by design: this data comes off disk, and a partially readable
        checkpoint should cost at most a few in-flight events.
        """
        active: list[ActiveEvent] = []
        for row in state.get("active") or []:
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row["channels"] = row.get("channels") or Channels()
            try:
                event = ActiveEvent(**row)
            except (TypeError, ValueError) as exc:
                log.warning("ignoring unreadable saved event (%s)", exc)
                continue
            if not event.finished:
                active.append(event)
        self.active = active

        time_to_next = state.get("time_to_next")
        try:
            self._time_to_next = None if time_to_next is None else float(time_to_next)
        except (TypeError, ValueError):
            self._time_to_next = None
        try:
            self.spawned = int(state.get("spawned", 0))
        except (TypeError, ValueError):
            self.spawned = 0

    def describe(self) -> str:
        if not self.active:
            return "none"
        return ", ".join(
            f"{e.kind}@{e.envelope():.2f}" for e in self.active
        )
