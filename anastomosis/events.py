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

import logging
import math
import random
from dataclasses import dataclass, field

from .config import EventParams
from . import gpu_params

log = logging.getLogger(__name__)


@dataclass
class ActiveEvent:
    """One in-flight perturbation."""

    x: float
    y: float
    radius: float
    peak: float
    channels: tuple[float, float, float, float]  # feed, kill, flow, hue
    attack: float
    hold: float
    release: float
    elapsed: float = 0.0
    kind: str = "bloom"

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
# amplitude. Values are (feed, kill, flow, hue) multipliers.
EVENT_KINDS: dict[str, tuple[float, float, float, float]] = {
    # A productive zone: more feed, less kill -- structure densifies.
    "bloom": (1.0, -0.55, 0.15, 0.35),
    # The opposite: material thins out and dissolves.
    "dieback": (-0.85, 0.7, 0.1, -0.25),
    # Mostly flow, so structure is carried and stretched rather than changed.
    "current": (0.1, 0.0, 1.0, 0.15),
    # A regional hue shift with little structural effect.
    "tint": (0.15, -0.1, 0.2, 1.0),
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
            channels=tuple(c * jitter() for c in base),  # type: ignore[arg-type]
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
            feed, kill, flow, hue = event.channels
            rows.append(
                {
                    "pos_x": event.x,
                    "pos_y": event.y,
                    "radius": event.radius,
                    "strength": strength,
                    "chan_feed": feed,
                    "chan_kill": kill,
                    "chan_flow": flow,
                    "chan_hue": hue,
                }
            )
        return rows, len(rows)

    def describe(self) -> str:
        if not self.active:
            return "none"
        return ", ".join(
            f"{e.kind}@{e.envelope():.2f}" for e in self.active
        )
