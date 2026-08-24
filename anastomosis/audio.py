"""The audio front end for the resonance mode. DESIGN.md §16.

Three pieces, split exactly where §16.2–§16.3 splits them:

* :class:`FeatureExtractor` — samples in, one :class:`AudioFeatures` record
  per hop out. Pure and deterministic in the sample stream: no wall-clock
  reads anywhere, every time constant advanced by sample count, so the same
  stream produces the same features whatever the chunking (§3's discipline,
  extended — audio is an input, not a clock).
* :func:`pick_capture_device` and :func:`pick_loopback_microphone` — the
  loopback-preferring device choices of §16.2, pure functions over device
  lists so the preference order is testable without an audio backend (or a
  Windows machine) in the room. The first serves the PortAudio route, the
  second the WASAPI-loopback route ``soundcard`` provides on Windows.
* :class:`AudioDrive` — owns the capture stream, the callback thread's
  bounded queue, and every degradation path. `sounddevice` is an optional
  extra and is imported lazily; its absence, a missing device, or a failed
  open is a status line for the panel, never an exception. The mode must
  never fail to open (§16.2(3)).

Nothing in the engine consumes this module yet — step 1 of §16.8 is
deliberately invisible, the way activation's step 1 was.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import statistics
import sys
import threading

import numpy as np

from . import config as config_module

log = logging.getLogger(__name__)

# -- framing (§16.3) --------------------------------------------------------

SAMPLE_RATE = 48_000
# ~21 ms at 48 kHz: ~47 feature frames a second, comfortably above any sim
# rate, so the tick loop always has a fresh record to poll.
HOP = 1024
WINDOW = 2048

# -- band edges, Hz ---------------------------------------------------------

BASS_BAND = (20.0, 150.0)
MID_BAND = (150.0, 2_000.0)
TREBLE_BAND = (2_000.0, 8_000.0)

# -- levels and gates -------------------------------------------------------

# -60 dBFS. Under this the room counts as quiet; feature targets go to zero
# and arrive there through the release followers.
SILENCE_RMS = 1e-3
# How long under the gate before the stream is *flagged* silent. The features
# are already decaying by then; the flag is for the panel and for consumers
# that want "no music" as a state rather than a threshold of their own.
SILENCE_SECONDS = 1.0
# The AGC reference never decays below this (-34 dBFS), so silence cannot be
# amplified into signal: with nothing playing, level is rms/floor ~ 0, not
# rms/rms ~ 1.
REF_FLOOR = 0.02
# Seconds; the decaying-peak AGC reference of §16.3. Slow enough that a
# phrase's dynamics survive normalisation, fast enough that a volume change
# stops mattering within half a minute.
REF_TAU = 20.0

# -- follower time constants, seconds (§16.3) -------------------------------

ATTACK_SECONDS = 0.08
RELEASE_SECONDS = 0.5
ONSET_RELEASE_SECONDS = 0.25

# -- onset detection (§16.3: log-domain flux, adaptive threshold) -----------

FLUX_HISTORY_HOPS = 43            # ~0.9 s of context for the median
FLUX_THRESHOLD_RATIO = 1.5
FLUX_THRESHOLD_BIAS = 0.05        # in compressed flux units; see _squash_flux
ONSET_REFRACTORY_SECONDS = 0.15

# -- the tempo estimate (§16.4: pace from the beat, in stream time) ---------

# An EMA over inter-onset intervals is the tempo proxy: honest (it is
# literally how often the music is hitting), cheap, and deterministic in the
# stream. The mapping to [0, 1]: one onset every two seconds or slower reads
# as 0, four a second -- a busy track counting fills and subdivisions -- as
# 1. A gap longer than this says nothing about tempo (a pause is not a slow
# song) and is left out of the average; the *current* gap still pulls the
# live estimate down, so a lull reads as slower immediately.
PACE_SLOW_HZ = 0.5
PACE_FAST_HZ = 4.0
PACE_EMA_ALPHA = 0.3
PACE_GAP_CAP_SECONDS = 30.0

# -- the waning estimate (§16.4: the music receding) ------------------------

# `waning` is how far the level has fallen from where it recently was: the
# fast level follower measured against a slow reference of itself. A song
# fading out, a breakdown, a long decrescendo all read as waning; ordinary
# verse/chorus dynamics (a drop of a third or so) stay under the fade door's
# threshold; a fade *in* reads as zero, because the fast level leads the
# slow one there. Scaled by how loud the music actually was, so a wobble in
# an already-quiet room is not a fade.
WANE_REF_TAU = 12.0
WANE_LOUDNESS_FLOOR = 0.3
# The waning follower's own attack, much slower than the ordinary 80 ms: a
# real fade develops over many seconds, so nothing musical is lost -- and a
# *hard cut* must not reach the fade threshold before the cut detector below
# has had its confirmation window, or every cut would draw a dieback ahead
# of its rift. From a full-scale recession this crosses the default 0.45
# threshold in ~1.2 s, after CUT_CONFIRM_SECONDS has settled the question.
WANE_ATTACK_SECONDS = 2.0
# Fade asks are spaced further apart than onset asks -- a fade is one
# musical moment, not a stream of them.
FADE_SPACING_FACTOR = 6.0

# -- the cut estimate (§16.4: the music severed) ----------------------------

# A hard cut -- a DJ cut, an abrupt ending, a caesura -- is not a fast fade:
# it is the silence gate engaging while the fast level was still loud, where
# a fade arrives at the gate already quiet. The level at the moment the gate
# engages is the discriminator (the follower's 0.5 s release means it still
# carries the pre-cut level on that hop), and a short confirmation window
# keeps an emphatic rest inside a phrase from counting. One firing per
# silence, re-armed when sound returns.
CUT_CONFIRM_SECONDS = 0.75
CUT_LOUDNESS_FLOOR = 0.3
SEVERED_RELEASE_SECONDS = 2.0
# Rifts are the heaviest gesture in the vocabulary, and are spaced like it.
RIFT_SPACING_FACTOR = 12.0

# The capture queue is bounded because the render thread can stall
# (checkpoint readback, a wedged compositor) and audio must never back
# memory up behind it; overflow drops the oldest blocks, which for a
# feature stream is self-healing (§16.2).
QUEUE_BLOCKS = 64

# Devices whose names mark them as "what the machine is playing", in
# preference order (§16.2): PulseAudio/PipeWire monitors, generic loopbacks,
# the macOS virtual devices, and the Windows driver mixes.
LOOPBACK_NAMES = (
    "monitor",
    "loopback",
    "blackhole",
    "soundflower",
    "stereo mix",
    "what u hear",
)


@dataclasses.dataclass(frozen=True)
class AudioFeatures:
    """One hop's worth of what the engine is allowed to know about the room.

    Every field is bounded to [0, 1] and already smooth (followed at the
    §16.3 time constants), so nothing downstream ever needs to defend
    against a step arriving from here. The zero record — which is also the
    default — is the identity: a drive holding it modulates nothing.
    """

    level: float = 0.0
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    flux: float = 0.0
    onset: float = 0.0
    # The music's tempo, as the smoothed onset rate mapped onto [0, 1]
    # (PACE_SLOW_HZ..PACE_FAST_HZ). What the drive paces the simulation's
    # own tempo -- the weather's reversion, the regimes' migration, an
    # event's envelope -- against (§16.4).
    pace: float = 0.0
    # How far the music has receded from where it recently was (§16.4): a
    # fade-out, a breakdown, a long decrescendo. The fade door's signal --
    # the moment a *dieback* fits the music -- and a feature that
    # deliberately outlives the sound for a few seconds: a fade's dieback
    # arrives *as* the music goes, which is the point.
    waning: float = 0.0
    # The music severed rather than receding (§16.4): a hard cut from loud
    # to silence, at the strength of what was cut. A pulse like `onset`,
    # and the rift door's signal -- the one moment severance fits.
    severed: float = 0.0
    silent: bool = True


class FeatureExtractor:
    """Samples to features, deterministically in the sample stream."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop: int = HOP,
        window: int = WINDOW,
    ) -> None:
        self.sample_rate = max(int(sample_rate), 4_000)
        self.hop = max(int(hop), 64)
        self.window = max(int(window), self.hop)
        self._dt = self.hop / self.sample_rate

        self._hann = np.hanning(self.window).astype(np.float32)
        # Scaled so an in-band sine of amplitude A sums to ~A across its
        # bins, which puts the band features in amplitude units before
        # normalisation.
        self._spec_gain = 2.0 / float(np.sum(self._hann))
        freqs = np.fft.rfftfreq(self.window, 1.0 / self.sample_rate)
        self._band_masks = [
            (freqs >= lo) & (freqs < hi)
            for lo, hi in (BASS_BAND, MID_BAND, TREBLE_BAND)
        ]

        self._buf = np.zeros(self.window, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._ref = REF_FLOOR
        self._env = [0.0, 0.0, 0.0, 0.0]  # level, bass, mid, treble
        self._flux_env = 0.0
        self._onset_env = 0.0
        self._prev_log_spec: np.ndarray | None = None
        self._flux_history: collections.deque[float] = collections.deque(
            maxlen=FLUX_HISTORY_HOPS
        )
        self._quiet_seconds = SILENCE_SECONDS  # born silent, not born loud
        self._hops_since_onset = 10**9
        self._onset_interval = math.inf  # EMA of seconds between onsets
        self._pace_env = 0.0
        self._slow_level = 0.0  # the reference `waning` measures against
        self._wane_env = 0.0
        self._was_under_gate = True
        self._cut_level = 0.0  # the fast level when the gate last engaged
        self._cut_fired_this_quiet = False
        self._severed_env = 0.0
        self._refractory_hops = max(
            1, math.ceil(ONSET_REFRACTORY_SECONDS / self._dt)
        )
        self.hops = 0
        self.onsets = 0
        self._features = AudioFeatures()

    # -- public -------------------------------------------------------------

    @property
    def features(self) -> AudioFeatures:
        return self._features

    @property
    def stream_seconds(self) -> float:
        """How much audio has been consumed, in seconds of *stream* time.

        Sample count over sample rate, so it advances only when audio does --
        the clock-free timebase everything drive-side that needs "how long
        since" measures against (§3, extended by §16.3).
        """
        return self.hops * self._dt

    def push(self, samples: np.ndarray) -> AudioFeatures:
        """Consume any amount of audio; return the latest features.

        Accepts mono ``(n,)`` or interleaved ``(n, channels)`` float arrays.
        Partial hops carry over to the next call, so chunking is invisible:
        the same stream in different block sizes produces the same feature
        sequence, which is what keeps the extractor a function of the stream
        rather than of whoever is feeding it.
        """
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1, dtype=np.float32)
        elif data.ndim != 1:
            data = data.reshape(-1)
        # Sanitisation at the door (§16.3): a broken driver's NaNs become
        # zeros and absurd amplitudes are clamped, so a hostile stream is at
        # worst a boring one.
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(data, -4.0, 4.0, out=data)

        if self._pending.size:
            data = np.concatenate([self._pending, data])
        whole = (data.size // self.hop) * self.hop
        for start in range(0, whole, self.hop):
            self._process_hop(data[start : start + self.hop])
        self._pending = data[whole:].copy()
        return self._features

    # -- the hop ------------------------------------------------------------

    def _process_hop(self, hop: np.ndarray) -> None:
        self.hops += 1
        self._hops_since_onset += 1
        n = self.hop
        self._buf[:-n] = self._buf[n:]
        self._buf[-n:] = hop

        # AC-coupled RMS: DC from a miswired capture path is not loudness.
        ac = hop - float(hop.mean())
        rms = float(np.sqrt(np.mean(ac * ac)))

        # The decaying-peak AGC reference (§16.3), floored so silence cannot
        # be amplified into signal.
        self._ref = max(
            self._ref * math.exp(-self._dt / REF_TAU), rms, REF_FLOOR
        )
        under_gate = rms < SILENCE_RMS
        if under_gate:
            self._quiet_seconds = min(
                self._quiet_seconds + self._dt, SILENCE_SECONDS * 10.0
            )
        else:
            self._quiet_seconds = 0.0

        mag = (
            np.abs(np.fft.rfft(self._buf * self._hann)).astype(np.float32)
            * self._spec_gain
        )

        # Continuous path: AGC-normalised, soft-compressed, followed.
        if under_gate:
            targets = (0.0, 0.0, 0.0, 0.0)
        else:
            band_amp = [
                float(np.sqrt(np.sum(np.square(mag[m])))) if m.any() else 0.0
                for m in self._band_masks
            ]
            targets = (
                _squash(rms / self._ref),
                _squash(2.0 * band_amp[0] / self._ref),
                _squash(2.0 * band_amp[1] / self._ref),
                _squash(2.0 * band_amp[2] / self._ref),
            )
        for i, target in enumerate(targets):
            self._env[i] = _follow(self._env[i], target, self._dt)

        # Transient path: flux on *log* magnitudes, unnormalised — the AGC
        # that makes quiet and loud equivalent for the followers erases
        # exactly the amplitude steps this path exists to see (§16.8, step 1
        # record). The adaptive median threshold absorbs level differences
        # instead.
        log_spec = np.log1p(mag * (1.0 / REF_FLOOR))
        if self._prev_log_spec is None:
            flux_raw = 0.0
        else:
            diff = log_spec - self._prev_log_spec
            flux_raw = float(np.sum(diff[diff > 0.0]))
        self._prev_log_spec = log_spec
        flux = _squash_flux(flux_raw)
        self._flux_history.append(flux)

        fired = 0.0
        if not under_gate and self._hops_since_onset >= self._refractory_hops:
            threshold = (
                FLUX_THRESHOLD_RATIO * statistics.median(self._flux_history)
                + FLUX_THRESHOLD_BIAS
            )
            if flux > threshold:
                fired = min(1.0, flux / threshold - 1.0)
                # The gap since the previous onset feeds the tempo estimate
                # -- unless it is long enough to be a pause rather than a
                # beat interval, which says nothing about tempo.
                gap = self._hops_since_onset * self._dt
                if gap < PACE_GAP_CAP_SECONDS:
                    if math.isfinite(self._onset_interval):
                        self._onset_interval += PACE_EMA_ALPHA * (
                            gap - self._onset_interval
                        )
                    else:
                        self._onset_interval = gap
                self._hops_since_onset = 0
                self.onsets += 1

        # Gated like the continuous targets: under the gate the flux the
        # detector computed is the noise floor differencing against itself,
        # and reporting it would make "silent" and "zero" disagree.
        self._flux_env = _follow(
            self._flux_env, 0.0 if under_gate else flux, self._dt
        )
        # The tempo estimate. The live gap is folded in through max(), so a
        # lull reads as slower immediately rather than when the next onset
        # finally lands; the gate uses the *sustained* silence clock rather
        # than one quiet hop, because the space between beats is not
        # silence.
        gap = self._hops_since_onset * self._dt
        effective = max(self._onset_interval, gap)
        if (
            not math.isfinite(effective)
            or self._quiet_seconds >= SILENCE_SECONDS
        ):
            pace_target = 0.0
        else:
            rate = 1.0 / max(effective, 1e-3)
            pace_target = min(
                1.0,
                max(0.0, (rate - PACE_SLOW_HZ) / (PACE_FAST_HZ - PACE_SLOW_HZ)),
            )
        self._pace_env = _follow(self._pace_env, pace_target, self._dt)
        # Instant attack, exponential release: a consumer polling slower
        # than the hop rate still sees the pulse.
        self._onset_env = max(
            self._onset_env * math.exp(-self._dt / ONSET_RELEASE_SECONDS),
            fired,
        )

        # The waning estimate: the fast level against a slow reference of
        # itself. Not gated on silence -- the tail end of a fade *is* quiet,
        # and the signal is meant to survive into it; what retires it is the
        # reference itself decaying, over WANE_REF_TAU seconds, and the
        # loudness scale falling away with it. Its follower rises at
        # WANE_ATTACK_SECONDS, not the ordinary attack, so a hard cut
        # resolves as a cut (below) before it can read as a fade.
        alpha = 1.0 - math.exp(-self._dt / WANE_REF_TAU)
        self._slow_level += alpha * (self._env[0] - self._slow_level)
        slow = self._slow_level
        receded = max(0.0, slow - self._env[0]) / slow if slow > 1e-6 else 0.0
        wane_target = receded * min(1.0, slow / WANE_LOUDNESS_FLOOR)
        tau = (
            WANE_ATTACK_SECONDS
            if wane_target > self._wane_env
            else RELEASE_SECONDS
        )
        self._wane_env += (wane_target - self._wane_env) * (
            1.0 - math.exp(-self._dt / tau)
        )

        # The cut estimate. The level follower's release lag means it still
        # carries the pre-cut loudness on the hop the gate engages, which is
        # what separates "the music was severed" from "the music arrived
        # here quietly, fading". Confirmed by a short window of sustained
        # silence so an emphatic rest inside a phrase does not count, fired
        # once per silence, re-armed when sound returns.
        if under_gate and not self._was_under_gate:
            self._cut_level = self._env[0]
        self._was_under_gate = under_gate
        if not under_gate:
            self._cut_fired_this_quiet = False
        self._severed_env *= math.exp(-self._dt / SEVERED_RELEASE_SECONDS)
        if (
            not self._cut_fired_this_quiet
            and self._quiet_seconds >= CUT_CONFIRM_SECONDS
            and self._cut_level >= CUT_LOUDNESS_FLOOR
            and slow >= CUT_LOUDNESS_FLOOR
        ):
            self._cut_fired_this_quiet = True
            self._severed_env = max(self._severed_env, min(self._cut_level, 1.0))

        self._features = AudioFeatures(
            level=_unit(self._env[0]),
            bass=_unit(self._env[1]),
            mid=_unit(self._env[2]),
            treble=_unit(self._env[3]),
            flux=_unit(self._flux_env),
            onset=_unit(self._onset_env),
            pace=_unit(self._pace_env),
            waning=_unit(self._wane_env),
            severed=_unit(self._severed_env),
            silent=self._quiet_seconds >= SILENCE_SECONDS,
        )


def _follow(current: float, target: float, dt: float) -> float:
    """Asymmetric envelope follower: quick up, slow down (§16.3)."""
    tau = ATTACK_SECONDS if target > current else RELEASE_SECONDS
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha


def _squash(x: float) -> float:
    """Soft-compress a non-negative ratio into [0, 1)."""
    return 1.0 - math.exp(-1.5 * max(x, 0.0))


def _squash_flux(x: float) -> float:
    return x / (x + 6.0) if x > 0.0 else 0.0


def _unit(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, float(x)))


# --------------------------------------------------------------------------
# Modulation — §16.4's first door, over a whitelist
# --------------------------------------------------------------------------

# Every parameter the audio drive may move, and the *only* parameters it may
# move. §14.1's channels, verbatim: motion (the flow the pigment rides and
# the network is sheared by), colour (chroma-budget spend, luminance-free by
# Oklab construction), and material (how fertile the weather is). What is
# absent is the point -- no agent parameter (§16.1: the organism keeps its
# autonomy), no luminance-architecture path, nothing in `safety.*` -- and
# test_the_whitelist_excludes_the_luminance_architecture holds the absence.
MODULATED_PATHS: tuple[str, ...] = (
    "flow.psi_gain",
    "flow.field_gain",
    # The simulation's own tempo, paced by the music's (the `pace` feature):
    # how fast the weather changes its mind and how fast regimes migrate.
    # Both are stable-form updates -- an EMA coefficient and a backward
    # semi-Lagrangian sample -- so the bound on their modulation is
    # perceptual, not numerical. `sim_hz` is deliberately not here (it is
    # frame pacing, owned by the budget governor), and neither is anything
    # of the agents'.
    "flow.psi_theta",
    "climate.advect_gain",
    "pigment.inject_rate",
    "render.chroma_activity_gain",
    "render.c_max",
    "render.hue_turns_per_hour",
)

# What an onset asks the scheduler for, by the band carrying the moment: the
# low end breathes structure, the mids shift the current the structure rides,
# the highs recolour. All three are the constructive kinds -- a beat should
# never be a dieback, and a rift on a drop is a §16.8 step 5 judgement for
# eyes, not a default.
ONSET_EVENT_KINDS = ("bloom", "current", "tint")

# What the *fade door* asks for (§16.4): the music receding is the moment a
# dieback fits -- the field thinning as the song goes. Never through the
# onset door.
FADE_EVENT_KIND = "dieback"

# And what the *cut door* asks for: a hard cut is the music being severed,
# and severance is what a rift is. Each destructive kind has exactly one
# musical gesture: fading -> thinning, severed -> torn. The drop stays
# constructive (a bloom, through the onset door).
CUT_EVENT_KIND = "rift"
SEVERED_ASK_FLOOR = 0.3


@dataclasses.dataclass(frozen=True)
class EventAsk:
    """One onset's request: which kind, and how the music shapes it.

    ``vigor`` and ``pace`` are hints in [0, 1] that the scheduler's
    ``trigger`` honours strictly inside the ranges its own RNG samples --
    a shaped event is never bigger, stronger, or faster-building than a
    randomly drawn one could have been (§16.4).
    """

    kind: str
    vigor: float
    pace: float


def modulate(
    params: "config_module.Params", features: AudioFeatures
) -> "config_module.Params":
    """The effective parameters for one frame, under what the room is doing.

    A pure function, applied after the ramp (its own speed limit is the
    §16.3 followers, which are allowed to be faster than the 8 s ramp) and
    before the engine. Three properties the tests hold it to:

    * **Identity at zero.** Features at the zero record return ``params``
      unchanged -- silence under resonance is the plain instrument (§16.1).
    * **The whitelist is the reach.** Only :data:`MODULATED_PATHS` differ
      between input and output, whatever the features say.
    * **Bounded by the standing arguments.** Multiplicative levers are capped
      by the gain ceilings in ``SAFETY_CEILINGS`` (the motion product stays
      inside the §14.8 step 2 sweep certificate), ``c_max`` approaches its
      own ceiling and never passes it, and the flash limiter is downstream
      of all of it as ever.

    ``params`` is not mutated: the touched sub-blocks are replaced, so the
    ramp's own state -- which is what the caller usually hands in -- is never
    written through.
    """
    audio = params.audio
    # Bass leads the motion and treble leads the colour, with overall level
    # behind both, so a bass-heavy passage surges more than it saturates and
    # a bright one saturates more than it surges.
    motion_drive = min(1.0, 0.6 * features.bass + 0.4 * features.level)
    colour_drive = min(1.0, 0.55 * features.treble + 0.45 * features.level)
    if (
        motion_drive <= 0.0
        and colour_drive <= 0.0
        and features.mid <= 0.0
        and features.treble <= 0.0
        and features.flux <= 0.0
        and features.pace <= 0.0
    ):
        return params

    motion = 1.0 + audio.motion_gain * motion_drive
    tempo = 1.0 + audio.tempo_gain * features.pace
    flow = dataclasses.replace(
        params.flow,
        psi_gain=params.flow.psi_gain * motion,
        field_gain=params.flow.field_gain * motion,
        psi_theta=params.flow.psi_theta * tempo,
    )
    climate = dataclasses.replace(
        params.climate,
        advect_gain=params.climate.advect_gain * tempo,
    )
    pigment = dataclasses.replace(
        params.pigment,
        # A mixing fraction, so its own ceiling is 1 whatever the gain says.
        inject_rate=min(
            params.pigment.inject_rate * (1.0 + audio.material_gain * features.mid),
            1.0,
        ),
    )
    c_ceiling = config_module.SAFETY_CEILINGS["render.c_max"][1]
    render = dataclasses.replace(
        params.render,
        chroma_activity_gain=params.render.chroma_activity_gain
        * (1.0 + audio.colour_gain * colour_drive),
        # Toward the ceiling, never past it: the same "approach, don't
        # reach" posture the activation curve takes (§14.8 step 3).
        c_max=min(
            params.render.c_max
            + (c_ceiling - params.render.c_max)
            * min(audio.colour_gain, 1.0)
            * features.treble,
            c_ceiling,
        ),
        hue_turns_per_hour=params.render.hue_turns_per_hour
        * (1.0 + audio.hue_gain * features.flux),
    )
    return dataclasses.replace(
        params, flow=flow, climate=climate, pigment=pigment, render=render
    )


def autonomous_arrivals(
    events: "config_module.EventParams", features: AudioFeatures
) -> "config_module.EventParams":
    """The event parameters the scheduler's *own* drawing should run under.

    While the room is playing, the arrivals are the music's (§16.4): the
    field's first shipped weeks taught this the direct way, when a §4.3
    dieback of the scheduler's own arrived mid-track and read as the
    visualizer misreading the music. So under sound this gates ``enabled``
    off -- which by the scheduler's documented semantics stops only its own
    drawing: onset, fade and cut asks still land through ``trigger``, and
    events already in flight finish their envelopes rather than being
    stranded. In silence it returns ``events`` untouched, and the field's
    own weather comes back: the §16.1 overlay contract, now applied to the
    one driver that would otherwise speak over the music.
    """
    if features.silent:
        return events
    return dataclasses.replace(events, enabled=False)


# --------------------------------------------------------------------------
# Device choice — §16.2, as a pure function so the order is testable
# --------------------------------------------------------------------------


def pick_capture_device(
    devices: list[dict],
    default_input: int | None,
    platform: str | None = None,
) -> tuple[int | None, str]:
    """Choose an input device, preferring the machine's own output.

    ``devices`` is shaped like ``sounddevice.query_devices()``: one mapping
    per device, in index order, with at least ``name`` and
    ``max_input_channels``. Returns ``(index, why)``; index is ``None`` when
    nothing can record at all. ``platform`` is accepted for symmetry with
    the §16.2 table but the vocabulary is small enough to search everywhere:
    a Linux box with BlackHole in the list has earned the surprise.
    """
    del platform
    inputs = [
        (i, d)
        for i, d in enumerate(devices)
        if int(d.get("max_input_channels", 0) or 0) > 0
    ]
    if not inputs:
        return None, "no input devices"

    for marker in LOOPBACK_NAMES:
        for i, d in inputs:
            if marker in str(d.get("name", "")).lower():
                return i, f"output loopback ({marker})"

    if default_input is not None and 0 <= int(default_input) < len(devices):
        d = devices[int(default_input)]
        if int(d.get("max_input_channels", 0) or 0) > 0:
            return int(default_input), "default input (microphone fallback)"

    i, _ = inputs[0]
    return i, "first input device (microphone fallback)"


def pick_loopback_microphone(mics, default_speaker_name, requested=None):
    """Choose a WASAPI loopback source from ``soundcard``'s microphone list.

    The Windows half of §16.2: PortAudio cannot capture what the machine is
    playing there, and the ``soundcard`` package can — its
    ``all_microphones(include_loopback=True)`` presents every output as a
    recordable device flagged ``isloopback``, named after the speaker it
    shadows. Pure over the ``name``/``isloopback`` attributes so the order
    is testable without a Windows machine in the room.

    Preference: a loopback matching ``requested`` (the config's
    ``audio_device``) when one is named; else the loopback shadowing the
    default speaker; else the first loopback. ``None`` — no loopbacks, or a
    request nothing here matches — means fall through to the PortAudio path,
    where the request gets its second chance against ordinary devices.
    """
    loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
    if requested is not None:
        needle = str(requested).lower()
        for mic in loopbacks:
            if needle in str(getattr(mic, "name", "")).lower():
                return mic
        return None
    if not loopbacks:
        return None
    wanted = str(default_speaker_name or "").lower()
    if wanted:
        for mic in loopbacks:
            name = str(getattr(mic, "name", "")).lower()
            if wanted in name or name in wanted:
                return mic
    return loopbacks[0]


# --------------------------------------------------------------------------
# Capture — §16.2, with every degradation path a status line
# --------------------------------------------------------------------------


class AudioDrive:
    """Owns capture and hands the render thread features on request.

    A capture thread appends raw blocks to a bounded deque; ``poll()``, on
    the render thread, drains it through the extractor. Between the two
    there is no lock the render thread can be made to wait on, so a wedged
    audio backend can starve the *features*, never the frame loop
    (§16.6(5)).

    Two capture backends, tried in order (§16.2). On Windows, the
    ``soundcard`` package first: it speaks WASAPI loopback directly, which
    is the machine's own output and the thing PortAudio cannot record
    there; its API is a blocking recorder rather than a callback, so this
    class owns a pump thread for it. Everywhere — and on Windows when
    ``soundcard`` is missing or finds no loopback — ``sounddevice``
    (PortAudio) with the name-heuristic device pick: monitors on
    PulseAudio/PipeWire, "Stereo Mix" where Windows drivers offer it,
    BlackHole on a routed Mac, then the microphone.

    Every way capture can be unavailable — the ``[audio]`` extra not
    installed, no device, a failed open, a stream that died — resolves to
    ``describe()`` saying so and ``poll()`` returning zeros. Automatic
    reopen after a mid-session device death is deliberately deferred to
    §16.8 step 2, where there is real hardware to test it against.
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        _sd=None,
        _sc=None,
        _platform: str | None = None,
    ) -> None:
        self._requested_device = device
        self._sample_rate = sample_rate
        self._sd = _sd  # injected by tests; lazily imported otherwise
        self._sc = _sc  # likewise, for the Windows loopback backend
        self._platform = _platform or sys.platform
        self._blocks: collections.deque[np.ndarray] = collections.deque(
            maxlen=QUEUE_BLOCKS
        )
        self._stream = None
        self._capture_thread: threading.Thread | None = None
        self._capture_stop = threading.Event()
        self._status = "not started"
        self._callback_faults = 0
        # All in stream seconds, never wall clock.
        self._last_event_ask = -math.inf
        self._last_fade_ask = -math.inf
        self._last_cut_ask = -math.inf
        self.extractor = FeatureExtractor(sample_rate)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> bool:
        """Open the best available capture route. Never raises."""
        if self._start_soundcard():
            return True
        return self._start_sounddevice()

    def _start_soundcard(self) -> bool:
        """The WASAPI loopback route, Windows only. False means fall through.

        Falling through is this method's whole error handling: nothing here
        writes a failure into ``describe()``, because the PortAudio route is
        still to be tried and gets to say what finally happened.
        """
        if self._platform != "win32":
            return False
        sc = self._sc
        if sc is None:
            try:
                import soundcard as sc  # type: ignore[no-redef]
            except Exception:
                log.info(
                    "audio: the soundcard package is not importable; "
                    "falling back to PortAudio capture"
                )
                return False
            self._sc = sc
        try:
            try:
                speaker_name = str(sc.default_speaker().name)
            except Exception:  # no default output is not fatal to loopback
                speaker_name = ""
            mics = list(sc.all_microphones(include_loopback=True))
            mic = pick_loopback_microphone(
                mics, speaker_name, self._requested_device)
            if mic is None:
                log.info(
                    "audio: no WASAPI loopback matched; falling back to "
                    "PortAudio capture"
                )
                return False
            rate = int(self._sample_rate)
            self.extractor = FeatureExtractor(rate)
            self._capture_stop.clear()
            why = (
                "configured audio.device"
                if self._requested_device is not None
                else "WASAPI loopback"
            )
            # Status before the thread: the pump reports its own death, and
            # a recorder that dies on its very first pull must not have that
            # report overwritten by this line landing second.
            self._status = f"listening to {mic.name} — {why}"
            self._capture_thread = threading.Thread(
                target=self._pump_soundcard,
                args=(mic, rate),
                name="anastomosis-audio",
                daemon=True,
            )
            self._capture_thread.start()
            log.info("audio: %s at %d Hz", self._status, rate)
            return True
        except Exception as exc:
            self._teardown()
            log.info(
                "audio: WASAPI loopback capture failed (%s); falling back "
                "to PortAudio capture", exc,
            )
            return False

    def _pump_soundcard(self, mic, rate: int) -> None:
        """The loopback recorder loop, on its own thread.

        ``soundcard`` records by blocking pull rather than callback, so this
        thread does what PortAudio's callback thread does for the other
        route: copy blocks into the bounded deque and nothing else. It holds
        no lock the render thread takes; a wedged recorder starves the
        features, never the frame loop (§16.6(5)).
        """
        try:
            channels = max(1, min(2, int(getattr(mic, "channels", 2) or 2)))
            with mic.recorder(
                samplerate=rate, channels=channels, blocksize=HOP
            ) as recorder:
                while not self._capture_stop.is_set():
                    block = recorder.record(numframes=HOP)
                    self._blocks.append(np.asarray(block, dtype=np.float32))
        except Exception as exc:
            if not self._capture_stop.is_set():
                self._status = (
                    f"capture stream stopped ({exc}); the field is on its own"
                )
                log.warning("audio: %s", self._status)

    def _start_sounddevice(self) -> bool:
        """The PortAudio route — every platform's fallback, most's first."""
        sd = self._sd
        if sd is None:
            try:
                import sounddevice as sd  # type: ignore[no-redef]
            except Exception:
                self._status = (
                    "no capture backend; install the [audio] extra "
                    "(pip install 'anastomosis[audio]')"
                )
                log.info("audio: %s", self._status)
                return False
            self._sd = sd

        try:
            devices = list(sd.query_devices())
            default = getattr(sd.default, "device", None)
            default_input = None
            if isinstance(default, (tuple, list)) and default:
                default_input = default[0]
            elif isinstance(default, int):
                default_input = default
            if default_input is not None and int(default_input) < 0:
                default_input = None

            if self._requested_device is not None:
                index: int | None = self._resolve_requested(devices)
                why = "configured audio.device"
            else:
                index, why = pick_capture_device(devices, default_input)
            if index is None:
                self._status = "no capture device found; the field is on its own"
                log.info("audio: %s", self._status)
                return False

            info = devices[index]
            rate = int(float(info.get("default_samplerate") or 0) or self._sample_rate)
            channels = min(2, max(1, int(info.get("max_input_channels", 1) or 1)))
            self.extractor = FeatureExtractor(rate)
            self._stream = sd.InputStream(
                device=index,
                channels=channels,
                samplerate=rate,
                blocksize=HOP,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            name = str(info.get("name", index))
            self._status = f"listening to {name} — {why}"
            log.info("audio: %s at %d Hz, %d channel(s)", self._status, rate, channels)
            return True
        except Exception as exc:
            self._teardown()
            self._status = f"audio capture failed: {exc}"
            log.warning("audio: %s", self._status)
            return False

    def _resolve_requested(self, devices: list[dict]) -> int | None:
        wanted = self._requested_device
        if isinstance(wanted, int):
            return wanted if 0 <= wanted < len(devices) else None
        needle = str(wanted).lower()
        for i, d in enumerate(devices):
            if needle in str(d.get("name", "")).lower() and int(
                d.get("max_input_channels", 0) or 0
            ) > 0:
                return i
        return None

    def stop(self) -> None:
        self._teardown()
        self._status = "stopped"

    def _teardown(self) -> None:
        self._capture_stop.set()
        thread, self._capture_thread = self._capture_thread, None
        if thread is not None and thread.is_alive():
            # The recorder blocks for at most a hop (~21 ms) per pull, so
            # this join is short; the timeout is for a wedged driver, which
            # a daemon thread must not let hold up shutdown.
            thread.join(timeout=2.0)
        stream, self._stream = self._stream, None
        if stream is not None:
            for step in (stream.stop, stream.close):
                try:
                    step()
                except Exception:  # a dying backend must not take the app
                    pass

    # -- data path ----------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        # PortAudio's thread: copy out and leave. The deque append is atomic
        # and bounded; everything else waits for poll() on the render thread.
        if status:
            self._callback_faults += 1
        self._blocks.append(np.array(indata, dtype=np.float32, copy=True))

    def poll(self) -> AudioFeatures:
        """Drain captured blocks through the extractor; return the latest.

        Called from the render thread, ideally once per frame. On an empty
        queue (or no stream at all) it returns the last features, which for
        a stream that has gone quiet are already decaying toward the zero
        record.
        """
        while True:
            try:
                block = self._blocks.popleft()
            except IndexError:
                break
            self.extractor.push(block)
        stream = self._stream
        if stream is not None and not getattr(stream, "active", True):
            self._status = "capture stream stopped; the field is on its own"
        # The loopback pump reports its own death with the exception that
        # caused it; this covers only a thread that ended without one.
        thread = self._capture_thread
        if (
            thread is not None
            and not thread.is_alive()
            and not self._capture_stop.is_set()
            and self._status.startswith("listening")
        ):
            self._status = "capture stream stopped; the field is on its own"
        return self.extractor.features

    def event_request(self, audio: "config_module.AudioParams") -> "EventAsk | None":
        """The event an onset is asking for right now, if any.

        §16.4's second door, drive-side half: decides *whether* to ask,
        *which* kind (by the band carrying the moment), and how the music
        wants it *shaped* -- ``vigor`` from the onset's own strength with the
        level behind it, so a harder hit is a stronger event, and ``pace``
        from the tempo estimate, so a faster track gets brisker envelopes.
        Everything about what the event then is belongs to the scheduler's
        ``trigger``, which honours the shape only inside the ranges its own
        RNG already samples. Asks are spaced by ``onset_spacing`` seconds of
        stream time -- sample count, never wall clock -- so a 140 BPM track
        asks a few times a minute and lets the concurrency cap do the rest,
        instead of producing a per-beat queue of refusals.
        """
        features = self.extractor.features
        now = self.extractor.stream_seconds

        if (
            features.onset >= audio.onset_threshold
            and now - self._last_event_ask >= audio.onset_spacing
        ):
            self._last_event_ask = now
            bloom, current, tint = ONSET_EVENT_KINDS
            if features.bass >= features.mid and features.bass >= features.treble:
                kind = bloom
            elif features.treble >= features.mid:
                kind = tint
            else:
                kind = current
            return EventAsk(
                kind=kind,
                vigor=min(1.0, 0.7 * features.onset + 0.3 * features.level),
                pace=features.pace,
            )

        # The cut door: the music severed asks for the rift. Checked before
        # the fade door, and the front end's slow waning attack guarantees
        # the order at the signal level too -- a hard cut confirms here
        # (CUT_CONFIRM_SECONDS) before its recession can read as a fade.
        # Firing stands the fade door down as well: one cut is one gesture,
        # not a rift with a dieback on its heels.
        if (
            features.severed >= SEVERED_ASK_FLOOR
            and now - self._last_cut_ask
            >= RIFT_SPACING_FACTOR * audio.onset_spacing
        ):
            self._last_cut_ask = now
            self._last_fade_ask = now
            return EventAsk(
                kind=CUT_EVENT_KIND,
                vigor=features.severed,
                pace=features.pace,
            )

        # The fade door: the music receding asks for the dieback the onset
        # door never may. Vigor is the depth of the fade -- a song going all
        # the way out earns the full gesture -- and pace is whatever tempo
        # remains, which as the music recedes is usually little: a fade's
        # dieback arrives long and slow, like the fade. Spaced further apart
        # than onset asks, because a fade is one musical moment.
        if (
            features.waning >= audio.fade_threshold
            and now - self._last_fade_ask
            >= FADE_SPACING_FACTOR * audio.onset_spacing
        ):
            self._last_fade_ask = now
            return EventAsk(
                kind=FADE_EVENT_KIND,
                vigor=features.waning,
                pace=features.pace,
            )
        return None

    # -- reporting ----------------------------------------------------------

    @property
    def features(self) -> AudioFeatures:
        return self.extractor.features

    def describe(self) -> str:
        """One line for the panel and the stall report (§16.6(5))."""
        return self._status
