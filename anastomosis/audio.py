"""The audio front end for the resonance mode. DESIGN.md §16.

Three pieces, split exactly where §16.2–§16.3 splits them:

* :class:`FeatureExtractor` — samples in, one :class:`AudioFeatures` record
  per hop out. Pure and deterministic in the sample stream: no wall-clock
  reads anywhere, every time constant advanced by sample count, so the same
  stream produces the same features whatever the chunking (§3's discipline,
  extended — audio is an input, not a clock).
* :func:`pick_capture_device` — the loopback-preferring device heuristic of
  §16.2, a pure function over a device list so the preference order is
  testable without an audio backend in the room.
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

import numpy as np

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
                self._hops_since_onset = 0
                self.onsets += 1

        # Gated like the continuous targets: under the gate the flux the
        # detector computed is the noise floor differencing against itself,
        # and reporting it would make "silent" and "zero" disagree.
        self._flux_env = _follow(
            self._flux_env, 0.0 if under_gate else flux, self._dt
        )
        # Instant attack, exponential release: a consumer polling slower
        # than the hop rate still sees the pulse.
        self._onset_env = max(
            self._onset_env * math.exp(-self._dt / ONSET_RELEASE_SECONDS),
            fired,
        )

        self._features = AudioFeatures(
            level=_unit(self._env[0]),
            bass=_unit(self._env[1]),
            mid=_unit(self._env[2]),
            treble=_unit(self._env[3]),
            flux=_unit(self._flux_env),
            onset=_unit(self._onset_env),
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


# --------------------------------------------------------------------------
# Capture — §16.2, with every degradation path a status line
# --------------------------------------------------------------------------


class AudioDrive:
    """Owns capture and hands the render thread features on request.

    The callback thread appends raw blocks to a bounded deque; ``poll()``,
    on the render thread, drains it through the extractor. Between the two
    there is no lock the render thread can be made to wait on, so a wedged
    audio backend can starve the *features*, never the frame loop
    (§16.6(5)).

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
    ) -> None:
        self._requested_device = device
        self._sample_rate = sample_rate
        self._sd = _sd  # injected by tests; lazily imported otherwise
        self._blocks: collections.deque[np.ndarray] = collections.deque(
            maxlen=QUEUE_BLOCKS
        )
        self._stream = None
        self._status = "not started"
        self._callback_faults = 0
        self.extractor = FeatureExtractor(sample_rate)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> bool:
        """Open the best available capture device. Never raises."""
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
        return self.extractor.features

    # -- reporting ----------------------------------------------------------

    @property
    def features(self) -> AudioFeatures:
        return self.extractor.features

    def describe(self) -> str:
        """One line for the panel and the stall report (§16.6(5))."""
        return self._status
