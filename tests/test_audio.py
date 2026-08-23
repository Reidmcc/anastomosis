"""The audio front end -- DESIGN.md §16, step 1.

The extractor is the load-bearing piece: everything the resonance mode will
ever do to the image passes through the handful of floats it emits, so these
tests pin the properties the design leans on rather than the numbers. Bounded
always, whatever the stream says (§16.3's sanitisation); deterministic in the
sample stream and indifferent to chunking (§3's discipline, extended); quiet
and loud sources equivalent after the AGC; silence a first-class state that
the features actually reach; onsets that fire on musical change, once, and
not on steady state. The capture side is tested for the §16.2 rule that
matters before any hardware exists: every way audio can be unavailable is a
status line, never an exception.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from anastomosis import audio


def _sine(freq: float, seconds: float, amplitude: float = 0.5,
          rate: int = audio.SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float64)
    return (amplitude * np.sin(2.0 * np.pi * freq * t / rate)).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# The extractor: silence
# ---------------------------------------------------------------------------


def test_born_silent_and_zero():
    """Before any audio arrives the record is the identity (§16.1)."""
    features = audio.FeatureExtractor().features
    assert features == audio.AudioFeatures()
    assert features.silent


def test_silence_stays_zero_and_flagged():
    """A quiet room is zeros, not the noise floor amplified (§16.3)."""
    extractor = audio.FeatureExtractor()
    noise = (np.random.default_rng(3).standard_normal(audio.SAMPLE_RATE * 2)
             * 1e-5).astype(np.float32)
    features = extractor.push(noise)
    assert features.silent
    for name in ("level", "bass", "mid", "treble", "flux", "onset"):
        assert getattr(features, name) == pytest.approx(0.0, abs=1e-3), name


def test_features_decay_to_zero_after_the_music_stops():
    """The field subsides through the release followers, then flags silent."""
    extractor = audio.FeatureExtractor()
    extractor.push(_sine(200.0, 2.0, amplitude=0.6))
    assert extractor.features.level > 0.3
    assert not extractor.features.silent

    features = extractor.push(np.zeros(audio.SAMPLE_RATE * 3, np.float32))
    assert features.silent
    for name in ("level", "bass", "mid", "treble", "onset"):
        assert getattr(features, name) < 0.05, name


# ---------------------------------------------------------------------------
# The extractor: bands and AGC
# ---------------------------------------------------------------------------


def test_a_bass_tone_reads_as_bass():
    features = audio.FeatureExtractor().push(_sine(60.0, 2.0))
    assert features.bass > 0.5
    assert features.bass > 4.0 * features.treble
    assert features.level > 0.3
    assert not features.silent


def test_a_treble_tone_reads_as_treble():
    features = audio.FeatureExtractor().push(_sine(5_000.0, 2.0))
    assert features.treble > 0.5
    assert features.treble > 4.0 * features.bass


def test_the_agc_makes_quiet_and_loud_sources_equivalent():
    """§16.2(2): the microphone fallback and a mastered-loud loopback must
    drive the same visual range, so after convergence the normalised level
    of a quiet steady source approaches the loud one's."""
    loud = audio.FeatureExtractor().push(_sine(440.0, 6.0, amplitude=0.9))
    quiet = audio.FeatureExtractor().push(_sine(440.0, 6.0, amplitude=0.05))
    assert loud.level > 0.4
    assert quiet.level > 0.6 * loud.level


# ---------------------------------------------------------------------------
# The extractor: onsets
# ---------------------------------------------------------------------------


def test_a_volume_step_fires_an_onset_and_steady_state_does_not():
    """The log-domain flux path exists to hear amplitude change; a steady
    tone after the step must go quiet again (§16.8 step 1 record)."""
    rate = audio.SAMPLE_RATE
    t = np.arange(int(4.2 * rate), dtype=np.float64)
    carrier = np.sin(2.0 * np.pi * 440.0 * t / rate)
    amplitude = np.where(t < 2 * rate, 0.02, 0.9)
    stream = (carrier * amplitude).astype(np.float32)

    extractor = audio.FeatureExtractor(rate)
    extractor.push(stream[: 2 * rate])
    settled = extractor.onsets

    step = 2 * rate + int(0.2 * rate)
    extractor.push(stream[2 * rate : step])
    assert extractor.onsets > settled
    assert extractor.features.onset > 0.0

    after_step = extractor.onsets
    extractor.push(stream[step:])
    assert extractor.onsets == after_step


def test_the_refractory_suppresses_a_double_fire():
    """Two transients inside the refractory window are one onset."""
    rate = audio.SAMPLE_RATE
    extractor = audio.FeatureExtractor(rate)
    extractor.push(np.zeros(rate, np.float32))
    base = extractor.onsets

    burst = _sine(440.0, 0.03, amplitude=0.9)
    gap = np.zeros(int(0.06 * rate), np.float32)
    extractor.push(np.concatenate([burst, gap, burst]))
    assert extractor.onsets - base == 1


def test_music_starting_from_silence_is_an_onset():
    extractor = audio.FeatureExtractor()
    extractor.push(np.zeros(audio.SAMPLE_RATE * 2, np.float32))
    extractor.push(_sine(300.0, 0.5, amplitude=0.7))
    assert extractor.onsets >= 1


# ---------------------------------------------------------------------------
# The extractor: the properties everything else leans on
# ---------------------------------------------------------------------------


def test_deterministic_in_the_sample_stream_whatever_the_chunking():
    """§3's discipline extended: the extractor is a function of the stream,
    not of wall time and not of whoever is feeding it."""
    rng = np.random.default_rng(11)
    stream = (rng.standard_normal(audio.SAMPLE_RATE * 3) * 0.3).astype(
        np.float32
    )

    one = audio.FeatureExtractor()
    one.push(stream)

    # The same stream in ragged chunks, none aligned to the hop.
    other = audio.FeatureExtractor()
    sizes = [100, 1023, 1, 4096, 777, 30_000]
    cursor = 0
    while cursor < len(stream):
        size = sizes[cursor % len(sizes)]
        other.push(stream[cursor : cursor + size])
        cursor += size

    assert one.hops == other.hops
    assert one.features == other.features
    assert one.onsets == other.onsets


def test_hostile_input_stays_bounded():
    """NaN, infinity and absurd amplitude from a broken driver come out as
    finite features in [0, 1] (§16.3 sanitisation, §16.6(1))."""
    extractor = audio.FeatureExtractor()
    hostile = np.full(audio.SAMPLE_RATE, 1e9, np.float32)
    hostile[::7] = np.nan
    hostile[::11] = np.inf
    hostile[::13] = -np.inf
    square = np.sign(_sine(20.0, 1.0)) * 1e6

    for block in (hostile, square.astype(np.float32)):
        features = extractor.push(block)
        for name in ("level", "bass", "mid", "treble", "flux", "onset"):
            value = getattr(features, name)
            assert np.isfinite(value), name
            assert 0.0 <= value <= 1.0, name


def test_stereo_and_odd_shapes_are_accepted():
    extractor = audio.FeatureExtractor()
    stereo = np.stack([_sine(60.0, 1.0), _sine(60.0, 1.0)], axis=1)
    features = extractor.push(stereo)
    assert features.bass > 0.5


# ---------------------------------------------------------------------------
# Device choice -- §16.2's preference order, without hardware
# ---------------------------------------------------------------------------


def _device(name: str, inputs: int = 2) -> dict:
    return {"name": name, "max_input_channels": inputs}


def test_a_monitor_beats_the_default_microphone():
    devices = [
        _device("Built-in Microphone"),
        _device("Monitor of Built-in Audio Analog Stereo"),
    ]
    index, why = audio.pick_capture_device(devices, default_input=0)
    assert index == 1
    assert "loopback" in why


def test_blackhole_beats_the_default_microphone():
    devices = [_device("MacBook Pro Microphone"), _device("BlackHole 2ch")]
    index, _ = audio.pick_capture_device(devices, default_input=0)
    assert index == 1


def test_stereo_mix_beats_the_default_microphone():
    devices = [
        _device("Microphone (Realtek Audio)"),
        _device("Stereo Mix (Realtek Audio)"),
    ]
    index, _ = audio.pick_capture_device(devices, default_input=0)
    assert index == 1


def test_no_loopback_falls_back_to_the_default_input():
    devices = [_device("Speakers", inputs=0), _device("Microphone")]
    index, why = audio.pick_capture_device(devices, default_input=1)
    assert index == 1
    assert "fallback" in why


def test_no_inputs_at_all_is_none_not_an_error():
    devices = [_device("Speakers", inputs=0)]
    index, why = audio.pick_capture_device(devices, default_input=None)
    assert index is None
    assert why == "no input devices"


def test_an_output_only_monitor_is_not_chosen():
    """The marker search must respect recordability, not just names."""
    devices = [_device("Monitor of Speakers", inputs=0), _device("Microphone")]
    index, _ = audio.pick_capture_device(devices, default_input=1)
    assert index == 1


# ---------------------------------------------------------------------------
# The drive -- every unavailability is a status line, never an exception
# ---------------------------------------------------------------------------


def test_missing_backend_degrades_to_a_status_line():
    """This environment has no sounddevice on purpose: the [audio] extra is
    optional, and §16.2(3) says its absence must leave the mode standing."""
    with pytest.raises(ImportError):
        import sounddevice  # noqa: F401

    drive = audio.AudioDrive()
    assert drive.start() is False
    assert "[audio]" in drive.describe()
    assert drive.poll() == audio.AudioFeatures()


class _StubStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.kwargs = kwargs
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


def _stub_sd(devices, default_input=0, fail_open=False):
    sd = types.SimpleNamespace()
    sd.default = types.SimpleNamespace(device=[default_input, -1])
    created = []

    def query_devices(index=None):
        return devices if index is None else devices[index]

    def input_stream(**kwargs):
        if fail_open:
            raise RuntimeError("device is busy")
        stream = _StubStream(**kwargs)
        created.append(stream)
        return stream

    sd.query_devices = query_devices
    sd.InputStream = input_stream
    sd.created = created
    return sd


def test_the_drive_picks_the_loopback_and_features_flow_end_to_end():
    devices = [
        {"name": "Microphone", "max_input_channels": 1,
         "default_samplerate": 48_000.0},
        {"name": "Monitor of Built-in Audio", "max_input_channels": 2,
         "default_samplerate": 48_000.0},
    ]
    sd = _stub_sd(devices, default_input=0)
    drive = audio.AudioDrive(_sd=sd)
    assert drive.start() is True
    assert "Monitor of Built-in Audio" in drive.describe()

    stream = sd.created[0]
    block = np.stack([_sine(100.0, 1.0, 0.6), _sine(100.0, 1.0, 0.6)], axis=1)
    stream.callback(block, len(block), None, None)
    features = drive.poll()
    assert features.bass > 0.4
    assert not features.silent

    drive.stop()
    assert not stream.active
    assert drive.describe() == "stopped"


def test_a_failed_open_is_a_status_line_not_an_exception():
    devices = [{"name": "Microphone", "max_input_channels": 1,
                "default_samplerate": 44_100.0}]
    drive = audio.AudioDrive(_sd=_stub_sd(devices, fail_open=True))
    assert drive.start() is False
    assert "failed" in drive.describe()
    assert drive.poll() == audio.AudioFeatures()


def test_a_configured_device_name_wins_over_the_heuristic():
    devices = [
        {"name": "Monitor of Built-in Audio", "max_input_channels": 2,
         "default_samplerate": 48_000.0},
        {"name": "USB Turntable", "max_input_channels": 2,
         "default_samplerate": 48_000.0},
    ]
    sd = _stub_sd(devices)
    drive = audio.AudioDrive(device="turntable", _sd=sd)
    assert drive.start() is True
    assert "USB Turntable" in drive.describe()
    assert "configured" in drive.describe()


def test_a_dead_stream_becomes_a_status_line_on_poll():
    devices = [{"name": "Monitor of Built-in Audio", "max_input_channels": 2,
                "default_samplerate": 48_000.0}]
    sd = _stub_sd(devices)
    drive = audio.AudioDrive(_sd=sd)
    assert drive.start() is True
    sd.created[0].active = False
    drive.poll()
    assert "on its own" in drive.describe()
