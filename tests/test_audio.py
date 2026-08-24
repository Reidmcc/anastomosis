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

import dataclasses
import types

import numpy as np
import pytest

from anastomosis import audio, config, events


def _float_paths(obj, prefix: str = ""):
    """Every numeric leaf of a Params tree, as dotted paths."""
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(value):
            yield from _float_paths(value, f"{path}.")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield path


LOUD = audio.AudioFeatures(
    level=1.0, bass=1.0, mid=1.0, treble=1.0, flux=1.0, onset=1.0, pace=1.0,
    silent=False,
)


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
    for name in ("level", "bass", "mid", "treble", "flux", "onset", "pace"):
        assert getattr(features, name) == pytest.approx(0.0, abs=1e-3), name


def test_features_decay_to_zero_after_the_music_stops():
    """The field subsides through the release followers, then flags silent."""
    extractor = audio.FeatureExtractor()
    extractor.push(_sine(200.0, 2.0, amplitude=0.6))
    assert extractor.features.level > 0.3
    assert not extractor.features.silent

    features = extractor.push(np.zeros(audio.SAMPLE_RATE * 3, np.float32))
    assert features.silent
    for name in ("level", "bass", "mid", "treble", "onset", "pace"):
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
        for name in ("level", "bass", "mid", "treble", "flux", "onset", "pace"):
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


# ---------------------------------------------------------------------------
# Modulation -- §16.4's first door, and the three properties it promises
# ---------------------------------------------------------------------------


def test_modulation_is_the_identity_at_silence():
    """§16.1: the drive is an overlay that is identity at zero, so silence
    under resonance is the plain instrument -- the same object, untouched."""
    params = config.Config(mode="resonance").resolve()
    assert audio.modulate(params, audio.AudioFeatures()) is params


def test_modulation_touches_only_the_whitelist():
    """The whitelist is the reach: under the loudest possible features, the
    diff between input and output parameters is MODULATED_PATHS exactly --
    nothing more (a hidden lever) and nothing less (a dead entry)."""
    params = config.Config(mode="resonance").resolve()
    modulated = audio.modulate(params, LOUD)
    moved = {
        path
        for path in _float_paths(params)
        if config.get_path(params, path) != config.get_path(modulated, path)
    }
    assert moved == set(audio.MODULATED_PATHS)
    # And the input was genuinely not written through (the ramp's state is
    # what callers hand in).
    assert params == config.Config(mode="resonance").resolve()


def test_the_whitelist_excludes_the_luminance_architecture():
    """§16.4's hard rule: audio buys motion, chroma and incident -- never
    luminance, never the safety stage, never the agents. The luminance set is
    collected from the macro tables rather than written out by hand, so a
    path added to the brightness or glow macros later is covered without
    anyone remembering this test exists."""
    luminance = {
        path
        for table in config.MODE_CURVES.values()
        for macro in ("brightness", "filament_glow")
        for path, *_ in table[macro]
    }
    forbidden = set(audio.MODULATED_PATHS) & luminance
    assert not forbidden, f"audio modulates luminance paths: {sorted(forbidden)}"
    for path in audio.MODULATED_PATHS:
        assert not path.startswith("safety."), path
        assert not path.startswith("agents."), path
        assert not path.startswith("events."), path


def test_modulated_chroma_never_passes_its_ceiling():
    """c_max approaches its SAFETY_CEILINGS bound and never crosses it, even
    from the activation-top starting point with the colour gain at its own
    ceiling."""
    cfg = config.Config(
        mode="resonance",
        macros=config.Macros(intensity=1.0),
        overrides={"audio.colour_gain": 100.0},  # clamped to its ceiling
    )
    params = cfg.resolve()
    ceiling = config.SAFETY_CEILINGS["render.c_max"][1]
    assert params.audio.colour_gain == config.SAFETY_CEILINGS["audio.colour_gain"][1]
    modulated = audio.modulate(params, LOUD)
    assert modulated.render.c_max <= ceiling + 1e-9
    assert modulated.render.c_max > params.render.c_max


def test_the_modulated_motion_tops_stay_inside_the_swept_certificate():
    """§16.6(2): curve top x (1 + motion ceiling) must stay inside the 6x
    envelope the §14.8 step 2 sweep certified. A retune of the resonance
    tempo tops or a raised gain ceiling fails here until tempo_sweep.py is
    re-run and the certificate extended."""
    reg = {p: hi for p, _lo, hi, _g in config.MACRO_CURVES["tempo"]}
    res = {p: hi for p, _lo, hi, _g in config.RESONANCE_CURVES["tempo"]}
    ceiling = config.SAFETY_CEILINGS["audio.motion_gain"][1]
    for path in ("flow.psi_gain", "flow.field_gain"):
        worst = res[path] * (1.0 + ceiling)
        assert worst <= 6.0 * reg[path] + 1e-9, (
            f"{path}: modulated top {worst:.2f} exceeds the swept "
            f"{6.0 * reg[path]:.2f}"
        )


def test_modulation_survives_gains_at_their_ceilings_with_hostile_features():
    """Every output is finite whatever the gains and features say."""
    cfg = config.Config(mode="resonance", overrides={
        "audio.motion_gain": 2.0, "audio.colour_gain": 2.0,
        "audio.material_gain": 2.0, "audio.hue_gain": 2.0,
    })
    params = cfg.resolve()
    modulated = audio.modulate(params, LOUD)
    for path in audio.MODULATED_PATHS:
        value = config.get_path(modulated, path)
        assert np.isfinite(value) and value >= 0.0, path
    assert modulated.pigment.inject_rate <= 1.0


# ---------------------------------------------------------------------------
# The event door -- §16.4's second, drive-side half
# ---------------------------------------------------------------------------


def test_onset_event_kinds_are_real_and_constructive():
    """A beat is never a dieback: the kinds the drive may ask for exist in
    the scheduler's table and exclude the destructive ones."""
    assert set(audio.ONSET_EVENT_KINDS) <= set(events.EVENT_KINDS)
    assert "dieback" not in audio.ONSET_EVENT_KINDS
    assert "rift" not in audio.ONSET_EVENT_KINDS


def test_event_requests_are_spaced_in_stream_time_and_typed_by_band():
    """One ask per onset_spacing seconds of *stream* time -- sample count,
    not wall clock -- and the kind follows the band carrying the moment."""
    rate = audio.SAMPLE_RATE
    drive = audio.AudioDrive()
    gains = config.AudioParams(onset_threshold=0.05, onset_spacing=2.0)

    # A bass hit out of silence: one ask, of the structural kind, carrying
    # the hit's own strength as vigor.
    drive.extractor.push(np.zeros(rate, np.float32))
    drive.extractor.push(_sine(60.0, 0.4, amplitude=0.9, rate=rate))
    ask = drive.event_request(gains)
    assert ask is not None and ask.kind == "bloom"
    assert 0.0 < ask.vigor <= 1.0
    assert 0.0 <= ask.pace <= 1.0
    # The onset envelope is still up, but the spacing gate holds.
    assert drive.event_request(gains) is None

    # Two stream-seconds later, a treble hit asks again, recoloured.
    drive.extractor.push(_sine(60.0, 2.1, amplitude=0.05, rate=rate))
    drive.extractor.push(_sine(6000.0, 0.4, amplitude=0.9, rate=rate))
    ask = drive.event_request(gains)
    assert ask is not None and ask.kind == "tint"

    # No onset, no ask, however long the stream runs.
    drive.extractor.push(_sine(6000.0, 3.0, amplitude=0.9, rate=rate))
    assert drive.event_request(gains) is None


# ---------------------------------------------------------------------------
# The Windows route -- WASAPI loopback through `soundcard`, without Windows
# ---------------------------------------------------------------------------


class _Mic:
    def __init__(self, name, isloopback=True, channels=2):
        self.name = name
        self.isloopback = isloopback
        self.channels = channels

    def recorder(self, samplerate, channels=None, blocksize=None):
        return _Recorder(samplerate)


class _Recorder:
    """A blocking recorder that plays an endless 100 Hz tone."""

    def __init__(self, samplerate):
        self._rate = samplerate
        self._phase = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        import time

        time.sleep(0.001)  # a real recorder blocks; a spinning stub would lie
        t = np.arange(self._phase, self._phase + numframes, dtype=np.float64)
        self._phase += numframes
        mono = 0.6 * np.sin(2.0 * np.pi * 100.0 * t / self._rate)
        return np.stack([mono, mono], axis=1).astype(np.float32)


def _stub_sc(mics, speaker="Speakers (Realtek High Definition Audio)"):
    sc = types.SimpleNamespace()
    sc.default_speaker = lambda: types.SimpleNamespace(name=speaker)
    sc.all_microphones = lambda include_loopback=False: (
        list(mics) if include_loopback
        else [m for m in mics if not m.isloopback]
    )
    return sc


def test_the_loopback_picker_prefers_the_default_speakers_shadow():
    mics = [
        _Mic("Microphone (USB Audio)", isloopback=False),
        _Mic("Headphones (2- Arctis)", isloopback=True),
        _Mic("Speakers (Realtek High Definition Audio)", isloopback=True),
    ]
    mic = audio.pick_loopback_microphone(
        mics, "Speakers (Realtek High Definition Audio)")
    assert mic is mics[2]


def test_the_loopback_picker_honours_a_requested_name_or_stands_aside():
    mics = [
        _Mic("Speakers (Realtek)", isloopback=True),
        _Mic("Headphones (Arctis)", isloopback=True),
    ]
    assert audio.pick_loopback_microphone(
        mics, "Speakers (Realtek)", requested="arctis") is mics[1]
    # A request nothing here matches falls through to the PortAudio route,
    # where it gets its second chance -- rather than being second-guessed.
    assert audio.pick_loopback_microphone(
        mics, "Speakers (Realtek)", requested="usb turntable") is None


def test_the_loopback_picker_returns_none_when_nothing_loops_back():
    mics = [_Mic("Microphone (USB Audio)", isloopback=False)]
    assert audio.pick_loopback_microphone(mics, "Speakers") is None


def test_on_windows_the_loopback_route_wins_and_features_flow():
    """End to end through the pump thread: start picks the default
    speaker's loopback, blocks reach the extractor, stop joins the thread."""
    import time

    mics = [
        _Mic("Microphone (USB Audio)", isloopback=False),
        _Mic("Speakers (Realtek High Definition Audio)", isloopback=True),
    ]
    drive = audio.AudioDrive(_sc=_stub_sc(mics), _platform="win32")
    assert drive.start() is True
    assert "Speakers (Realtek" in drive.describe()
    assert "loopback" in drive.describe()

    features = drive.poll()
    deadline = time.monotonic() + 5.0
    while features.level == 0.0 and time.monotonic() < deadline:
        time.sleep(0.02)
        features = drive.poll()
    assert features.bass > 0.3
    assert not features.silent

    drive.stop()
    assert drive._capture_thread is None
    assert drive.describe() == "stopped"


def test_off_windows_the_portaudio_route_is_used_even_with_soundcard_present():
    """The soundcard route is Windows-only by design: monitors already serve
    Linux through PortAudio, and gating on the platform keeps one machine
    from walking two capture stacks."""
    mics = [_Mic("Monitor of Built-in Audio", isloopback=True)]
    devices = [{"name": "Monitor of Built-in Audio", "max_input_channels": 2,
                "default_samplerate": 48_000.0}]
    sd = _stub_sd(devices)
    drive = audio.AudioDrive(
        _sc=_stub_sc(mics), _sd=sd, _platform="linux")
    assert drive.start() is True
    assert len(sd.created) == 1, "PortAudio was not the route taken"
    assert drive._capture_thread is None
    drive.stop()


def test_on_windows_without_loopbacks_stereo_mix_is_the_fallback():
    """soundcard finding nothing must fall through, not fail: the PortAudio
    heuristic still knows 'Stereo Mix' when a driver offers it."""
    devices = [
        {"name": "Microphone (Realtek)", "max_input_channels": 1,
         "default_samplerate": 48_000.0},
        {"name": "Stereo Mix (Realtek)", "max_input_channels": 2,
         "default_samplerate": 48_000.0},
    ]
    sd = _stub_sd(devices)
    drive = audio.AudioDrive(
        _sc=_stub_sc([_Mic("Microphone", isloopback=False)]),
        _sd=sd, _platform="win32")
    assert drive.start() is True
    assert "Stereo Mix" in drive.describe()
    drive.stop()


def test_a_broken_soundcard_backend_still_reaches_portaudio():
    """An exception inside the loopback route is a fall-through with a log
    line, never the drive's final word."""
    sc = types.SimpleNamespace()

    def explode(*a, **k):
        raise RuntimeError("COM said no")

    sc.default_speaker = explode
    sc.all_microphones = explode
    devices = [{"name": "Microphone", "max_input_channels": 1,
                "default_samplerate": 44_100.0}]
    drive = audio.AudioDrive(_sc=sc, _sd=_stub_sd(devices), _platform="win32")
    assert drive.start() is True
    assert "Microphone" in drive.describe()
    drive.stop()


def test_a_dead_pump_thread_becomes_a_status_line():
    """A recorder that dies mid-session (§16.6(5)): the pump writes what
    happened into the status the panel and the stall report read."""
    import time

    class _DyingMic(_Mic):
        def recorder(self, samplerate, channels=None, blocksize=None):
            raise RuntimeError("device invalidated")

    drive = audio.AudioDrive(
        _sc=_stub_sc([_DyingMic("Speakers (Realtek)", isloopback=True)]),
        _platform="win32")
    assert drive.start() is True  # the open is asynchronous; death is later
    deadline = time.monotonic() + 5.0
    while "stopped" not in drive.describe() and time.monotonic() < deadline:
        time.sleep(0.01)
    drive.poll()
    assert "stopped" in drive.describe()
    assert "device invalidated" in drive.describe()
    drive.stop()


# ---------------------------------------------------------------------------
# The tempo estimate -- pace from the beat, in stream time (§16.4)
# ---------------------------------------------------------------------------


def _beat_train(period: float, seconds: float,
                rate: int = audio.SAMPLE_RATE) -> np.ndarray:
    """A 440 Hz carrier pulsing to 0.9 for 50 ms at each period start,
    resting at 0.05 between -- beats over a quiet bed, not beats over
    silence, so the silence gate stays out of the picture."""
    t = np.arange(int(seconds * rate), dtype=np.float64)
    carrier = np.sin(2.0 * np.pi * 440.0 * t / rate)
    phase = np.mod(t / rate, period)
    amplitude = np.where(phase < 0.05, 0.9, 0.05)
    return (carrier * amplitude).astype(np.float32)


def test_pace_tracks_the_beat_rate():
    """A fast train reads fast, a slow one slow, and the mapping is
    monotone between them -- the tempo estimate the envelopes ride."""
    fast = audio.FeatureExtractor().push(_beat_train(0.3, 8.0))
    slow = audio.FeatureExtractor().push(_beat_train(1.5, 8.0))
    assert fast.pace > 0.5, fast
    assert slow.pace < 0.25, slow
    assert fast.pace > slow.pace


def test_pace_subsides_in_a_lull_without_waiting_for_the_next_onset():
    """The live gap pulls the estimate down: when the beat stops, pace
    falls, even though no new onset ever arrives to update the average."""
    extractor = audio.FeatureExtractor()
    extractor.push(_beat_train(0.3, 8.0))
    busy = extractor.features.pace
    extractor.push(_sine(440.0, 10.0, amplitude=0.05))
    assert extractor.features.pace < 0.5 * busy


def test_pace_is_zero_before_any_second_onset():
    """One onset is an arrival, not a tempo."""
    extractor = audio.FeatureExtractor()
    extractor.push(np.zeros(audio.SAMPLE_RATE, np.float32))
    extractor.push(_sine(440.0, 1.0, amplitude=0.8))
    assert extractor.features.pace == pytest.approx(0.0, abs=1e-6)
