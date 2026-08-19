"""Long-duration properties: non-repetition, numerical survival, no leaks.

These are the tests that speak to the actual use case. A visual that is
beautiful for ten minutes and dead, saturated, or subtly looping after six
hours would fail the brief completely, and none of that is visible in a short
interactive check.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
import wgpu

import morphology
from anastomosis import config, engine as engine_module, events, gpu_params, shaders

MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Non-repetition, tested at the mechanism rather than by observation
# ---------------------------------------------------------------------------


def test_no_shader_treats_time_as_a_float_phase():
    """The central rule from DESIGN.md §3, enforced statically.

    Slow variation must come from integrating increments into a stored field,
    never from evaluating a function of a clock. A float-valued tick would both
    reintroduce periodicity and quantise visibly: at 30 Hz, one day of runtime
    puts an f32 at ~0.008 resolution.

    A grep is crude, but this is a property that is easy to reintroduce by
    accident and effectively impossible to notice by looking at the output for
    an hour.
    """
    offenders = []
    pattern = re.compile(
        r"f32\s*\(\s*(params\.tick|render\.frame|params\.seed|render\.seed)\s*\)"
    )
    for name in shaders.all_shader_names():
        source = shaders.load(name)
        for lineno, line in enumerate(source.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "tick/seed converted to float, which makes the result a function of "
        "time rather than a stateful process:\n  " + "\n  ".join(offenders)
    )


def test_no_shader_uses_a_trig_function_of_the_tick():
    """The same rule, catching the `sin(t)` formulation directly."""
    offenders = []
    pattern = re.compile(r"(sin|cos)\s*\([^)]*\b(tick|frame)\b")
    for name in shaders.all_shader_names():
        for lineno, line in enumerate(shaders.load(name).splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "periodic function of the tick counter found:\n  " + "\n  ".join(offenders)
    )


def _pcg(values: np.ndarray) -> np.ndarray:
    """The PRNG from common.wgsl, in numpy."""
    v = values.astype(np.uint64) & MASK32
    state = (v * 747796405 + 2891336453) & MASK32
    shift = ((state >> np.uint64(28)) + 4) & MASK32
    word = (((state >> shift) ^ state) * 277803737) & MASK32
    return ((word >> np.uint64(22)) ^ word) & MASK32


def test_prng_has_no_short_period():
    """A repeating PRNG would make the whole aperiodicity argument moot."""
    counters = np.arange(1_000_000, dtype=np.uint64)
    values = _pcg(counters)
    unique = len(np.unique(values))
    # A 32-bit output over 1e6 draws has ~1.2e5 expected collisions by the
    # birthday bound; far fewer uniques than that would indicate structure.
    assert unique > 950_000, f"only {unique} distinct outputs from 1e6 counters"

    # And consecutive counters must not be correlated, which is the property
    # that matters when hashing (pixel, tick).
    correlation = np.corrcoef(values[:-1].astype(float), values[1:].astype(float))[0, 1]
    assert abs(correlation) < 0.01, f"successive outputs correlate at {correlation:.4f}"


def test_ou_process_is_stationary_and_aperiodic():
    """The climate/psi drift model: mean-reverting, bounded, non-repeating."""
    rng = np.random.default_rng(0)
    theta, sigma = 0.0016, 0.055
    value = 0.0
    series = []
    for _ in range(200_000):
        value = value * (1.0 - theta) + rng.normal() * sigma
        value = float(np.clip(value, -1.0, 1.0))
        series.append(value)
    series = np.asarray(series)

    assert abs(series.mean()) < 0.1, "OU process drifted off its mean"
    assert series.std() > 0.05, "OU process collapsed to its mean"
    assert np.isfinite(series).all()

    # No periodic component: autocorrelation must decay and stay decayed.
    centred = series - series.mean()
    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    spectrum[0] = 0.0
    peak = spectrum.max()
    assert peak < spectrum.sum() * 0.02, (
        "the drift has a dominant frequency component, i.e. it oscillates"
    )


def test_the_feature_size_walk_is_bounded_and_tempo_independent():
    """The global half of the morphology drift, DESIGN.md §4.7.

    Three properties, all load-bearing. The walk must be bounded, because it
    scales the diffusion rate and an unbounded excursion would park the whole
    field against a clamp -- a frozen texture again, just a different one. It
    must be unit-variance, because that is where the amplitude comes from: the
    parameter is expressed as a fraction of `du`, and a walk whose spread was
    not one would silently mean something else. And neither property may depend
    on the tick rate: the tempo macro moves ``sim_hz`` by more than 2x, and if
    the spread widened with it, turning up the tempo would quietly change how
    far feature size roams.

    Run against a short time constant so the sample covers a few hundred
    correlation times without a few hundred thousand iterations; the
    normalisation under test is a function of dt/tau, not of either alone.
    """
    tau = 30.0
    duration = 12_000.0  # seconds, i.e. 400 time constants
    spreads = {}
    for sim_hz in (12.0, 26.0):
        engine = engine_module.Engine.__new__(engine_module.Engine)
        engine._du_walk = 0.0
        engine._walk_rng = np.random.default_rng(4)
        dt = 1.0 / sim_hz
        series = np.empty(int(duration * sim_hz))
        for i in range(series.size):
            engine._advance_du_walk(dt, tau)
            series[i] = engine._du_walk

        assert np.isfinite(series).all()
        assert abs(series.mean()) < 0.2, (
            f"the walk drifted off its mean ({series.mean():.2f} at {sim_hz} Hz)"
        )
        assert np.abs(series).max() <= 2.0, "the walk left its bound"
        spreads[sim_hz] = float(series.std())

    for sim_hz, spread in spreads.items():
        assert 0.85 < spread < 1.15, (
            f"the walk is not unit-variance at {sim_hz} Hz (s.d. {spread:.2f}); "
            f"its amplitude no longer means a fraction of du"
        )

    # Whatever the walk does, the diffusion rate it produces stays in the band.
    reaction = config.Config().resolve().reaction
    for extreme in (-2.0, 2.0):
        du = reaction.du * (1.0 + reaction.du_walk * extreme)
        assert reaction.du_min <= config.clamp_du(du, reaction) <= reaction.du_max


def _parallax_series(fps: float, seconds: float, params, seed: int = 5):
    """The viewpoint's x offset over a run, sampled once per frame."""
    engine = engine_module.Engine.__new__(engine_module.Engine)
    engine._parallax = None
    engine._parallax_walk = []
    engine._parallax_lag = []
    engine._parallax_rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    series = np.empty(int(seconds * fps))
    for i in range(series.size):
        engine._update_parallax(params, dt, 1)
        series[i] = engine._parallax[0][0]
    return series


def test_the_viewpoint_drift_actually_uses_its_travel():
    """Motion parallax is the strongest depth cue either backend has, and it is
    only a cue if the viewpoint moves.

    This is a regression test for a walk that was numerically inert. A fixed
    per-frame decay of 0.02 against a `sqrt(dt)` noise term gave a stationary
    spread of 3e-4 across a travel of 1: at `render.parallax = 0.02` that is a
    viewpoint displaced by a few hundredths of a pixel, so the `depth` macro's
    parallax leg -- and the slab's entire camera motion -- did nothing at all.
    Nothing about the image looks wrong when that happens, which is exactly why
    it needs asserting rather than watching for.

    Bounded and frame-rate independent for the same reasons the feature-size
    walk is: `render.parallax` has to mean a maximum, and the frame rate is the
    one thing in the system the viewpoint must not take its excursion from.
    """
    params = config.Config().resolve()
    spreads = {}
    for fps in (30.0, 60.0):
        series = _parallax_series(fps, 2400.0, params)
        assert np.isfinite(series).all()
        assert np.abs(series).max() <= 1.0, "the offset left its bound"
        assert np.abs(series).max() > 0.5, (
            f"the viewpoint never used more than half its travel at {fps} Hz; "
            "parallax is switched off in all but name"
        )
        spreads[fps] = float(series.std())

    for fps, spread in spreads.items():
        assert 0.30 < spread < 0.60, (
            f"the drift's spread is {spread:.2f} of its travel at {fps} Hz; "
            "it should sit in the middle and reach the ends occasionally"
        )
    assert abs(spreads[30.0] - spreads[60.0]) < 0.10, (
        "how far the viewpoint roams depends on the frame rate")


def test_the_viewpoint_drifts_without_ever_shimmering():
    """The travel above has to be spent slowly, or it is per-pixel noise.

    An Ornstein-Uhlenbeck process is smooth in its envelope and *white* in its
    increments, which is fine for an amplitude and disastrous for a position:
    the raw walk moves the image about half a pixel per frame, at random, which
    is the temporal noise this application exists not to produce. The lag the
    walk reaches the viewpoint through is what fixes that, and this is the
    number it was chosen against.

    Expressed in pixels of a 1440p display, since a fraction of the field's
    extent says nothing about whether the eye can see it.
    """
    width = 2560.0
    params = config.Config().resolve()
    for fps in (30.0, 60.0):
        series = _parallax_series(fps, 1200.0, params)
        step = np.abs(np.diff(series)) * params.render.parallax * width
        assert step.max() < 0.5, (
            f"the viewpoint jumped {step.max():.2f} px in one frame at {fps} Hz")
        assert step.mean() < 0.05, (
            f"the viewpoint moves {step.mean():.3f} px per frame at {fps} Hz, "
            "which is a shake rather than a drift")
        # ... and the speed that leaves is a property of the walk, not of the
        # frame rate: the same drift seen twice as often moves half as far each
        # time.
        assert 0.2 < step.mean() * fps < 3.0, (
            f"the viewpoint travels {step.mean() * fps:.2f} px/s at {fps} Hz")


def test_the_feature_size_band_is_used_but_not_camped_on():
    """The drift must span the du band without spending its time at the clamp.

    Both failure directions matter. A drift too small for the band leaves the
    feature size effectively fixed, which is the texture this whole mechanism
    exists to break up. A drift too large for it parks a large fraction of the
    field against a bound, where the deviation can no longer act -- uniform
    again, just at a different size.

    The climate does not reach the +-1 it is clamped to; it settles around
    ``morphology.CLIMATE_SD``. So the fraction is computed from that, treating
    the deviation as normal -- which understates it slightly, since the field
    is spatially smoothed and its tails are thinner than a Gaussian's.
    """
    params = config.Config().resolve()
    reaction, climate = params.reaction, params.climate
    sigma = morphology.CLIMATE_SD

    # A one-sigma region must vary, and vary enough to matter: the length scale
    # moves roughly 1.7 -> 3.5 across the whole du band, so a deviation of a
    # few percent would be invisible.
    typical = math.exp(climate.range_du * sigma)
    assert 1.2 < typical < 1.8, (
        f"a one-sigma region's diffusion rate differs from the mean by "
        f"x{typical:.2f}; feature size is either barely varying or swinging "
        f"across the whole band at every excursion"
    )
    assert (
        reaction.du_min < reaction.du / typical
        and reaction.du * typical < reaction.du_max
    ), "a one-sigma region is already at a du clamp"

    # And the band must be reachable, but only in the tails.
    def tail(bound: float) -> float:
        """Fraction of a normal climate whose deviation clamps at `bound`."""
        z = abs(math.log(bound / reaction.du)) / (climate.range_du * sigma)
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    clamped = tail(reaction.du_min) + tail(reaction.du_max)
    assert 0.01 < clamped < 0.15, (
        f"{clamped:.1%} of the field sits at a du clamp; the band is either "
        f"unreachable (so the survival floor is untested) or so easily "
        f"reached that much of the screen is pinned to one feature size"
    )


def test_repulsion_is_reachable_but_is_not_the_common_case():
    """Where the anti-fusion crossing sits, against the climate that drives it.

    The commitment axis runs *through* fusion rather than peaking at it: under
    1 the agent follows the filament it sensed, at 1 it drives straight through
    and fuses, past 1 it veers away (agents.wgsl). So the whole mechanism is
    the position of that crossing relative to the climate amplitude, and both
    failure directions are quiet ones. Out of reach, and the layer is one-way
    again -- attraction with no repulsion, a topology that can only accrete.
    Too easily reached, and the network never gets to close a loop anywhere.

    As with the diffusion band, the climate does not reach the +-1 it is
    clamped to; it settles near ``morphology.CLIMATE_SD``.
    """
    params = config.Config().resolve()
    agents, climate = params.agents, params.climate
    sigma = morphology.CLIMATE_SD

    assert agents.fusion_bias < 1.0 <= agents.fusion_max, (
        f"the commitment clamp ({agents.fusion_max:.2f}) no longer reaches "
        f"past 1, so no region can repel however hard the climate pushes"
    )

    def repelling(bias: float) -> float:
        """Fraction of a normal climate whose commitment crosses 1."""
        z = (1.0 - bias) / (climate.range_repel * sigma)
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    fraction = repelling(agents.fusion_bias)
    assert 0.02 < fraction < 0.12, (
        f"{fraction:.1%} of the field is repelling at any moment; the zones "
        f"where the network comes apart are either vanishing or are most of "
        f"the screen"
    )

    # The intensity macro moves the base bias, so it moves the crossing with
    # it. That coupling is intended -- a denser field should knit and unknit
    # harder -- but it must not take either end out of the band.
    lo, hi = next(
        (low, high) for path, low, high, _ in config.MACRO_CURVES["intensity"]
        if path == "agents.fusion_bias"
    )
    for bias in (lo, hi):
        assert 0.005 < repelling(bias) < 0.25, (
            f"at a fusion bias of {bias:.2f} -- an end of the intensity macro "
            f"-- {repelling(bias):.1%} of the field repels"
        )


@pytest.mark.slow
def test_flux_pruning_returns_the_mass_it_removes(gpu_device):
    """The property that decides whether pruning survives the homeostat.

    Raising decay on abandoned strands is a mass sink, and the controller has a
    lever pointed straight at it: corr_decay. Left uncompensated, the homeostat
    lowers decay globally until the trail mass comes back -- a weaker network
    everywhere and no severance anywhere, which is the opposite of the point.
    The removed mass is measured in the reduce pass and handed back through the
    agent deposit, so the field ends up the same weight and the controller has
    nothing to correct.

    Asserted against a prune_gain = 0 control run rather than against absolute
    numbers, since what matters is the difference the term makes.
    """
    device, _ = gpu_device
    # 1400 ticks, not 700: this is a statement about the steady state, and the
    # field takes longer to reach one now that the network is less concentrated
    # (DESIGN.md §4.9). Measured across two seeds, the mass difference is 2-3%
    # from 1400 ticks on; at 700 one seed was still 29% short of the control
    # purely because it had not finished building.
    width, height, ticks = 128, 128, 1400

    def run(prune_gain: float) -> dict:
        params = config.Config().resolve()
        params.render.layers = 1
        params.agents.prune_gain = prune_gain
        engine = engine_module.Engine(device, width, height, params, seed=23)
        for _ in range(ticks):
            engine.tick(params, [])
        layer = engine.layers[0]
        texture = layer.trail.textures[layer.trail.index]
        raw = device.queue.read_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
            (width, height, 1),
        )
        trail = np.frombuffer(raw, dtype=np.float16).reshape(height, width, 4)
        stats = np.frombuffer(
            device.queue.read_buffer(engine.stats_buf), dtype=gpu_params.STATS_DTYPE
        )[0]
        return {
            "mass": float(trail[..., 0].astype(np.float32).mean()),
            "prune_return": float(stats["prune_return"]),
            "corr_decay": float(stats["corr_decay"]),
            "mean_v": float(stats["mean_v"]),
        }

    control = run(0.0)
    pruned = run(3.0)

    assert control["prune_return"] == 0.0, (
        "the prune term is not inert at zero gain, so it is not off by default"
    )
    assert pruned["prune_return"] > 0.05, (
        f"pruning at gain 3 removed almost nothing "
        f"({pruned['prune_return']:.4f} of throughput); the deficit measure has "
        f"stopped discriminating -- check income_rate against trail_decay"
    )

    drift = abs(pruned["mass"] - control["mass"]) / max(control["mass"], 1e-9)
    assert drift < 0.25, (
        f"trail mass moved {drift:.1%} against the unpruned control "
        f"({pruned['mass']:.4f} against {control['mass']:.4f}); the return "
        f"accounting is not putting back what the prune term takes"
    )

    # The controller is the thing that must not notice. corr_decay is clamped to
    # +-0.010, so a term it was fighting would show up as a large fraction of
    # that, not as a rounding difference.
    fight = abs(pruned["corr_decay"] - control["corr_decay"])
    assert fight < 0.001, (
        f"corr_decay moved by {fight:.5f} when pruning was switched on; the "
        f"homeostat is compensating for it, which is what cancels the effect"
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_every_event_channel_reaches_the_climate_field():
    """A packed channel that nothing reads is invisible in every other way.

    The GPU record, the host-side :class:`events.Channels` and the shader that
    consumes them are three lists that have to agree, and a mismatch produces
    no error anywhere -- the event simply does less than it says. That is how
    §4.7 found the events layer at all: `dieback` could thin material but not
    sever anything, because severance lives in channels ``EVENT_FIELDS`` did
    not have.
    """
    packed = {name for name, _ in gpu_params.EVENT_FIELDS if name.startswith("chan_")}
    declared = {f"chan_{name}" for name in events.Channels._fields}
    assert packed == declared, (
        f"host and GPU event channels disagree: {packed ^ declared}"
    )

    source = shaders.load("climate.wgsl")
    unread = {name for name in packed if f"ev.{name}" not in source}
    assert not unread, (
        f"event channels packed but never applied to the climate field: "
        f"{sorted(unread)}"
    )


def test_an_event_kind_can_sever_the_network():
    """Not merely thin it -- the distinction §4.7 step 4 turns on.

    Feed and kill move how much material there is. They leave the topology
    exactly as it was: the same mesh, fainter. Severance needs the trail decay,
    the pruning term and the agents' junction behaviour, and an event that
    reaches only feed and kill cannot touch any of them.
    """
    severing = {
        kind: channels for kind, channels in events.EVENT_KINDS.items()
        if channels.decay or channels.prune or channels.repel
    }
    assert severing, (
        "no event kind reaches the severance channels, so events can only ever "
        "thin material -- the topology is one-way again"
    )

    # And a severing kind must not also move mass, which is not the fussy
    # constraint it looks like. An event *adds* its amplitude to the climate
    # every tick against a mean reversion of 0.0016, so any channel a kind
    # names is pinned at the clamp for the length of the envelope -- the
    # coefficient shapes the ramp and nothing else. A "small" feed term is
    # therefore a full one, measured: it took the reaction inside the disc down
    # by a third and the whole field's mass by 16%, which reaches the image as
    # a slow global brightness swing through the exposure governor.
    for kind, channels in severing.items():
        assert channels.feed == 0.0 and channels.kill == 0.0, (
            f"the {kind} event moves feed by {channels.feed:+.2f} and kill by "
            f"{channels.kill:+.2f} as well as severing; there is no such thing "
            f"as a small event channel, so this is a dieback with extra steps"
        )


def test_event_envelope_never_steps():
    params = config.EventParams()
    scheduler = events.EventScheduler(seed=1)
    event = scheduler._spawn(params)

    dt = 1.0 / 20.0
    previous = event.envelope()
    largest = 0.0
    while not event.finished:
        event.elapsed += dt
        current = event.envelope()
        largest = max(largest, abs(current - previous))
        previous = current
    # A raised cosine over tens of seconds cannot move fast per tick.
    assert largest < 0.02, f"event envelope stepped by {largest:.4f} in one tick"


def test_event_envelope_starts_and_ends_at_zero():
    scheduler = events.EventScheduler(seed=2)
    event = scheduler._spawn(config.EventParams())
    assert event.envelope() == pytest.approx(0.0, abs=1e-9)
    event.elapsed = event.duration
    assert event.envelope() == pytest.approx(0.0, abs=1e-6)


def test_events_stay_under_the_flash_area_threshold():
    """Event radius is capped at the WCAG area threshold, with margin."""
    params = config.EventParams()
    scheduler = events.EventScheduler(seed=3)
    for _ in range(200):
        event = scheduler._spawn(params)
        area = np.pi * event.radius**2
        assert area < 0.25, f"event covers {area:.1%} of the screen"


def test_event_arrivals_are_memoryless():
    """Poisson arrivals, so the next event is never predictable from the last."""
    params = config.EventParams(rate_per_hour=60.0, max_concurrent=8)
    scheduler = events.EventScheduler(seed=4)
    dt = 1.0
    spawns = []
    for step in range(20_000):
        before = scheduler.spawned
        scheduler.update(dt, params)
        if scheduler.spawned > before:
            spawns.append(step)
    assert len(spawns) > 50, "too few events to judge"
    gaps = np.diff(spawns)
    # Exponential inter-arrivals have std ~ mean; a fixed cadence would have
    # std near zero.
    assert gaps.std() > gaps.mean() * 0.5, (
        f"arrival gaps look regular: mean {gaps.mean():.1f}, std {gaps.std():.1f}"
    )


def test_events_can_be_disabled():
    params = config.EventParams(enabled=False)
    scheduler = events.EventScheduler(seed=5)
    for _ in range(10_000):
        scheduler.update(1.0, params)
    assert scheduler.spawned == 0


def test_only_a_rift_drives_the_severance_channels_of_the_climate(gpu_device):
    """The plumbing, end to end, without waiting for the field to respond.

    Cheap and deterministic where the structural consequence is neither: what
    the climate field holds inside an event's disc is a property of the event
    pass alone, so it can be read after a few hundred ticks. `dieback` is the
    control -- it is the strongest thinning event there is, and it must leave
    every one of these channels alone, because thinning material and severing
    a network are different things.
    """
    device, _ = gpu_device
    size = 64

    def climate_inside(kind: str | None) -> tuple[float, float]:
        params = config.Config().resolve()
        params.render.layers = 1
        engine = engine_module.Engine(device, size, size, params, seed=13)
        rows = []
        if kind is not None:
            scheduler = events.EventScheduler(seed=0)
            scheduler.active = [
                events.ActiveEvent(
                    x=0.5, y=0.5, radius=0.24, peak=params.events.strength,
                    channels=events.EVENT_KINDS[kind],
                    attack=1.0, hold=1e6, release=1.0, kind=kind,
                )
            ]
            scheduler.active[0].elapsed = 2.0
            rows, _ = scheduler.pack(8)
        for _ in range(400):
            engine.tick(params, rows)

        layer = engine.layers[0]
        width, height = params.climate.width, params.climate.height
        read = {}
        for name, pair in (("b", layer.climate_b), ("c", layer.climate_c)):
            texture = pair.textures[pair.index]
            raw = device.queue.read_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
                (width, height, 1),
            )
            read[name] = np.frombuffer(raw, dtype=np.float16).reshape(
                height, width, 4).astype(np.float32)
        middle = (slice(height // 2 - 2, height // 2 + 3),
                  slice(width // 2 - 2, width // 2 + 3))
        # decay (climate_b.y) and repel (climate_c.z).
        return float(read["b"][middle][..., 1].mean()), float(read["c"][middle][..., 2].mean())

    quiet = climate_inside(None)
    thinning = climate_inside("dieback")
    severing = climate_inside("rift")

    for name, value in zip(("decay", "repel"), quiet):
        assert abs(value) < 0.5, (
            f"the {name} channel sits at {value:+.2f} with no event at all; "
            f"the drift alone is as strong as an event"
        )
    for name, value in zip(("decay", "repel"), thinning):
        assert abs(value) < 0.5, (
            f"a dieback moved the {name} channel to {value:+.2f}; it is "
            f"supposed to thin material, not take the network apart"
        )
    for name, value in zip(("decay", "repel"), severing):
        assert value > 0.9, (
            f"a rift left the {name} channel at {value:+.2f}; the severance "
            f"channels are not reaching the climate field"
        )


@pytest.mark.slow
def test_a_rift_event_takes_the_network_apart_and_the_network_comes_back(gpu_device):
    """What a rift does to the field, against a no-event control on one seed.

    Three claims, and the second is as important as the first:

    * the network inside the disc comes apart -- which is the thing `dieback`
      cannot do at any amplitude, since nothing flows from the reaction back to
      the trail;
    * the *material* is left alone. A rift moves how the material is connected,
      not how much of it there is. An event that moved mass would reach the
      image as a slow global brightness swing through the exposure governor,
      which is the interaction DESIGN.md §4.7 flags as the real risk;
    * it heals. V = 0 is an absorbing state for Gray-Scott, so an event that
      could empty a quarter of the screen and leave it empty would be a worse
      failure than the monotony §4.7 exists to fix, because it would be
      permanent.

    Two runs of the same seed rather than an inside/outside ratio within one:
    the disc's own ground is not representative of the field -- it can be a
    dense hub or a void -- and the control run is the only fair reference for
    what would have been there anyway.

    The attack and release are compressed but the hold is not, and that is not
    a free choice. The direct effect of the raised decay is about a fifth of
    the trail; the rest is the feedback behind it -- a thinned strand stops
    being found, which thins it further -- and that takes on the order of a
    thousand ticks to develop.

    The seed is fixed because the *magnitude* of the severance depends on what
    the disc lands on, and legitimately so: the feedback runs furthest where
    the network is thinnest. What the seed must guarantee is that there is a
    network under the disc at all, which is why the control's own level is
    asserted before the ratio is: on bare ground both runs measure a few
    thousandths and their ratio is noise over noise. Surveyed across seven
    seeds, the ones that land on material (control trail 0.16 to 0.27) give a
    severance of 0.73 to 0.83, and the ones that land on a void (0.001 to 0.015)
    give anything between 0.30 and 1.30 with no signal in it.

    This test was previously anchored to a seed whose disc turned out to sit on
    a void, at a threshold only reachable there. Fixing the flow's wrap seam
    (DESIGN.md §4.8) re-rolled the field and moved that disc onto different
    ground, which is what exposed it; bounding the sensing reach (§4.9) re-rolled
    it again, hence the seed here. The effect size has been stable across all of
    it: 0.77 on a dense hub before either change, 0.73 and 0.83 after the first,
    0.78 here after the second.
    """
    device, _ = gpu_device
    size, radius = 128, 0.24
    warm, attack, hold, release, recover = 800, 200, 2000, 400, 2000

    ys, xs = np.mgrid[0:size, 0:size]
    distance = np.hypot((xs + 0.5) / size - 0.5, (ys + 0.5) / size - 0.5)
    inside = distance < radius * 0.5

    def run(with_event: bool) -> dict[str, tuple[float, float]]:
        params = config.Config().resolve()
        params.render.layers = 1
        engine = engine_module.Engine(device, size, size, params, seed=13)
        event = events.ActiveEvent(
            x=0.5, y=0.5, radius=radius, peak=params.events.strength,
            channels=events.EVENT_KINDS["rift"],
            attack=attack, hold=hold, release=release, kind="rift",
        )
        scheduler = events.EventScheduler(seed=0)

        def sample() -> tuple[float, float]:
            layer = engine.layers[0]
            means = []
            for pair, channel in ((layer.trail, 0), (layer.reaction, 1)):
                texture = pair.textures[pair.index]
                raw = device.queue.read_texture(
                    {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                    {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
                    (size, size, 1),
                )
                field = np.frombuffer(raw, dtype=np.float16).reshape(
                    size, size, 4)[..., channel].astype(np.float32)
                means.append(float(field[inside].mean()))
            return means[0], means[1]

        marks: dict[str, tuple[float, float]] = {}
        for tick in range(warm + attack + hold + release + recover):
            rows: list[dict] = []
            if with_event and tick >= warm:
                event.elapsed = tick - warm
                if not event.finished:
                    scheduler.active = [event]
                    rows, _ = scheduler.pack(8)
            engine.tick(params, rows)
            if tick == warm + attack + hold - 1:
                marks["during"] = sample()
            if tick == warm + attack + hold + release + recover - 1:
                marks["after"] = sample()
        return marks

    control = run(False)
    rifted = run(True)

    # Precondition, not a claim about rifts: a severance measured on ground
    # with nothing on it is a ratio of two roundings.
    assert control["during"][0] > 0.05, (
        f"the control run has almost no network inside the disc "
        f"({control['during'][0]:.4f}), so the ratio below would measure "
        f"nothing -- the seed needs to land the disc on material"
    )

    trail_during = rifted["during"][0] / max(control["during"][0], 1e-9)
    assert trail_during < 0.9, (
        f"the network inside the rift is at {trail_during:.2f} of the control "
        f"run's, so the severance channels barely reached it"
    )

    material = rifted["during"][1] / max(control["during"][1], 1e-9)
    assert material > 0.75, (
        f"the reaction inside the rift fell to {material:.2f} of the control's; "
        f"a rift is supposed to sever the network, not thin the material -- "
        f"check whether the kind has picked up a feed or kill channel, which "
        f"an event pins at its clamp however small the coefficient looks"
    )

    trail_after = rifted["after"][0] / max(control["after"][0], 1e-9)
    assert trail_after > 0.35, (
        f"the network inside the rift is still at {trail_after:.2f} of the "
        f"control's well after the envelope released; a rift is leaving "
        f"permanently bare ground"
    )
    assert rifted["after"][1] / max(control["after"][1], 1e-9) > 0.7, (
        f"the reaction inside the rift did not come back "
        f"({rifted['after'][1]:.4f} against a control's {control['after'][1]:.4f})"
    )


# ---------------------------------------------------------------------------
# Condensation of the agent layer
# ---------------------------------------------------------------------------


def _agent_layer_state(device, engine) -> tuple[float, float, float]:
    """Axis alignment, band share, and the trail's peak-to-median ratio.

    ``axis`` is ``|mean exp(4i.heading)|``, which is 1.0 when every agent runs
    along +-x or +-y and 0 when the headings are spread; the fourth harmonic is
    the one that does not care which of the four directions an agent picked.
    ``band`` is the largest share of the population inside any eight-cell row or
    column. Together they separate the two things that can go wrong: a
    population that has aligned, and a population that has piled up.
    """
    layer = engine.layers[0]
    width, height = layer.spec.width, layer.spec.height
    agents = np.frombuffer(
        device.queue.read_buffer(layer.agents_buf),
        dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("heading", "<f4"),
            ("rng", "<u4"), ("recent", "<f4"), ("age", "<f4")]),
    )
    axis = float(np.abs(np.mean(np.exp(4j * np.mod(agents["heading"], 2 * np.pi)))))
    rows, _ = np.histogram(agents["y"], bins=height // 8, range=(0, height))
    columns, _ = np.histogram(agents["x"], bins=width // 8, range=(0, width))
    band = float(max(rows.max(), columns.max()) / len(agents))

    raw = device.queue.read_texture(
        {"texture": layer.trail.textures[layer.trail.index], "mip_level": 0,
         "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, 1),
    )
    trail = np.frombuffer(raw, dtype=np.float16).reshape(
        height, width, 4)[..., 0].astype(np.float64)
    # Per axis, then the worse of the two. Pooling the row and column profiles
    # instead would hide exactly the case being measured: a strand running the
    # width of the field leaves every *row* alike, so the pooled median lands
    # between the two distributions and flattens the ratio by an order of
    # magnitude.
    peak = 0.0
    for profile in (trail.mean(axis=1), trail.mean(axis=0)):
        peak = max(peak, float(profile.max() / max(np.median(profile), 1e-9)))
    return axis, band, peak


def _run_agent_layer(device, ticks, seed, reach=None):
    """Run one layer and report what its agent population has become."""
    params = config.Config().resolve()
    params.render.layers = 1
    if reach is not None:
        params.agents.sensor_distance, params.agents.sensor_reach_max = reach[:2]
        params.climate.range_sensor_distance = reach[2]
    engine = engine_module.Engine(device, 192, 128, params, seed=seed)
    scheduler = events.EventScheduler(seed=3)
    for _ in range(ticks):
        rows, _ = scheduler.pack(8)
        engine.tick(params, rows)
    return _agent_layer_state(device, engine)


@pytest.mark.slow
def test_the_agent_layer_does_not_condense_onto_one_strand(gpu_device):
    """DESIGN.md §4.9, and the reason the sensing reach is bounded at all.

    Left unbounded, trail following straightens the strand an agent is on --
    sensors placed well ahead of it cut every corner -- and a straight strand on
    a torus closes on itself after one lap, so it is reinforced every lap while
    nothing else is. Trail following has no capacity limit and no exit but
    `max_age`, so that strand then takes the whole population: within ~1000
    ticks every agent runs along one axis-aligned line and the rest of the field
    is bare. That is what the reported vertical and horizontal lines were.

    The control arm is the reach as it shipped, because a collapse metric that
    has never seen a collapse is not evidence. It is not a claim that those
    numbers are uniquely bad -- anything past a ratio of about four does it.
    """
    device, _ = gpu_device

    axis, _band, peak = _run_agent_layer(device, 1100, seed=5)
    assert axis < 0.35, (
        f"agent headings are {axis:.2f} aligned to the axes, so the layer is "
        f"laying the straight strands that close on the torus"
    )
    assert peak < 150.0, (
        f"the trail's busiest row or column is {peak:.0f}x the median, i.e. the "
        f"field has emptied into one strand"
    )

    # The reach as it shipped: 8.0 cells, no ceiling, +-3.0 cells of climate.
    was = _run_agent_layer(device, 1100, seed=5, reach=(8.0, 99.0, 3.0))
    assert was[0] > 0.6 and was[2] > 150.0, (
        f"the unbounded reach no longer condenses (axis {was[0]:.2f}, band "
        f"{was[1]:.2f}, peak {was[2]:.0f}), so the assertions above are not "
        f"testing anything"
    )


# ---------------------------------------------------------------------------
# GPU soak
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_long_run_stays_finite_and_in_band(gpu_device, offscreen_target):
    """The core long-duration property: it neither dies nor saturates."""
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 1

    width, height = 128, 96
    engine = engine_module.Engine(device, width, height, params, seed=7)
    target, fmt = offscreen_target(width, height)
    scheduler = events.EventScheduler(seed=8)

    history = []
    for _ in range(1200):
        rows, _ = scheduler.pack(8)
        engine.tick(params, rows)
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)
        scheduler.update(1.0 / params.sim_hz, params.events)
        history.append(engine.read_stats()["mean_v"])

    image = engine.read_final_rgba()
    assert np.isfinite(image).all(), "output contains non-finite values"

    stats = engine.read_stats()
    for key in ("mean_v", "var_v", "mean_activity", "exposure"):
        assert np.isfinite(stats[key]), f"{key} is not finite"

    assert stats["mean_v"] > 0.02, f"field died: mean_v {stats['mean_v']:.5f}"
    assert stats["mean_v"] < 0.6, f"field saturated: mean_v {stats['mean_v']:.5f}"
    assert stats["mean_activity"] > 1e-4, (
        f"field settled: activity {stats['mean_activity']:.6f}"
    )

    # The homeostat should have moved the field toward its target, not away.
    target_mass = params.homeostat.target_mass
    start_error = abs(history[50] - target_mass)
    end_error = abs(history[-1] - target_mass)
    assert end_error < start_error, (
        f"homeostat did not converge: error went {start_error:.4f} -> {end_error:.4f}"
    )


@pytest.mark.slow
def test_steady_state_allocates_nothing(gpu_device, offscreen_target):
    """A run of 10^7 frames will find any per-frame allocation.

    Bind groups are created on demand and cached by resource identity, so the
    cache must stop growing once every ping-pong parity has been seen.
    """
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 2

    width, height = 96, 64
    engine = engine_module.Engine(device, width, height, params, seed=9)
    target, fmt = offscreen_target(width, height)

    def step():
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    # The sanitise pass runs every 60 ticks and flips field parity, so the full
    # set of ping-pong combinations only appears after a couple of cycles.
    for _ in range(150):
        step()
    settled = len(engine._bind_cache)

    for _ in range(150):
        step()
    assert len(engine._bind_cache) == settled, (
        f"bind group cache grew from {settled} to {len(engine._bind_cache)}; "
        "something is allocating per frame"
    )


@pytest.mark.slow
def test_recovers_from_a_corrupted_field(gpu_device, offscreen_target):
    """A NaN must not be able to destroy the run permanently.

    This is the failure mode with no natural recovery: one non-finite texel
    propagates through diffusion and takes the whole field with it, and a
    session two days in has no way back.
    """
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 1

    width, height = 96, 64
    engine = engine_module.Engine(device, width, height, params, seed=10)
    target, fmt = offscreen_target(width, height)

    for _ in range(20):
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    # Inject NaN and infinity across a band of the reaction field.
    layer = engine.layers[0]
    poison = np.zeros((height, width, 4), dtype=np.float32)
    poison[..., 0] = 1.0
    poison[: height // 4, :, 1] = np.nan
    poison[height // 4 : height // 2, :, 1] = np.inf
    device.queue.write_texture(
        {"texture": layer.reaction.textures[layer.reaction.index],
         "mip_level": 0, "origin": (0, 0, 0)},
        np.ascontiguousarray(poison.astype(np.float16)),
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, 1),
    )

    for _ in range(90):  # past at least one sanitise pass
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    image = engine.read_final_rgba()
    assert np.isfinite(image).all(), "non-finite values reached the output"
    stats = engine.read_stats()
    assert np.isfinite(stats["mean_v"]), "field statistics are still poisoned"
    assert stats["mean_v"] > 0.0, "field did not recover"
