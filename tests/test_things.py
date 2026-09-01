"""The Small Strange Things backend -- DESIGN.md §18.

What the port has to prove is different in kind from the other backends'
suites: §18.1 is a bill of rights, so most of these tests are the souls
under assertion --

* nothing ever dies, and the cap is a ceiling on births, never a cull;
* bonds never break, and at most three form;
* only the curious seek friends;
* traits rolled at birth never change;
* a click adds Things where it points;
* the world wraps, "sort of", exactly as the founding file wrapped it;
* a checkpoint resumes bit-identically -- ages, friendships, the canvas
  and the breath's phase included (§4.6);
* and the flash bound holds under this backend exactly as under the other
  three, with the sparkles at their most enthusiastic.

The reference implementation is docs/founding/small_strange_thing.html;
where an expected value below looks odd (0.7 alpha, hue+60, ±25 scatter),
it is the founding file's.
"""

from __future__ import annotations

import numpy as np
import pytest
import wgpu

import reference as R
from anastomosis import checkpoint, config
from anastomosis.backend import aspect_correction
from anastomosis.things import (
    NO_FRIEND,
    THING_ALIVE,
    THING_DTYPE,
    ThingsEngine,
    ThingsGeometry,
)

WIDTH, HEIGHT = 128, 96

# Same figure, same reasoning as test_flash_safety.py.
F16_TOLERANCE = 0.0005


def _resolve(overrides=None, **macros) -> config.Params:
    return config.Config(
        backend="things",
        macros=config.Macros(**macros),
        overrides=dict(overrides or {}),
    ).resolve()


def _engine(gpu_device, params, seed=4242, width=WIDTH, height=HEIGHT):
    device, _ = gpu_device
    return ThingsEngine(device, width, height, params, seed=seed)


def _population(engine) -> np.ndarray:
    raw = engine.device.queue.read_buffer(engine.things.cur)
    return np.frombuffer(raw, dtype=THING_DTYPE).copy()


def _write_population(engine, pop: np.ndarray) -> None:
    for buffer in engine.things.buffers:
        engine.device.queue.write_buffer(buffer, 0, pop.tobytes())


def _alive(pop: np.ndarray) -> np.ndarray:
    return (pop["flags"] & THING_ALIVE) != 0


def _canvas(engine) -> np.ndarray:
    pair = engine.canvas
    return checkpoint._read_texture(engine.device, pair.textures[pair.index])


def _blank_population(capacity: int) -> np.ndarray:
    pop = np.zeros(capacity, dtype=THING_DTYPE)
    pop["friend0"] = NO_FRIEND
    pop["friend1"] = NO_FRIEND
    pop["friend2"] = NO_FRIEND
    return pop


def _crowd(engine, count, spacing=2.0, curiosity=1.0, shyness=0.0):
    """A hand-placed population: `count` Things in a tight line."""
    g = engine.geometry
    pop = _blank_population(g.capacity)
    for i in range(count):
        pop["x"][i] = g.width * 0.5 + i * spacing
        pop["y"][i] = g.height * 0.5
        pop["size"][i] = 2.0
        pop["speed"][i] = 0.3
        pop["hue"][i] = (i * 37.0) % 360.0
        pop["curiosity"][i] = curiosity
        pop["shyness"][i] = shyness
        pop["flags"][i] = THING_ALIVE
    _write_population(engine, pop)
    return pop


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_geometry_derives_and_bounds_itself():
    params = _resolve()
    geometry = ThingsGeometry.derive(WIDTH, HEIGHT, params)
    assert geometry.width % 32 == 0
    assert geometry.capacity == params.things.capacity
    assert geometry.problems() == []
    assert "Thing slots" in geometry.describe()

    # A file claiming a metropolis is unusable, not an allocation: the cap
    # is the "resolution up, population not" law (§18.1 soul 4).
    bent = ThingsGeometry(width=geometry.width, height=geometry.height,
                          capacity=1_000_000)
    assert bent.problems()


# ---------------------------------------------------------------------------
# The souls (§18.1)
# ---------------------------------------------------------------------------


def test_nothing_ever_dies(gpu_device):
    """Soul 5. Whoever is alive stays alive, tick after tick, forever ==
    for as long as a test can afford to look."""
    params = _resolve()
    engine = _engine(gpu_device, params, seed=7)
    was_alive = _alive(_population(engine))
    assert int(was_alive.sum()) == params.things.seed_count

    for _ in range(300):
        engine.tick(params)
        now_alive = _alive(_population(engine))
        assert bool(np.all(now_alive[was_alive])), (
            "a Thing died; the port invented mortality"
        )
        was_alive = now_alive


def test_the_village_is_capped_softly(gpu_device):
    """Soul 4. The population reaches the cap and never exceeds it -- and
    the cap arrives as scarcity of empty slots, not as a cull."""
    params = _resolve(overrides={
        "things.capacity": 24,
        "things.spawn_rate": 60.0,       # a very eager lottery
        "things.mature_seconds": 0.0,
    })
    engine = _engine(gpu_device, params, seed=11)
    for _ in range(200):
        engine.tick(params)
        count = int(_alive(_population(engine)).sum())
        assert count <= 24
    assert int(_alive(_population(engine)).sum()) == 24, (
        "an eager village should have filled its cap"
    )


def test_bonds_never_break_and_survive_emigration(gpu_device):
    """Soul 2's load-bearing half. Friendships, once recorded, are never
    unrecorded -- however far the pair wanders apart afterwards."""
    params = _resolve(overrides={
        # No new friendships and no births: the recorded bonds are the
        # only social state, so any change is a break.
        "things.friend_rate": 0.0,
        "things.spawn_rate": 0.0,
    })
    engine = _engine(gpu_device, params, seed=13)
    g = engine.geometry
    pop = _crowd(engine, 6)
    # Hand-recorded friendships, including a duplicate -- the founding
    # file allows the same neighbour twice (§18.1 soul 2), so the port
    # must preserve one without collapsing it.
    pop["friend0"][0] = 1
    pop["friend1"][0] = 2
    pop["friend2"][0] = 1
    pop["friend_count"][0] = 3
    pop["friend0"][3] = 5
    pop["friend_count"][3] = 1
    # And send one friend far away: the bond must hold at any distance.
    pop["x"][1] = 1.0
    pop["y"][1] = 1.0
    _write_population(engine, pop)

    for _ in range(240):
        engine.tick(params)
    after = _population(engine)
    assert after["friend_count"][0] == 3
    assert after["friend0"][0] == 1
    assert after["friend1"][0] == 2
    assert after["friend2"][0] == 1, "the duplicate bond was tidied away"
    assert after["friend_count"][3] == 1
    assert after["friend0"][3] == 5


def test_at_most_three_friends_ever(gpu_device):
    """Soul 2's cap, under the most sociable conditions reachable."""
    params = _resolve(overrides={
        "things.friend_rate": 60.0,
        "things.spawn_rate": 0.0,
    })
    engine = _engine(gpu_device, params, seed=17)
    _crowd(engine, 12, spacing=1.5)
    for _ in range(120):
        engine.tick(params)
    after = _population(engine)
    assert int(after["friend_count"].max()) <= 3
    # And in a crowd this tight, the curious did all find friends.
    assert int(after["friend_count"][:12].min()) == 3


def test_only_the_curious_seek(gpu_device):
    """Soul 1's one behavioural consequence: curiosity must outweigh
    shyness or no bond ever forms, however close the crowd."""
    params = _resolve(overrides={
        "things.friend_rate": 60.0,
        "things.spawn_rate": 0.0,
    })
    engine = _engine(gpu_device, params, seed=19)
    _crowd(engine, 10, spacing=1.5, curiosity=0.2, shyness=0.9)
    for _ in range(120):
        engine.tick(params)
    after = _population(engine)
    assert int(after["friend_count"].sum()) == 0, (
        "a shy Thing sought friends"
    )


def test_traits_are_fixed_for_life(gpu_device):
    """Soul 1. The five birth traits never move again."""
    params = _resolve()
    engine = _engine(gpu_device, params, seed=23)
    before = _population(engine)
    mask = _alive(before)
    for _ in range(200):
        engine.tick(params)
    after = _population(engine)
    for trait in ("hue", "size", "speed", "curiosity", "shyness"):
        assert np.array_equal(before[trait][mask], after[trait][mask]), (
            f"{trait} drifted; traits are rolled at birth and fixed for life"
        )


def test_ages_advance_in_whole_ticks(gpu_device):
    """Soul 10 plus the §17 lesson: age is a u32 tick count, so it can
    never stall the way an f16 accumulation did."""
    params = _resolve(overrides={"things.spawn_rate": 0.0})
    engine = _engine(gpu_device, params, seed=29)
    before = _population(engine)
    mask = _alive(before)
    ticks = 150
    for _ in range(ticks):
        engine.tick(params)
    after = _population(engine)
    assert bool(np.all(after["age"][mask] == before["age"][mask] + ticks))


def test_the_world_wraps_sort_of(gpu_device):
    """Soul 8, conserved quirk included: an edge crossing teleports to the
    far edge (the founding wrap sets, it does not fold), and positions
    always land back inside the field."""
    params = _resolve(overrides={"things.spawn_rate": 0.0})
    engine = _engine(gpu_device, params, seed=31)
    g = engine.geometry
    pop = _blank_population(g.capacity)
    pop["x"][0] = g.width - 0.01
    pop["y"][0] = 0.01
    pop["size"][0] = 2.0
    pop["speed"][0] = 0.6
    pop["curiosity"][0] = 0.0
    pop["shyness"][0] = 1.0
    pop["flags"][0] = THING_ALIVE
    _write_population(engine, pop)

    for _ in range(400):
        engine.tick(params)
        after = _population(engine)
        assert 0.0 <= float(after["x"][0]) <= g.width
        assert 0.0 <= float(after["y"][0]) <= g.height


def test_a_click_adds_things_where_it_points(gpu_device):
    """Soul 9, the participation verb. A click in window-normalised
    coordinates lands the founding three Things at the mapped canvas
    point, inside the founding scatter."""
    params = _resolve(overrides={"things.spawn_rate": 0.0})
    engine = _engine(gpu_device, params, seed=37)
    g = engine.geometry
    _write_population(engine, _blank_population(g.capacity))

    u, v = 0.25, 0.7
    engine.queue_click(u, v)
    engine.tick(params)

    pop = _population(engine)
    alive = _alive(pop)
    assert int(alive.sum()) == params.things.per_click == 3

    sx, sy = aspect_correction(
        engine.width, engine.height, g.width, g.height)
    expected_x = (((u - 0.5) * sx + 0.5) % 1.0) * g.width
    expected_y = (((v - 0.5) * sy + 0.5) % 1.0) * g.height
    scatter = params.things.click_scatter + 1e-3
    for x, y in zip(pop["x"][alive], pop["y"][alive]):
        assert abs(float(x) - expected_x) <= scatter
        assert abs(float(y) - expected_y) <= scatter
    # Newborns fade in from nothing (§18.5): age zero this tick.
    assert bool(np.all(pop["age"][alive] == 0))


def test_clicks_queue_rather_than_drop(gpu_device):
    """A burst of clicks is enthusiasm: consumed a few per tick, in
    order, none lost inside the queue's bound."""
    params = _resolve(overrides={"things.spawn_rate": 0.0})
    engine = _engine(gpu_device, params, seed=41)
    g = engine.geometry
    _write_population(engine, _blank_population(g.capacity))
    for _ in range(6):
        engine.queue_click(0.5, 0.5)
    for _ in range(3):
        engine.tick(params)
    pop = _population(engine)
    assert int(_alive(pop).sum()) == 6 * params.things.per_click


# ---------------------------------------------------------------------------
# The canvas (soul 7)
# ---------------------------------------------------------------------------


def test_the_canvas_holds_ghosts_and_lets_them_go(gpu_device):
    """The founding fade: everywhere a Thing has been stays lit for a
    while, and an abandoned haunt decays toward dark rather than being
    erased by anything."""
    params = _resolve(overrides={"things.spawn_rate": 0.0,
                                 "things.sparkle_rate": 0.0})
    engine = _engine(gpu_device, params, seed=43)
    _crowd(engine, 4, spacing=3.0, curiosity=0.0, shyness=1.0)
    for _ in range(60):
        engine.tick(params)
    lit = _canvas(engine)[..., :3].astype(np.float32)
    assert float(lit.max()) > 0.05, "the Things left no light at all"

    # Empty the world (a test's privilege, not a simulation path) and the
    # ghosts fade on the configured clock, never stepping.
    _write_population(engine, _blank_population(engine.geometry.capacity))
    levels = []
    for _ in range(8):
        for _ in range(10):
            engine.tick(params)
        levels.append(float(_canvas(engine)[..., :3].astype(np.float32).max()))
    assert all(b <= a + 1e-4 for a, b in zip(levels, levels[1:])), (
        "an abandoned ghost brightened"
    )
    assert levels[-1] < levels[0] * 0.2, "the ghosts never fade"


# ---------------------------------------------------------------------------
# Persistence (§18.2, §4.6)
# ---------------------------------------------------------------------------


def test_a_resumed_world_evolves_bit_identically(gpu_device):
    """The §4.6 discipline: the one check that catches a piece of state
    quietly left out -- for this backend that means ages, friendships, the
    canvas and the breath's phase as much as anything."""
    params = _resolve()
    original = _engine(gpu_device, params, seed=47)
    for _ in range(80):
        original.tick(params)
    original.queue_click(0.4, 0.4)
    original.tick(params)

    snapshot = checkpoint.capture(original, sim_hz=params.sim_hz)
    assert snapshot.meta["backend"] == "things"

    geometry = checkpoint.required_geometry(snapshot, backend="things")
    assert geometry == original.geometry
    # And the other backends must refuse it rather than misread it.
    assert checkpoint.required_geometry(snapshot, backend="layered") is None
    assert checkpoint.required_geometry(snapshot, backend="rhizotron") is None

    device, _ = gpu_device
    resumed = ThingsEngine(
        device, WIDTH, HEIGHT, params, seed=1, geometry=geometry)
    assert checkpoint.restore(resumed, snapshot)
    assert resumed.tick_count == original.tick_count
    assert resumed.pulse_state() == original.pulse_state()

    for _ in range(20):
        original.tick(params)
        resumed.tick(params)
    assert np.array_equal(_population(original), _population(resumed)), (
        "a resumed village diverged from the one it was captured from"
    )
    assert np.array_equal(_canvas(original), _canvas(resumed)), (
        "a resumed canvas diverged -- a ghost depended on something the "
        "checkpoint does not carry"
    )


def test_a_world_is_not_a_soil_column(gpu_device):
    """The layouts must not misread each other's files."""
    params = _resolve()
    engine = _engine(gpu_device, params, seed=53)
    engine.tick(params)
    snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)
    assert checkpoint.checkpoint_backend(snapshot) == "things"
    assert checkpoint.required_geometry(snapshot, backend="volumetric") is None


# ---------------------------------------------------------------------------
# The flash bound (§7, §18.5)
# ---------------------------------------------------------------------------


def test_the_flash_bound_holds_with_everything_sparkling(
    gpu_device, offscreen_target
):
    """§18.5: sparkles are the mode's only fast luminance actor, so the
    adversarial case is all of them at once -- sparkle rate at its
    ceiling, a full crowd, clicks landing -- and the per-pixel Oklab
    lightness step must still respect the limiter's budget."""
    params = _resolve(overrides={
        "things.sparkle_rate": 30.0,
        "things.sparkle_amp": 1.0,
        "things.friend_rate": 10.0,
        "things.spawn_rate": 10.0,
        "things.mature_seconds": 0.0,
    })
    engine = _engine(gpu_device, params, seed=59)
    _crowd(engine, 40, spacing=1.0)
    target, fmt = offscreen_target(WIDTH, HEIGHT)

    # Let the governor and the history converge before judging (§17.5's
    # instrument lesson, kept).
    for _ in range(60):
        engine.tick(params)
        engine.render(params, frac=1.0, target_view=target, target_format=fmt)

    previous = R.lightness(engine.read_final_rgba()[..., :3])
    limit = params.safety.max_luma_delta + F16_TOLERANCE
    for index in range(40):
        if index % 10 == 0:
            engine.queue_click(0.5, 0.5)
        engine.tick(params)
        engine.render(params, frac=1.0, target_view=target, target_format=fmt)
        current = R.lightness(engine.read_final_rgba()[..., :3])
        step = float(np.abs(current - previous).max())
        assert step <= limit, (
            f"frame {index}: per-pixel lightness stepped {step:.4f}, "
            f"budget {limit:.4f}"
        )
        previous = current


def test_the_exposure_governor_never_amplifies(gpu_device, offscreen_target):
    """§18.4: brightness is the census. However sparse the world, the
    resolved config pins amplification off and the measured exposure
    stays at or below one."""
    params = _resolve()
    assert params.safety.exposure_max == 1.0
    engine = _engine(gpu_device, params, seed=61)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    for _ in range(40):
        engine.tick(params)
        engine.render(params, frac=1.0, target_view=target, target_format=fmt)
    assert engine.read_stats()["exposure"] <= 1.0 + 1e-6
