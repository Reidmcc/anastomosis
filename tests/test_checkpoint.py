"""Checkpointing: does resuming actually continue the same simulation?

The property that matters is not "a file was written" but that a restored engine
evolves identically to the one it was captured from. If any piece of accumulated
state were missed -- an OU field, the homeostat integrators, the agent array --
the two would diverge on the very next tick, and that divergence is what the
first test looks for.

The other half is failure behaviour. This file is read at startup and may be
stale, truncated, or foreign, and none of those may stop the application from
opening. A *different window size* is deliberately not in that list any more:
the launch builds its engine at the geometry the saved field needs instead of
asking whether the field fits the window it found, and the tests below are what
hold that distinction in place.
"""

from __future__ import annotations

import json
import time

import numpy as np
import panelstub
import pytest

from anastomosis import (
    checkpoint,
    config,
    engine as engine_module,
    events,
    shaders,
)

SIZE = (128, 96)


def _params(layers: int = 2) -> config.Params:
    return config.Config(overrides={"render.layers": layers}).resolve()


def _make_engine(gpu_device, params, seed=1234, size=SIZE):
    device, _ = gpu_device
    return engine_module.Engine(device, size[0], size[1], params, seed=seed)


def _state(engine) -> dict[str, np.ndarray]:
    """Every mutable field on the GPU, whether or not the checkpoint saves it.

    Deliberately a superset of the snapshot: the fields left out are left out
    because the engine rewrites them every tick, and comparing them across a
    resume is what turns that claim into something the suite checks.
    """
    state = checkpoint.capture(engine).arrays
    for layer in engine.layers:
        for name in checkpoint.DERIVED_FIELDS:
            state[f"layer{layer.spec.index}.{name}"] = checkpoint._read_texture(
                engine.device, getattr(layer, name)
            )
    return state


# ---------------------------------------------------------------------------
# The central property
# ---------------------------------------------------------------------------


def test_a_restored_engine_continues_the_same_simulation(gpu_device, tmp_path):
    """The whole point: resuming must not be a restart in disguise.

    Both engines are advanced by the same further ticks, driven by the same
    events, after the handover. State left out of the checkpoint shows up here as
    a divergence in whichever field it feeds -- a much stronger check than
    comparing the captured arrays against themselves.
    """
    params = _params()
    # A high arrival rate so the climate field is genuinely perturbed by events
    # before the handover, rather than resting at its OU baseline.
    params.events.rate_per_hour = 4000.0

    original = _make_engine(gpu_device, params, seed=99)
    scheduler = events.EventScheduler(seed=7)
    tick_interval = 1.0 / params.sim_hz
    for _ in range(12):
        scheduler.update(tick_interval, params.events)
        original.tick(params, scheduler.pack(8)[0])
    assert scheduler.active, "the fixture should have events in flight"

    path = tmp_path / "checkpoint.npz"
    checkpoint.save(
        path,
        checkpoint.capture(original, scheduler=scheduler, sim_hz=params.sim_hz),
    )

    # Deliberately a different seed and a different event stream, so failing to
    # restore either is caught rather than masked.
    resumed = _make_engine(gpu_device, params, seed=555)
    resumed_scheduler = events.EventScheduler(seed=31)
    loaded = checkpoint.load(path)
    assert loaded is not None
    assert checkpoint.restore(resumed, loaded, scheduler=resumed_scheduler)

    assert resumed.tick_count == original.tick_count
    assert resumed.seed == original.seed
    assert resumed.hue_phase == pytest.approx(original.hue_phase, abs=1e-6)
    assert resumed_scheduler.active == scheduler.active

    # Advance both on identical event rows. The scheduler RNG is deliberately not
    # checkpointed, so beyond this point the two event streams would diverge --
    # what is under test here is the engine's state, not the scheduler's.
    rows = scheduler.pack(8)[0]
    for _ in range(5):
        original.tick(params, rows)
        resumed.tick(params, rows)

    before, after = _state(original), _state(resumed)
    assert set(before) == set(after)
    for name in before:
        # Bit-identical: same shaders, same inputs, same device. A mismatch means
        # some piece of state was reconstructed rather than restored -- or, for
        # the fields the snapshot skips, that one of them has quietly started
        # carrying state between ticks and now has to be saved.
        assert np.array_equal(before[name], after[name]), (
            f"{name} diverged after resuming, so it is not fully checkpointed"
        )
    assert any(name.endswith(".velocity") for name in before), (
        "the derived fields must be part of this comparison"
    )


def test_the_trail_pass_drains_the_deposit_accumulator():
    """Why the deposit buffer is not in the checkpoint.

    ``atomicExchange`` both reads and clears it, so between ticks it is
    guaranteed empty and there is nothing to save. If that ever changed, the
    final tick's agent deposits would be silently lost on every resume, so the
    invariant is asserted here rather than left as a comment.
    """
    source = shaders.load("trail.wgsl")
    assert "atomicExchange(&deposit_buf[index], 0u)" in source


# ---------------------------------------------------------------------------
# Refusing gracefully
# ---------------------------------------------------------------------------


def test_restoring_into_an_engine_of_another_shape_is_refused(gpu_device):
    """The upload is still shape-checked, whatever the launch decided to build.

    The launch path avoids this case by building the engine the checkpoint asks
    for; if something ever gets that wrong, a refusal is the failure mode, not a
    mismatched write to the GPU.
    """
    params = _params()
    original = _make_engine(gpu_device, params)
    for _ in range(2):
        original.tick(params)
    snapshot = checkpoint.capture(original, sim_hz=params.sim_hz)

    smaller = _make_engine(gpu_device, params, size=(96, 64))
    problems = checkpoint.compatibility_problems(smaller, snapshot)
    assert problems, "a size mismatch must be detected"
    assert any("128x96" in problem for problem in problems)

    assert not checkpoint.restore(smaller, snapshot)
    # Nothing was applied, so the engine still holds its own seeded state.
    assert smaller.tick_count == 0


def test_a_layer_count_change_is_refused(gpu_device):
    """Layer 0's shapes are identical either way, so only the metadata catches this."""
    original = _make_engine(gpu_device, _params(layers=2))
    snapshot = checkpoint.capture(original, sim_hz=20.0)
    other = _make_engine(gpu_device, _params(layers=1))
    assert not checkpoint.restore(other, snapshot)


# ---------------------------------------------------------------------------
# The geometry a checkpoint asks to be launched into
# ---------------------------------------------------------------------------


def test_the_required_geometry_is_the_one_it_was_captured_from(gpu_device):
    """What the launch reads out of the file to decide what to build."""
    engine = _make_engine(gpu_device, _params())
    snapshot = checkpoint.capture(engine, sim_hz=20.0)
    required = checkpoint.required_geometry(snapshot)
    assert required == engine.geometry
    assert not required.differences(engine.geometry)


def test_a_geometry_this_engine_could_not_build_is_ignored():
    """Geometry out of a file decides how much memory a launch allocates.

    So it is bounds-checked before anything is created from it -- otherwise a
    corrupt or foreign file could turn the launch into an allocation failure,
    which is precisely the "cannot open the application" outcome the whole
    module exists to avoid.
    """
    snapshot = checkpoint.Checkpoint(meta={
        "version": checkpoint.FORMAT_VERSION,
        "geometry": {
            "sim_width": 10**6,
            "sim_height": 10**6,
            "layers": [{
                "index": 0, "width": 10**6, "height": 10**6,
                "agent_count": 10**12, "psi_width": 256, "psi_height": 256,
                "climate_width": 64, "climate_height": 36,
            }],
        },
    })
    assert checkpoint.required_geometry(snapshot) is None


def test_metadata_that_describes_no_geometry_is_ignored():
    for block in ("nonsense", {}, {"sim_width": 128, "sim_height": 96},
                  {"sim_width": 128, "sim_height": 96, "layers": ["nope"]}):
        snapshot = checkpoint.Checkpoint(
            meta={"version": checkpoint.FORMAT_VERSION, "geometry": block}
        )
        assert checkpoint.required_geometry(snapshot) is None


def test_a_version_1_checkpoint_is_still_readable(gpu_device):
    """Version 1 kept the same sizes in two other keys. Nobody loses a field to
    an upgrade over that."""
    engine = _make_engine(gpu_device, _params())
    engine.tick(_params())
    snapshot = checkpoint.capture(engine, sim_hz=20.0)

    meta = json.loads(json.dumps(snapshot.meta))
    geometry = meta.pop("geometry")
    meta["version"] = 1
    meta["engine"]["width"] = geometry["sim_width"]
    meta["engine"]["height"] = geometry["sim_height"]
    meta["layers"] = geometry["layers"]
    old = checkpoint.Checkpoint(meta=meta, arrays=snapshot.arrays)

    assert checkpoint.required_geometry(old) == engine.geometry
    resumed = _make_engine(gpu_device, _params(), seed=7)
    assert checkpoint.restore(resumed, old)
    assert resumed.tick_count == engine.tick_count


def test_a_version_2_checkpoint_without_the_morphology_climate_is_readable(
    gpu_device,
):
    """The morphology climate pair postdates version 2, so those files lack it.

    A field that a build did not have cannot be demanded of a file that build
    wrote. The resume keeps the freshly seeded `climate_c` and loses that one
    channel's history -- which drifts back to a normal spread within a couple of
    OU timescales -- rather than losing the hours of field that came with it.
    """
    params = _params()
    engine = _make_engine(gpu_device, params)
    engine.tick(params)
    snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)

    old = checkpoint.Checkpoint(
        meta=json.loads(json.dumps(snapshot.meta)),
        arrays={
            name: array
            for name, array in snapshot.arrays.items()
            if not name.endswith(".climate_c")
        },
    )
    old.meta["version"] = 2

    resumed = _make_engine(gpu_device, params, seed=7)
    seeded = [
        checkpoint._read_texture(
            resumed.device, layer.climate_c.textures[layer.climate_c.index]
        )
        for layer in resumed.layers
    ]
    assert checkpoint.restore(resumed, old)
    assert resumed.tick_count == engine.tick_count
    for layer, before in zip(resumed.layers, seeded):
        after = checkpoint._read_texture(
            resumed.device, layer.climate_c.textures[layer.climate_c.index]
        )
        assert np.array_equal(before, after), (
            "a field the file predates must be left as the engine seeded it"
        )


def test_the_morphology_climate_is_part_of_a_current_checkpoint(gpu_device):
    """It is stateful -- advected, diffused and OU-stepped from its own previous
    value -- so leaving it out is a resume that continues a different field."""
    engine = _make_engine(gpu_device, _params())
    snapshot = checkpoint.capture(engine, sim_hz=20.0)
    assert "climate_c" in checkpoint.PAIR_FIELDS
    for layer in engine.layers:
        pair = layer.climate_c
        assert snapshot.arrays[f"layer{layer.spec.index}.climate_c"].shape == (
            pair.textures[0].height, pair.textures[0].width, 4
        )


def test_the_feature_size_walk_is_restored_rather_than_recentred(gpu_device):
    """The walk on the reaction's diffusion rate is engine state like any other.

    Dropping it would put a resumed field back at the middle of its band and let
    it wander somewhere else from there -- a slow change in feature size that no
    field comparison at the moment of the resume would catch.
    """
    params = _params()
    engine = _make_engine(gpu_device, params, seed=3)
    for _ in range(20):
        engine.tick(params)
    assert engine._du_walk != 0.0, "the fixture should have walked somewhere"
    snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)

    resumed = _make_engine(gpu_device, params, seed=808)
    assert checkpoint.restore(resumed, snapshot)
    assert resumed._du_walk == engine._du_walk
    # And the stream it is driven by, so the walk continues rather than starting
    # a different path from the same point.
    for _ in range(5):
        engine.tick(params)
        resumed.tick(params)
    assert resumed._du_walk == engine._du_walk


def test_a_resumed_viewpoint_carries_on_from_where_it_was(gpu_device):
    """The camera drift is saved as an offset, and the states behind it are not.

    That is deliberate -- the offset is the part that was on screen, and the
    walk and the lag that produced it are derivable from it. But it means a
    resumed engine has an offset with nothing driving it, and the failure mode
    if the deriving is skipped is not a jump: it is a viewpoint frozen exactly
    where it stopped, for the rest of the session. Nothing about a still image
    looks wrong, so it is asserted here rather than looked for.
    """
    params = _params()
    engine = _make_engine(gpu_device, params, seed=3)
    for _ in range(200):
        engine._update_parallax(params, 1.0 / 30.0, len(engine.layers))
    was = [list(pair) for pair in engine._parallax]
    assert max(abs(x) for pair in was for x in pair) > 1e-3, (
        "the fixture should have drifted somewhere")
    snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)

    resumed = _make_engine(gpu_device, params, seed=808)
    assert checkpoint.restore(resumed, snapshot)
    assert resumed._parallax == was

    resumed._update_parallax(params, 1.0 / 30.0, len(resumed.layers))
    moved = [
        abs(now - before)
        for pair, old in zip(resumed._parallax, was)
        for now, before in zip(pair, old)
    ]
    assert max(moved) > 0.0, "the resumed viewpoint is frozen where it stopped"
    # Continuing, not restarting: one frame moves it by a frame's worth.
    assert max(moved) < 0.05, "the resumed viewpoint jumped rather than carried on"


def test_a_walk_state_this_build_cannot_use_is_not_fatal(gpu_device):
    """Every field of the metadata is untrusted; none of them may cost the field."""
    params = _params()
    engine = _make_engine(gpu_device, params, seed=3)
    engine.tick(params)
    snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)
    snapshot.meta["engine"]["walk_rng"] = {"bit_generator": "NotARealGenerator"}
    snapshot.meta["engine"]["du_walk"] = "not a number"

    resumed = _make_engine(gpu_device, params, seed=808)
    assert checkpoint.restore(resumed, snapshot)
    assert resumed.tick_count == engine.tick_count
    assert resumed._du_walk == 0.0
    resumed.tick(params)  # and the fresh stream still drives it


def test_a_future_format_version_is_refused(gpu_device):
    engine = _make_engine(gpu_device, _params())
    snapshot = checkpoint.capture(engine, sim_hz=20.0)
    snapshot.meta["version"] = checkpoint.FORMAT_VERSION + 1
    assert checkpoint.compatibility_problems(engine, snapshot)
    assert not checkpoint.restore(engine, snapshot)


def test_a_missing_array_is_refused_before_anything_is_uploaded(gpu_device):
    """Shapes are validated against the live resources, not just the metadata."""
    engine = _make_engine(gpu_device, _params())
    snapshot = checkpoint.capture(engine, sim_hz=20.0)
    del snapshot.arrays["layer0.pigment"]
    assert not checkpoint.restore(engine, snapshot)


def test_a_truncated_file_is_ignored_rather_than_fatal(tmp_path):
    """Startup reads this file. It must never be able to stop the app opening."""
    path = tmp_path / "checkpoint.npz"
    path.write_bytes(b"PK\x03\x04 not really a checkpoint")
    assert checkpoint.load(path) is None


def test_a_missing_file_is_not_an_error(tmp_path):
    assert checkpoint.load(tmp_path / "nothing-here.npz") is None


def test_a_file_without_metadata_is_ignored(tmp_path):
    path = tmp_path / "checkpoint.npz"
    with path.open("wb") as handle:
        np.savez(handle, stats=np.zeros(4, dtype=np.uint8))
    assert checkpoint.load(path) is None


def test_load_does_not_permit_pickled_objects(tmp_path):
    """Nothing on disk may execute anything at startup."""
    path = tmp_path / "checkpoint.npz"
    with path.open("wb") as handle:
        np.savez(handle, meta=np.array({"version": 1}, dtype=object))
    assert checkpoint.load(path) is None


# ---------------------------------------------------------------------------
# Persistence mechanics
# ---------------------------------------------------------------------------


def test_saving_is_atomic_and_leaves_no_temporary_behind(gpu_device, tmp_path):
    """A crash mid-write must leave the previous checkpoint, not a broken one."""
    params = _params()
    engine = _make_engine(gpu_device, params)
    path = tmp_path / "checkpoint.npz"
    checkpoint.save(path, checkpoint.capture(engine, sim_hz=params.sim_hz))
    engine.tick(params)
    checkpoint.save(path, checkpoint.capture(engine, sim_hz=params.sim_hz))

    assert path.exists()
    assert not (tmp_path / "checkpoint.npz.tmp").exists()
    assert checkpoint.load(path).tick_count == 1


def test_discard_removes_the_checkpoint_and_any_interrupted_write(tmp_path):
    path = tmp_path / "checkpoint.npz"
    tmp = tmp_path / "checkpoint.npz.tmp"
    path.write_bytes(b"x")
    tmp.write_bytes(b"y")
    checkpoint.discard(path)
    assert not path.exists() and not tmp.exists()
    checkpoint.discard(path)  # idempotent, so a reset never fails on a stale file


def test_background_saver_writes_and_can_be_waited_on(gpu_device, tmp_path):
    engine = _make_engine(gpu_device, _params())
    path = tmp_path / "checkpoint.npz"
    saver = checkpoint.BackgroundSaver()
    assert saver.submit(path, checkpoint.capture(engine, sim_hz=20.0))
    saver.join()
    assert not saver.busy
    assert checkpoint.load(path) is not None


def test_metadata_is_plain_json(gpu_device):
    """Kept JSON-able, so the format stays inspectable and pickle-free."""
    engine = _make_engine(gpu_device, _params())
    snapshot = checkpoint.capture(
        engine, scheduler=events.EventScheduler(seed=1), sim_hz=20.0
    )
    assert json.loads(json.dumps(snapshot.meta)) == snapshot.meta


def test_describe_reports_how_much_simulation_was_saved():
    snapshot = checkpoint.Checkpoint(meta={
        "version": checkpoint.FORMAT_VERSION,
        "created": time.time() - 120.0,
        "sim_hz": 20.0,
        "engine": {"tick_count": 20 * (3600 + 12 * 60)},
    })
    assert "1h 12m" in snapshot.describe()
    assert "2m ago" in snapshot.describe()


def test_describe_survives_metadata_it_cannot_use():
    assert checkpoint.Checkpoint().describe()


# ---------------------------------------------------------------------------
# The application wiring
#
# Driven through an offscreen canvas so the whole path -- setup, resume, the
# interval save, the save on close, reset -- is exercised rather than just the
# pieces it is built from. The test device is injected, and both the config and
# the checkpoint are pointed at tmp_path, so nothing touches the real user state.
# ---------------------------------------------------------------------------


def _offscreen(gpu_device, monkeypatch, tmp_path, loop=None, size=SIZE, **options):
    from rendercanvas.offscreen import RenderCanvas, loop as offscreen_loop
    from anastomosis import app as app_module, device as device_module

    device, info = gpu_device
    monkeypatch.setattr(
        device_module, "request_device", lambda *a, **k: (device, info)
    )
    # Hot reload is a separate concern and would leave a watcher thread behind.
    monkeypatch.setattr(app_module.Application, "_start_hot_reload", lambda self: None)
    monkeypatch.setattr(
        app_module.Application,
        "_make_canvas",
        lambda self: (
            RenderCanvas(size=size, update_mode="manual"),
            loop or offscreen_loop,
            False,
        ),
    )
    return app_module.Application(app_module.AppOptions(
        width=size[0], height=size[1], ui=False,
        config_path=tmp_path / "config.toml",
        checkpoint_path=tmp_path / "checkpoint.npz",
        **options,
    ))


def test_a_new_session_picks_up_where_the_last_one_left_off(
    gpu_device, monkeypatch, tmp_path
):
    """The requirement, end to end: launch, run, save, relaunch, continue."""
    first = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    first.setup()
    assert first.resumed_from is None
    for _ in range(4):
        first.draw_frame()
    ticks = first.engine.tick_count
    assert ticks > 0
    assert first.save_checkpoint(blocking=True)
    assert "saved" in first.checkpoint_status()

    second = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    second.setup()
    assert second.resumed_from is not None
    assert second.engine.tick_count == ticks
    second.draw_frame()  # and it runs on from there


def test_a_new_window_size_resumes_the_field_rather_than_resetting_it(
    gpu_device, monkeypatch, tmp_path
):
    """The requirement: launch in the state the saved field needs.

    The window is presentation, so a session that reopens at a different size
    keeps the field and simply shows it at the new size -- exactly what a
    mid-session resize already does -- instead of discarding hours of maturity
    over a mismatch that never concerned the simulation.
    """
    first = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    first.setup()
    for _ in range(4):
        first.draw_frame()
    ticks = first.engine.tick_count
    assert first.save_checkpoint(blocking=True)

    bigger = (SIZE[0] * 2, SIZE[1] + 64)
    second = _offscreen(
        gpu_device, monkeypatch, tmp_path, size=bigger, checkpoint_seconds=0.0
    )
    second.setup()

    assert second.resumed_from is not None, "the saved field must survive a resize"
    assert second.engine.tick_count == ticks
    # The simulation is in the shape the checkpoint needed; only the output
    # targets followed the window.
    assert (second.engine.sim_width, second.engine.sim_height) == SIZE
    assert (second.engine.width, second.engine.height) == bigger
    second.draw_frame()  # and it renders into the new window


def test_a_structural_config_change_no_longer_throws_the_field_away(
    gpu_device, monkeypatch, tmp_path
):
    """Layer count follows the config, and the config may have been edited.

    The saved field wins for as long as it exists; the new value takes effect
    when there is a new field to apply it to.
    """
    first = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    first.setup()
    first.draw_frame()
    layers = len(first.engine.layers)
    first.save_checkpoint(blocking=True)

    (tmp_path / "config.toml").write_text(
        '[overrides]\n"render.layers" = 1\n', encoding="utf-8"
    )
    second = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    second.setup()
    assert second.params.render.layers == 1, "the edited config must be in force"
    assert second.resumed_from is not None
    assert len(second.engine.layers) == layers, (
        "the resumed field keeps the geometry it was grown at"
    )

    # ...and the new layer count lands on the next field, which is what the
    # reset is for.
    second.reset_simulation()
    assert len(second.engine.layers) == 1


def test_an_unusable_saved_state_still_starts_a_session(
    gpu_device, monkeypatch, tmp_path
):
    """Reading the checkpoint now decides how the engine is built, so a foreign
    file must not be able to take the launch down with it."""
    app = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    app.setup()
    app.draw_frame()
    app.save_checkpoint(blocking=True)

    snapshot = checkpoint.load(tmp_path / "checkpoint.npz")
    snapshot.meta["geometry"]["layers"][0]["width"] = 10**6
    checkpoint.save(tmp_path / "checkpoint.npz", snapshot)

    fresh = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    fresh.setup()
    assert fresh.resumed_from is None
    assert fresh.engine.tick_count == 0
    assert (fresh.engine.sim_width, fresh.engine.sim_height) == SIZE
    fresh.draw_frame()


def test_a_saved_geometry_this_device_refuses_falls_back_to_a_fresh_field(
    gpu_device, monkeypatch, tmp_path
):
    """Plausible geometry can still fail to allocate -- a file from a machine
    with more memory, say. The session opens anyway."""
    first = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    first.setup()
    first.draw_frame()
    first.save_checkpoint(blocking=True)

    real_engine = engine_module.Engine
    attempts: list = []

    def refuses_the_first_shape(*args, **kwargs):
        attempts.append(kwargs.get("geometry"))
        if len(attempts) == 1:
            raise MemoryError("not enough device memory")
        return real_engine(*args, **kwargs)

    monkeypatch.setattr(engine_module, "Engine", refuses_the_first_shape)
    app = _offscreen(
        gpu_device, monkeypatch, tmp_path, size=(192, 160), checkpoint_seconds=0.0
    )
    app.setup()

    assert [g.sim_width for g in attempts] == [SIZE[0], 192], (
        "the saved geometry is tried first, then the one this window implies"
    )
    assert app.resumed_from is None
    assert app.engine.tick_count == 0
    app.draw_frame()


def test_reset_starts_a_new_field_and_drops_the_saved_one(
    gpu_device, monkeypatch, tmp_path
):
    """What the control panel button does. The old field must not come back."""
    app = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    app.setup()
    for _ in range(4):
        app.draw_frame()
    app.save_checkpoint(blocking=True)
    before = app.engine

    app.reset_simulation()
    assert app.engine is not before
    assert app.engine.tick_count == 0
    assert not (tmp_path / "checkpoint.npz").exists()
    assert app.resumed_from is None
    app.draw_frame()  # the new field runs


def test_starting_with_reset_ignores_a_saved_field(gpu_device, monkeypatch, tmp_path):
    """``--reset`` on the command line, i.e. resume=False."""
    first = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.0)
    first.setup()
    for _ in range(3):
        first.draw_frame()
    first.save_checkpoint(blocking=True)

    fresh = _offscreen(
        gpu_device, monkeypatch, tmp_path, resume=False, checkpoint_seconds=0.0
    )
    fresh.setup()
    assert fresh.resumed_from is None
    assert fresh.engine.tick_count == 0


def test_a_running_session_checkpoints_on_its_own(gpu_device, monkeypatch, tmp_path):
    """The periodic save, with the interval shortened so a test can see it."""
    app = _offscreen(gpu_device, monkeypatch, tmp_path, checkpoint_seconds=0.01)
    app.setup()
    app.draw_frame()
    app.draw_frame()
    app._saver.join()
    assert (tmp_path / "checkpoint.npz").exists()


def test_closing_the_application_saves_the_state(gpu_device, monkeypatch, tmp_path):
    """Closing the window ends most sessions, so it must checkpoint like a tick."""
    holder: dict = {}

    class ClosingLoop:
        """Stands in for the user watching for a while and then closing the window."""

        def run(self):
            for _ in range(3):
                holder["app"].draw_frame()

    app = _offscreen(gpu_device, monkeypatch, tmp_path, loop=ClosingLoop())
    holder["app"] = app
    app.run()

    saved = checkpoint.load(tmp_path / "checkpoint.npz")
    assert saved is not None
    assert saved.tick_count == app.engine.tick_count


def test_opting_out_writes_nothing(gpu_device, monkeypatch, tmp_path):
    app = _offscreen(
        gpu_device, monkeypatch, tmp_path, checkpoint=False, checkpoint_seconds=0.01
    )
    app.setup()
    for _ in range(3):
        app.draw_frame()
    assert not app.save_checkpoint(blocking=True)
    assert not (tmp_path / "checkpoint.npz").exists()
    assert app.checkpoint_status() == "off"


# ---------------------------------------------------------------------------
# The reset affordance in the control panel
# ---------------------------------------------------------------------------


class _StubApp(panelstub.PanelApp):
    """The shared panel stand-in, counting what this module watches."""

    def __init__(self):
        super().__init__()
        self.resets = 0
        self.fullscreen_toggles = 0

    def toggle_fullscreen(self) -> bool:
        self.fullscreen_toggles += 1
        return False

    def checkpoint_status(self):
        return "saved 3m ago"

    def reset_simulation(self):
        self.resets += 1


def _panel(monkeypatch, answer):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6", reason="the control panel needs the [ui] extra")
    from PySide6 import QtWidgets

    from anastomosis.ui.control_panel import ControlPanel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # The confirmation is modal, so it is answered for the test rather than shown.
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: answer)
    )
    app = _StubApp()
    panel = ControlPanel(app)
    button = next(
        widget for widget in panel.findChildren(QtWidgets.QPushButton)
        if widget.text() == "Reset simulation"
    )
    return app, panel, button


def test_confirming_the_reset_resets_the_simulation(monkeypatch):
    """The button the requirement asks for, wired to the application."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app, _panel_widget, button = _panel(monkeypatch, QtWidgets.QMessageBox.Yes)
    button.click()
    assert app.resets == 1


def test_declining_the_reset_leaves_the_field_alone(monkeypatch):
    """A mature field is unrecoverable once discarded, so the dialog must mean it."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app, _panel_widget, button = _panel(monkeypatch, QtWidgets.QMessageBox.No)
    button.click()
    assert app.resets == 0


def test_the_panel_reports_the_saved_state(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app, panel, _button = _panel(monkeypatch, QtWidgets.QMessageBox.No)

    class Engine:
        tick_count = 1200
        # The status line names the shape the field is actually running at, so
        # a stand-in engine has to have one.
        geometry = engine_module.Geometry.derive(
            *panelstub.SIZE, config.Config().resolve())

        def read_stats(self):
            return {"mean_v": 0.12, "mean_activity": 0.0012}

    app.engine = Engine()
    panel._refresh_status()
    assert panel.status_labels["checkpoint"].text() == "saved 3m ago"


# ---------------------------------------------------------------------------
# Event scheduler state -- no GPU needed
# ---------------------------------------------------------------------------


def test_in_flight_events_survive_a_round_trip():
    """Dropping a mid-envelope event would be exactly the step envelopes prevent."""
    params = config.EventParams(rate_per_hour=100000.0)
    scheduler = events.EventScheduler(seed=3)
    for _ in range(20):
        scheduler.update(1.0, params)
    assert scheduler.active, "the fixture needs at least one active event"

    resumed = events.EventScheduler(seed=99)
    resumed.load_state(json.loads(json.dumps(scheduler.state())))

    assert resumed.active == scheduler.active
    assert resumed.describe() == scheduler.describe()
    assert resumed.spawned == scheduler.spawned
    assert resumed.pack(8) == scheduler.pack(8)


def test_restored_events_carry_named_channels():
    """`pack` reads the channels by name, so a restored event must have names.

    JSON stores the channels positionally and a plain list of floats survives
    everything the round-trip test checks -- equality, `describe`, arithmetic --
    and then takes down every frame in `pack`, which is the draw callback: a
    resumed session that renders nothing but a grey window.
    """
    scheduler = events.EventScheduler(seed=3)
    scheduler.load_state({
        "active": [{
            "x": 0.5, "y": 0.5, "radius": 0.1, "peak": 1.0,
            "channels": [1.0, -0.5, 0.25, 0.125, 0.0, 0.0, 0.0],
            "attack": 10.0, "hold": 10.0, "release": 10.0,
            "elapsed": 10.0, "kind": "bloom",
        }],
    })
    (restored,) = scheduler.active
    assert isinstance(restored.channels, events.Channels)

    (row,), count = scheduler.pack(8)
    assert count == 1
    assert row["chan_feed"] == pytest.approx(1.0)
    assert row["chan_kill"] == pytest.approx(-0.5)
    assert row["chan_repel"] == pytest.approx(0.0)


def test_a_checkpoint_from_before_the_severance_channels_still_loads():
    """Channel count is a version marker: short means old, not corrupt.

    A checkpoint written before rift events existed carries four channels. It
    describes a perfectly valid bloom, and dropping it would lose an in-flight
    envelope for no reason -- the channels it never heard of simply do nothing.
    """
    scheduler = events.EventScheduler(seed=3)
    scheduler.load_state({
        "active": [{
            "x": 0.5, "y": 0.5, "radius": 0.1, "peak": 1.0,
            "channels": [1.0, -0.55, 0.15, 0.35],
            "attack": 10.0, "hold": 10.0, "release": 10.0,
            "elapsed": 10.0, "kind": "bloom",
        }],
    })
    (restored,) = scheduler.active
    assert restored.channels == events.Channels(
        feed=1.0, kill=-0.55, flow=0.15, hue=0.35
    )
    assert restored.channels.decay == 0.0
    (row,), _ = scheduler.pack(8)
    assert row["chan_decay"] == pytest.approx(0.0)


def test_finished_events_are_not_resurrected():
    scheduler = events.EventScheduler(seed=3)
    scheduler.load_state({
        "active": [{
            "x": 0.5, "y": 0.5, "radius": 0.1, "peak": 1.0,
            "channels": [1.0, 0.0, 0.0, 0.0],
            "attack": 1.0, "hold": 1.0, "release": 1.0,
            "elapsed": 99.0, "kind": "bloom",
        }],
        "time_to_next": 12.0,
        "spawned": 4,
    })
    assert scheduler.active == []
    assert scheduler._time_to_next == pytest.approx(12.0)


def test_unreadable_event_state_costs_at_most_the_events():
    """A partly readable checkpoint must not take the whole restore down."""
    scheduler = events.EventScheduler(seed=3)
    scheduler.load_state({
        "active": ["not an event", {"nonsense": 1}],
        "time_to_next": "soon",
        "spawned": None,
    })
    assert scheduler.active == []
    assert scheduler._time_to_next is None
    assert scheduler.spawned == 0
    # And it still schedules normally afterwards.
    scheduler.update(1.0, config.EventParams())


# ---------------------------------------------------------------------------
# Defaults, which are themselves part of the requirement
# ---------------------------------------------------------------------------


def test_state_is_saved_every_five_minutes_by_default():
    assert checkpoint.DEFAULT_INTERVAL_SECONDS == 300.0


def test_resuming_is_the_default_behaviour():
    from anastomosis.app import AppOptions

    options = AppOptions()
    assert options.checkpoint and options.resume
    assert options.checkpoint_seconds == checkpoint.DEFAULT_INTERVAL_SECONDS


def test_the_command_line_can_reset_and_opt_out():
    from anastomosis.__main__ import build_parser

    args = build_parser().parse_args([])
    assert not args.reset and not args.no_checkpoint
    args = build_parser().parse_args(["--reset", "--checkpoint-interval", "60"])
    assert args.reset and args.checkpoint_interval == 60.0


def test_the_checkpoint_lives_in_the_state_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert checkpoint.default_checkpoint_path() == (
        tmp_path / "anastomosis" / "checkpoint.npz"
    )
