"""Running on a laptop's integrated GPU -- DESIGN.md §8.3.

The claim this file exists to hold is in two halves, and they fail in
completely different ways, so they are asserted separately.

The first half is that it *runs*: the pipeline is core WebGPU throughout, so
every adapter that implements the specification's guaranteed minima can build
it. That is a property of the shaders, it is invisible from a machine with a
generous adapter -- the software adapter used in CI reports limits far above
the minima, as does every discrete card -- and it would be broken by a single
extra binding in a single pass, silently, on a device nobody testing has. So
it is asserted against the WGSL source rather than against a device.

The second half is that it runs *acceptably*, which is a question about cost
and is answered by defaults: how much is simulated, what the governor does
when a frame runs long, what happens on a battery, and what happens when the
device goes away -- which on a laptop is a lid closing rather than a driver
bug. None of those needs a GPU to assert either, because all of them are
decisions made in Python about numbers.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from anastomosis import app as app_module
from anastomosis import config as config_module
from anastomosis import device as device_module
from anastomosis import engine as engine_module
from anastomosis import power as power_module
from anastomosis import shaders


# ---------------------------------------------------------------------------
# Core WebGPU limits: the portability half
# ---------------------------------------------------------------------------

# The guaranteed minima from the WebGPU specification's default limits -- what
# an adapter is allowed to report and still be a conforming implementation.
# Integrated GPUs are the machines that actually sit near them; the software
# adapter and every discrete card report far higher, which is exactly why a
# regression here would never show up in CI as anything but this test.
CORE_LIMITS = {
    "storage_textures": 4,   # maxStorageTexturesPerShaderStage
    "storage_buffers": 8,    # maxStorageBuffersPerShaderStage
    "sampled_textures": 16,  # maxSampledTexturesPerShaderStage
    "samplers": 16,          # maxSamplersPerShaderStage
    "uniform_buffers": 12,   # maxUniformBuffersPerShaderStage
    "bind_groups": 4,        # maxBindGroups
    "workgroup_storage": 16384,   # maxComputeWorkgroupStorageSize, bytes
    "workgroup_invocations": 256,  # maxComputeInvocationsPerWorkgroup
}

# maxComputeWorkgroupSizeX / Y / Z.
CORE_WORKGROUP_SIZE = (256, 256, 64)

_BINDING = re.compile(
    r"@group\((\d+)\)\s*@binding\(\d+\)\s*var(?:<(?P<space>[^>]*)>)?\s*"
    r"\w+\s*:\s*(?P<type>[\w<]+)"
)
_WORKGROUP_VAR = re.compile(
    r"var<workgroup>\s*\w+\s*:\s*array<\s*(?P<of>[\w<>]+?)\s*,\s*(?P<n>\d+)\s*>")
_WORKGROUP_SIZE = re.compile(
    r"@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?(?:,\s*(\d+)\s*)?\)")

# Byte sizes of the workgroup-array element types actually used here. A type
# not in this table is a new one, and the KeyError is the point: a silent
# default would be a limit checked against a number nobody computed.
_ELEMENT_BYTES = {"vec4<f32>": 16, "vec2<f32>": 8, "f32": 4, "u32": 4, "i32": 4}


def _counts(source: str) -> dict[str, int]:
    """Bindings in one expanded module, by the limit each is charged against."""
    counts = {key: 0 for key in CORE_LIMITS}
    groups = set()
    for match in _BINDING.finditer(source):
        groups.add(int(match.group(1)))
        space = (match.group("space") or "").strip()
        kind = match.group("type")
        if space.startswith("storage"):
            counts["storage_buffers"] += 1
        elif space.startswith("uniform"):
            counts["uniform_buffers"] += 1
        elif kind.startswith("texture_storage"):
            counts["storage_textures"] += 1
        elif kind.startswith("texture_"):
            counts["sampled_textures"] += 1
        elif kind.startswith("sampler"):
            counts["samplers"] += 1
    counts["bind_groups"] = len(groups)

    storage = 0
    for match in _WORKGROUP_VAR.finditer(source):
        storage += _ELEMENT_BYTES[match.group("of")] * int(match.group("n"))
    counts["workgroup_storage"] = storage
    return counts


@pytest.mark.parametrize("name", shaders.all_shader_names())
def test_every_shader_fits_core_webgpu_limits(name):
    """No pass may need an adapter better than the specification guarantees.

    `device.request_device` asks for no raised limits and no optional
    features, deliberately, so that one build runs on Vulkan, Metal and DX12
    and on integrated and discrete GPUs alike. That promise is only as good as
    the passes: one more storage texture in one climate pass and the
    application stops opening on the machines this section is about, while
    continuing to work perfectly everywhere it is developed.
    """
    source = shaders.load(name)
    counts = _counts(source)
    for key, ceiling in CORE_LIMITS.items():
        assert counts[key] <= ceiling, (
            f"{name} uses {counts[key]} {key}, above core WebGPU's {ceiling}"
        )


@pytest.mark.parametrize("name", shaders.all_shader_names())
def test_every_workgroup_fits_core_webgpu_limits(name):
    """Workgroup shape, against maxComputeInvocationsPerWorkgroup and the axes."""
    for match in _WORKGROUP_SIZE.finditer(shaders.load(name)):
        dims = tuple(int(d) if d else 1 for d in match.groups())
        product = dims[0] * dims[1] * dims[2]
        assert product <= CORE_LIMITS["workgroup_invocations"], (
            f"{name} dispatches {product} invocations per workgroup"
        )
        for axis, (size, ceiling) in enumerate(zip(dims, CORE_WORKGROUP_SIZE)):
            assert size <= ceiling, f"{name} workgroup axis {axis} is {size}"


def test_no_optional_features_or_raised_limits_are_requested():
    """The source of the promise the two tests above protect.

    Asserted on the text because there is nothing else to assert it on: a
    device request that quietly grew a `required_features` would keep working
    on every machine anybody develops this on.
    """
    source = pathlib.Path(device_module.__file__).read_text(encoding="utf-8")
    assert "required_features" not in source
    assert "required_limits" not in source


# ---------------------------------------------------------------------------
# Which GPU the session asks for
# ---------------------------------------------------------------------------


def test_auto_asks_the_platform_rather_than_for_the_discrete_card():
    """The default must not be "high-performance".

    On a laptop with switchable graphics that is a request for the discrete
    GPU, which is the wrong answer for a program sized to leave the machine
    usable and expected to run for days on a battery.
    """
    assert device_module.DEFAULT_GPU_CHOICE == "auto"
    assert device_module._PREFERENCE["auto"] is None
    assert device_module._PREFERENCE["integrated"] == "low-power"
    assert device_module._PREFERENCE["discrete"] == "high-performance"


@pytest.mark.parametrize("given,expected", [
    ("integrated", "integrated"), ("iGPU", "integrated"), ("dgpu", "discrete"),
    ("DISCRETE", "discrete"), (None, "auto"), ("", "auto"), ("nonsense", "auto"),
])
def test_gpu_choices_are_normalised(given, expected):
    assert device_module.normalise_gpu_choice(given) == expected


def test_no_adapter_is_a_message_rather_than_an_attribute_error(monkeypatch):
    """A driver with nothing to offer returns None rather than raising.

    Reading `.info` off it reported an AttributeError from inside the device
    module -- true, useless, and three frames from anything that says "this
    machine has no working GPU driver".
    """
    monkeypatch.setattr(
        device_module.wgpu.gpu, "request_adapter_sync",
        lambda **kwargs: None, raising=False)
    with pytest.raises(device_module.NoAdapter) as caught:
        device_module.request_device()
    assert "adapter" in str(caught.value).lower()


# ---------------------------------------------------------------------------
# The cell ceiling
# ---------------------------------------------------------------------------


def _cells(geometry) -> int:
    return sum(layer.width * layer.height for layer in geometry.layers)


def _derive(width, height, budget=0, scale=1.0):
    params = config_module.Config().resolve()
    params.render.base_scale = scale
    params.render.cell_budget = budget
    return engine_module.Geometry.derive(width, height, params)


def test_no_ceiling_is_the_shipped_default():
    """The card of §8.1 keeps its native 1440p front layer.

    The ceiling exists for machines without that card's headroom, and §8.1
    spends the headroom on exactly this: "front layer simulated at native
    1440p (no upscale) for fine filament detail". A default that quietly
    stopped doing so would be a regression on the target hardware.
    """
    assert config_module.RenderParams().cell_budget == 0
    front = _derive(2560, 1440).layers[0]
    assert (front.width, front.height) == (2560, 1440)


def test_the_ceiling_leaves_a_window_that_already_fits_alone():
    """It is a ceiling, not a target: it may shrink and must never grow.

    A 1080p panel is 2.72 M cells against the integrated budget's 3 M, which
    is the case the number was chosen for -- the commonest laptop display
    simulates at native resolution and never notices this exists.
    """
    budget = config_module.INTEGRATED_CELL_BUDGET
    for width, height in [(1280, 720), (1600, 900), (1920, 1080)]:
        plain = _derive(width, height)
        capped = _derive(width, height, budget=budget)
        assert _cells(capped) == _cells(plain)
        assert capped.layers[0].width == plain.layers[0].width


@pytest.mark.parametrize("width,height", [
    (2560, 1440), (2560, 1600), (2880, 1800), (3840, 2160),
])
def test_the_ceiling_binds_on_the_panels_it_is_for(width, height):
    """And binds on the *stack*, since that is what costs bandwidth."""
    budget = config_module.INTEGRATED_CELL_BUDGET
    assert _cells(_derive(width, height)) > budget
    capped = _derive(width, height, budget=budget)
    assert _cells(capped) <= budget
    # Still a whole stack at the configured layer count -- the ceiling shrinks
    # the layers, it does not drop any.
    assert len(capped.layers) == config_module.RenderParams().layers


def test_the_ceiling_applies_to_the_rhizotron_too():
    """It is one layer rather than three, and a full-window one.

    On a 1600p panel the soil pane is more cells than the stack's front sheet,
    and its passes read the same shared memory, so leaving it at native would
    have sized two of the three backends and not the third.
    """
    from anastomosis import rhizotron as rhizotron_module

    params = config_module.Config().resolve()
    plain = rhizotron_module.RhizotronGeometry.derive(2560, 1600, params)
    params.render.cell_budget = config_module.INTEGRATED_CELL_BUDGET
    capped = rhizotron_module.RhizotronGeometry.derive(2560, 1600, params)

    assert plain.width * plain.height > config_module.INTEGRATED_CELL_BUDGET
    assert capped.width * capped.height <= config_module.INTEGRATED_CELL_BUDGET
    assert capped.problems() == []

    # And leaves a window that already fits alone, as everywhere else.
    small = rhizotron_module.RhizotronGeometry.derive(1920, 1080, params)
    params.render.cell_budget = 0
    assert small == rhizotron_module.RhizotronGeometry.derive(1920, 1080, params)


def test_the_ceiling_composes_with_base_scale_rather_than_replacing_it():
    """`base_scale` is applied first and still means what it says.

    Someone who has asked for half resolution has asked for half resolution;
    the ceiling may take more away and may never hand any back.
    """
    budget = config_module.INTEGRATED_CELL_BUDGET
    half = _derive(2560, 1600, scale=0.5)
    assert _cells(half) < budget  # so the ceiling has nothing to do
    assert _cells(_derive(2560, 1600, budget=budget, scale=0.5)) == _cells(half)


def test_an_unreachable_ceiling_terminates_at_the_layer_floors():
    """A ceiling under the per-layer floors cannot be met, and must not hang.

    `validate` raises a positive ceiling to `MIN_CELL_BUDGET` for exactly this
    reason, but the geometry is also handed values out of a checkpoint and out
    of hand-edited files, so the search itself has to stop.
    """
    geometry = _derive(2560, 1600, budget=1)
    assert geometry.problems() == []
    assert all(layer.width >= 64 and layer.height >= 32 for layer in geometry.layers)


def test_a_positive_ceiling_is_raised_to_one_that_can_be_met():
    params = config_module.Config().resolve()
    params.render.cell_budget = 3
    assert config_module.validate(params).render.cell_budget == \
        config_module.MIN_CELL_BUDGET
    params.render.cell_budget = -5
    assert config_module.validate(params).render.cell_budget == 0


def test_the_ceiling_is_structural_and_snaps_through_the_ramp():
    """It decides the shape of a field, so it may never be caught mid-ramp.

    `Geometry.derive` reads it from the *live* parameters, and a reset that
    landed while a hot reload was still ramping would otherwise grow the field
    at a size nobody chose.
    """
    params = config_module.Config().resolve()
    ramp = config_module.ParamRamp(params)
    target = config_module.Config().resolve()
    target.render.cell_budget = config_module.INTEGRATED_CELL_BUDGET
    ramp.set_target(target)
    current = ramp.update(0.001)  # far shorter than any ramp time constant
    assert current.render.cell_budget == config_module.INTEGRATED_CELL_BUDGET


# ---------------------------------------------------------------------------
# The budget governor's two levers
# ---------------------------------------------------------------------------


class _Pacing:
    """An Application with only what the governor and the rate logic touch."""

    def __init__(self, max_fps=30, sim_hz=20.0, battery=False):
        self.params = config_module.Config().resolve()
        self.params.max_fps = max_fps
        self.params.sim_hz = sim_hz
        self.power = power_module.PowerSource()
        self.power.on_battery = battery
        self._frame_times = []
        self._sim_hz_scale = 1.0
        self._fps_scale = 1.0
        self._present_fps = max_fps
        self._battery = battery
        self.rates = []

        class Canvas:
            def set_update_mode(inner, mode, *, max_fps=None):
                self.rates.append(max_fps)

        self.canvas = Canvas()

    # The methods under test, bound off the real class.
    _on_battery = app_module.Application._on_battery
    _uncapped_fps = app_module.Application._uncapped_fps
    _target_fps = app_module.Application._target_fps
    _apply_frame_rate = app_module.Application._apply_frame_rate
    _governor = app_module.Application._governor
    effective_sim_hz = app_module.Application.effective_sim_hz

    def feed(self, frame_time, frames=30):
        for _ in range(frames):
            self._governor(frame_time)


def test_the_tick_rate_is_the_first_lever():
    """Because the interpolator hides it completely (§8)."""
    pacing = _Pacing()
    pacing.feed(0.100)  # 100 ms frames against a 33 ms slot
    assert pacing._sim_hz_scale < 1.0
    assert pacing._fps_scale == 1.0


def test_the_frame_rate_is_the_second_lever_and_only_the_second():
    """The failure this exists for: a frame that is render-bound.

    Lowering the tick rate against one recovers nothing at all -- the render
    pass costs exactly what it cost -- so the governor used to walk the tick
    rate to its floor, degrade the motion, and leave the frame just as long.
    """
    pacing = _Pacing()
    for _ in range(400):
        pacing.feed(0.100, frames=1)
    assert pacing._sim_hz_scale == pytest.approx(app_module.SIM_SCALE_FLOOR)
    assert pacing._fps_scale < 1.0
    pacing._apply_frame_rate()
    assert pacing._present_fps < 30


def test_recovery_gives_back_the_visible_degradation_first():
    """Frame rate before tick rate: one is visible and the other is not."""
    pacing = _Pacing()
    for _ in range(400):
        pacing.feed(0.100, frames=1)
    assert pacing._fps_scale < 1.0
    sim_at_the_bottom = pacing._sim_hz_scale

    pacing._frame_times.clear()
    pacing.feed(0.002, frames=60)  # frames now cost nothing
    assert pacing._fps_scale > app_module.FPS_SCALE_FLOOR
    assert pacing._sim_hz_scale == pytest.approx(sim_at_the_bottom)


def test_the_two_levers_do_not_chase_each_other():
    """Recovery is judged against the slot at the *full* rate.

    Judged against the reduced slot, dropping to 20 fps would itself look like
    headroom -- the slot is half again as long -- and the rate would go
    straight back up, over budget, forever.
    """
    pacing = _Pacing()
    for _ in range(400):
        pacing.feed(0.040, frames=1)  # 40 ms: over a 33 ms slot, under a 50 ms one
    pacing._apply_frame_rate()
    lowered = pacing._present_fps
    assert lowered < 30
    for _ in range(400):
        pacing.feed(0.040, frames=1)
    pacing._apply_frame_rate()
    assert pacing._present_fps <= lowered


def test_the_governor_never_presents_faster_than_the_configured_cap():
    """Which is the safety-relevant direction. DESIGN.md §7.

    `config.validate` sizes the per-frame lightness allowance against
    `max_fps`, so the flash bound is only sound while the presented rate stays
    at or below it. Presenting fewer frames only slows the worst case; more
    would multiply the allowance by a rate it was never checked against.
    """
    pacing = _Pacing()
    pacing._fps_scale = 4.0  # as if something had let it run away
    pacing._apply_frame_rate()
    assert pacing._present_fps <= pacing.params.max_fps


def test_nothing_presents_below_the_floor():
    pacing = _Pacing()
    pacing._fps_scale = 0.0
    assert pacing._target_fps() >= app_module.MIN_PRESENT_FPS


def test_the_rate_reaches_the_canvas_only_when_it_changes():
    """It is pushed from `draw_frame`, thirty times a second."""
    pacing = _Pacing()
    pacing._apply_frame_rate()
    pacing._apply_frame_rate()
    assert pacing.rates == []
    pacing._fps_scale = 0.5
    pacing._apply_frame_rate()
    pacing._apply_frame_rate()
    assert pacing.rates == [15.0]


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def test_a_machine_that_will_not_say_counts_as_mains():
    """The same rule the window poll follows for a window that will not answer.

    A desktop reports no power source at all, and throttling it for a battery
    it does not have is the one failure here nobody would ever diagnose.
    """
    source = power_module.PowerSource(probe=lambda: None)
    source.poll()
    assert source.mains is None
    assert source.on_battery is False
    assert "assuming mains" in source.describe()


def test_battery_lowers_both_rates_and_mains_restores_them():
    on_mains = _Pacing()
    on_battery = _Pacing(battery=True)
    assert on_battery.effective_sim_hz() < on_mains.effective_sim_hz()
    assert on_battery._uncapped_fps() < on_mains._uncapped_fps()
    assert on_battery._uncapped_fps() == \
        on_battery.params.power.battery_max_fps


def test_the_battery_backoff_can_be_switched_off_live():
    """Read from the parameters every frame rather than latched at startup."""
    pacing = _Pacing(battery=True)
    assert pacing._on_battery()
    pacing.params.power.battery_backoff = False
    assert not pacing._on_battery()
    assert pacing.effective_sim_hz() == pytest.approx(pacing.params.sim_hz)


def test_battery_never_raises_the_frame_cap():
    """A config asking for more on battery than on mains gets the mains number."""
    params = config_module.Config().resolve()
    params.max_fps = 24
    params.power.battery_max_fps = 60
    assert config_module.validate(params).power.battery_max_fps == 24


def test_the_battery_probe_never_raises_whatever_the_platform_does():
    def explode():
        raise OSError("no such thing")

    source = power_module.PowerSource(probe=explode)
    source.poll()
    assert source.on_battery is False


def test_linux_reads_an_online_mains_supply(tmp_path):
    """A docked laptop lists several supplies and one of them carries power."""
    root = tmp_path / "power_supply"
    for name, kind, online in [
        ("BAT0", "Battery", None), ("ADP0", "Mains", "0"), ("ADP1", "Mains", "1"),
    ]:
        supply = root / name
        supply.mkdir(parents=True)
        (supply / "type").write_text(kind)
        if online is not None:
            (supply / "online").write_text(online)

    assert power_module._linux_on_mains(root) is True


# ---------------------------------------------------------------------------
# Device loss
# ---------------------------------------------------------------------------


class _SilentWatchdog:
    """The stall watchdog's one method `_on_device_lost` reaches for."""

    def dump(self, why):
        return None


class _Rebuildable:
    """An Application with the collaborators `_rebuild_device` replaces."""

    def __init__(self, fail_times=0):
        self.options = app_module.AppOptions(width=800, height=600)
        self.params = config_module.Config().resolve()
        self.device = object()
        self.device_info = None
        self.engine = object()
        self.present_context = object()
        self.canvas = None
        self.resumed_from = "an hour ago"
        self._device_lost = "driver reset"
        self._device_losses = 0
        self._device_retry_at = float("-inf")
        self._size = (800, 600)
        self._accumulator = 5.0
        self._sim_hz_scale = 0.4
        self._fps_scale = 0.5
        self._frame_times = [0.1] * 30
        self._last_time = 0.0
        self._last_checkpoint = 0.0
        self._fail_times = fail_times
        self.engines_started = 0
        self.surfaces_configured = 0
        self.lost_events = []
        self._stopped = False
        self._stop_requested = False
        self.watchdog = _SilentWatchdog()

    _rebuild_device = app_module.Application._rebuild_device
    _listen_for_device_loss = app_module.Application._listen_for_device_loss

    def _on_device_lost(self, event, device=None):
        self.lost_events.append(device)

    def _configure_surface(self):
        self.surfaces_configured += 1

    def _start_engine(self, width, height):
        self.engines_started += 1
        self.engine = object()
        self.resumed_from = "a checkpoint"


@pytest.fixture
def fake_device(monkeypatch):
    info = device_module.DeviceInfo(
        vendor="acme", device="iGPU", adapter_type="IntegratedGPU",
        backend="Vulkan", is_software=False, is_integrated=True)

    def request_device(gpu="auto", force_fallback=False):
        return object(), info

    monkeypatch.setattr(device_module, "request_device", request_device)
    monkeypatch.setattr(device_module, "install_error_handler", lambda *a: None)
    return info


def test_a_lost_device_is_replaced_rather_than_reported(fake_device):
    """§13 recorded this as scaffolded and untested. It is the thing a laptop
    needs most: the same event is a lid closing, and it happens nightly."""
    app = _Rebuildable()
    assert app._rebuild_device() is True
    assert app._device_lost is None
    assert app._device_losses == 1
    assert app.engines_started == 1
    assert app.surfaces_configured == 1
    assert app.device_info is fake_device


def test_a_replaced_device_saying_goodbye_does_not_restart_the_cycle(fake_device):
    """Dropping the old device can itself raise a lost event.

    Acted on, it would take the session straight back into the rebuild it had
    just come out of, every two seconds, for as long as it was left running.
    So the listener is bound to the device it was installed on and an event
    from any other one is dropped.
    """
    app = _Rebuildable()
    stale = app.device
    app._rebuild_device()
    assert app.device is not stale

    handler = app_module.Application._on_device_lost
    handler(app, object(), stale)
    assert app._device_lost is None  # ignored: not the device we are holding

    handler(app, "the real thing", app.device)
    assert app._device_lost == "the real thing"


def test_the_rebuild_clears_pacing_measured_against_the_dead_device(fake_device):
    app = _Rebuildable()
    app._rebuild_device()
    assert app._accumulator == 0.0
    assert app._sim_hz_scale == 1.0
    assert app._fps_scale == 1.0
    assert app._frame_times == []


def test_a_driver_that_is_still_resetting_is_retried_rather_than_given_up_on(
    monkeypatch,
):
    attempts = []

    def refuse(gpu="auto", force_fallback=False):
        attempts.append(gpu)
        raise RuntimeError("device is lost")

    monkeypatch.setattr(device_module, "request_device", refuse)
    app = _Rebuildable()
    assert app._rebuild_device() is False
    # Still pending, so the frame loop comes back to it.
    assert app._device_lost == "driver reset"
    assert app._device_losses == 0
    assert len(attempts) == 1


def test_the_retry_is_not_attempted_every_frame(monkeypatch):
    """A driver mid-reset refuses for as long as it takes, and a log written
    thirty times a second about it is a log nobody can read."""
    monkeypatch.setattr(
        device_module, "request_device",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("nope")))
    app = _Rebuildable()
    app._rebuild_device()
    attempts_before = app._device_retry_at
    for _ in range(50):  # a second and a half of frames
        assert app._rebuild_device() is False
    assert app._device_retry_at == attempts_before


def test_a_reset_session_comes_back_to_its_own_field_not_a_fresh_one(monkeypatch):
    """`--reset` is a statement about the launch, not a standing instruction.

    A device lost an hour into a `--reset` session must not throw the hour
    away, which is what re-reading the launch flag would do.
    """
    from anastomosis import checkpoint as checkpoint_module

    saved = object()
    monkeypatch.setattr(checkpoint_module, "load", lambda path: saved)

    app = app_module.Application.__new__(app_module.Application)
    app.options = app_module.AppOptions(resume=False)
    app.checkpoint_path = pathlib.Path("checkpoint.npz")
    app._resume = False
    assert app._saved_checkpoint() is None

    # One save later there is a field of this session's own on disk, and that
    # is the one a rebuild returns to.
    app._resume = True
    assert app._saved_checkpoint() is saved


def test_saving_a_checkpoint_is_what_makes_a_reset_session_resumable(monkeypatch):
    """The flip lives in `save_checkpoint`, so it cannot be reasoned about
    separately from the fact that made it true."""
    from anastomosis import checkpoint as checkpoint_module

    app = app_module.Application.__new__(app_module.Application)
    app.options = app_module.AppOptions(resume=False, checkpoint=True)
    app.engine = object()
    app.scheduler = None
    app.params = config_module.Config().resolve()
    app.checkpoint_path = pathlib.Path("checkpoint.npz")
    app._resume = False
    app._saver = checkpoint_module.BackgroundSaver()
    app._last_checkpoint = 0.0
    app._checkpoint_saved_at = None

    monkeypatch.setattr(checkpoint_module, "capture", lambda *a, **k: {})
    monkeypatch.setattr(checkpoint_module.BackgroundSaver, "submit",
                        lambda self, path, snapshot: None)

    assert app.save_checkpoint() is True
    assert app._resume is True


def test_waiting_for_a_driver_is_not_reported_as_a_wedged_loop(tmp_path, monkeypatch):
    """A rebuild that has not taken yet must not also read as a freeze.

    The loop is running and knows exactly what it is waiting for, and it said
    so in the log. A stall report every thirty seconds for the duration is the
    report-nobody-reads failure §8.2 is built to avoid, on the one occasion the
    application already knows what is wrong.
    """
    monkeypatch.setattr(
        device_module, "request_device",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("still resetting")))

    app = app_module.Application(app_module.AppOptions(
        width=96, height=72, ui=False, checkpoint=False, stall_seconds=0.0,
        config_path=tmp_path / "config.toml",
        diagnostics_dir=tmp_path / "diagnostics",
    ))
    try:
        app._size = (96, 72)
        app._device_lost = "driver reset"
        before = app.watchdog.frames

        app.draw_frame()

        assert app.watchdog.frames == before + 1  # the loop is alive
        assert app.watchdog.phase == "idle"       # and not stuck in a phase
        assert app._device_lost == "driver reset"  # still pending
    finally:
        app.power.stop()


# ---------------------------------------------------------------------------
# What a stall report says about all of it
# ---------------------------------------------------------------------------


def test_the_stall_report_separates_a_slow_laptop_from_an_unplugged_one(tmp_path):
    """Otherwise the two look identical from outside: a window moving less.

    §8.2's whole argument is that a freeze has to leave evidence taken while
    it is still happening. A session that is merely doing what it was told to
    on battery produces the same symptom -- and, without these lines, the same
    report.
    """
    app = app_module.Application(app_module.AppOptions(
        width=96, height=72, ui=False, checkpoint=False, stall_seconds=0.0,
        config_path=tmp_path / "config.toml",
        diagnostics_dir=tmp_path / "diagnostics",
    ))
    try:
        app.power.mains = False
        app.power.on_battery = True

        snapshot = app.diagnostic_snapshot()

        assert snapshot["power"] == "battery"
        assert "Hz sim" in snapshot["rates"]
        assert "fps presented" in snapshot["rates"]
        assert snapshot["device lost"] == "no"
    finally:
        app.power.stop()
