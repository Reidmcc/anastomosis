"""The GPU-niceness policy: bounded queue depth and preemption gaps.

Issue #40: a free-running tick loop holds the WDDM hardware queue nearly
continuously, and the desktop compositor's small jobs miss their deadlines
behind it. The policy is timing by nature, but nothing here asserts on a
wall clock: the sleep is a seam (`GpuNice.sleep`), so these tests record
what the policy *would* have done and stay deterministic on any adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from anastomosis import config
from anastomosis import engine as engine_module
from anastomosis import nice


# ---------------------------------------------------------------------------
# Stand-ins: a device whose adapter class and drain calls we control
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self, adapter_type: str):
        self.info = {"adapter_type": adapter_type}


class _FakeDevice:
    """Records drains; reports whichever adapter class the test wants."""

    def __init__(self, adapter_type: str = "DiscreteGPU"):
        self.adapter = _FakeAdapter(adapter_type)
        self.drains = 0

    def _poll_wait(self):
        self.drains += 1


@dataclass
class _RecordingSleep:
    calls: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


# ---------------------------------------------------------------------------
# The policy itself
# ---------------------------------------------------------------------------


def test_disabled_policy_touches_nothing():
    device = _FakeDevice()
    sleep = _RecordingSleep()
    policy = nice.GpuNice(enabled=False, sleep=sleep)
    policy.after_submit(device)
    assert device.drains == 0
    assert sleep.calls == []


def test_enabled_policy_drains_then_yields():
    device = _FakeDevice()
    sleep = _RecordingSleep()
    policy = nice.GpuNice(enabled=True, yield_seconds=0.004, sleep=sleep)
    policy.after_submit(device)
    assert device.drains == 1
    assert sleep.calls == [0.004]


def test_zero_yield_still_bounds_the_queue():
    """A drain with no sleep is a meaningful setting -- most of the latency
    win at more of the throughput -- so a zero yield must not skip it."""
    device = _FakeDevice()
    sleep = _RecordingSleep()
    policy = nice.GpuNice(enabled=True, yield_seconds=0.0, sleep=sleep)
    policy.after_submit(device)
    assert device.drains == 1
    assert sleep.calls == []


def test_auto_is_on_for_hardware_and_off_for_software():
    hardware = _FakeDevice("DiscreteGPU")
    software = _FakeDevice("cpu")
    assert nice.GpuNice().applies_to(hardware) is True
    assert nice.GpuNice().applies_to(software) is False


def test_auto_resolves_once_and_acts_accordingly():
    sleep = _RecordingSleep()
    policy = nice.GpuNice(sleep=sleep)
    software = _FakeDevice("cpu")
    policy.after_submit(software)
    assert software.drains == 0
    assert sleep.calls == []


def test_explicit_choice_beats_the_adapter_class():
    """`enabled=True` on a software adapter still applies: the CI soak job
    may deliberately want the pacing, and an explicit answer is an answer."""
    device = _FakeDevice("cpu")
    sleep = _RecordingSleep()
    policy = nice.GpuNice(enabled=True, sleep=sleep)
    policy.after_submit(device)
    assert device.drains == 1
    assert sleep.calls == [nice.DEFAULT_YIELD_SECONDS]


# ---------------------------------------------------------------------------
# The seam: every backend's tick must route its submit through the policy
# ---------------------------------------------------------------------------


class _RecordingPolicy:
    def __init__(self):
        self.submits = 0

    def after_submit(self, device):
        self.submits += 1


def test_backend_defaults_to_the_auto_policy(gpu_device):
    device, info = gpu_device
    params = config.Config().resolve()
    engine = engine_module.Engine(device, 256, 160, params, seed=3)
    assert isinstance(engine.nice, nice.GpuNice)
    assert engine.nice.enabled is None  # auto
    # And auto means: nice exactly when the adapter is real hardware --
    # the compositor-starving case -- never on CI's software adapter.
    assert engine.nice.applies_to(device) is (not info.is_software)


def test_tick_submits_through_the_niceness_seam(gpu_device):
    device, _ = gpu_device
    params = config.Config().resolve()
    engine = engine_module.Engine(device, 256, 160, params, seed=3)
    policy = _RecordingPolicy()
    engine.nice = policy
    engine.tick(params)
    engine.tick(params)
    assert policy.submits == 2


# ---------------------------------------------------------------------------
# The config surface
# ---------------------------------------------------------------------------


def test_gpu_nice_round_trips_through_the_config_file(tmp_path):
    path = tmp_path / "config.toml"
    config.save(config.Config(gpu_nice=True), path)
    assert config.load(path).gpu_nice is True
    config.save(config.Config(gpu_nice=False), path)
    assert config.load(path).gpu_nice is False


def test_a_config_file_predating_the_flag_means_off(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('preset_name = "default"\n', encoding="utf-8")
    assert config.load(path).gpu_nice is False


def test_cli_flag_reaches_app_options():
    from anastomosis.__main__ import build_parser

    args = build_parser().parse_args(["--gpu-nice"])
    assert args.gpu_nice is True
    # Omitted means None -- "use the config's answer" -- not False, or the
    # CLI would silently override a config that asked for niceness.
    args = build_parser().parse_args([])
    assert args.gpu_nice is None
