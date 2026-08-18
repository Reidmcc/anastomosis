"""The agent layer's junction behaviour and its respawn -- DESIGN.md §4.7.

Both mechanisms here exist because the layer was topologically one-way. Agents
fused and reinforced, decay was uniform and traffic-blind, and respawn was
solitary, so every structure had to grow off a structure that already existed
and nothing could ever come apart. That is what left the field with a texture
whose *character* never changed.

There is no numpy reference for the agent pass -- it is a stateful compute
shader over a storage buffer, not an arithmetic kernel -- so these dispatch
``agents.wgsl`` directly against a hand-built world where the right answer is
known by construction: one filament, one agent, one thing it must do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import wgpu

from anastomosis import config, gpu_params, shaders

SIZE = 64

AGENT_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32), ("heading", np.float32),
    ("rng", np.uint32), ("recent", np.float32), ("age", np.float32),
])


def _run_agents(device, trail, agents, values, repel=0.0):
    """Dispatch agents.wgsl once; return the agent buffer and the deposits.

    `repel` is the climate `repel` channel (climate_c.z), uploaded flat at the
    simulation's own resolution so the shader's bilinear sample lands on texel
    centres and the test is about the steering rather than about interpolation.
    """
    size = trail.shape[0]
    module = device.create_shader_module(code=shaders.load("agents.wgsl"))
    pipeline = device.create_compute_pipeline(
        layout=wgpu.enums.AutoLayoutMode.auto,
        compute={"module": module, "entry_point": "main"},
    )
    usage = (
        wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.STORAGE_BINDING
        | wgpu.TextureUsage.COPY_SRC
        | wgpu.TextureUsage.COPY_DST
    )

    def make_texture(array):
        texture = device.create_texture(
            size=(size, size, 1), format=wgpu.TextureFormat.rgba16float, usage=usage
        )
        device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(array.astype(np.float16)),
            {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
            (size, size, 1),
        )
        return texture

    field = np.zeros((size, size, 4), dtype=np.float32)
    field[..., 0] = trail
    trail_tex = make_texture(field)
    flat = make_texture(np.zeros_like(field))
    climate_c = np.zeros_like(field)
    climate_c[..., 2] = repel
    climate_c_tex = make_texture(climate_c)

    packed = dict(values)
    packed.setdefault("dims_x", size)
    packed.setdefault("dims_y", size)
    packed.setdefault("agent_count", len(agents))
    params_buf = device.create_buffer_with_data(
        data=gpu_params.pack(gpu_params.SIM_DTYPE, packed).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    agents_buf = device.create_buffer_with_data(
        data=agents.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    deposit_buf = device.create_buffer_with_data(
        data=np.zeros(size * size, dtype=np.uint32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    stats_buf = device.create_buffer_with_data(
        data=np.zeros(1, dtype=gpu_params.STATS_DTYPE).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    sampler = device.create_sampler(
        address_mode_u="repeat", address_mode_v="repeat",
        mag_filter="linear", min_filter="linear",
    )

    def buffer(buf):
        return {"buffer": buf, "offset": 0, "size": buf.size}

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": buffer(params_buf)},
            {"binding": 1, "resource": buffer(agents_buf)},
            {"binding": 2, "resource": trail_tex.create_view()},
            {"binding": 3, "resource": buffer(deposit_buf)},
            {"binding": 4, "resource": flat.create_view()},
            {"binding": 5, "resource": flat.create_view()},
            {"binding": 6, "resource": sampler},
            {"binding": 7, "resource": buffer(stats_buf)},
            {"binding": 8, "resource": climate_c_tex.create_view()},
        ],
    )
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind_group)
    cpass.dispatch_workgroups(math.ceil(len(agents) / 64), 1, 1)
    cpass.end()
    device.queue.submit([encoder.finish()])

    out = np.frombuffer(
        device.queue.read_buffer(agents_buf), dtype=AGENT_DTYPE).copy()
    deposits = np.frombuffer(
        device.queue.read_buffer(deposit_buf), dtype=np.uint32
    ).reshape(size, size)
    return out, deposits


def _steering_params(agents_cfg, **over):
    """Parameters that isolate the steering: no jitter, no respawn."""
    values = {
        "tick": 0, "seed": 1,
        "range_repel": config.Config().resolve().climate.range_repel,
        "speed": agents_cfg.speed,
        "sensor_angle": agents_cfg.sensor_angle,
        "sensor_distance": agents_cfg.sensor_distance,
        "turn_rate": agents_cfg.turn_rate,
        # Jitter is stochastic steering, which is the whole point of the
        # parameter and exactly what must not be in the measurement.
        "jitter": 0.0,
        "deposit": agents_cfg.deposit,
        "fusion_bias": agents_cfg.fusion_bias,
        "fusion_max": agents_cfg.fusion_max,
        "starve_threshold": 0.0,  # no respawn: the agent must survive the tick
        "max_age": 1e9,
        "found_fraction": 0.0,
        "found_period": 1,
        "found_site_cells": 1,
        "found_radius": 0.0,
    }
    values.update(over)
    return values


def _blob(centre, sigma=1.5, size=SIZE):
    """One filament, as a smooth blob -- no texel edges for a sensor to catch."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = np.abs(xs - centre[0])
    dy = np.abs(ys - centre[1])
    dx = np.minimum(dx, size - dx)  # toroidal, like the domain
    dy = np.minimum(dy, size - dy)
    return np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2)).astype(np.float32)


def _one_agent(pos, heading, recent=0.02):
    agent = np.zeros(1, dtype=AGENT_DTYPE)
    agent["x"], agent["y"] = pos
    agent["heading"] = heading
    agent["rng"] = 12345
    # Well below what the blob reads, so `excess` saturates the smoothstep and
    # the commitment is at full strength -- this is a test of the clamp, not of
    # the ramp into it.
    agent["recent"] = recent
    return agent


def _turned(before, after):
    """Signed heading change, on the shortest arc."""
    delta = float(after["heading"][0]) - float(before["heading"][0])
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


@pytest.fixture(scope="module")
def flanking_world():
    """An agent at 16,16 heading +x, with a filament off its left sensor."""
    cfg = config.Config().resolve().agents
    angle, dist = cfg.sensor_angle, cfg.sensor_distance
    left = (16.0 + dist * math.cos(angle), 16.0 + dist * math.sin(angle))
    return cfg, _blob(left), _one_agent((16.0, 16.0), 0.0)


def test_fusion_bias_still_commits_to_a_junction(gpu_device, flanking_world):
    """The shipped behaviour, unchanged where the climate is neutral.

    Widening the commitment clamp past 1.0 must not move the default look: the
    base bias is well under 1, so the clamp is only reached where the climate
    `repel` channel or a rift event asks for it.
    """
    device, _ = gpu_device
    cfg, trail, agent = flanking_world

    after, _ = _run_agents(device, trail, agent, _steering_params(cfg), repel=0.0)
    turned = _turned(agent, after)

    expected = cfg.turn_rate * (1.0 - cfg.fusion_bias)
    assert turned == pytest.approx(expected, rel=0.05), (
        f"turned {turned:+.4f} toward the filament, expected {expected:+.4f}; "
        f"the fusion bias is no longer damping the turn as it did"
    )


def test_a_repelling_region_turns_agents_away_from_a_junction(
    gpu_device, flanking_world
):
    """Anti-fusion: the steering term changes sign past a commitment of 1.

    This is the mechanism §4.7 identifies as missing. With commitment clamped
    below 1 the layer has attraction and no repulsion -- every fusion adds a
    cycle to the network and nothing can ever remove one -- so the topology can
    only accrete, whatever else changes.
    """
    device, _ = gpu_device
    cfg, trail, agent = flanking_world
    climate = config.Config().resolve().climate

    after, _ = _run_agents(device, trail, agent, _steering_params(cfg), repel=1.0)
    turned = _turned(agent, after)

    commitment = min(cfg.fusion_bias + climate.range_repel, cfg.fusion_max)
    assert commitment > 1.0, (
        f"the repel channel cannot drive commitment past 1 "
        f"({commitment:.2f}); anti-fusion is unreachable from the climate"
    )
    assert turned < 0.0, (
        f"the agent turned {turned:+.4f}, i.e. still toward the filament, in a "
        f"fully repelling region"
    )
    assert turned == pytest.approx(cfg.turn_rate * (1.0 - commitment), rel=0.05)


def test_a_repelling_region_deflects_a_head_on_approach(gpu_device):
    """The case the sign flip alone does not cover, and the common one.

    When the forward sensor is the strongest the steering term is already zero,
    so scaling it expresses nothing and the agent drives into the junction it
    is supposed to be avoiding -- and an agent that has been committing to
    junctions is, by then, pointed straight at one. It must break toward the
    weaker flank instead.
    """
    device, _ = gpu_device
    cfg = config.Config().resolve().agents
    dist = cfg.sensor_distance

    # Just off the forward axis, toward the right sensor: the forward reading
    # stays the largest, and the right flank is the one to avoid.
    offset = -0.10
    trail = _blob((16.0 + dist * math.cos(offset), 16.0 + dist * math.sin(offset)))
    agent = _one_agent((16.0, 16.0), 0.0)

    neutral, _ = _run_agents(device, trail, agent, _steering_params(cfg), repel=0.0)
    repelled, _ = _run_agents(device, trail, agent, _steering_params(cfg), repel=1.0)

    assert _turned(agent, neutral) == pytest.approx(0.0, abs=1e-6), (
        "the control is not actually a head-on approach: the forward sensor "
        "should be the strongest, which leaves the turn at zero"
    )
    assert _turned(agent, repelled) > 0.05, (
        f"a head-on agent turned {_turned(agent, repelled):+.4f} in a fully "
        f"repelling region; it is fusing with what it should be avoiding"
    )


def _respawn_world(cfg, tick, found_fraction, size=SIZE, agents=1024, **over):
    """Every agent starving, so the whole buffer respawns in one dispatch."""
    rng = np.random.default_rng(5)
    buffer = np.zeros(agents, dtype=AGENT_DTYPE)
    buffer["x"] = rng.random(agents).astype(np.float32) * size
    buffer["y"] = rng.random(agents).astype(np.float32) * size
    buffer["heading"] = rng.random(agents).astype(np.float32) * 2.0 * math.pi
    buffer["rng"] = rng.integers(1, 2**31, agents, dtype=np.uint32)
    buffer["recent"] = 0.0  # starving, by construction
    values = _steering_params(
        cfg,
        tick=tick,
        starve_threshold=cfg.starve_threshold,
        found_fraction=found_fraction,
        found_period=cfg.found_period,
        found_site_cells=size * size,  # one site, so the cohort is unambiguous
        found_radius=cfg.found_radius,
    )
    values.update(over)
    return buffer, values


def _spread(positions, size=SIZE):
    """Toroidal r.m.s. distance from the circular mean of a point set."""
    angles = positions / size * 2.0 * math.pi
    centre = np.angle(np.exp(1j * angles).mean(axis=0))
    delta = (angles - centre + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.sqrt((delta**2).mean(axis=0)).mean() * size / (2.0 * math.pi))


def test_founding_respawn_lands_a_cohort_together(gpu_device):
    """Respawn has to be collective or it cannot found anything.

    A lone agent deposits orders of magnitude less per tick than decay removes,
    so a solitary respawn onto bare ground vanishes without trace and all
    growth accretes onto the network that is already there. A cohort sharing a
    site is what makes new structure possible somewhere else -- and that is
    what the resorbed mass of the flux-pruning term needs in order to become
    turnover rather than concentration (DESIGN.md §4.7).
    """
    device, _ = gpu_device
    cfg = config.Config().resolve().agents
    trail = np.zeros((SIZE, SIZE), dtype=np.float32)

    solitary, values = _respawn_world(cfg, tick=0, found_fraction=0.0)
    scattered, _ = _run_agents(device, trail, solitary, values)
    cohort, values = _respawn_world(cfg, tick=0, found_fraction=1.0)
    gathered, _ = _run_agents(device, trail, cohort, values)

    loose = _spread(np.stack([scattered["x"], scattered["y"]], axis=1))
    tight = _spread(np.stack([gathered["x"], gathered["y"]], axis=1))

    assert loose > SIZE * 0.2, (
        f"the uniform-respawn control is not uniform (spread {loose:.1f} cells "
        f"on a {SIZE}-cell torus)"
    )
    assert tight < 3.0 * cfg.found_radius, (
        f"founding respawns are {tight:.1f} cells apart against a founding "
        f"radius of {cfg.found_radius:.1f}; they are not landing together"
    )


def test_founding_sites_move_between_epochs(gpu_device):
    """A cohort must be an epoch, not a permanent address.

    A fixed site would be a hole punched in the same place forever, which is a
    static feature -- the failure this section exists to fix, in miniature.
    """
    device, _ = gpu_device
    cfg = config.Config().resolve().agents
    trail = np.zeros((SIZE, SIZE), dtype=np.float32)

    centres = []
    for epoch in range(6):
        buffer, values = _respawn_world(
            cfg, tick=epoch * cfg.found_period, found_fraction=1.0, agents=256)
        after, _ = _run_agents(device, trail, buffer, values)
        centres.append((float(after["x"].mean()), float(after["y"].mean())))

    distinct = {(round(x), round(y)) for x, y in centres}
    assert len(distinct) == len(centres), (
        f"founding sites repeat across epochs: {sorted(distinct)}"
    )

    # And within an epoch it must *not* move, or there is no cohort.
    buffer, values = _respawn_world(cfg, tick=1, found_fraction=1.0, agents=256)
    early, _ = _run_agents(device, trail, buffer, values)
    buffer, values = _respawn_world(
        cfg, tick=cfg.found_period - 1, found_fraction=1.0, agents=256)
    late, _ = _run_agents(device, trail, buffer, values)
    together = _spread(np.stack([
        np.concatenate([early["x"], late["x"]]),
        np.concatenate([early["y"], late["y"]]),
    ], axis=1))
    assert together < 3.0 * cfg.found_radius, (
        f"the site moved within one epoch (spread {together:.1f} cells), so "
        f"arrivals never accumulate anywhere"
    )


def test_founding_sites_prefer_bare_ground(gpu_device):
    """Landing on the network would only reinforce it, which respawn already did.

    Sampled across epochs rather than within one, because the choice is the
    barest of four hashed candidates: any single draw can be unlucky and the
    claim is about the distribution.
    """
    device, _ = gpu_device
    cfg = config.Config().resolve().agents
    rng = np.random.default_rng(9)
    # Smooth, so a site is judged on a neighbourhood rather than on one texel,
    # and heavily structured, so there is real ground to choose between.
    trail = rng.random((SIZE, SIZE)).astype(np.float32)
    for _ in range(4):
        trail = 0.5 * trail + 0.125 * (
            np.roll(trail, 1, 0) + np.roll(trail, -1, 0)
            + np.roll(trail, 1, 1) + np.roll(trail, -1, 1)
        )
    trail -= trail.min()
    trail /= trail.max()

    landed = []
    for epoch in range(40):
        buffer, values = _respawn_world(
            cfg, tick=epoch * cfg.found_period, found_fraction=1.0, agents=64,
            found_radius=0.0)  # no scatter: measure the site itself
        after, _ = _run_agents(device, trail, buffer, values)
        x = int(round(float(after["x"][0]))) % SIZE
        y = int(round(float(after["y"][0]))) % SIZE
        landed.append(float(trail[y, x]))

    # The barest of four uniform draws sits around the field's lower quintile,
    # so the median is a loose bound that still fails outright if the choice is
    # uniform -- which is the failure worth catching.
    chosen = float(np.mean(landed))
    median = float(np.median(trail))
    assert chosen < median, (
        f"founding sites sit on trail averaging {chosen:.3f}, no better than "
        f"the field median of {median:.3f}; they are not seeking bare ground"
    )
