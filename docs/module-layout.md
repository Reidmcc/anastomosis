> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 11. Module layout

The sketch below is the shape this was planned in; the built layout is flatter
(no `sim/` and `gfx/` packages) and has one addition the sketch does not name.
The two depth backends live in `engine.py` (layered) and `volume.py` (slab), and
everything they share — the output chain from the exposure governor to the
present blit, the flash-safety stage, the parameter mapping, and the device-side
plumbing — lives in `backend.py`. That module is what makes "a clean swap rather
than a fork" (§5.1) true rather than merely intended: the safety stage in
particular is a guarantee enforced by construction, and two copies of it free to
drift apart would be the most expensive duplication in the application.

```
anastomosis/
  __main__.py           entry point
  app.py                canvas, event loop, pacing, hot-reload
  backend.py            shared: output chain, safety stage, parameter mapping
  volume.py             the volumetric slab backend (§5.1)
  device.py             adapter selection, feature detection, device-lost recovery
  config.py             dataclasses, TOML load/save, validation, safety ceilings
  macros.py             macro → primitive curves, parameter ramping
  sim/
    scheduler.py        tick pacing, substeps, interpolation state
    layers.py           per-layer resource sets
    passes.py           pipeline + bind group construction
    homeostat.py        PI controller config, telemetry readback ring
    events.py           Poisson slow-event scheduler
  gfx/
    composite.py        layer compositing, parallax, DOF
    grade.py            Oklab colour mapping
    safety.py           slew limiter, exposure governor, dither
  shaders/
    common/             rng.wgsl, noise.wgsl, oklab.wgsl, sampling.wgsl
    climate.wgsl  agents.wgsl  reaction.wgsl  advect.wgsl  curl.wgsl
    blur.wgsl  couple.wgsl  reduce.wgsl  sanitize.wgsl
    interpolate.wgsl  composite.wgsl  grade.wgsl  safety.wgsl
  checkpoint.py         periodic save/restore of simulation state
  ui/                   control surface (TBD)
tests/
  test_flash_safety.py  test_soak.py  test_parity.py  test_config.py
  test_regime.py  test_morphology.py  test_agents.py  test_resize.py
  test_ui_backend.py
  reference.py  morphology.py        numpy reference + measurement, not tests
  test_checkpoint.py
```

**Dependencies:** `wgpu>=0.32`, `rendercanvas>=2.7`, `glfw`, `numpy`, `tomlkit`,
`watchdog`. Python ≥3.11 (wgpu-py requirement). No heavy frameworks.

Ping-pong texture pairs throughout (sampled read + storage write) rather than
read-write storage textures, which are an optional WebGPU feature — keeps the whole
thing on core WebGPU and portable across Vulkan/Metal/DX12.
