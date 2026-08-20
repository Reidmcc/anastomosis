> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 1. Design constraints, restated as engineering requirements

| Stated need | Engineering requirement |
|---|---|
| Fluid continuous motion with depth | Velocity-field advection (not just cellular update); multi-layer composite with parallax + depth attenuation |
| No visual punctuation or flashing | **Hard** per-pixel slew limit on the final image, motion-compensated; exposure governor; no thresholds anywhere in shading |
| Unpredictable, never loops | All slow variation driven by *stateful random walks*, never by a function of wall-clock time; counter-based PRNG with unbounded period |
| Slow, reactive colour change | Colour derived from simulation state in Oklab, with a drifting palette anchor; heavy temporal lowpass |
| Cap 30 FPS | `rendercanvas` `update_mode="continuous", max_fps=30`, vsync on |
| Leave GPU headroom | Sim decoupled from render (sim ~15 Hz, render 30 Hz, motion-compensated interpolation); sim at fraction of display resolution; explicit frame budget governor |
| Adjustable parameters | TOML config as source of truth, hot-reloaded; ~8 macro knobs over ~40 primitives; presets |

**Target hardware:** RTX 3080, 2560×1440. Sized in §8.1; 4K is explicitly not a
requirement, which is what makes a native-resolution front layer affordable.

Two requirements dominate everything else and deserve to be called out before the
architecture, because they are the ones that are *hard*:

**(a) Not looping is easy. Not settling is hard.** Reaction-diffusion, Physarum, and
Lenia all have attractors. Left alone, every one of them either dies, saturates, or
reaches a quasi-static texture within minutes to hours. An application that must be
interesting for eight hours cannot rely on the simulation's own dynamics. The
architectural answer is §4: the governing parameters are themselves a slowly
drifting spatial field, so the system is never solving the same equation twice.

**(b) "No flashing" is a safety property, not a style.** It should be *enforced by
construction at the output stage*, not merely avoided by taste in the simulation
stage. A parameter regime nobody tested, a numerical blow-up, a NaN — any of these
could otherwise produce exactly the thing the application must never do. §7 makes
it a bounded, testable invariant that holds regardless of what the simulation does.
