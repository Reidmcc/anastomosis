> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 12. Build order

1. Skeleton: canvas at 30 FPS, device management, config load + hot reload, one
   full-screen pass. Verify pacing and GPU load on the target machine early.
2. Single-layer Physarum + trail decay. Confirm agent cost and visual character.
3. Velocity field + semi-Lagrangian pigment advection. **This is the step that
   determines whether the "fluid" requirement is met** — worth evaluating before
   building on top of it.
4. Reaction–diffusion coupling.
5. Climate field + OU drift + homeostat. First point at which a long soak test is
   meaningful.
6. Oklab grading + full safety stage + flash-safety test.
7. Multi-layer depth compositing.
8. Sim/render decoupling + motion-compensated interpolation + budget governor.
9. Macros, presets, control UI.
10. Checkpointing, device-loss recovery, long soak.

Steps 1–6 produce something already usable for its purpose.
