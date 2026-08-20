> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 3. Never loops, and never *can* loop

The naive approach — `noise(x, y, t)` for slow variation — is wrong here for two
reasons. It is periodic in practice (any tileable noise repeats; any non-tileable
one drifts into float precision loss), and at `t = 86400 s` an `f32` has ~0.008
resolution, so after one day the animation quantises visibly.

Instead, **every slow-varying quantity is a stateful process, integrated forward**:

- **Ornstein–Uhlenbeck random walk** for each global scalar:
  `x ← x + θ(μ − x)·dt + σ·√dt·N(0,1)`, computed on-GPU in a single-workgroup pass.
  Mean-reverting, so it stays in a sane band; aperiodic by construction; bounded
  variance; no dependence on absolute time.
- The **climate field** (§4) is itself advected and diffused each tick, so it is a
  stateful PDE, not a function of `t`.
- Randomness comes from a **counter-based PRNG** (PCG-family hash of
  `(pixel_id, frame_counter, stream_id)`) seeded from OS entropy at launch. The
  counter is `u64` split across two `u32`s; at 30 Hz the period exceeds the age of
  the universe.
- `frame_counter` is a `u32`/`u64` integer, never a float, and is used only as hash
  input — never as a phase. Nothing anywhere is `sin(t)`.

The prohibition is on periodicity **in time**. Noise that tiles in *space* is a
different matter, and is in fact required: the domain is a torus, and a spatial
increment that does not close on it leaves a seam in whatever integrates it
(§4.8). The spatial lattice tiles; the value drawn from it is still re-hashed
every tick, so the increment is as white in time as it ever was.

The consequence is stronger than "does not loop": there is no periodic component in
the system at all, and no state that recurs, because the state space is being
explored by a diffusion process rather than traversed by a trajectory.
