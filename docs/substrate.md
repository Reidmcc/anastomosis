> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 2. Substrate: a three-system hybrid

None of the three named systems alone hits the brief.

- **Physarum** gives literal anastomosis — filaments that seek and fuse — but its
  agent deposits are point-like and produce fine-grained shimmer (visual
  punctuation), and its networks stabilise or die.
- **Reaction–diffusion** gives organic texture and self-maintaining structure, but
  crawls rather than flows, and Gray–Scott settles into a steady state across most
  of its parameter space.
- **Lenia** gives beautiful smooth morphology, but its interesting regimes are
  narrow and metastable — it dies or explodes on long horizons.

The design uses each for what it is good at, in a stack of coupled fields:

```
climate field  (64×36, very slow)     ── governs every parameter below, per-region
      │
      ├─► agents (Physarum)           ── filament seeking, fusion, network topology
      │        │ soft deposit
      │        ▼
      ├─► trail field  T              ── hyphal density
      │        │ feeds
      │        ▼
      ├─► reaction field (U,V)        ── Gray–Scott-ish, gives texture *within* filaments
      │        │
      │        ▼
      └─► velocity field  v = ∇×ψ     ── incompressible flow; ψ from climate + blurred field
               │ advects
               ▼
           pigment field  P            ── what is actually shaded; carries colour history
```

The key structural choice is the **pigment field advected by a divergence-free
velocity field**. This is what produces "fluid continuous motion" as opposed to the
crawling, twitchy quality that raw RD and raw Physarum both have. Structures are
*carried* rather than recomputed. Because `v = curl(ψ)` it is incompressible by
construction, so pigment neither piles up nor drains — no bright accumulation
spots, no washing out. Semi-Lagrangian advection with bilinear (`textureSampleLevel`
works in compute shaders) is unconditionally stable at any timestep, which matters
for a process that must never blow up.

`ψ = a·curl_noise(climate) + b·blur(V)` — so the flow is partly imposed weather and
partly the structure's own field pushing itself around. That feedback is a
significant source of the non-predictability in requirement (a).

### Anastomosis specifically

Fusion is an emergent property of Physarum sensing, but it can be encouraged
explicitly, which makes the visual signature much stronger:

- Agents sense `T` at three points ahead; standard.
- Add a **fusion bias**: when the sensed value exceeds the agent's own recent
  deposit history, reduce the turn angle sharply (the filament commits to the
  junction rather than glancing off). Cheap, and it is what turns a tangle into a
  network.
- Agent deposits are **soft splats** (a small Gaussian, or bilinear-weighted into 4
  texels), never a single-texel write. This is a flashing-safety measure as much as
  an aesthetic one: a hard write is a one-pixel step change.
- Deposit magnitude is kept well below the field's decay rate per tick, so no single
  agent event is individually visible. Structure emerges from thousands of
  reinforcements, which is inherently gradual.
