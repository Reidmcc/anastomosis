# Anastomosis — Design

A long-running generative visual field for self-regulation and stimming. Built on
`wgpu-py` + WGSL. Designed to run for days on a secondary display while the machine
is used for other work.

The name comes from hyphal anastomosis: the fusion of fungal filaments into a
network. That is the target behaviour — filaments that grow, seek, touch, and fuse,
inside a slowly breathing medium.

---

## How this document is organised

This file is an index. The design and its investigations live in `docs/`, one
file per section, so that a reader can pull exactly the part they need instead
of the whole thing. **Section numbers are preserved inside the files**: a
reference like "DESIGN.md §4.7" — used throughout the docs, the code comments,
and the tests — resolves via the table below, and means the file listed against
§4.7 there.

## Index

- **§1 — [docs/constraints.md](docs/constraints.md)**
  The stated needs restated as engineering requirements, and the two hard ones
  called out: a simulation that must not *settle* over multi-day runs, and "no
  flashing" as a safety property enforced by construction rather than by taste.

- **§2 — [docs/substrate.md](docs/substrate.md)**
  The three-system hybrid: Physarum agents and their trail field, a
  Gray–Scott-ish reaction field, and a divergence-free flow advecting the
  pigment field that is actually shaded — plus the fusion bias and soft
  deposits that make anastomosis a visible signature.

- **§3 — [docs/non-repetition.md](docs/non-repetition.md)**
  Why nothing anywhere is a function of wall-clock time: stateful
  Ornstein–Uhlenbeck walks, counter-based PRNG, integer frame counters, and why
  noise that tiles in *space* is still required (the domain is a torus).

- **§4 — [docs/homeostasis.md](docs/homeostasis.md)** (§4.1–§4.6)
  The long-duration core: the climate field (per-region governing parameters,
  advected and diffused), the GPU-resident homeostat (a PI controller with wide
  deadbands), Poisson slow events, the measured live band in feed/kill space,
  the absorbing state at V = 0 and the trail-seeding fix, and numerical
  survival — NaN quarantine, precision choices, checkpointing, device loss.

- **§4.7 — [docs/morphology.md](docs/morphology.md)**
  The longest investigation: the "uniform field of dots" failure the homeostat
  cannot see (also a trypophobia trigger, so a functional defect). Feature-size
  drift via `du`, the third climate pair, the flux-pruning postmortem,
  anti-fusion / rift events / founding respawn, the closed-loop feature-size
  (ℓ) controller, the shading rebalance, deposit capacity, shading knee and
  trail advection — and "the network that was never there": the sensing
  saturation (`sense_cap`) that finally made the trail layer grow a network
  instead of stationary knots.

- **§4.8 — [docs/wrap-seam.md](docs/wrap-seam.md)**
  Line-like structures near the window edges, traced to non-tiling noise
  forcing the vector potential: a permanent kink at the domain seam becomes a
  standing jet. Fix: tiled value noise with a quadrature crossfade between
  periods.

- **§4.9 — [docs/sensing-reach.md](docs/sensing-reach.md)**
  The agent population condensing onto a single axis-aligned strand that wraps
  the torus. The controlling quantity is the ratio of sensing reach to trail
  width; the fix bounds it everywhere it can be set, and `found_radius` had to
  follow.

- **§5 — [docs/depth.md](docs/depth.md)** (incl. §5.1)
  Depth, both backends. The layered 2.5D stack (per-layer resolution, parallax
  drift, DOF, atmosphere, Beer–Lambert compositing, cross-layer coupling) and
  §5.1, the volumetric slab: a 3-torus, a stored vector potential for exactly
  divergence-free 3D flow, depth anisotropy, cone-based 3D sensing, and the
  thickness and lateral-detail knobs.

- **§6 — [docs/colour.md](docs/colour.md)**
  All colour work in Oklab/OkLCh, driven by simulation state rather than a
  clock; lightness/chroma clamps, gamut mapping at constant lightness, and
  blue-noise dithering against banding.

- **§7 — [docs/flash-safety.md](docs/flash-safety.md)**
  The non-negotiable output stage: motion-compensated per-pixel slew limit,
  exposure governor, temporal IIR; exactly what is guaranteed, the WCAG
  arithmetic behind the 1% default and 1.2% ceiling, and the gamut-mapping
  leak that had to be closed.

- **§8 — [docs/frame-pacing.md](docs/frame-pacing.md)** (incl. §8.1–§8.2)
  Sim decoupled from render with motion-compensated interpolation, the frame
  budget governor, the RTX 3080 / 1440p cost budget (§8.1), and
  secondary-display behaviour: fullscreen, vsync, sleep/unplug, and why a
  window resize rebuilds the presentation chain only. Then the failure all of
  that pacing does not cover (§8.2): a loop that stops rather than one that
  runs slow, which produces no evidence at all unless something takes it while
  the freeze is still happening.

- **§9 — [docs/parameters.md](docs/parameters.md)**
  The control surface: eight macros over ~40 primitives, presets, TOML as the
  hot-reloaded source of truth, every change ramped — and why event rate was
  split out of intensity.

- **§10 — [docs/testing.md](docs/testing.md)**
  What the real QA is: the flash-safety assertion, soak tests, non-repetition
  checks, numeric parity against NumPy references, morphology checks, and the
  no-allocation check.

- **§11 — [docs/module-layout.md](docs/module-layout.md)**
  Module layout as planned and as built — why the shared output chain and
  safety stage live in `backend.py` — plus dependencies and the ping-pong
  texture policy.

- **§12 — [docs/build-order.md](docs/build-order.md)**
  The ten-step build sequence the project was planned in.

- **§13 — [docs/status.md](docs/status.md)**
  Implementation status: what is built and verified headless, what is not
  (device-loss rebuild path), and the judgements that still need a real GPU
  and a pair of eyes — including the defaults most likely to want moving.

- **§14 — [docs/activation-mode.md](docs/activation-mode.md)**
  The second mode: the same instrument retuned for sensory *seeking* rather
  than settling, inside an unchanged safety envelope. Where activation's
  energy is allowed to come from (motion, chroma, incident density — never
  luminance dynamics), per-mode curve tables, the measurements that set the
  endpoints, the polychrome palette, and a step-by-step record of what each
  build step actually did — including the gamut leak the mode-slam test
  found in §7's stage.

- **§15 — [docs/root-mode.md](docs/root-mode.md)** *(building: step 1 of
  §15.11 — soil, descent, moisture — is built; the roots are not yet)*
  A third backend on a second metaphor: the plant root. A rhizotron pane —
  stratified Munsell-coloured soil, pale roots steered by tropisms, moisture
  percolating after rain, and a window that sinks with the growing front over
  a ring-buffered, never-repeating soil column. Why it is a backend rather
  than a mode, the three inversions that make it visually unmistakable for
  the fungal field (ramification, gravity, matrix), succession as the
  never-settles mechanism, its safety analysis, and its build order.

## Where to look first

- Changing **agent behaviour** (`agents.wgsl`, `vol_agents.wgsl`): §2, §4.7,
  §4.9.
- Changing the **reaction, climate or homeostat**: §4, §4.7.
- Changing anything in the **output chain** (grading, compositing, raymarch,
  safety): §6, §7 — §7's guarantee must survive any change.
- Changing a **macro curve or a preset**: §9, and §14 — there are two curve
  tables, one per mode, and they must keep driving the same paths.
- Changing the **volumetric slab**: §5, §13 (its open questions).
- Adding **state that persists between ticks**: the checkpointing rules in
  §4.6.
- Wondering **why a test asserts what it does**: §10, then the section the
  test's docstring names.
