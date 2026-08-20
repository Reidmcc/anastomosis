> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

### 4.9 Sensing reach — the condensation the layer was sitting inside

§4.8 fixed a seam that *nucleated* lines at the field's edge. It did not fix the
lines. What makes a line is one step further in: at the sensing reach the layer
shipped with, the agent population condenses onto a **single axis-aligned strand
that wraps the torus**, and everything else goes bare.

Measured at 256×160 with one layer, seed 5, over 2000 ticks — `axis` is
`|mean exp(4iθ)|` over agent headings (1.0 = every agent running along ±x or
±y), `band` is the share of agents inside the busiest eight-cell row or column,
and `peak` is the ratio of the strongest trail row to the median row:

| `sensor_distance` | `trail_diffuse` | ratio | axis | band | peak | outcome |
|---|---|---|---|---|---|---|
| 8.0 *(shipped)* | 1.375 | 5.8 | **0.87** | **0.97** | 340 | one strand, everything else bare |
| 5.5 | 1.375 | 4.0 | 0.35 | 0.24 | 23 | holding, but concentration still climbing |
| 4.5 | 1.375 | 3.3 | 0.30 | 0.35 | 49 | distributed network |
| 3.5 | 1.375 | 2.5 | 0.01 | 0.37 | 16 | distributed network |
| 7.0 | 1.90 | 3.7 | 0.01 | 0.41 | 43 | distributed network |
| 12.0 | 1.90 | 6.3 | 0.88 | 0.06 | 1.2 | axis-locked, no structure at all |

The controlling quantity is not the sensing distance but its **ratio to the
width of what it senses**. Above ~5 the layer condenses; at 3.7 and below it
holds a network. Both of the last two rows have the same absolute reach as rows
above them and behave in opposite ways, which is what identifies the ratio
rather than the distance.

**Why the ratio.** An agent whose sensors reach much further than a strand is
wide stops resolving the strand it is standing on and starts steering at
whatever it can see from a distance. The paths that produces are straight — and
a straight path on a torus is the one path that closes on itself, after exactly
one lap. A closed path is re-walked, and re-walking deposits, so it reinforces
every lap while nothing else does. From there it is winner-take-all: trail
following has no capacity limit and no exit except `max_age`, so a strand that
starts ahead captures every agent that crosses it, the ambient falls away
beneath the starve threshold, and within ~1500 ticks the whole population is on
one strand. Straight and axis-aligned because those are the shortest closed
geodesics; horizontal on one seed, vertical on another, and that is the only
difference between them.

This is the mechanism behind the reported artefact: vertical and horizontal
line-like structures that split, rejoin and partially disappear but never go
away. They behaved like part of the simulation because they *were* the
simulation — by the end, all of it.

**What it is not.** Four candidates were ruled out by measurement rather than by
argument, each a 2000-tick run at the same size and seed:

- *the §4.8 seam*: the pre-fix and post-fix builds condense identically
  (axis 0.869 both, at tick 2000), differing only in which axis wins. The seam
  biased *where* a line nucleated, near the field edge, which is why the
  artefact first read as an edge effect; it had nothing to do with whether one
  formed;
- *jitter*: tripled to 0.30, the layer still condenses (axis 0.68, band 0.94);
- *the anti-fusion channel*: `range_repel` moved either way changes nothing;
- *commitment persisting after arrival*: gating the fusion `excess` on the trail
  underfoot, so the commitment releases the moment an agent reaches what it
  sensed, leaves the collapse untouched (axis 0.86, band 0.95). Worth recording
  because it is the plausible-sounding fix, and it is wrong: the straight runs
  are not the approach, they are the strand.

`fusion_bias` does move it — at zero the axis alignment goes to 0.014 — but it
does so by removing anastomosis, which is the application. It is the reach that
has to change.

**The threshold moves with the field.** A smaller torus condenses more easily —
the closing lap is shorter, there are fewer agents, and fewer strands fit — so
the ratio that is comfortable at 256×160 is not automatically comfortable
below it. At 192×128 a ratio of 3.2 still ended half-condensed (axis 0.72, band
0.78) where 2.5 did not (axis 0.25, band 0.28). The operating point is set from
the smaller field, since that is the harder case and the real one is larger
still.

**Fix.** Bound the sensing reach against the width of what it senses, at every
place that can set it:

- the `scale` macro moves `sensor_distance` and `trail_diffuse` *together*
  (2.2–4.9 against 0.85–1.90), holding the ratio at ~2.6 along its whole length.
  It used to sweep 4.0–12.0 against the same diffusion, so the ratio ran 4.7 at
  one end to 6.3 at the other and the macro was over the threshold everywhere,
  the default included;
- `range_sensor_distance` drops from ±3.0 cells to ±0.8, which is the spread the
  climate can carry inside the safe band rather than one that leaves it;
- `agents.wgsl` clamps the result against `sensor_reach_max × trail_diffuse`
  (3.2×), mirrored by `config.clamp_sensor_distance` in the same way
  `clamp_du` mirrors the reaction's. The clamp is what makes it safe rather than
  merely likely: the climate reaches its own bounds in the tails, and a region
  that wandered over the threshold for a few hundred ticks would nucleate a line
  that outlives the excursion.

With that, the reach spans 1.4–5.7 cells across the whole macro range and the
whole climate excursion, and the ratio stays inside 1.7–3.2 throughout.

**A second length had to follow it.** Shortening the reach broke the founding
respawn of §4.7 step 4, and silently: a cohort is scattered around its site with
a Gaussian radius of `found_radius`, and at 6.0 cells against a reach of 3.6 its
members land outside each other's sensing. A cohort that cannot find itself
never establishes, so bare ground stopped being recolonised at all — a rifted
disc sat at 1–3% of the control's trail for 5400 ticks after the envelope
released, which is precisely the "permanently bare ground" the rift test exists
to catch. `found_radius` now moves with the `scale` macro alongside the reach
(1.6–3.5 against 2.2–4.9, i.e. ~0.72× throughout) and defaults to 2.55, which
heals the same disc back to 80% and, as a second-order benefit, disperses the
population better than the wide cohort did: on the seed that concentrated worst,
the busiest band falls from 0.77 of the agents to 0.26.

**After.** Two seeds at 192×128 — the harder of the two sizes — over 2000 ticks:
axis alignment sits at **0.003–0.06** against 0.87 before, i.e. the headings are
no longer aligned to anything, and the trail's peak-to-median rises to ~58 and
comes back down to ~14. One seed still gathers 0.6–0.77 of the agents into a
band for a while, but with no axis alignment under it: that is a dense *region*
of network, not a line, and it disperses over the run rather than tightening.
Structure forms, concentrates, and comes apart again, which is the behaviour
§4.7 spent its length arguing for and could not get from the morphology levers
alone.

**Cost.** Shorter sensing makes the network finer and more reticulate, and the
`scale` macro still moves feature size — through `du`, `trail_diffuse` and the
flow's noise scale — over its full range. What it no longer does is move the
layer across a bifurcation.

**Tests.** `test_config.py` asserts the ratio invariant across the whole macro
range and the whole climate deviation, including the clamp, which costs nothing
and is the guard that actually holds the line. `test_soak.py` runs the layer to
1100 ticks and asserts the population has neither aligned to the axes nor
emptied the field into one strand, with the shipped-at-the-time reach as a
positive control in the same test, because a collapse metric that has never seen
a collapse is not evidence.
