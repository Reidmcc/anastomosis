> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 9. Parameters and control surface

~40 primitive parameters, but exposing 40 sliders is a worse interface than exposing
6 good ones. Two tiers:

**Macros** (the normal interface):

| Macro | Effect |
|---|---|
| Intensity | overall activity, contrast, agent count |
| Scale | feature size across all layers |
| Stability | how sturdy the network is: whether a lasting network forms and holds, or the filaments keep re-forming forever (§4.7) |
| Tempo | sim rate, flow strength, drift rates |
| Palette | hue anchor, hue rotation rate, chroma cap |
| Brightness | luminance ceiling and exposure target |
| Depth | layer separation, DOF, atmospheric falloff |
| Parallax | how far the viewpoint drifts, and how briskly |
| Event rate | mean arrival interval of the slow events, and nothing else (§4.3) |

Event rate was originally folded into intensity, and separating them is the one
change to this table worth arguing for. The two answer different questions. How
much material is on screen and how often it is disturbed are independent things
to want — a dense field left alone for an hour at a time is coherent, and so is
a sparse one that keeps being interrupted — and while they were one knob, nobody
could ask for either. It is also the macro most likely to be adjusted *for* a
state rather than for a look: "not right now" is a thing to be able to say to
the events without also dimming the network. Presets carry the rate their
intensity used to imply, so the split moved nothing; configs written before it
have their old rate recovered from their intensity on load, for the same reason.

Stability is the other macro that had to be argued into the table, and its
argument is the mirror image of event rate's. The shipped tuning deliberately
never lets a coherent network settle out — §4.7 spent its whole length making
sure the field cannot go monotonous — but "filaments that fuse, unfuse and
keep moving forever" and "a network that forms and holds" are both coherent
things to want from a regulation aid, and until this knob only the first was
askable. The travel starts at the shipped character (an untouched slider
changes nothing) and firms from there, by moving the four primitives the
volatility actually stands on rather than only one of them:

* `agents.prune_gain` 0 → 3 (gamma 0.6) — flux pruning, §4.7 step 3: the
  mechanism that was built, measured, and shipped off because its measured
  effect (persistence 0.11 → 0.27 at a 1050-tick lag, the same mass in a
  third less area) was a defect against the churn brief. It is exactly the
  effect this knob is for. The top is 3.0 and not the first gain that shows
  an effect, because §4.7's stability record runs the other way around: one
  run in four at gain 1.5 fell into a sparse state, while 3 and 5 held
  across every seed tried; the gamma moves the travel through the low gains
  quickly.
* `climate.range_repel` 2.6 → 0.9 — how much of the field is actively coming
  apart at junctions (~6% at any moment at the default, migrating with the
  climate). 0.9 takes ordinary weather out of reach of the repulsion
  crossing while a rift event, which pins its channel at the 1.0 clamp,
  still lands past it from the lowest `fusion_bias` up — so events keep the
  authority to take a sturdy network apart, and the test on that floor is
  `test_a_rift_can_still_take_a_sturdy_network_apart`.
* `agents.found_fraction` 0.55 → 0.30 — founding cohorts are how the network
  *moves house* (§4.7 step 4); the sturdy end biases respawns back toward
  accretion. Not lower, because founding is also how a rifted disc heals.
* `agents.jitter` 0.10 → 0.07 — the per-tick steering noise that keeps
  strands wandering, settled without approaching a regime nobody has run.

Measured on the software adapter (128² and 256², 2 550–2 850 ticks, three or
four seeds per point, paired by seed; flow frozen for the persistence
figures, since at the shipped `trail_advect` the whole network rides the flow
and pointwise correlation mostly measures the weather). What is established:
the sturdy end holds total trail mass within ~4% of the control in every
paired seed — the prune return accounting working exactly as
`test_flux_pruning_returns_the_mass_it_removes` demands, so the homeostat has
nothing to fight — mass-per-area rises (the same mass in a smaller
footprint), the §4.9 condensation guard stays at the control's level, and no
pruned run at any point of the travel fell into the sparse state §4.7
recorded at gain 1.5 on the older build. The persistence gain itself is
directional rather than decisive at this resolution and duration: the share
of the network's footprint still occupied 1 050 ticks later rises about 20%
in the mean (0.34 → 0.40, three of four paired seeds), while the raw
pointwise autocorrelation §4.7 once quoted no longer discriminates — the
step-4 founding and repulsion machinery, which postdates those figures,
churns both variants. That is the same measurement story steps 4 and 6 of
§4.7 ended on, and it ends the same way: the mechanism and its invariants
are verified here, and where along the travel is *right* is a judgement for
real eyes on real hardware (§13). The knob exists so that question can be
asked from the panel instead of from a hand-edited override.

Each macro drives many primitives through a curve defined in the config. Presets
(named macro settings) are first-class — this is a regulation tool, so *quickly
getting back to the one that worked* matters more than fine-grained tweaking.

**Primitives** available in the config file for anyone who wants them.

**Mechanism:** TOML file as the single source of truth, hot-reloaded on change
(watchdog); every parameter change is **ramped, never stepped** (250 ms–5 s
depending on the parameter) so adjusting a slider can't itself cause punctuation.
Invalid values are clamped with a logged warning rather than crashing a
long-running session.

The control UI itself is an open question — see the questions below.
