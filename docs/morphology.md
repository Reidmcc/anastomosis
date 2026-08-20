> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

### 4.7 Morphological monotony — the failure mode the homeostat cannot see

Everything above is about staying *alive*. The first real viewing found a
different failure: after a few minutes the field reaches a texture whose
**character** never changes again. A dense population of small, similar-sized,
round features, holding steady indefinitely. The simulation is doing exactly what
§4.2 asks — mass, variance and activity all in band, never settling, never
repeating — and it is still the same picture it was an hour ago.

This also has an accessibility dimension the brief did not anticipate. A regular
lattice of similar-sized, high-contrast, round holes is a common trypophobia
trigger, and anastomosis is a process that walks straight into that geometry.
The application is meant to be a regulation aid, so a texture a proportion of
viewers find repellent is a functional defect, not a matter of taste.

**The reaction layer is a monodisperse spot field with a pinned length scale.**
Measured offline (`tests/morphology.py`) at the §4.4 regime, 160² torus, 6000
ticks after warm-up:

| quantity | value |
|---|---|
| component count | 255–297 (mean 270, s.d. 10 — ±4%) |
| length scale ℓ = mean V / mean\|∇V\| | 2.20–2.31 cells |
| holes (components − Euler characteristic) | 2 — it is a spot field, not a mesh |

Both the count and the size are constant to within a few percent for the whole
run. On the 1440p front layer that is ~250 features across the screen, all the
same size. The regularity is the problem, not the density.

**Three causes, and they are independent.**

1. *Nothing drives the length scale.* `du`/`dv` are constant except for the
   `scale` macro. A Gray–Scott regime at fixed diffusion has one characteristic
   wavelength, so the feature size is pinned by construction.
2. *The homeostat is blind to arrangement.* Mass, variance and activity are all
   invariant under rearrangement (`homeostat.wgsl`). A field can be perfectly
   on-band and morphologically frozen; the controller has no term that objects,
   and in defending its three measures it defends the texture along with them.
3. *The agent layer is topologically one-way.* `commitment` in `agents.wgsl` is
   clamped to `[0, 0.92]`, so it only ever *reduces* the turn — there is
   attraction and no repulsion. Trail decay is uniform and traffic-independent,
   so nothing can remove a strand; decay hits trunk and twig alike while agents
   preferentially reinforce whatever is already strong. Every fusion adds a
   cycle and no mechanism destroys one. This is anastomosis without autolysis,
   and it is only half of what real hyphal networks do: fungi resorb unused
   hyphae, and Physarum prunes low-flux tubes. Two lesser contributors sit
   alongside it — respawn is uniform and solitary (`agents.wgsl`), so a
   respawned agent can never found anything and all growth accretes onto the
   existing network; and only pigment is advected, so the fluid motion is in the
   colour carrier while the structure sits still and never experiences shear.

**The lever is `du`, and the reason is that it is nearly orthogonal to
everything the controller measures.** Static sweep at `feed=0.018, kill=0.051`,
ratio `dv/du` held at 0.50:

| du | features | ℓ | mean V | area | activity |
|---|---|---|---|---|---|
| 0.12 | 712 | 1.69 | 0.129 | 0.170 | 0.00046 |
| 0.21 (shipped) | 266 | 2.30 | 0.114 | 0.116 | 0.00139 |
| 0.26 | 177 | 2.56 | 0.108 | 0.104 | 0.00155 |
| 0.32 | 157 | 2.80 | 0.114 | 0.122 | 0.00153 |
| 0.40 | 112 | 3.05 | 0.102 | 0.114 | 0.00166 |
| 0.50 | 83 | 3.44 | 0.096 | 0.098 | 0.00170 |

An 8.6× change in feature count for a 26% change in mass. Across the usable
band below (du 0.17–0.40) it is tighter still: mass 0.102–0.114 and covered
area 0.104–0.122, both inside the noise of the fixed-`du` control. The two obvious alternatives are both worse: `kill`
0.046→0.056 gives ℓ 5.80→1.67 but drags mean V 0.178→0.084, and the homeostat
already owns kill; the diffusion *ratio* `dv/du` 0.36→0.64 gives 598→105
components but moves mass 0.144→0.084. Both fight the controller, and anything
that moves mass also moves the exposure governor, which turns a morphology
change into a slow global luminance swing — precisely the coordinated global
change §4.3 forbids.

Walking `du` between 0.16 and 0.34 over 3000 ticks, against a fixed-`du` control:

| | components | ℓ | mean V | activity |
|---|---|---|---|---|
| fixed | 255–297 (s.d. 10) | 2.20–2.31 | 0.1141 | 0.00135 |
| drifting | 147–395 (s.d. 68) | 2.02–2.85 | 0.1120 | 0.00142 |

A 2.7× swing in feature count — structures genuinely merging and splitting
throughout — while mass and activity stay within a few percent of the control
and inside both homeostat deadbands (mass [0.083, 0.153], activity
[0.00084, 0.00156]). Coarsening merges adjacent cells; refinement splits them.

The usable band is **du ∈ [0.17, 0.40]** at fixed ratio. Below 0.17 activity
falls under the homeostat's floor (0.00044 at du = 0.12) and the controller
starts fighting the drift with kill; above ~0.42 the explicit-diffusion headroom
starts being spent, although du = 0.5 still ran clean (`dt·du` = 0.43 against a
limit near 1.0 for this averaging-form Laplacian).

**It must be spatial, not global.** A globally coherent breathing of feature size
is coordinated global change of exactly the kind §4.2 warns about. Driven through
the climate field instead, coarse and fine regions coexist and migrate — which
additionally destroys the *uniformity of size*, and uniformity is the actual
trigger. This is worth stating plainly because it corrects the obvious framing:
breakup alone does not fix the texture. A churning field of uniformly-sized
holes is still a field of uniformly-sized holes. Varying the size fixes the
texture; breakup fixes the monotony. One lever happens to serve both.

Both `climate_a` and `climate_b` are fully allocated, so this wants a third
climate pair — 64×36 `rgba16f` is ~9 KB and free — with channels
`(scale, prune, fusion, spare)`, which is enough for every mechanism below.

**The missing half of anastomosis: flux-based pruning.** Give the trail field the
ability to lose an edge, by storing an income EMA in `trail.g` — the trail
texture is `rgba16float` and only `.r` is used, so this costs no new texture and
no extra bandwidth — and raising decay where income falls short of expenditure:

```wgsl
let income  = mix(prev_income, deposited, income_rate);        // trail.g
let deficit = clamp(1.0 - income / (decay * previous + 1e-6), 0.0, 1.0);
let decay_eff = decay * (1.0 + prune_gain * (deficit - 0.5));  // note the centring
```

Once a strand thins below sensing range agents stop finding it, which starves it
further: a positive feedback that *severs* the edge and merges two cells into
one. That is the coarsening half of a foam, and with fusion still running the
result is stationary churn rather than monotone refinement.

The centring is load-bearing and was not obvious. Uncentred at `prune_gain = 6`
the term removed 68% of trail mass; centred on the mean deficit it removed 19%.
An uncentred version is a net mass sink, so the homeostat cancels it through
`corr_decay` — yielding a globally weaker network and no severance, the exact
opposite of the intent. Use 0.5 as the reference, or carry a mean-deficit term
in the stats buffer if it needs to be exact. The trail blur smooths all four
channels, so the income channel is diffused for free, which is wanted: per-texel
agent arrivals are Poisson-spiky.

**Three smaller additions.** *Anti-fusion*: widen the `commitment` clamp past 1.0
so `turn * (1.0 - commitment)` goes negative and the agent turns *away* from a
junction, driven per-region from the new climate channel — migrating zones where
the network visibly comes apart while it fuses elsewhere. *Rift events*:
`EVENT_FIELDS` reaches only feed, kill, flow and hue, so `dieback` can thin
material but cannot sever anything; adding `chan_decay` → `climate_b.y` and
`chan_prune`/`chan_fusion` → the new pair buys an event kind that raises decay
and prune and negates fusion across a region under the usual long envelope.
*Trail advection*: the largest payoff for dynamism and the largest risk — shear
would stretch and pinch filaments, and the semi-Lagrangian pass already exists,
but it changes the agent↔trail feedback qualitatively and can push the reaction
into stripe instabilities. Last, behind a gain that starts at zero.

**The homeostat needs a morphological input**, or it remains unable to
distinguish a live pattern from a frozen one. `ℓ = mean V / mean|∇V|` costs one
term in `reduce.wgsl` (partials go from one `vec4` per tile to two, so the buffer
stride goes 16 → 32 bytes). Then either hold ℓ in a band using the global `du`,
or — better, and more in keeping with §4.2 — let the ℓ *setpoint* be a slow
bounded OU walk, τ ≈ 5–15 min, so coarsening and refinement become a continuous
cycle rather than a defended equilibrium. Measured ℓ spans 1.7–3.5 across
du 0.12–0.5, so a setpoint band of ℓ ∈ [2.0, 3.0] maps onto du ∈ [0.17, 0.40].
Split it the way feed and kill already are: the controller owns the global mean,
the climate field carries the deviation.

**What this does to the safety argument: nothing, by construction.** All of it is
upstream of §7, which bounds the output regardless. Two things still need
watching. The WCAG *area* criterion in `test_flash_safety.py` — the fraction of
pixels changing by ≥10% in one frame — will rise as churn increases, and trail
advection is the change most likely to move it. And the exposure governor is the
real interaction risk: any lever that moves mass produces a slow global
brightness cycle, which is the concrete reason to prefer `du` over `kill` or the
diffusion ratio.

**Testing.** The complaint should be encoded numerically for the same reason the
flash threshold was (§7): `holes = components − χ`, where the Euler
characteristic is a vectorised count of 2×2 pixel patterns — pure NumPy, no
scipy. Assert that hole count is non-monotone over a long run, and that feature
count has a coefficient of variation above a floor. The fixed-`du` control fails
that at s.d./mean = 0.04; the drifting one passes at 0.30. The drifting figures
move a little between runs — the reaction is chaotic, so a change of 1e-5 in
`dv` reshuffles which structures merge — but the statistics are stable, which is
what a test can assert. `gray_scott_step`
already takes `du`/`dv` as scalars and should widen to arrays, so a
climate-varying `du` stays covered by `test_parity.py`.

**Build order.** Each step is useful on its own and the first is throwaway:

1. ~~A global OU on `du` in `_sim_values`, as a spike~~ — **built**, and kept
   rather than thrown away; see below.
2. ~~The third climate pair, with `du` deviation per region.~~ **Built.** *This
   is the step that addresses the texture itself.*
3. ~~Flux pruning in `trail.g`, centred.~~ **Built, and off by default** — it
   does not do what this section predicted. See below.
4. ~~Rift events and anti-fusion, both riding the channels from (2).~~
   **Built**, together with the founding respawn step 3's postmortem asked
   for; see below.
5. ~~ℓ in the reduce pass, with a drifting setpoint.~~ **Built**, and it is
   the step that turns the mean from an open-loop guess into something the
   controller can be held to; see below.
6. ~~Trail advection, behind a knob, once the rest is tuned.~~ **Built**,
   alongside two mechanisms this list never anticipated, because the dots
   turned out to live in a different field than every step above assumed;
   see below.

~~Steps 1–3 should carry most of the value: polydisperse, migrating feature
sizes plus genuine edge severance.~~ Step 2 carries the value. Step 3 works
mechanically and delivers no visible benefit; the reasons are worth keeping and
are recorded below.

**What steps 1 and 2 turned out to be.** They are not two mechanisms but one
split the way feed and kill already are — a global mean and a per-region
deviation around it. The spike was worth keeping in that role: a unit-variance
OU walk on the mean (`Backend._advance_ell_walk`, τ = 7 min; ±7% on `du` per
standard deviation as originally built, ±9% on the length scale it now asks
for), with the climate deviation on top of it. The walk alone is
explicitly *not* the fix, for the reason given above — it moves every feature on
screen the same way at the same time and leaves them all the same size as each
other — but as the carrier of the mean it is what step 5 hands over to the
controller, so the plumbing is the same plumbing. It is still that walk, still
unit-variance, still bounded and still checkpointed with its stream; what
changed in step 5 is what it multiplies.

**The climate field realises about a tenth of its nominal range, and this had to
be measured before anything could be calibrated.** Every `range_*` parameter is
the deviation at a climate value of 1, and the field is clamped to [-1, 1], so
the natural reading is that `range_feed = 0.008` means ±0.008. It does not. The
OU drive is spatially white and the per-tick diffusion (§4.1) removes almost all
of the injected power immediately, so the field settles at **s.d. ≈ 0.11 with
extremes near ±0.44** — measured off a running engine over ticks 1200–3600, and
the same for every channel. Every existing range is therefore delivering roughly
a tenth of its apparent amplitude. That is not being changed here: those values
were tuned by eye against the behaviour they actually produce, and rescaling
them would change the shipped look for no benefit. But `range_du` had to be
calibrated against the realised amplitude rather than the nominal one, and any
future measurement of a climate range has to do the same or it overstates the
effect by an order of magnitude.

**The deviation is geometric, not additive.** `du` is driven by the `scale`
macro, so a fixed offset would mean a different spread at each end of that knob.
More importantly the survivable band is not symmetric around the base — a factor
of two of headroom above, and 0.57 below — so an additive deviation spends its
downside against the floor while its upside is still unused. Measured, at
matched spread of feature size: additive puts 13% of texels on a clamp,
geometric 4%. Both terms are geometric, so the walk and the regional deviation
compose exactly rather than interacting. `dv` rides on `du` throughout; holding
the ratio is what keeps the lever off the homeostat's measures, and varying the
two independently would reintroduce the mass movement that ruled the ratio out
as a lever in the first place.

**The floor is measured, not chosen.** The sweep in the table above stops at du
= 0.12; below that, at fixed feed and kill: 0.10 gives mean V 0.128 with
activity 1.5×10⁻⁴, 0.08 gives 0.097 and 1.1×10⁻⁴, and **0.06 collapses** — mean
V 0.015 and activity indistinguishable from zero. `du_min = 0.12` sits a factor
of two above that collapse and is the lowest point with any real measurement
behind it. The ceiling, 0.42, has more headroom than it needs: du = 0.50 still
ran clean.

**One thing the analysis did not anticipate: the region has to be big enough to
hold a wavelength.** Measured at a climate-texel-to-cell ratio of 10 the effect
largely cancels — regions a few features across cannot establish their own
length scale and average back to one. At 24 it is clean, and the engine's real
ratio is ~40 (64×36 over a 1440p layer), so this is comfortable rather than
marginal. It does mean an offline measurement has to be run at a realistic ratio
or it understates the effect.

Measured against a fixed-`du` control, isolated reaction (`tests/morphology.py`,
192² torus, 4000 ticks, one climate snapshot at the realised amplitude):

| | local ℓ spread (c.v.) | corr(local ℓ, local `du`) | mean V | activity |
|---|---|---|---|---|
| fixed | 0.081 | — | 0.1116 | 0.00135 |
| climate-varying | 0.118 | 0.88 | 0.1175 | 0.00116 |

and with the climate drifting under the field (160², 4000 ticks after warm-up):

| | components | c.v. |
|---|---|---|
| fixed | 263–302 | 0.037 |
| drifting | 241–388 | 0.157 |

The correlation is the number that says the mechanism is doing what it claims
rather than merely adding noise: where the climate asks for coarse structure,
coarse structure appears. Mass and activity both stay inside the homeostat
deadbands and within a few percent of the control, which is the property the
whole choice of `du` rests on — and they stay there at both extremes of the
walk (at ±2 s.d.: mean V 0.121 and 0.113, activity 1.01×10⁻³ and 1.32×10⁻³).
That last check is what set the walk's amplitude: at ±20% rather than ±7%, the
low end takes activity to 9.0×10⁻⁴ against a deadband floor of 8.4×10⁻⁴, and a
homeostat that starts correcting for the drift would cancel it.

The spatial swing in feature count is smaller than the 2.7× the global drift
produced, and that is expected — a coarsening region and a refining one partly
cancel in a whole-field count. The count is the weaker of the two measures here;
the spread of the *local* length scale is the one that speaks to the complaint,
since uniformity is the trigger.

Two caveats worth recording. About 5% of texels sit on a `du` clamp at any
moment, which is intended — the band is meant to be reachable — but those
regions have no local variation left, so the figure is worth watching if the
range is ever widened. And in the *full* engine, rather than the isolated
reaction, the local length scale already varies (c.v. 0.15) from the agent and
feed/kill machinery, so this lever is one contributor among several and the
effect is not cleanly separable there. That is why the measurements above are
made on the reaction alone, as §4.7's original analysis was.

The third channel of the new pair is read by the agent layer as of step 4,
under the name `repel` rather than `fusion`; the section below says why the
name had to change.

### What step 3 actually did

The flux pruning above is built: an income EMA in `trail.g`, decay raised where
income falls short of expenditure, the whole thing measurable from `trail.b` and
gated per region by the `prune` climate channel. It is off by default
(`agents.prune_gain = 0`), because measuring it changed the conclusion three
times. All three corrections are worth recording, because the reasoning that
produced the original prescription looks sound and is wrong in specific ways.

**The income EMA has to be faster than the trail's own decay, not slower.** The
first implementation used `income_rate = 0.05` against a trail decay of 0.055
and the deficit came out at **0.02 across the entire field** — the term was
inert and would have shipped looking like it worked. The reason is structural:
the trail *is* a decaying integral of deposits, so an EMA with the same time
constant reproduces it exactly and `income / (decay · trail)` is 1 by
construction. The deficit only carries information when income is measured over
a window shorter than the trail's memory, so that a strand's *recent* traffic
can be compared against its *accumulated* size. At 0.15 the deficit spreads
across the full range, median 0.51.

**The centring must not be local, and 0.5 is not the reference.** The
mass-weighted mean deficit is 0.21 at the default intensity — but 0.07 at the
top of the intensity macro, where the network concentrates two thirds of its
mass into 2.6% of texels. So the reference is not a constant and cannot be one;
it is now measured in the reduce pass, which is why the per-tile partials went
from one `vec4` to two (the stride change §4.7 anticipated for step 5's ℓ term;
step 5 has since taken the third lane of that second `vec4`).

That fixed the accounting and broke something worse. Centring the term locally —
raising decay on starved strands and lowering it on well-fed ones, exactly as
prescribed above — **froze the field solid**. Well-fed strands have a deficit
near zero, so centring hands them several times the memory of everything else,
and the network locks into whatever shape it happened to hold. Measured, the
trail's autocorrelation at a 1050-tick lag went from 0.11 unpruned to **0.72**
centred. That is a far worse failure than the monotony this section exists to
fix, and it is not a tuning problem: any mass-neutral multiplicative scheme has
to give back what it takes, and giving it back through decay means giving it to
whatever is already strongest.

The fix is to return the mass somewhere else. Pruning is now one-sided — decay
is only ever raised — and the removed fraction of the field's throughput is
measured and handed back through the *agent deposit*, so it reappears wherever
traffic currently is rather than wherever mass already is. That is also the
better model: fungi translocate what they resorb to the growing tips. Persistence
at a 1050-tick lag comes back to 0.27.

**And with all of that right, it still does not deliver churn.** Measured at
256², against a `prune_gain = 0` control:

| | trail mass | mass / area | trail autocorr, lag 1050 | reaction autocorr, lag 600 |
|---|---|---|---|---|
| off | 0.097 | 1.12 | 0.11 | 0.03 |
| gain 3 | 0.098 | 1.44 | 0.27 | 0.02 |

Mass is preserved and `corr_decay` does not move at all between the two, so the
accounting works and the homeostat genuinely does not fight it. The network
concentrates — the same mass in a third less area, which is a real coarsening.
But the trail gets *more* persistent, not less, and the reaction field, which is
what the texture complaint was actually about, does not change at all: mean V,
feature count, ℓ and the local-ℓ spread are all within noise of the control.

The reason is a selection effect rather than a mechanism failure. Pruning
removes the parts of the field that were changing, so what survives is by
definition the persistent part. The prediction above — "with fusion still
running the result is stationary churn rather than monotone refinement" —
assumed a cellular network where severing an edge merges two cells and fusion
creates new ones. The trail field is not cellular: measured across thresholds
and at 160², 256² and 384², the web has essentially no closed loops (holes 0–2),
so there are no cells to merge and severance just deletes a wisp.

There is also a stability cost. One run in four at `prune_gain = 1.5` fell into
a sparse state — trail mass around 0.045 against a normal 0.095 — and stayed
there for the rest of the run. Gains of 3 and 5 were stable across four seeds
each, but a rare one-way transition is precisely what §4.5 and §4.6 exist to
prevent over a multi-day run.

**What would make it earn its place.** The returned mass currently goes through
the agent deposit, and agents live on the network, so the mass cycles within the
structure that is already there. The accretion bias identified as a lesser
contributor above — "respawn is uniform and solitary, so a respawned agent can
never found anything" — is what closes that loop. If a founding respawn let the
returned mass start new structure on bare ground, pruning would become genuine
turnover rather than concentration: material resorbed here, reinvested there.
That makes step 4 the interesting one, and it makes step 3 a component of step 4
rather than a step in its own right. The mechanism stays in the tree, tested and
inert, waiting for that.

### What step 4 actually did

Three mechanisms rather than two: anti-fusion and rift events as prescribed,
plus the founding respawn that step 3's postmortem identified as the thing
standing between flux pruning and genuine turnover. All three are on by
default; flux pruning still is not.

**Anti-fusion, and the axis it turned out to sit on.** The prescription is what
shipped — the `commitment` clamp reaches past 1.0, at `fusion_max = 1.85`, so
that `turn * (1.0 - commitment)` changes sign — with two corrections found in
the building.

The first is that this is not a *fusion* axis, and a channel named for fusion
would have been named for the wrong thing. Commitment runs from 0, where the
agent turns toward what it sensed and follows it, through 1, where the turn is
cancelled and it drives straight through the junction and fuses, to above 1,
where the turn reverses and it veers away. Fusion is a *point* on that axis
rather than an end of it, and both sides of the point fuse less. A single
signed channel has to pick a monotone direction, so the channel carries
deflection toward the junction and is named `repel`. Note also that the old
0.92 ceiling was never reached in the first place — the bias tops out at 0.72
at the far end of the intensity macro — so widening the clamp does nothing on
its own. What it does is make the region past the crossing reachable at all.

The deviation is additive, unlike `range_du` and `range_prune`, and for a
reason specific to this axis: what matters is a fixed crossing at 1.0 rather
than a ratio, so an additive deviation puts that crossing at a fixed distance
in climate units and the fraction of the field that repels can be read off the
amplitude. At `range_repel = 2.6` against the realised climate amplitude of
0.11 (measured for step 2, and the same for every channel) the crossing sits
1.6 s.d. out: about 6% of the field is coming apart at any moment, and those
zones migrate with the climate.

The second correction is that the sign flip alone does not cover the case that
matters most. When the forward sensor is the strongest reading the steering
term is *already* zero — that is what committing to a junction means — so
multiplying it by a negative number leaves it zero and the agent drives
straight into the thing it is supposed to be avoiding. And an agent that has
been committing to junctions is, by construction, pointed at one: head-on is
the common case here, not the corner case. A repelling region therefore breaks
the agent toward the weaker flank explicitly, at the strength the sign-flipped
term would have had.

**Founding respawn.** Respawns now land in cohorts — 55% of them — at a site
shared by everything respawning in the same 240-tick epoch, chosen as the
barest of four hashed candidates. There is no communication between
invocations and none is needed: every agent hashes the same (epoch, site) and
gets the same answer. The one thing that needed care is that the number of
sites is a *density* (cells per site) rather than a count. Agent count scales
with layer area, so a fixed number of sites would put a hundred times more
traffic on each one at 1440p than on a 128-cell test layer, and a founding
would arrive as a flare rather than as a founding.

**Rift events, and what an event amplitude actually means.** Three channels
added to the event record — `chan_decay` into `climate_b.y`, `chan_prune` and
`chan_repel` into the new pair — and one kind that drives all three.

Measuring it turned up something about the *whole* event system that had not
been noticed: **an event pins every channel it names**. The climate is
mean-reverting at `theta = 0.0016` and an event *adds* its amplitude every
tick, so a sustained contribution settles at `amplitude / theta` — hundreds of
times the clamp. Every channel a kind names therefore sits at ±1 through the
whole hold, whatever coefficient it was given; the coefficient shapes the ramp
and nothing else. (The spatial raised cosine survives only as a rim, though the
climate's own diffusion softens it; the temporal envelope is unaffected, since
the plateau is approached over the ~625-tick reversion time.)

That turned a tidy-looking choice into a wrong one. The kind originally carried
a small feed and kill "so the reaction is not what is being taken apart" —
which, pinned, meant feed at its clamp for the length of the event. Measured
against a no-event control at 96²: with feed and kill, the reaction inside the
disc fell to 0.65 of its surroundings and the whole field's mass fell 16%; with
the identical severance channels and no feed or kill, the reaction sat at 0.93
against a control's 0.97 and the effect on the trail was *exactly the same*.
So the kind now names the three severance channels and nothing else. That is
what makes it structural: it changes how the material is connected, not how
much of it there is, which is also what keeps it clear of the exposure governor
— the interaction §4.7 flagged as the real risk.

Traced at 128² through a shipped-length envelope, as the disc's middle over
ground the event never reached, against a no-event control on the same seed:

| | trail inside / outside | V inside / outside | global mean V |
|---|---|---|---|
| control | 0.10 – 0.13 | 0.9 – 1.2 | 0.115 – 0.125 |
| dieback | 0.10 – 0.13 | 0.13 – 0.37 | 0.085 – 0.104 |
| rift | 0.024 – 0.06 | 0.9 – 1.2 | 0.115 – 0.125 |

(The rift's two right-hand columns are the control's, which is the point of the
row: measured against the control run's *own* disc, the reaction inside a rift
sits at 0.97 and 1.00 of it on the two seeds tried, while the trail falls to
0.32 and 0.77.)

The dieback row for the trail is not merely similar to the control, it is
*identical at every sample*, and that is the sharpest possible statement of the
gap this kind fills. Nothing flows from the reaction back to the trail, so an
event reaching only feed and kill cannot touch the network at all, whatever it
does to the material. The two kinds are now orthogonal: dieback moves how much
there is, rift moves how it is joined up.

Decomposed at 96², against the same control (trail inside/outside 3.92, V
0.97): the raised decay does most of the work (3.22), the repulsion adds to it
(3.63 alone, 3.08 together), and neither moves the reaction outside the
run-to-run spread (1.25 and 1.11). The `prune` channel is inert, since
`prune_gain` is zero — it is wired so that switching pruning on would sharpen
rifts rather than needing a second change.

Severance is strongest where the network is thinnest, by a factor of two or
more. In a sparse region the raised decay thins a strand, agents stop finding
it, and that thins it further — the feedback §4.7 predicted — and the trail
inside falls to 0.32 of the control's. Where the disc lands on a dense hub the
same event takes off 0.77, and at 96² about a fifth. That is the right shape
for the mechanism: it takes apart what is weakly held and leaves trunks alone.
It also means the effect needs on the order of a thousand ticks at strength to
develop, since the direct effect of the decay term alone is only that fifth.
The reaction sat at 0.97 and 1.00 of the control in both, which is the claim
that does not depend on where the disc lands.

And it heals. Traced for 4000 ticks past the release, the reaction inside the
disc is back to parity with its surroundings about 1700 ticks after the hold
ends — before the envelope has finished releasing — and the trail comes back to
0.53 and 0.92 of the control's on the two seeds tried, the network being the
slower of the two to return. This half is not optional: V = 0 is an absorbing
state for Gray–Scott, and an event that could empty a quarter of the screen and
leave it empty would be a worse failure than the monotony this section exists
to fix, because it would be permanent. It is asserted in `test_soak.py`.

**What could not be measured, which is most of what step 4 claims.** The
mechanisms are verified individually — that agents in a repelling region turn
away from a junction and break off a head-on approach, that respawns land
together and on bare ground, that a rift severs and heals — and every invariant
they had to respect is unmoved: across every configuration and seed run here,
trail mass sits at 0.088–0.093, mean V at 0.123–0.144, activity at
0.0014–0.0015, and the homeostat's `corr_decay` does not move.

The system-level claim — that this makes the trail layer churn — did not
survive its own measurement. Every statistic tried (structural turnover at a
1000-tick lag, per-texel change rate, threshold-crossing rate, field
autocorrelation) has a run-to-run spread comparable to or larger than the
difference between configurations, in *both* directions. At 160² over 4000
sampled ticks, four seeds gave structural turnover 0.058 → 0.090 with the
mechanisms on, which looks like the doubling one would want; the spread between
seeds is 0.049, so it is not distinguishable from nothing. Nine seeds per arm
at 128², using a per-texel rate rather than a pattern correlation, gave a mean
change of nothing at all: 0.127 against 0.130, with individual runs spanning
0.04 to 0.22 in both arms. Nor can the runs be genuinely paired to
recover power: founding respawn draws extra random numbers per agent, so
turning it on shifts every subsequent draw and two runs of "the same seed"
share only their first few ticks. The honest summary is that the field's
aggregate behaviour on a 128–160 cell test layer is dominated by which
dynamical state a given run wandered into, and that resolving an effect of this
size would need tens of independent runs per arm at a resolution the software
adapter cannot reach in reasonable time. What the tests assert is therefore the
mechanism and the invariants, not the aggregate.

They are on by default anyway, and that is a deliberate call rather than an
oversight: the argument for them is structural rather than statistical — the
agent layer could previously only add edges and could only grow onto what it
already had — and nothing measured says they do harm, across nine seeds per
arm, with the field neither collapsing nor freezing and the homeostat never
reaching for a correction. Whether they earn their place perceptually is one
more question for the first viewing on real hardware (§13), alongside the ones
already waiting there.

### What step 5 actually did

Reported from a real viewing, and the reason this step stopped waiting: the
field still reaches a state that is a mostly uniform array of dots which change
very little — the §4.7 texture, in the picture rather than in the statistics.
Less pronounced under the volumetric backend, and worse at *low* intensity,
which is the observation that identifies the cause. The intensity macro scales
down exactly the couplings that perturb the reaction off its own attractor
(`agents.density`, `agents.deposit`, `reaction.trail_feed_gain`), so at the
quiet end the reaction is running as nearly free Gray–Scott, and free
Gray–Scott in this regime is a monodisperse spot lattice.

Two separate things were wrong, and only one of them was in the simulation.

**Nothing was closed on the texture.** Steps 1–4 all *ask*. The global walk asks
for a diffusion rate, the climate asks for a per-region deviation, the events
ask for a severance — and if the field declines, nothing anywhere notices,
because §4.2's three measures are invariant under rearrangement and cannot see
a frozen arrangement. The measured spread the climate deviation bought is real
and is also small: local ℓ c.v. 0.081 → 0.118, i.e. features about 12% apart in
size when the trigger is *uniformity*. The walk on the global mean was smaller
still: ±7% in `du` is ±3% in ℓ.

So ℓ now goes into the reduce pass, as this section proposed, and a controller
is closed on it. The measurement costs the four neighbour loads for a central
difference — matching `tests/morphology.py` exactly, since every number in this
section is quoted in that ℓ — summed into the third lane of the second `vec4`,
and one division in `homeostat.wgsl`.

**The setpoint is referenced to the field, not to a number.** The obvious
design — hold ℓ at a constant — is wrong here in a way worth recording, because
it looks right. The `scale` macro moves `du` *deliberately*, so a controller
defending an absolute ℓ would cancel the macro outright; and the value the full
engine settles at is not the value the isolated reaction does, because the
agent layer and the feed/kill machinery move ℓ as well (this section already
records the local ℓ spread as 0.15 in the engine against 0.081 in the reaction
alone). There is no constant to use, and calibrating one against the software
adapter would have been calibrating against the wrong field.

So the setpoint is `reference + walk`, where the reference is a slow average of
the field's own ℓ. The loop therefore has no opinion about where feature size
should sit, only that it should move — which is exactly the complaint restated
as a control objective.

Two details of that are load-bearing and neither was obvious.

*The modulation is subtracted before the reference averages it.* Otherwise the
reference chases its own output: ℓ tracks the setpoint, the reference averages
ℓ, the setpoint is built from the reference, and the walk gets integrated into
a drift with no fixed point at all. Averaging `ln ℓ − offset` instead leaves the
reference tracking the field's *natural* length scale, which is what a baseline
is supposed to be. Measured with the reference frozen after seeding and the
amplitude widened to make the effect unmissable, the two arms of a sustained
±2 s.d. demand landed on ℓ 2.48 and 2.48 — the reference had absorbed the
offset while it was being established, and the loop had nothing left to ask
for. That is the correct behaviour for a *constant* offset, and it is also the
sharpest possible statement that this loop answers changes in the demand rather
than its level. The demand it is actually given changes on a seven-minute time
constant against a reference that averages over thirty, which is where the
separation comes from.

*The reference has to start fast and then stop being fast.* Seeded from the
first measurement and left at its 30-minute time constant, its first value is
its value for the next half hour; measured, `corr_du` reached its clamp within
1600 ticks of a cold start and stayed there, running the whole warm-up at
`du × 1.43`. Seeded as a running mean and left that way, it converges faster
than the walk moves and cancels the modulation entirely. It is therefore a
running mean for 600 ticks and an exponential after that. The loop is
additionally gated on the mass deadband the homeostat already computes: a field
still growing into its band has a length scale that is going to change for
reasons that are nothing to do with this loop, and so does one a dieback has
just emptied.

**The plant, measured rather than assumed.** ℓ against `du` at the shipped
feed/kill, through `reduce.wgsl` and `homeostat.wgsl` themselves and agreeing
with the numpy reference to f16 rounding:

| du | 0.146 | 0.170 | 0.2097 | 0.260 | 0.301 | 0.380 |
|---|---|---|---|---|---|---|
| ℓ | 1.96 | 2.10 | 2.32 | 2.57 | 2.75 | 3.09 |

An exponent of 0.47 — ℓ goes as `sqrt(du)`, which is what a diffusion length
should do — over the span the controller's own bound permits. In the *full*
engine it measured 0.61, the agent layer contributing the difference. That
exponent is what sizes the gain and it is asserted, because a run that measured
it negative would make the loop a positive feedback that drives `du` to a
bound.

Three time constants, and the ordering between them is the design: the
reaction's own response to a change in `du` is a few hundred ticks, the loop is
90 s, the setpoint walk is 420 s, and the reference is 1800 s. Faster than the
walk so it tracks rather than lags; slower than the plant so it is not chasing
the reaction's own dynamics; far slower again for the reference, or it absorbs
what it is the baseline for.

**What it does, paired.** One field grown to maturity, checkpointed, and
restored into three engines that differ only in what they ask for — which is
the paired comparison this section records as impossible for the step-4
mechanisms, and it is available here because the demand is the only thing that
differs and it draws no random numbers. 128², 5000 ticks after the step:

| demand | ℓ | `corr_du` | mean V | activity |
|---|---|---|---|---|
| −1.5 s.d. | 2.286 | −0.349 | 0.1356 | 0.00117 |
| 0 | 2.469 | −0.159 | 0.1332 | 0.00131 |
| +1.5 s.d. | 2.687 | +0.041 | 0.1314 | 0.00142 |

Feature size separates monotonically and mass moves 3% doing it, with activity
inside the deadband in every arm — which is the property the whole choice of
`du` as the lever rests on, now demonstrated on the running engine rather than
on the isolated reaction. Across the walk's full range the actuator bound
allows ℓ to span about ×1.4, and feature count goes as roughly ℓ⁻², so ×2 in
count: comparable to the 2.7× the offline drift experiment above produced, and
against the ×1.15 in ℓ the open-loop walk it replaces was delivering.

The middle row is worth reading. At zero demand the controller is holding
`corr_du` at −0.16, opposing a slow rise in the field's own ℓ that the
reference has not caught up with yet. That is the loop working as specified —
deviations from the baseline get corrected, and the baseline follows over half
an hour — but it does eat into the headroom on one side, which is why the −1.5
arm reached the clamp.

**The other half was not in the simulation at all.** `advect.wgsl` builds the
density it shades as `density_from_v · V + density_from_trail · trail`, clamped
to 1, and at 2.9 against 0.85 a *single* reaction spot cleared that ceiling on
its own with no filament under it: mean V is 0.118 and a spot reaches 0.3–0.4.
So every feature on screen was drawn as a flat-topped disc with a hard rim.
The lattice of similar-sized round holes was not merely being passed through by
the last stage before colour — it was being picked out and clipped, while the
network the piece is named after contributed about 8% of the mean density.

Two changes, both in shading and neither touching the simulation. The weights
are rebalanced to 1.9 against 1.25, which puts the ceiling *between* the two
distributions instead of below both: the reaction's extreme now reaches 0.86 of
it and clips nowhere, and the network reaches it on its strongest 1% and
nowhere near its top 10%. Both bounds matter and the second is the
non-obvious one — past a trail weight of about 1.3 the network fills the ceiling
by itself over a real fraction of the field, and where it does, the reaction's
contribution is simply discarded. That would move the clipping from the spots to
the filaments rather than removing it.

And the reaction's contribution is gated on there being network under it,
through the same saturating `trail / (1 + trail)` the trail-feed coupling itself
uses, because the reaction is doing two different things: on a filament it is
the internal texture §2 wants from the coupling, and away from one it is free
Gray–Scott in its spot regime, which is the monodisperse lattice answering to
nothing.

**How hard to gate is a trade-off, and it does not run the way the obvious
measure suggests.** Rendered through the full output chain at 160², after the
exposure governor has settled:

| | mean L | exposure | bright components | local ℓ c.v. |
|---|---|---|---|---|
| before, 2.9 / 0.85, no gate | 0.159 | 0.56 | 147 | 0.191 |
| 1.9 / 1.25, no gate | 0.156 | 0.99 | 70 | 0.222 |
| 1.9 / 1.25, gate 0.25 *(shipped)* | 0.151 | 1.38 | 25 | 0.203 |
| 1.9 / 1.25, gate 0.40 | 0.147 | 1.72 | 11 | 0.183 |

The component count is the measure that speaks to "an array of dots": the bright
material goes from 147 separate blobs to 25, and a connected filigree is not the
geometry the trigger is about. But it keeps improving as the gate rises, and the
*other* measure does not. Uniformity is the actual trigger, and the spread of
local feature size peaks ungated, is still above the unrebalanced control at
0.25, and falls back through it by 0.40. Gating harder wins the obvious measure
by hiding the reaction — and the reaction is what carries the variation in
feature size that the whole of the rest of this section works to produce. So the
gate ships at 0.25, where the blob count has most of its improvement and the
size spread is intact.

That is a genuinely uncomfortable place to have to choose from: two measures of
the same complaint, pulling in opposite directions, with no viewer to break the
tie. `pigment.v_needs_trail` is the knob, and §13 names it as the default most
likely to want moving.

**And it helps least where the complaint was worst, which has to be said
plainly.** The report was that the dots are more pronounced at low intensity.
They are, measurably — and the shading change does much less about it there:

| intensity | | mean L | exposure | bright components | local ℓ c.v. |
|---|---|---|---|---|---|
| 0.24 | before | 0.162 | 0.69 | 192 | 0.135 |
| 0.24 | after | 0.149 | 2.04 | 138 | 0.139 |
| 0.50 | before | 0.159 | 0.56 | 147 | 0.191 |
| 0.50 | after | 0.151 | 1.38 | 25 | 0.203 |

The top-left pair is the complaint in numbers: turning the intensity down gives
*more* separate blobs and a *third less* variation in their size. And the gate
takes 28% off the blob count there against 83% at the default, because it can
only hand the picture to the network when there is a network to hand it to. At
the quiet end there is not: `agents.density`, `agents.deposit` and
`reaction.trail_feed_gain` are all near the bottom of their curves, so the
reaction is barely coupled to the filaments and most of what is on screen is
free Gray–Scott. Gating then dims the field roughly uniformly and the exposure
governor puts it back, which is the "wrong correction applied to the wrong
thing" `raymarch.wgsl` warns about, arriving from a different direction.

The feature-size loop does still act there — it is upstream of all of this, and
`ell` is identical in both arms of each intensity, which is the check that says
the shading change is a shading change. So the *"changes very little"* half of
the complaint is answered at every intensity and the *"array of dots"* half is
answered mostly at the top of the range.

What would answer it at the bottom is not in this step: it is that the intensity
macro currently scales down the very couplings that hold the reaction off its
attractor, so "quiet" and "decoupled" are the same setting. Putting a floor
under `trail_feed_gain` and `deposit`, or pointing intensity at the render-side
quantities instead, would make quiet mean dimmer and sparser rather than more
monodisperse. That is a change to what a shipped macro means, so it wants
deciding rather than assuming.

**The exposure interaction, which is the one this section has always flagged as
the real risk.** Less clipped area means less bright area, so the mean density
falls by about a third and the governor has to make it up: its multiplier goes
from 0.56 to 1.38, which is nowhere near either bound, and the settled image
lightness lands on target as before. What it costs is *time*. Brightening is the
deliberately slow direction — `exposure_attack` is a third of `exposure_release`,
because the unsafe direction is always "gets brighter" — so the governor now
takes appreciably longer to walk up to its target at startup than it did when
the image arrived nearly bright enough already. That is a slower fade-in, not a
different settled level, and it is the price of the ceiling no longer doing the
tone mapping.

**What is still not answered.** Whether any of this is enough, which is the same
question §13 has been carrying: these are numbers about a field, and the
complaint is about a picture. The two measures that speak most directly to it —
the component count of the shaded density, and the spread of local feature size
— both move in the right direction and by a lot, but neither has a threshold
behind it that anyone has validated against an actual viewer. Step 6 (trail
advection) remains the largest untried lever, and is now more attractive than it
was: with the shading gated on the network, shear that stretches and pinches
filaments would reach the image far more directly than it would have when the
picture was made of spots.

### What the dots turned out to be, and what step 6 actually did

Step 5 shipped, was watched on real hardware, and the report came back: sizes
now vary, but the dots are still there — hundreds of them, still mostly
circular. Attributing the bright blobs to a field and a layer, instead of
assuming, settled it in one measurement: the blobs are **73% trail term**, at
the front layer's own scale, and the network is holding **46% of its mass in
its top 2% of texels**. The white dots are not Gray–Scott spots. They are
*trail hubs* — the ordinary winner-take-all of trail following, which §4.9
names in passing ("no capacity limit and no exit but `max_age`") and nothing
anywhere counteracted. The reaction, which every §4.7 mechanism so far acts
on, is the *background* texture — elongated, varied, and fine. Everything
built above this line moved the field that was not the dots. (Founding
respawn was ruled out as the hub source: a `found_fraction = 0` fork is
statistically identical to its control.)

A hub is round because agent congregation is isotropic plus a Gaussian blur,
and stationary because nothing moves the trail field. So the fixes are a
capacity, a carrier, and a knee.

**Deposit capacity.** A deposit landing on trail at `deposit_cap` is halved
(`1 / (1 + trail / cap)`), so hubs stop out-competing while filaments — an
order of magnitude below the cap — barely notice, and founding cohorts on bare
ground are untouched. What the capacity withholds is tracked as an EMA in the
trail texture's spare `.a` channel, summed in the reduce pass against the
income EMA, and handed back through the agent deposit exactly as flux
pruning's removal is: a redistribution from hubs to wherever traffic is, not a
sink.

Two calibration findings, both of which invert naive intuition:

- *Lower is not stronger.* At cap 1.2, across three seeds, the top-2% mass
  share falls 0.49 → 0.33 and the bright-blob count roughly halves, with trail
  mass matching the uncapped control to 1% and `corr_decay` unmoved — the
  prune postmortem's full checklist. At cap 0.6 the return **pins its clamp**
  and the capacity becomes exactly the sink it must not be: mass falls 11% and
  the hubs survive *better* than at 1.2. The reason is that deposits land on
  the network by construction — agents ride the strands they follow — so the
  deposit-weighted trail level is several times the field mean and the
  equilibrium withheld/landed ratio is well above one. The return's bound is
  3, sized from that, and `cap_return` is in the telemetry line because a
  pinned return is the failure to watch for.
- At cap 2.0 the effect fades (top-2% share 0.42): the band is real on both
  sides.

**Trail advection — step 6, at last.** The trail rides the velocity field at
`trail_advect` of the pigment's rate, all four channels together, in the same
semi-Lagrangian form. The velocity pass moved ahead of the trail pass in the
tick for it, so `velocity` stays a derived field — written every tick before
anything reads it — rather than becoming checkpoint state; the
structure-following flow component consequently reads the previous tick's
reaction, one diffusion step behind, which nothing can see.

What is verified is the mechanism and the invariants, in step 4's tradition: a
blob under a known velocity translates by exactly `velocity · advect_dt ·
trail_advect` per tick with the carry conserving mass to 0.5%, and across
seeds the sweep shows mass, mean V, activity and every homeostat correction
unmoved with it on. The aggregate — whether the network's 400-tick
autocorrelation falls — did **not** resolve above run-to-run variance at test
resolution (0.89 and 0.99 on two seeds at gain 0.5), exactly as step 4's churn
did not; §13's caveat applies. One measured cost is accepted: with the trail
sliding under the depositors the capacity de-hubs somewhat less (top-2% share
0.33 → 0.39), which is the price of the hubs being moving objects rather than
fixed ones.

**The knee.** The step-5 rebalance stopped the *reaction* clipping and thereby
handed the ceiling to the hubs — a third of the bright-blob texels sat clipped
flat. The trail's rendered term is now `knee · tanh(trail / knee)`: within 12%
of linear at filament level, bounded at `knee` above it, so no amount of hub
mass renders as a hard-rimmed white disc. Chosen against dumped fields —
knee 0.45 takes the clipped fraction of blob texels from 32% to 3% while
moving filament brightness by under 0.003 — and it dims hubs rather than
removes them, which is the capacity's job. The two compose: the capacity
thins the hubs' mass, the knee stops whatever remains from clipping, and the
advection keeps it moving.

Together, on the pigment structure term at test scale: bright-blob count 72 →
34 (three-seed means, capacity alone; 56 with advection on), no invariant
moved, and the picture's brightest object is now the network. Whether that is
*enough* is the same §13 question as ever — these are numbers about a field,
and the complaint is about a picture.

**One interaction found by a failing test, and what it turned out to mean.**
With the trail mobile, the rift soak test's severance ratio went to noise —
and isolating it cleared the suspect the arithmetic pointed at: the capacity
does not blunt rifts at all (severance 0.80 with it, 0.79 without, same seed
and ground). What advection does is dissolve the *measurement*: severance is a
statement about the ground under a fixed disc, and a mobile network has no
such ground — the same seed's disc sits on trail at 0.57 with advection off
and 0.002 with it on, because the network had drifted elsewhere. The rift
itself still works at shipped defaults whenever there is ground to sever:
on a seed whose disc lands on material, severance measures 0.53 with the
usual heal behind it. Its appearance shifts from "a gap grows and heals"
toward "a zone the network thins while crossing", which may well be the
better look and is a §13 question. The test now asserts the mechanism with
the trail held still, and says why.

---

### The network that was never there

The step-6 mechanisms shipped, were watched, and the dots were still there —
brighter-ordered, more varied, and still hundreds of persistent light discs.
The report that settled it came with a falsifying experiment already run:
gating the reaction's rendered term onto the network (`v_needs_trail = 1.0`)
made the picture *worse* — nothing but dots — and removing the reaction from
the density term entirely (`density_from_v = 0`) changed nothing visible.
Whatever the image was made of, it was the trail.

Dumping the raw fields from a from-scratch run at shipped defaults said the
rest. **The trail layer holds no network at all.** Filaments appear in the
first few hundred ticks — confirmed on real hardware as well as on the
software adapter — and then every agent ends in a round stationary knot;
from tick ~1000 onward the layer is knots on black, stable indefinitely,
p99/mean concentration ~16 with a third of its mass in its top 2% of texels.
The reaction field alongside it is an elongated organic labyrinth — the most
network-like thing in the system. Every mechanism in this section acted on a
"network" that did not exist: the capacity and the knee made the knots
dimmer and the advection made them drift, which is exactly the improvement
that was reported, and none of it could make them be a network. And this is
not a regression: the same probe on the pre-step-5 and pre-step-6 builds
grows the same knot field, same seed, same layout. Founding respawn and
trail advection were each ruled out by disabling them — knots regardless.

**The cause is that sensing is unbounded, and the capacity bounded the wrong
half.** `deposit_cap` limits what a hub can *store*; nothing limited what it
could *attract*. A knot at trail 1.5 out-competes a filament at 0.15 from
anywhere in sensor range, ten to one, forever — so every strand loses its
agents to the nearest knot, thins below sensing range, and dissolves, while
the knot's members orbit it at their minimum turning radius
(`speed / turn_rate` ≈ 2.8 cells, which is exactly the knot size). The
returned capacity mass made the attractor no weaker, because attraction was
never a function of what landed, only of what was sensed.

**The fix is to saturate the sensing.** `sample_trail` clamps what the
sensors read at `agents.sense_cap` (zero disables), so a healthy filament is
exactly as attractive as any hub — and the dispersal comes free: inside a
saturated plateau the three sensors tie, a tie reads as "keep going", and an
agent drives straight out of a knot instead of orbiting in it. Measured
against an uncapped control (same seed, 320×180 and 128×128, 4000 ticks):
the layer grows an **anastomosing network** — strands, junctions, closed
loops, coarsening and reconnecting through the whole run — with trail mass
identical to the control (0.095, within 1%), p99/mean down from ~16 to ~4.7,
and the top-2% mass share from 0.37 to 0.10. The obvious parameter-space
alternative, a turn rate above the sensor angle plus more jitter, was tried
and is far worse: sparse lone strands with knots between them.

Three calibration notes. **The cap is a ratio, not a level, and the quiet
end is why.** The first cut shipped an absolute cap scaled with the deposit
alone, and at `intensity = 0` it drew something new: *rings* — agents
orbiting the rim of their own saturated deposit. The intensity macro halves
the agent density as well as the deposit, so the level a filament can
sustain falls with the product of the two, and an absolute cap that was 2×
filament level at the default was ~8× the equilibrium at the quiet end: the
plateau survived only at knot cores, and the cap turned knots into donuts
instead of strands. The equilibrium mean trail is exactly
`density · deposit / trail_decay` (0.091 predicted, 0.095 measured at
defaults), so `sense_cap` is that multiple — 3.3, i.e. ~0.30 absolute at
defaults — and each backend packs the absolute value from its own agent
density in `_physics_values`. Measured at the ends of the intensity macro,
p99/mean against the uncapped control on the same seed: 5.0 against 13–19 at
the quiet end (a sparse, wispy network; the deposit-scaled cap's ring state
sat at ~10), 4.7–5.0 against 14–16 at the dense end. This is also what makes
the volumetric backend right for free: a voxel sees a third of a cell's
traffic, and a shared absolute cap would have been inert there. The cap is
a liveness bound when enabled: `recent` is an EMA of sensed — and therefore
capped — values, so a cap at or below `starve_threshold` reads the whole
population as starving and it respawns forever; the packed value is floored
well clear. And the fusion `excess` now saturates with everything else, so
commitment near hubs is gentler — junction behaviour among ordinary
filaments, which live below the cap, is unchanged, and the network in the
measured runs visibly fuses.

What this reopens is welcome rather than costly: the capacity, the knee, and
the §4.7 morphology levers were all tuned against a knot field, and hubs now
barely form to need them. They stay — the capacity still bounds the residual
concentration the cap's plateau cannot see, and the knee still bounds the
render — but their measured thresholds live in tests that pin sensing off
(`sense_cap = 0`), so each mechanism is still checked in the regime it was
built for. The from-scratch network claim has its own soak test, both
regimes asserted against each other.

One honest caveat, in §13's tradition: every number here is from the
software adapter at test scale, and the claim that matters — that the
*picture* is now a living network rather than a field of dots — is a
first-viewing question. The mechanism, the invariants, and the from-scratch
morphology are what the tests can hold.
