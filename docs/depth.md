> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 5. Depth

On the target hardware (§8.1) the GPU budget does *not* rule out true volumetric
raymarching — a 1440p 48-step march is ~5 Gsamples/s, which is a few percent of an
RTX 3080's texture throughput. The reasons to start with **layered 2.5D** are
implementation risk and lateral resolution, not cost:

- a 2D grid at 1440p has far finer filament detail than any affordable 3D grid;
- 3D Physarum needs different sensing and steering, and is much harder to tune;
- the layered path validates the colour, safety, and pacing stages, which are
  identical under either depth backend.

So: layered 2.5D first, volumetric slab as a **planned alternate depth backend**
once the rest is proven (§5.1), not as a speculative stretch goal. Both are now
built, and the choice between them is the user's -- a selector in the control
panel, a `backend` key in the config, a `--backend` flag. The default is still
the layered path.

- 3 independent simulation layers at different spatial scales and tempos (back layer
  large/slow, front layer fine/quicker).
- **Resolution follows depth of field.** Back layers are blurred by DOF, so
  simulating them at full resolution computes detail that is then discarded. Front
  layer at native 1440p, mid at 1/2 linear, back at 1/4 — total ~4.9 M cells instead
  of 11.1 M, with no visible loss. Agent counts scale with layer resolution the same
  way.
- Composited back-to-front with Beer–Lambert transmittance, so nearer material
  genuinely occludes and tints what is behind it, rather than just alpha-blending.
- **Parallax** offset per layer driven by an extremely slow drift (and optionally by
  cursor position, if the cursor is on that display — probably not, since the point
  is the user is working elsewhere).

  Two things about that drift were wrong for as long as it existed, and both are
  worth recording because neither is visible as a fault — a viewpoint that does
  not move produces a perfectly good-looking image. It was an Ornstein–Uhlenbeck
  walk with a fixed *per-frame* decay of 0.02 against a `sqrt(dt)` noise term,
  which gave it a stationary spread of 3e-4 across a travel of 1: at
  `render.parallax = 0.02` that is a viewpoint displaced by a few hundredths of a
  pixel, and over two hours of simulation it never left a thousandth of its
  range. Parallax was, in effect, switched off in both backends — including the
  volumetric one, where it is the *whole* of the camera's motion. It is now
  written in the same form as the feature-size walk (`_advance_du_walk`):
  unit-variance, `sqrt(1 - (1-θ)²)` noise, θ from `dt/τ`, so the excursion is a
  property of the walk and not of the frame rate.

  The second correction is that an OU process is smooth in its envelope and
  *white* in its increments. That is fine for an amplitude and disastrous for a
  position: the raw walk moves the image about half a pixel per frame, at
  random, which is precisely the per-pixel temporal noise §7 and the dither
  design exist to avoid. The walk therefore reaches the viewpoint through one
  first-order lag at a sixth of its own time constant, which makes the position
  C¹ and costs almost none of the travel — measured over an hour at 1440p, the
  frame-to-frame step falls from 0.54 px to 0.020 px while the peak excursion
  stays at 49 px of 50. What is left is about 0.6 px/s, which is what the
  parallax actually is.

  What that fix did *not* settle is whether the amount of parallax on offer was
  ever enough to see, since the range it moved over — 0.006 to 0.038 of the
  screen's width — was chosen against a mechanism that did not work and is
  therefore not evidence of anything. It is now its own macro, split out of
  `depth` exactly as `event_rate` was split out of `intensity` and for the same
  reason: the two answer different questions. Everything left under `depth` is
  a shading trick applied to a *normalised* depth and says the same thing about
  the far face however far away it is; parallax is the only cue that comes from
  the scene moving. The new knob reaches a quarter of the screen's width at
  about 8 px/s, and a config written before the split takes the new default
  rather than inheriting the old one — there is nothing to carry across from a
  setting that did nothing.

  **And the volumetric backend holds itself to what its slab has earned.**
  Parallax is thickness times the tangent of the viewing angle, which is the
  geometry underneath the whole complaint that a thin slab reads flat: 48 voxels
  against 512 is a sheet of paper, and there is not much depth to be found by
  walking around a sheet of paper. Swinging further does not buy more depth, it
  buys the same slab seen edge-on — a long oblique smear through it, with a path
  length (and so an optical depth) that grows with the swing. So the drift's
  amplitude is scaled — not clipped, which would put a corner in the viewpoint's
  path — to `PARALLAX_MAX_TANGENT` (0.8, about 39°) times the thickness. At the
  default slab that caps the travel at about 8% of the width; at 144 voxels deep
  it is 22%. The thickness knob and the parallax knob therefore compound, and
  the control panel's readout reports the *effective* travel so that a knob
  which has stopped doing anything visibly stops moving.

  The safety stage is deliberately not told about the viewpoint. It reprojects
  its history through the screen-space velocity of the *material* (§7), so a
  moving camera leaves an uncompensated residual — 0.02 px per frame against
  material moving several, three orders of magnitude below anything the limiter
  responds to. At the top of the parallax knob it is 0.3 px per frame, which is
  the same argument with less margin — and still not a visible artefact, because
  a rate limiter tracking a steadily moving edge settles at a constant lag
  rather than suppressing the motion: the displayed edge trails the true one by
  the sub-pixel offset at which demand equals budget. What is bounded here is
  the *rate* of change, and a smooth pan is exactly the case that costs it
  nothing.
- **Depth-of-field**: per-layer blur radius increases with distance. Because the
  layers are separate render targets this is one cheap separable blur each, not a
  gather.
- Atmospheric attenuation: distant layers lose chroma and contrast toward a
  background colour. This is doing most of the perceptual work.
- **Weak cross-layer coupling**: layer *k*'s field slightly perturbs layer *k+1*'s
  climate. The layers therefore aren't independent processes that happen to be
  stacked — structures loosely echo through depth, which reads as a single volume
  rather than three sheets.

### 5.1 Volumetric slab (alternate backend)

A thin slab — `512 × 288 × 48` ≈ 7.1 M voxels at the default thickness — keeps usable
lateral resolution while giving genuine volume. 48 depth slices is enough for parallax
and occlusion given that DOF blurs the far field anyway, and it is where the thickness
starts rather than where it stays: see *How thick* below.

- Sim: 7.1 M voxels × 6 passes × 20 Hz ≈ 850 M voxel-ops/s ≈ **20 GB/s** — ~3% of a
  3080's bandwidth.
- Render: 3.7 M px × 48 steps × 30 Hz ≈ **5.3 Gsamples/s** of 3D trilinear. Ray
  coherence is excellent (near-orthographic camera, no secondary rays), so this sits
  around 2% of texture throughput.

What it buys over layers: real Beer–Lambert attenuation through a continuous medium,
self-shadowing from a single soft light, and structures that pass smoothly in front
of and behind each other rather than living in three discrete sheets. For "fluid
continuous motion with depth" it is materially better — but only once the 2D system
is tuned and proven, since it makes every parameter harder to reason about.

The output stages (§6, §7) are unchanged between backends, so this is a clean swap
rather than a fork.

#### What building it actually settled

Five decisions were not obvious from the sketch above, and each is the answer to
a question the layered path never had to ask.

**The slab is a 3-torus, not a box.** Every argument for a toroidal domain in
two dimensions applies unchanged to the third: a reflecting or no-flux face
accumulates material against itself over a run measured in days, and agents that
bounce off a wall pile up along it. So all three axes wrap and no pass anywhere
has a boundary case. The cost is paid once, in the renderer: material leaving the
near face reappears at the far one, so the march applies a raised-cosine window
over both faces and the arrival and departure are fades rather than steps. That
window is also the reason the toroidal choice is free rather than merely cheap —
the far face is the fogged, dimmed, blurred end of the volume anyway.

**Divergence-free flow needs a stored potential, not a sum of velocities.** In
the plane, `v = curl(ψ)` for a scalar ψ, and the two components — imposed weather
and structure-following — can be built as velocities and added, because the sum
of two curls is a curl. In three dimensions the structure term is not the curl of
anything obvious: "flow along the iso-surfaces of V" wants `∇V × ê`, which is
divergence-free only when `ê` is constant — and a constant `ê` gives the slab a
permanent preferred plane with every filament in it sliding the same way. So the
vector potential `A` is assembled into a texture at full resolution and
differentiated in a second pass. `div(curl(A))` then cancels term for term under
central differences, because the difference operators commute — exactly zero, not
zero to within a discretisation error. Measured on the running field, the residual
divergence is 0.04% of the velocity magnitude, which is the `float16` quantisation
of the velocity texture and nothing else.

Two things ride on that licence, and neither would be exactly divergence-free
applied to a velocity: the climate's per-region flow gain (multiplying a velocity
by a varying gain adds `∇g·v` to the divergence, and material piles up wherever
the gain falls off), and the depth anisotropy below.

**Motion has to be anisotropic, and the anisotropy belongs on the potential.**
The slab is four or five feature-widths deep. Isotropic flow at a speed the agent
layer needs carries material through the entire thickness in a couple of seconds,
and the depth axis reads as churn rather than as depth. Weighting the *lateral*
components of `A` by a factor scales `v_z = ∂ₓA_y − ∂_yA_x` by roughly that factor
while leaving the lateral velocity dominated by the terms in `A_z` — and leaves
the field a curl, so it stays exactly divergence-free. Scaling `v_z` directly does
not. Agents get the same treatment applied to their heading rather than their
step, so that sensing follows motion instead of probing at angles they will not
take.

**3D Physarum sensing is a rolled cone.** Heading is a unit vector, since there is
no scalar turn in three dimensions, and steering is a normalised interpolation
toward (or, past the anti-fusion crossing, away from) a sensed direction — which
is a rotation in the plane the two span and needs no basis to express. Four flank
sensors sit on a cone of half-angle `sensor_angle`, rolled by a per-agent random
phase every tick: four *fixed* flanks give the population a preferred lattice of
turn directions, which the left/right pair in 2D cannot do. The fusion bias, the
sign change at commitment 1.0, and the head-on special case all carry over
unchanged in meaning.

**The climate needs the third axis too.** A 2D climate stretched through depth
would give every structure at a given (x, y) the same feed, the same kill and the
same feature size however far apart in z, and the volume would read as a thick
sheet. A regime is a region of the slab.

#### How thick

Forty-eight voxels is a slab a ray crosses one or two filaments of, and that is
about as subtle as genuine volume gets: the mechanisms are all working, and there
is not much material for them to work *on*. So the thickness is a knob — the only
piece of the slab's geometry the control panel moves — running from 8 voxels up to
the shorter lateral axis, which is 288 on a 16:9 display. It is structural, so it
grows a new field; and unlike the backend switch it cannot keep the old one, since
a slab of a different depth is a differently shaped array and nothing resamples one
into the other.

What it costs is memory, linearly and almost exactly: ~92 bytes per voxel, so ~13 MB
per slice at 512 × 288, from ~650 MB at the default to ~3.9 GB at the ceiling. The
panel quotes the figure beside the slider rather than leaving it to be discovered.

What it buys stops arriving before the ceiling does, and for a reason worth stating:
extinction is calibrated *per filament* (see `raymarch.wgsl`), so a ray's optical
depth grows with the number of filaments it crosses, which grows with the thickness.
Six times the depth is roughly six times the accumulated optical depth, and past some
thickness — dependent on the `intensity` macro, since that sets how much material
there is — the near structure is opaque enough that the far face contributes nothing
the eye can find. Beyond that point the extra voxels are memory spent on material
the camera cannot see. The ceiling is where the shape stops being a slab; the useful
range ends somewhere below it, and where exactly is a judgement for a real GPU.

Making the thickness a knob also forced a re-expression that was overdue. Three of
the march's quantities were fractions of the slab's depth, which is harmless while
the depth is fixed and wrong once it moves: held as fractions, a slab six times
deeper would fade six times as much of its crisp near face at the toroidal seam, and
send its shadow rays six times as far on the same six steps — blotches rather than
shading. Both were calibrated against a *filament*, not against the slab, so both are
now lengths in voxels (`depth_window_voxels`, `shadow_voxels`) and come out the same
absolute size at every thickness. The third is the march itself, which takes one step
per slice up to a ceiling (`volume.steps`, 160), because a thick slab marched at a
thin slab's step count is a ray stepping over whole filaments. Only the amount of
material a ray passes through is left to vary with the knob, which is the whole point
of moving it.

The one thing genuinely lost is lateral resolution, and it is the reason the
layered path remains the default: 512 voxels across against 2560, so filaments are
about five display pixels wide rather than one. Motion is unaffected in the way
that matters — features are five times larger *and* move five times faster in
absolute terms, so feature-widths per second, which is what the eye reads at this
timescale, is the same under both. Memory is the other real cost: ~650 MB of
`rgba16float` against ~480 MB of field state for the 1440p stack, and up to
~3.9 GB if the thickness is taken to its ceiling. (This comparison read
"~90 MB" for the stack until §8.3 had to price both against an integrated
GPU's shared memory and found the stack undercounted: eleven full-resolution
textures a layer, not three or four. The slab is still the expensive one, by
less than it looked.)

**That loss is a budget decision, not a structural one, and it is exposed as a
setting.** `config.VOLUME_DETAIL` offers the slab at 512, 768 or 1024 voxels
across — `volume_detail = "standard" | "fine" | "finest"` — and the width is the
only thing that moves: the thickness is its own knob and this leaves it where
it is, while the height keeps following the window, so the voxels stay cubic and
every argument above survives unchanged. What changes is how many display pixels
a filament covers, which at 1440p goes from about five to about two and a half.

The two slab settings are independent but not unrelated, and the one coupling is
in the right direction: the thickness ceiling is the shorter lateral axis, so a
wider slab is also allowed to be a deeper one — 288 voxels at `standard`, 576 at
`finest`. Memory is the product of both, which is why the panel prices the pair
rather than either alone.

The asymmetry in what that costs is the reason it is worth offering. Voxel count
goes with the square of the width, and three things follow it: the per-tick
passes over the volume, the interpolation pass, and memory — so `finest` is ~4×
the simulation of `standard`, at ~2.7 GB at the default thickness. The **render** side does not follow it
at all. The march is one ray per output pixel and its step count is tied to the
slab's *thickness*, which this setting does not touch, so a 1440p frame costs
the same at every size and the 5.3 Gsamples/s above is unchanged. Against the ~3% of bandwidth the sim costs at the
standard size, even `finest` leaves the headroom check of §8.1 intact.

Unlike the backend choice this cannot preserve what it replaces: there is one
volumetric field, and a slab of a different width is not the same field
presented differently — every voxel of it is a different voxel. So a size change
is a reset, and is presented to the user as one.
