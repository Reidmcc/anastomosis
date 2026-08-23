> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 16. Resonance — the field as a music visualizer

> This section began as a proposal. §16.1–§16.7 are the design as proposed;
> the "what step *n* actually did" records under §16.8 are what has shipped,
> deviations included.

The observation that prompted this section is fair and worth conceding at the
top: activation is regulation with the intensity turned up. Deliberately so —
§14.2 chose "same macros, different endpoints" precisely because nothing about
sensory seeking wanted different *mechanisms* — but the consequence is that the
two modes differ in degree, not in kind. A genuinely distinct third mode
cannot come from a third curve table, because a curve table only says where
the knobs reach. What makes a mode different in kind is a different answer to
a deeper question: **where does the variation come from?**

Today the answer is: from inside. Every slow change on screen is endogenous —
OU walks integrated into stored fields (§3), a climate advecting its own
weather (§4.1), Poisson arrivals nobody scheduled (§4.3), a homeostat leaning
gently on the means (§4.2). The proposal here is a mode whose variation is
**exogenous**: the machine's own audio output — whatever the computer is
playing — captured, reduced to a handful of slowly-usable features, and wired
into the places the endogenous drivers already reach. The field stops
generating its own weather and starts *listening* to the room's.

The proposed key is `mode = "resonance"`, because that is the relationship:
the field does not display the music, it resonates with it.

### 16.1 What kind of thing it is — the taxonomy, extended

The activation and rhizotron work sharpened a two-way taxonomy: `backend`
decides what the engine *is* (structural, applies to a new field), `mode`
decides where the knobs reach (non-structural, a live ramped transition).
Resonance is not a backend — it wants no new substrate, no new geometry, no
new way of drawing; the fungal field is exactly the right instrument. And it
is more than a tuning — a curve table cannot hear. It is the third kind of
thing: a **drive source**. The taxonomy becomes:

- `backend` — what the engine is;
- `mode` — where the knobs reach;
- *drive* — where the slow variation comes from: endogenous (the OU walks,
  the scheduler's own arrivals) or exogenous (an audio stream).

User-facingly it should still be a mode — a third entry in the panel's Mode
selector — because "which application is this" must stay one selector, and
because the drive needs a tuning to modulate *around*: resonance resolves the
macros through its own curve table (starting as a copy of activation's, the
same way activation's started as a copy of regulation's), and the audio drive
rides on top of what the table resolves. Internally the two halves stay
separate: the table is data beside the other two in `MODE_CURVES`, and the
drive is a new object with one job.

Two boundary decisions, both taken the strict way:

**The drive is an overlay, not a replacement.** The OU walks, the climate,
the homeostat and the scheduler all keep running underneath, exactly as in
the other modes. Audio *adds* modulation on top of what they produce. This is
not timidity; it is what makes the mode degrade correctly. Music stops, the
features decay to zero, the modulation becomes the identity — and the field
is simply itself again, breathing on its endogenous drivers, rather than
freezing into whatever posture the last chorus left it in. Silence is not an
error state in a mode built for a machine that plays music sometimes; it is
most of many days, and the design has to be good *at* it. (A replacement
design would also have put the homeostat's world at stake for nothing: §4.2's
bands assume the endogenous drive exists to correct around.)

**The agents are not audio targets.** The request was precise — "the
non-agent part of the visualization" — and the architecture agrees with it
for its own reasons. What the music drives is the *medium*: the flow, the
pigment, the colour, the climate, the events. The agent layer's own
parameters — speed, sensing, deposit, fusion — stay under the macros and the
climate, untouched by the features. The network therefore keeps its autonomy
and *rides* audio-driven weather: bass surges the current the filaments are
sheared by (`trail_advect` already couples structure to flow), events bloom
where a downbeat landed, colour floods with the treble — and the organism
responds as an organism, on its own timescale. That seam is what keeps this
anastomosis-with-music rather than a spectrum display wearing filaments: the
music moves the weather; the network lives in the weather.

### 16.2 Getting the audio — the platform question answered honestly

"Whatever the computer is playing" is a loopback capture, and the three
platforms are not equally willing:

| Platform | Route to the machine's own output | Honesty |
|---|---|---|
| Linux (PulseAudio / PipeWire) | Every output sink has a `.monitor` source; any recording API sees it as an ordinary input device. | Solid. Works out of the box on essentially every desktop distribution. |
| Windows (WASAPI) | Loopback capture exists in the OS, but PortAudio as shipped (and therefore `sounddevice`) does not expose it. Some drivers expose a "Stereo Mix" input device that is the same thing. | Partial. "Stereo Mix" when present; otherwise a dedicated backend (the `soundcard` package speaks WASAPI loopback directly) is the likely step-2 answer. |
| macOS (CoreAudio) | No public loopback API. The established route is a user-installed virtual output device (BlackHole) set as the system output, which then appears as an input. | Requires one manual setup step, documented, or the fallback below. |

The strategy that follows from the table, in order:

1. **Prefer a device that is the machine's own output**: an input whose name
   marks it as a monitor/loopback (`monitor`, `loopback`, `blackhole`,
   `soundflower`, `stereo mix`, `what u hear` — the vocabulary is small and
   stable), chosen by heuristic, overridable by an explicit `audio.device`
   in the config for the cases the heuristic cannot see.
2. **Fall back to the default input** — the microphone. A laptop playing
   through its speakers with the mic listening is still a music visualizer,
   room-coupled; the AGC of §16.3 exists partly so that this path produces
   the same feature ranges the loopback path does.
3. **Fall back to silence.** No capture library installed (`sounddevice` is
   an optional `[audio]` extra, in the same spirit as `[ui]`), no input
   device, a device that dies mid-session — the drive reports its status in
   plain words for the panel to show, the features sit at zero, and the mode
   keeps running as the overlay-identity of §16.1. The mode never fails to
   open, for the same reason a damaged checkpoint never stops a launch
   (§4.6): degradation beats refusal in a tool built for long sessions.

Capture runs on the audio backend's own callback thread, hands raw blocks
across a bounded lock-free queue, and everything downstream of the queue runs
on the render thread at poll time. The queue is bounded because the render
thread can stall (checkpoint readback, a compositor wedge) and audio must
never be the thing that backs memory up behind it; overflow drops the oldest
blocks, which for a feature stream is self-healing.

One sentence on privacy, because a microphone fallback earns it: samples are
reduced to a handful of floats within milliseconds of arriving, nothing is
ever written to disk or included in a checkpoint or diagnostic report, and
nothing leaves the process. The features are the whole product of capture.

### 16.3 From samples to features — the front end

The engine cannot use audio; it can use a small number of smooth, bounded,
slowly-varying numbers. The front end reduces the stream to exactly that,
and every choice in it serves one of three masters: the no-punctuation
discipline (nothing downstream may step), the AGC question (quiet and loud
sources must modulate alike), and §3's clock discipline (the extractor is a
deterministic function of the *sample stream* — it contains no wall-clock
reads, and all of its time constants are advanced by sample count, so the
same stream produces the same features, always, testably).

Concretely (values as shipped in step 1; all tunable):

- **Framing.** Mono downmix, hops of 1024 samples (~21 ms at 48 kHz — ~47
  feature frames/s, comfortably above any sim rate), each analysed against a
  2048-sample Hann window.
- **Sanitisation first.** NaN and infinity from a broken driver become
  zeros, amplitude is clamped, DC is removed — the same posture as the §4.6
  NaN quarantine, applied at the door. A hostile stream must be at worst a
  boring one.
- **Level and three bands** — bass 20–150 Hz, mid 150 Hz–2 kHz, treble
  2–8 kHz — each normalised by a slow automatic gain reference (a decaying
  peak of the RMS, ~20 s time constant, floored so silence cannot amplify
  noise into signal) and soft-compressed into [0, 1). The AGC is what makes
  a whispering stream and a mastered-loud one drive the same visual range,
  and its floor plus the silence gate is what makes "quiet" mean *zero*
  rather than "the noise floor, amplified".
- **Envelope followers** on all four: ~80 ms attack, ~500 ms release. The
  attack bounds how fast any downstream parameter can be asked to move —
  the followers are the first of several lowpasses between a drum hit and
  a pixel — and the release is what makes the field *subside* after a
  phrase rather than cutting.
- **Onsets by spectral flux, in the log domain.** The half-wave-rectified
  positive difference of successive log-magnitude spectra, against an
  adaptive threshold (a multiple of the recent median), with a ~150 ms
  refractory. Log-domain deliberately: the AGC that serves the continuous
  features would eat amplitude steps (a quiet verse and a loud chorus
  normalise to the same spectrum), and transients are precisely the thing
  the continuous path is built to smooth away. Two paths, two jobs: the
  followers carry *how much is happening*, the flux carries *that something
  just happened*.
- **A silence gate.** RMS under −60 dBFS for a second or more flags the
  stream silent; the feature targets go to zero and arrive there through
  the release followers. Silent is a first-class state, not an absence.

The output is one small immutable record per hop — level, bass, mid, treble,
flux, onset strength, silent — every field in [0, 1], every field already
smooth. Nothing downstream ever sees a sample.

### 16.4 Where the features enter — three doors, all of them existing seams

The engine already has exactly the seams this needs, which is most of the
argument that the mode belongs in this codebase. Audio enters through three,
and through nothing else:

**1. Continuous modulation of resolved parameters.** Each frame the app
resolves config → ramp → `Params` (`app.draw_frame`). The drive adds one
step: `params = drive.modulate(params)` — a pure function from the ramped
params and the current features to the effective params for this frame,
applied *after* the ramp (the modulation must be allowed to be faster than
the 8 s ramp; its own speed limit is the followers) and *before* the engine,
with `validate()` and the §4.4 reaction clamps re-applied to what it
produces, so the ceilings hold by the same construction they always have.
The target list is a **whitelist**, and it is §14.1's list of channels the
safety argument leaves free, verbatim:

- *Motion*: `flow.psi_gain`, `flow.field_gain` scaled up by bass and level —
  the current surges on the low end, and the network is sheared by it via
  `trail_advect`. This is the ~100 ms path that makes the field read as
  moving *with* the music.
- *Colour*: `render.chroma_activity_gain`, `render.c_max` (toward its own
  ceiling, never past), `render.polychrome` lifted by treble and level;
  hue rotation rate nudged by flux, so a busy passage wheels the palette
  and a still one lets it rest. All chroma-budget spend, per §14.1(2).
- *Material*: `pigment.inject_rate` and the climate's range parameters
  breathing with the mids — louder passages are more fertile weather.

And a hard rule enforced by the same construction: **the luminance
architecture is not on the list.** `render.background_luma`, `l_max`,
`filament_luma`, `glow_gamma`, `safety.*`, `exposure_target` — none is a
modulation target, and a test asserts the whitelist's intersection with the
brightness and glow macro paths and the safety table is empty. Audio buys
motion, chroma and incident, never luminance — the beat cannot be spent on
the one channel §14.1 closed.

**2. Onsets become events, through the front door.** An onset above
threshold asks the scheduler for an event via the same `trigger` the panel's
buttons use — the same `_spawn`, the same jittered radius and amplitude, the
same raised-cosine envelopes (activation's 15 s attack floor), the same
`max_concurrent` honoured by refusal rather than queueing, the kind chosen
by which band moved (a bass onset breathes a bloom or a current shift, a
treble onset a tint). §4.3's machinery was built so that *nothing* an event
source does can punctuate, and it does not care that the source is a kick
drum rather than an RNG. A 140 BPM track does not spawn 140 events a minute;
it saturates the concurrency cap and becomes a field that is always inside
weather — §4.3's own framing of the fast end, arrived at from outside the
process. The one scheduler change resonance wants is a per-mode refractory
on triggered arrivals so consecutive onsets prefer *distinct* regions, which
is a placement bias, not a new event kind.

**3. Bands into depth** (a later step, and optional): the layered backend
has per-layer parameter blocks already, so bass can lean on the back layer's
flow and treble on the front sheet's — the stack becomes a spectrum, back to
front, and parallax makes the low end literally *behind* the high end. The
volumetric analogue (band → depth bins of the slab) falls out of the same
shape. This is the step that would make the mode's geometry unmistakable,
and it costs nothing structural because the per-layer split already exists.

What does *not* get a door: the reaction's `feed`/`kill` beyond the existing
climate ranges (§4.4 measured the live band; a bass drop must not walk a
region off the map), the homeostat (it corrects the means under everything,
audio included, and that is the point of it), and the agents (§16.1).

### 16.5 The latency question — what "with the music" can honestly mean

Everything in this engine is built to be slow, and a visualizer reads as
connected to the music only if *something* answers within roughly a tenth of
a second. These two facts have to be reconciled in the open.

The budget, on the motion path: ~21 ms of hop, ~80 ms of follower attack, up
to one sim tick (33 ms at the 30 Hz activation top) for the modulated flow
gain to act, and the interpolator's sub-tick blend. Call it **120–150 ms
from a drum hit to the current visibly surging** — musically, on the beat's
tail rather than its face. The chroma path is similar: the chroma slew
allowance (0.030/frame against a 0.100 ceiling) crosses a large chroma
distance in three or four frames once the demand arrives. The event path is
deliberately slower — 15 s attack at the floor — and reads as the *phrase*
level rather than the beat level, which is what events are for.

And one path can never answer at all: **luminance**. At the 1%/frame bound a
10% lightness excursion takes 333 ms by construction, in every mode, forever
(§7). A beat-strobing visualizer — lights flashing on the kick — is the one
thing this application is constitutionally incapable of being, and that is
its identity, not its bug: the promise on the README's first screen outranks
the mode. What resonance offers instead is the field *dancing*: surging,
leaning, flooding with colour, blooming on the downbeats — a visualizer for
someone who wants the music made visible without being made stroboscopic,
which is, for this application's audience specifically, rather the point.
The README should say this in as many words when the mode ships.

Two engineering notes the budget imposes. Resonance's tempo curve should
top out at activation's 30 Hz sim rate, since a 12 Hz regulation-bottom
tick would add 80 ms of quantisation to every path. And the budget governor
(§8) may throttle `sim_hz` under load exactly as it does today — the mode
degrades to slightly laggier, never to unsafe.

### 16.6 Safety analysis — what resonance actually risks

The per-pixel bound is not at risk, for the standing reason: the limiter is
downstream of everything, modulation included. The honest risks:

1. **Adversarial audio is adversarial parameter movement.** A full-scale
   square wave at 20 Hz is the worst thing a stream can say, and after the
   front end it is: features slamming between bounded values at follower
   speed. That is precisely the input class the flash-safety suite already
   distrusts and slams (§10, §14.8 step 1 — where the mode-switch slam
   found a real leak). The suite gains a twin: features stepped between
   extremes every frame, flow off so the per-pixel bound is exact, and the
   WCAG-area assertion under a synthesised worst-case stream. No new
   argument is needed — only the existing argument aimed at the new input.
2. **The WCAG area criterion, again, on the motion path.** Step 2 of §14.8
   certified the motion axis to 6× the regulation tops with the area
   fraction at 0.0%; activation's tops sit at 1.6–1.9×. The rule for
   resonance: modulated tops — curve top × maximum modulation gain — stay
   inside the swept 6×, asserted by test exactly as activation's tops are,
   and re-swept with `tempo_sweep.py` if tuning ever wants past it.
3. **Events cannot punctuate, but placement can cluster.** Onset-triggered
   events inherit every §4.3 constraint. The residual risk is several
   onsets landing events in one region within a few seconds — summed
   envelopes approaching the area cap locally. The concurrency cap already
   bounds the sum; the placement bias of §16.4(2) spreads it; the soak
   suite's event-overlap assertions run with a dense synthetic onset
   stream.
4. **The homeostat under sustained loud music.** Continuous modulation of
   `pigment.inject_rate` and the climate ranges shifts what the controller
   measures, and a controller centred for silence could lean against an
   evening of drum and bass with `corr_decay` for the whole session —
   §14.3's question, re-asked. Same answer: measure (run the busy corner
   under a sustained synthetic loud stream, watch the corrections), and
   the fix if needed is per-mode targets, which §14 already priced.
5. **The failure the watchdog cannot see.** Capture adds a thread and a C
   library. The callback writes to a bounded queue and takes no Python
   locks the render thread holds, so a wedged audio backend can starve the
   *features*, never the frame loop; `poll()` returns the last features on
   an empty queue. The stall report (§8.2) gains the drive's status line so
   a dead stream is visible in diagnostics rather than a mystery of a
   suddenly-still field.
6. **Photosensitivity posture, restated.** Same bound, same enforcement,
   same caveat, now for a mode whose input is beat-structured. The bound is
   what makes that sentence safe to write: the periodicity of music reaches
   the screen only through channels the flash argument does not price, and
   the one channel it prices is closed to audio by whitelist and to
   everything by the limiter.

### 16.7 What does not change — the invariants, gathered

- `SAFETY_CEILINGS`: one table, now serving three modes; `max_luma_delta`
  and the per-second budget untouched.
- The output chain in `backend.py`: limiter, exposure governor, gamut
  mapping, dither — shared, untouched, downstream of modulation as of
  everything.
- **No functions of the clock, still.** The extractor is driven by sample
  count; the modulation is driven by features; neither reads time. Audio is
  an input, not a clock — the periodicity a track carries into the image is
  the user's chosen input, confined to the mode that asks for it, and the
  other modes' non-repetition promise is untouched because the drive is an
  overlay that is identity at zero.
- The checkpoint format. Nothing about the drive is checkpoint state: the
  AGC reference and follower states rebuild from live audio in seconds, and
  are deliberately left out on the same reasoning as the event scheduler's
  RNG (§4.6 — memoryless enough that a fresh start is indistinguishable).
  `mode = "resonance"` is non-structural like the other modes, one field
  per backend serves all three, and a field saved under resonance resumes
  cleanly into regulation.
- The rhizotron keeps one tuning (§15): under that backend the Mode
  selector stays greyed at regulation, resonance included. If the root
  world ever wants rain that answers the rain in the music, that is its
  own proposal with its own safety record, not a default.
- Events: enveloped, localised, area-capped, concurrency-capped, applied to
  climate only — whatever asks for them.

### 16.8 Build order

Each step lands something usable alone; the first is deliberately invisible,
exactly as activation's was:

1. **The audio front end.** `audio.py`: device discovery with the loopback
   preference of §16.2, capture behind the optional `[audio]` extra with
   every degradation path of §16.2(3), and the feature extractor of §16.3 —
   deterministic in the sample stream, testable headless against
   synthesised signals, no engine wiring. Tests: band separation, AGC
   convergence, onset firing and refractory, silence gating, hostile-input
   boundedness, stream-determinism, and the no-library degradation path.
2. **The two load-bearing measurements, on real hardware** (the development
   environment has neither GPU nor sound): the platform capture inventory —
   which of §16.2's routes actually opens on each machine, and whether
   Windows needs the `soundcard` backend — and the end-to-end latency of
   §16.5's budget, hop to visible surge, measured rather than summed.
3. **Mode plumbing.** `resonance` in `MODES` with its table a copy of
   activation's, panel selector entry plus the drive's status line, preset
   mode-tagging, the three-way switch-slam test. No audible effect on the
   image yet beyond what the table copy implies (none).
4. **The modulation layer and the event door.** `modulate()` over the
   whitelist with clamps re-applied, onset → `trigger` with the placement
   bias, the silence-identity test (features at zero ⇒ bit-identical
   params), the feature-slam flash test, the whitelist/luminance
   intersection test, and the homeostat lean measurement of §16.6(4).
5. **Endpoint tuning and presets, by eyes** (§13's standing rule): the
   modulation gains, the onset threshold's musical feel, and the first
   resonance presets. The README section, including §16.5's sentence about
   what this visualizer will never be.
6. **Bands into depth** (§16.4(3)), behind gains that default to the flat
   coupling, judged on real hardware; the volumetric variant if the layered
   one earns it.

### What step 1 actually did

Built as specified in §16.2–§16.3: `anastomosis/audio.py` holds the pure
feature extractor (`FeatureExtractor`, one immutable `AudioFeatures` record
per 1024-sample hop), the device-preference heuristic as a pure function
(`pick_capture_device`), and the capture wrapper (`AudioDrive`) that owns
the callback thread, the bounded queue and every degradation path —
`sounddevice` is imported lazily and its absence is a status line, not an
exception. Nothing imports `audio.py` yet; the module is load-bearing only
through its tests, which is what step 1 promised.

Two small design points the implementation settled. The continuous features
and the onset detector genuinely wanted different normalisations — the AGC
that makes quiet and loud sources equivalent for the followers erases
exactly the amplitude steps the onset detector exists to see, so flux runs
in the log-magnitude domain instead, unnormalised, with the adaptive median
threshold absorbing level differences (§16.3's "two paths, two jobs",
discovered the hard way in the first sketch and worth pinning: a
*normalised*-flux onset detector cannot hear a volume change, and a volume
change is a musical event). And determinism in the sample stream turned out
to be free to test but easy to break — the streaming test (same samples,
different chunk boundaries, identical features) is the one that guards the
property §3 cares about, and it caught an off-by-a-hop in the first
window-buffer implementation before it ever reached review.

The capture-side automatic reopen after a device dies mid-session is
deliberately *not* in step 1: it needs wall-clock-free backoff (poll-count
based) and a real dead device to test against, so it belongs with step 2's
hardware pass. Until then a dead stream is a visible status line and a
silent, breathing field — the degradation §16.2 requires.
