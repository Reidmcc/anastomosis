> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 9. Parameters and control surface

~40 primitive parameters, but exposing 40 sliders is a worse interface than exposing
6 good ones. Two tiers:

**Macros** (the normal interface):

| Macro | Effect |
|---|---|
| Intensity | overall activity, contrast, agent count |
| Scale | feature size across all layers |
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
