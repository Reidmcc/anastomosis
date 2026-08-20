> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 6. Colour

All colour work happens in **Oklab / OkLCh**, not sRGB or HSV. This is not
fastidiousness — it is a requirement of the brief. Interpolating a hue rotation in
sRGB or HSV swings through large *perceived* lightness excursions (the classic
blue→yellow brightness surge), which is exactly the punctuation the application must
not produce. In Oklab, lightness is separable and can be capped independently.

**Colour is a function of simulation state, not of a clock:**

| Perceptual channel | Driven by |
|---|---|
| Lightness `L` | pigment density, with layer depth attenuation |
| Chroma `C` | heavily lowpassed local activity — busy regions saturate, quiet regions desaturate toward the background |
| Hue `h` | local field orientation (`atan2` of `∇V`) + reaction-species ratio `U/V`, offset by a global drifting anchor |

What that pigment density is *made of* is decided one stage earlier, in
`advect.wgsl`, and it turned out to matter more than anything in this section:
weighted as it originally was, the density handed to the colour stage was
essentially the reaction field with the filament network as a rounding error,
and every reaction spot arrived already clipped against its ceiling. §4.7 step 5
has the measurements. Nothing here changed; what changed is that the thing being
graded is now the network rather than a lattice of discs.

The hue anchor is one channel of the climate field, so hue varies *spatially* as
well as drifting globally — different regions sit in different parts of the palette
and those regions migrate. Global hue rotation defaults to one full turn per ~45
minutes (tunable), slow enough to be imperceptible moment-to-moment while making a
glance ten minutes later clearly different.

Constraints applied after mapping and before output:

- `L` and `C` clamped to configured ranges — a hard bound on both brightness and
  saturation, enforced at the last stage.
- Gamut-mapped back to sRGB by chroma reduction at constant `L` and `h`, so clipping
  can never change perceived brightness.
- **Blue-noise dithering before quantisation.** This matters more than it sounds:
  an 8-bit display showing a very slowly drifting smooth gradient produces visible
  banding, and worse, *crawling* band boundaries as the gradient moves — a moving
  hard edge, which is precisely a form of visual punctuation. A void-and-cluster
  blue-noise mask, animated per-frame, removes it. If a 10-bit or HDR surface is
  available, use it and reduce dither amplitude accordingly.
