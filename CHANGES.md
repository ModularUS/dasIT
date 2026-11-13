# Changes

## 2.0.1

`analytic` and `oversample` compute with an FFT, which treats a record as periodic. A signal at the end of a record
therefore showed up at its start: a burst cut in half by the end of a 780-sample record put 54 % of its peak into
the envelope of the first rows, and 14 % into the oversampled signal there. Both functions now continue the record
with zeros by half its length on both sides and cut back afterwards (0.006 % and below 0.2 % in the same case).
Signals inside a record are unchanged. The reference frame of the example changes by 0.2 % of its maximum in the
top rows, where the echo of the transducer surface at the start of the record no longer leaks, and by less than
0.01 % below row 100.

## 2.0.0

dasIT 2.0 is a new implementation. The classes of dasIT 1.x (`dasIT.src`, `dasIT.features`, `dasIT.visualization`,
`RFDataloader`, `TDloader`, `TGCloader`) are removed, and there is no compatibility layer. The 1.x code is the git
tag `v1.1.0`.

### New

| Module | Content |
|---|---|
| `dasIT.beamform` | `Beamformer`: delay-and-sum from explicit inputs in SI units, as one sparse matrix or as a loop over the elements, one or several transmits per image |
| `dasIT.geometry` | element positions, plane-wave transmit delays, axes of scanner-beamformed lines |
| `dasIT.signal` | band-pass, analytic signal, oversampling, envelope, log compression |
| `dasIT.plot` | `show_image` |
| `dasIT.data.loader` | `load_rf`, shots and frames on separate axes |

### What 2.0 computes differently from 1.x

Measured on the bundled example (CIRS 054GS phantom, one plane wave).

| 1.x | 2.0 |
|---|---|
| element positions stretched by 192/191 | elements exactly one pitch apart. Lateral scale error +0.55 % becomes +0.02 %. |
| delay rounded to the nearest sample, at four samples per period | linear interpolation after 4x oversampling. Amplitude error of a point 21 % becomes 1.2 %. |
| 10-tap filter that passed 36 % of a constant offset | Butterworth band-pass, zero phase |
| receive aperture from the elevation focus (z / 1.27), Hann window with two unused end elements | aperture z / f-number (default 1.5), every element inside it used |
| pixels not normalised, brightness grows with the aperture | optional division by the sum of the weights |
| record cropped to 1416 samples, first 40 samples set to zero | the whole record |
| time of the first sample taken as zero | time zero as an input. For the example +0.62 microseconds from the acquisition settings, the image sits 0.48 mm deeper. |
| raw TGC control values multiplied onto the data | no gain applied |
| lateral interpolation of the envelope after beamforming, which produced negative values | finer lateral grid beamformed directly |
| several plane-wave angles failed | coherent compounding of any number of transmits |
| log compression returned not-a-number for an empty frame | an empty frame maps to the lower limit |

### Open findings on the example recording

They are present in 1.x and 2.0 alike and belong to the recording or its description.

- The two wires at nominal 20 and 40 mm depth are 19.1 mm apart in the image (depth scale -4.5 %). Sampling rate
  (20.83 MHz) and speed of sound (1540 m/s) are confirmed by the acquisition settings.
- The wires are about 1.0 mm wide laterally. The expectation for f/1.5 at 5.2 MHz is 0.45 to 0.65 mm.
- The near-field wires sit about 1.2 mm deeper than their nominal depth below the phantom surface. A layer of
  coupling gel would explain it. It was not measured.
- The acquisition fired three plane waves per frame (-12, 0, +12 degrees). The example file holds one of them per
  frame, the 0 degree transmit according to its author.
