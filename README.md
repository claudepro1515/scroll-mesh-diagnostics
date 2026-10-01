# Two offline checks for published tifxyz meshes: self-intersection and ink-label spillover

Two small, GPU-free, model-free tools that audit data the Vesuvius Challenge community and the
organizers have already published, without needing a CT scan of the whole scroll: a topology
checker for `tifxyz` spiral-fit meshes, and an ink-label audit for the curated segments that ship
`ink-labels/` alongside `supervision/`. Both run on `x.tif`/`y.tif`/`z.tif` (or `.zarr`) arrays you
can pull straight from the official S3 data bucket, need only numpy, and finish in seconds to
tens of seconds per mesh or segment.

## At a glance

| tool | question it answers | real result |
|---|---|---|
| `tools/mesh_topology_check.py` | Does a published winding fold back on itself (self-intersect) along its own sheet? | 0 folds in the curated control (PHerc0139 w035, 62,929 interior vertices); two public community fits of eligible scrolls do fold: armando-gaona's PHerc0211 fit, 97/120 windings, up to 20% of vertices on the innermost windings (w010–w012); rodriguescarson's PHerc0826 fit, 90/90 windings, 3–7% typical, matching the author's own published satisfaction metric (11.1%) |
| `tools/ink_label_audit.py` | Does a curated segment's published `ink-labels/` agree with its own `supervision/` mask (and, where published twice, across resolutions)? | 7 of 8 curated PHerc0139 segments with public ink labels show only normal edge bleed; segment w035 has large blocks (up to 28,753 px, median 252 px from the nearest supervised area) of `ink=yes` outside any area a human annotator ever reviewed, confirmed identically at both of its two independently published resolutions (2.399 µm and 9.362 µm) |
| `tools/umbilicus_radius.py` | Support tool: physical radius (voxels, converted to a common scale) of a winding to the scroll's own axis, so windings from different scrolls or resolutions can be compared on a like-for-like basis | Used to show that the community fits' most-folded windings sit at a physical radius (113–211 vox) that no public curated segment of any scroll reaches today (PHerc0139's innermost, w023, is at 1193–1223 vox) — so a companion self-intersection check between neighbouring windings is documented but held back, honestly, for lack of a comparable control (see Limitations) |

## Why these checks

Both of the 2026 open problems on topology ("devise methods to verify and repair the topology of
segmentation meshes") and on ink detection generalization ask for exactly this kind of check: a
way to tell a good mesh or a good label from a bad one using only what is already public, without
requiring a new model or GPU time. The July 2026 progress prizes rewarded small tools in the same
spirit (a zarr v3 reader, an ink-validation harness, a metadata checker) — these two fit the same
category, applied to different data than any of those.

## Method

### `mesh_topology_check.py`

A `tifxyz` winding (`wNNN`) is a sheet parametrized on a grid: row ≈ the scroll's Z axis, column ≈
position across the sheet at that Z. `row_fold_report` walks each row and, at every interior vertex,
takes the cosine of the angle between the incoming and outgoing 3D step. A smooth sheet keeps that
cosine near +1; a vertex where the sheet folds back on itself drops toward −1. Vertices below
`--min-cos` (default −0.3, ≈107°) are flagged; steps shorter than `--min-step` (0.5 vox) are
excluded as near-duplicate-point noise, not real geometry.

Calibrated with 8 synthetic cases before touching real data (a smooth quarter-circle and a straight
line: 0 folds; a deliberately inserted perfect fold: caught at 100%, cosine < −0.99) — this caught
and fixed a real sign bug (the first version flagged smooth curves and missed the inserted fold).
Then run, unchanged, against three real meshes: the curated control and the two community fits
above. The result is not noise: it is spatially concentrated (armando-gaona's folds concentrate on
the innermost windings, dropping to <1% by w015) and agrees with an independent signal the fit's own
author published (rodriguescarson's `satisfaction_metrics_fitted.json` reports only 11.1% of tracks
"satisfied" for the same fit that this tool flags as folded on every winding).

### `ink_label_audit.py`

Curated segments can ship an `inklabels.zarr` (per-pixel ink/no-ink) alongside a `supervision.zarr`
(the area a human annotator actually reviewed). Ink flagged outside the supervised area is only
meaningful if it is genuinely far from the reviewed boundary — a few pixels of brush overshoot at
the edge is normal and not a finding. `spillover_locality` measures that distance with a bounded
distance transform (cropped to the supervision bounding box, down-sampled before the transform for
the largest arrays so the check stays under a minute instead of exhausting memory on a
33280×30000 px label array) and only flags segments with ink more than a fixed **physical** margin
(560 µm, converted to pixels per resolution) and more than 500 px away. `cross_resolution_agreement`
separately compares the two independent publications of the same physical segment, where both
exist, by the fraction of ink inside the supervised area (not pixel-to-pixel, which would require
aligning two independently flattened meshes).

Two real method bugs were found and fixed against synthetic tests before any real-data result was
reported: (a) a first version used a fixed **pixel** margin, which gave different verdicts for the
*same* segment at its two different resolutions (60 px is ~3.9× more tolerance at 9.362 µm than at
2.399 µm) — fixed by fixing the margin in microns and converting per resolution; (b) a first memory
optimization for the distance transform still reconstructed a full-resolution array (`np.kron`) and
cost exactly what it was meant to save — fixed by indexing the already-downsampled distance array
directly at the (rescaled) query coordinates. With both fixes, the six segments published at two
resolutions agree with themselves, and the full census (8 segments, 12 segment×resolution
combinations) runs in about 65 seconds, CPU only.

## Reproduce

```
pip install -r requirements.txt
python -m pytest tools/tests/          # 46 synthetic tests, no network, no real data needed
```

The real-data numbers in the tables above were produced by downloading each mesh or segment's
`x.tif`/`y.tif`/`z.tif` (or `.zarr` arrays) from the official S3 data bucket
(`vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com`) and running the corresponding function
directly; no notebook or GPU is involved. `tools/mesh_topology_check.py` and
`tools/ink_label_audit.py` both separate their pure, tested functions from the handful of
network-fetch helpers, so the synthetic test suite needs no network access at all.

## Limitations, stated honestly

- `mesh_topology_check.py`'s row-fold check only looks for folds **within** one winding's own
  sheet. A cheaper but real question — do two **neighbouring** windings cross each other in 3D —
  needs a physically comparable control at the same radius as the windings under suspicion.
  `umbilicus_radius.py` shows that, with today's public data, no such control exists (the one
  scroll with public curated segments, PHerc0139, has no segment closer than 1193 vox to its axis,
  while the community windings in question sit at 113–211 vox) — so that extension is documented,
  not shipped as a verdict, until the challenge publishes a more internal curated segment for any
  scroll.
- Very narrow windings (few valid columns, tens of measured vertices) give noisy fold fractions;
  summarize by weighting on `interior_vertices_measured`, not a plain average across windings.
- `--min-cos` is calibrated on synthetic geometry, not on a real mesh with a known-true fold
  (no such ground truth is public) — the evidence that it measures something real is indirect but
  convergent across three independent sources: zero on the known-good control, spatially
  concentrated on the community fit's most-compressed region, and correlated with that fit's own
  independently published quality metric.
- The ink-label audit's one real finding (segment w035) does not invalidate any earlier use of that
  segment that already restricted itself to a crop inside the supervised area; it only means a
  *future* use of w035 as ink ground truth must filter explicitly by `supervision==1`, or it will
  silently inherit roughly 62,000 unreviewed pixels.

## Authorship and AI disclosure

Written and run by Claude (an AI model by Anthropic) working for Marco Zarate Castro; every number
above comes from code in this repo run on public data. Companion submissions:
<https://github.com/claudepro1515/first-letters-scan-atlas> (September) and
<https://github.com/claudepro1515/first-letters-fit-audit> (spiral-fit audit).

## License

MIT (see `LICENSE`).
