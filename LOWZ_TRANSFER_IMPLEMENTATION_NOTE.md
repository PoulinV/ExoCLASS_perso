# Low-redshift transfer implementation

Status: 2026-09-07

## Bottom line

The ExoCLASS consumer now implements the complete transfer-function block
matrix supplied in August 2026. It uses the new high-redshift tables, the
summed low-redshift tables, and the high-injection/low-deposition bridge.

No extra factor of eight and no additional `H(z_inj)/H(z_dep)` factor is
applied in ExoCLASS. The email accompanying the files states that the
`*_summed_*` products already contain the timestep conversion. Those factors
were relevant to earlier unsummed diagnostic files, not to the active inputs.

The causal heating matrix is

```text
                              injection
                         low z          high z
deposition low z        low-low         bridge
           high z       0 (acausal)     high-high
```

The residual non-y distortion is likewise the sum of the low-injection table,
the high-injection bridge, and the high-redshift table.

## Active files

- High-redshift deposition channels:
  `DarkAgesModule/transfer_functions/original/tf_final_summed_Ch1.dat`
  through `Ch5.dat`, replaced from `original/new_tfs/`.
- Low-low heat:
  `tf_heat_eps-7_mass_scan_summed_lowz.dat` (`12 x 20 x 12`).
- High-to-low heat bridge:
  `tf_heat_eps-7_mass_scan_summed_highinj_lowdep.dat`
  (`13 x 20 x 52`).
- Low-injection non-y:
  `tf_nony_eps-7_mass_scan_summed_lowz.dat` (`12 x 500 x 20`).
- High-injection non-y bridge:
  `tf_nony_eps-7_mass_scan_summed_highinj_lowdep.dat`
  (`52 x 500 x 20`).

The transfer loaders place entries by their explicit coordinates. This is
essential because the bridge files serialize the high-redshift injection axis
in decreasing order.

These large tables are intentionally distributed outside git. Their expected
locations and SHA-256 digests are recorded in `LOWZ_TRANSFER_TABLES.sha256`.

## User-facing modes

- `legacy`: high-redshift calculation only.
- `extend`: production extension using the new low-z files below the old
  coverage, the high-z result in the overlap, and both bridges. Despite its
  name, this does **not** use the superseded sparse low-z tables.
- `extend-new`: same complete matrix, but uses the low-z table throughout the
  overlap. This is mainly an overlap diagnostic because the low-z calculation
  is intended for the post-reionization range.
- `low-only`: diagnostic isolation of the low-injection/low-deposition block.
  It excludes the high-z result and both bridges and gives the high-z heating
  tail an explicit zero. It must not be used as the production combined model.

For the production extension, use

```ini
lowz_transfer_mode = extend
```

## Convolution implemented

The low-low and bridge heating blocks are convolved independently on their own
injection grids. The bridge normalization is evaluated on its 13-point
deposition grid, not on its 52-point injection grid. The two low-deposition
answers are then aligned by coordinate and added before joining the high-z
channels.

The joined CLASS table carries an explicit low/high handoff redshift in its
header. CLASS interpolates the low-z rows and the gap to the first high-z node
linearly, then uses a separate natural spline for the high-z block. This
hybrid avoids cubic ringing across the large handoff jump while leaving
`legacy` interpolation unchanged. It does not smooth or renormalize the
supplied transfer-function values themselves.

For non-y distortions, each table is convolved over its injection-redshift
grid, with the logarithmic-redshift integration weight. The input Dirac
spectrum contains two particles, while these transfer tables are normalized
per pair event, so the residual calculation applies one factor of `1/2`. This
factor is covered on and off the tabulated energy grid by regression tests.

The y distortion is not read from the non-y tables. CLASS evolves the gas
temperature using the heating history returned by the same selected heating
blocks and computes y from that thermal history.

For redshift-resolved fractions, `classy` exposes CLASS's exact-y integrand,
y branching ratio, and integration weights. The scan subtracts the matched
baseline history point by point and integrates directly over the requested
redshift interval. This avoids cancellation between completed y integrals.

## Check against draft Figures 6--9

`scripts/reproduce_lowz_draft_figures.py` is the reproducible audit.

| draft figure | result |
|---|---|
| 6 | Reproduced directly from the active low-z and high-z non-y tables at the four plotted energies. |
| 7 | Reproduced directly by summing each heating table over deposition cells. A regression test fixes the four curves numerically. |
| 8 | Not reconstructible from the collapsed consumer tables. It requires the producer's electron, photon, and heat histories at every timestep. |
| 9 | Reproduced as a low-z `low-only` CLASS run plus the archived direct-DarkHistory high-z spectra and matched baseline. |

For Figure 9, subtracting a full `extend` run from a `legacy` run is wrong:
the high-z heating changes the gas temperature carried into the low-z era.
`low-only` resets that high-z source and therefore isolates the plotted
low-injection contribution without thermal-memory contamination. The
production `extend` calculation still includes the bridge.

The reproduced low-z peak amplitudes divided by peaks digitized from the
standalone draft PDF are

```text
1.032 MeV  0.890
1.7   MeV  0.975
7.2   MeV  1.100
450   MeV  1.130
32    GeV  1.108
2.3   TeV  1.275
```

Thus all six low-z curves have the draft shape and agree in peak amplitude to
within 2.5--27.5%. The first point must use the exact table threshold
`m_chi = 2*(m_e+5 keV) = 1.0319978922 MeV`; using `1.03 MeV` puts the electron
below the tabulated energy range and creates a spurious 35% deficit.

Five archived direct-DarkHistory high-z curves match the draft peaks to
0.1--4.7%. The first archived point is `1.05 MeV`, not the draft's threshold
mass, and is a factor 1.91 higher; an exact high-z reproduction requires the
original threshold-mass DarkHistory spectrum or a rerun at that mass.

## Energy-budget check

The Figure 7 heating discontinuity is present in the supplied tables. One
active high-z curve reaches `1.12363`; ExoCLASS does not create that value.
The hybrid interpolation described above prevents this discontinuity from
producing negative low-z heating through spline overshoot, but it deliberately
does not decide whether the jump amplitude is physically correct.

The apparent factor of two in the first Figure-8 point is not present in the
active transfer tables. That point is an initial-state bookkeeping overlap in
which the plotted electron and photon inventories are both near one; the next
point is `1.054`. Independently, the active summed low-z heat table has a
maximum heat-to-pair-kinetic-budget ratio of `1.0383`, not two. No empirical
factor is applied downstream.

## Reproduction command

```bash
MPLCONFIGDIR=/tmp/mpl-lowz-draft PYTHONPATH=DarkAgesModule \
  /usr/local/anaconda3/bin/python scripts/reproduce_lowz_draft_figures.py \
  --run-class --jobs 2 \
  --darkhistory-high-dir output/sectionV \
  --darkhistory-baseline \
  /Users/vpoulin/Dropbox/Labo/papers/CLASS_DarkHistory/sectionV_data/scan/no_puchwein_y_residual/output/baseline_slow_legacy__sd_distortions.dat
```

Products are written to `output/lowz_draft_figures/`; PDFs are also copied to
`output/pdf/`. `reproduction_metadata.json` records table checksums, numerical
diagnostics, and the exact Figure 9 source choice.

## What collaborators still need to provide or confirm

1. Archive the per-timestep arrays used for Figure 8: `rs`, `elec_eng`,
   `elec_spec`, `phot_spec`, and deposited heat, for both photoionization
   choices.
2. Provide or regenerate the high-z direct-DarkHistory spectrum at
   `1.0319978922 MeV` if exact reproduction of the first Figure 9 curve is
   required.
