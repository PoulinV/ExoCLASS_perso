# Low-z transfer functions: agent handoff

Status: 2026-09-07

Repository:
`/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/ExoCLASS_perso`

## Current conclusion

The new summed transfer functions are implemented, including the formerly
missing high-injection/low-deposition block. The consumer must use the four
summed low-z products exactly as delivered. Do not apply the Hubble-ratio or
factor-of-eight correction discussed for older unsummed diagnostic files.

The implementation reproduces draft Figures 6 and 7 directly, and it contains
the range-isolation machinery needed for Figure 9. Figure 8 is producer-side
and cannot be recovered from collapsed transfer tables.

Read `LOWZ_TRANSFER_IMPLEMENTATION_NOTE.md` first for the concise scientific
status and the current reproduction command. This file records the code path
for another agent.

## Inputs and their meaning

The collaborator's August 2026 email identifies these as the important
consumer products:

```text
tf_heat_eps-7_mass_scan_summed_lowz.dat
tf_nony_eps-7_mass_scan_summed_lowz.dat
tf_heat_eps-7_mass_scan_summed_highinj_lowdep.dat
tf_nony_eps-7_mass_scan_summed_highinj_lowdep.dat
```

The first two describe low-redshift injections. The latter two carry particles
injected on the high-z grid that deposit or survive into the low-z regime.
The heating files are already timestep-converted and summed onto cells with
`Delta ln(1+z) ~= 0.128`. The residual files named `summed` are numerically
identical to their unsuffixed companions because they have no deposition axis
to coarsen.

The new high-z deposition tables are under
`DarkAgesModule/transfer_functions/original/new_tfs/`; their active copies are
`tf_final_summed_Ch1.dat` through `Ch5.dat`.

The complete causal heating matrix is

```text
                              injection
                         low z          high z
deposition low z        low-low         bridge
           high z       zero            high-high
```

The zero block is causal: an injection at a later time cannot deposit at an
earlier time.

## Code path

### Loading

`DarkAgesModule/DarkAges/__init__.py`

- `_deposition_transfer_paths()` loads the active high-z `Ch1`--`Ch5` files.
- `get_lowz_heat_transfer_functions()` loads low-low heat and the heat bridge.
- `get_lowz_spectral_distortions_transfer_functions()` loads low-injection
  non-y and the non-y bridge.
- `_spectral_distortions_paths()` loads the active high-z non-y table. The
  `DARKAGES_SD_TRANSFER_FILE` environment variable can override it for audits.

`DarkAgesModule/DarkAges/transfer.py` and
`DarkAgesModule/DarkAges/spectral_distortions.py` reshape tables by coordinate,
not row order. This is required because the bridge injection rows are written
in decreasing redshift.

### Heating

`DarkAgesModule/DarkAges/recipes.py::loading_from_specfiles()` performs the calculation.

1. Build the high-z injection model on the 52-point high grid.
2. Build the low-z injection model on the 12-point low grid.
3. Convolve the low-low heat block with the selected low injections.
4. Convolve the bridge with the selected high injections. Its injection model
   has 52 entries, but its deposition normalization is separately evaluated on
   the 13-point bridge deposition grid.
5. Align bridge and low-low deposition rows by coordinate and add them.
6. Join this low-z heat result to all five high-z deposition channels.

The joined table includes a `lowz_interpolation_handoff_z` comment. The CLASS
reader uses linear interpolation for every row below that boundary, including
the gap between the two blocks, and constructs an independent natural spline
from the first high-z row upward. Without this split, the natural spline rings
across large jumps and can make a positive `f_heat` table negative near the
handoff. `legacy` tables contain no marker and retain their original spline.

For decay, the legacy convolution is schematically

```text
f_heat(r_d) = sum_i T_heat(d,i) [H(r_d)/H(r_i)] S_i / N_d,
r = 1+z.
```

The stored summed products are already `T_heat`. An additional
`H(r_i)/H(r_d)` conversion here would undo their producer-side timestep
conversion.

Only heat is supplied below the handoff. H ionization, He ionization,
Lyman-alpha, and continuum deposition are therefore set to zero on the low-z
rows, while all five channels remain present on the high-z rows.

### Residual non-y distortion

`DarkAgesModule/DarkAges/recipes.py::compute_distortions()` calculates the
high, bridge, and low residuals separately and adds them after interpolation in
log frequency. `spectral_distortion_today()` performs the energy and
injection-redshift integrals.

The Dirac injection spectrum contains two daughters. The transfer spectrum is
normalized per pair event, so `spectral_distortion_today()` multiplies the
particle-count result by `1/2`. Tests cover a Dirac line on and off the energy
grid and continuous spectra in both energy-integration schemes.

### y distortion

The low-z non-y table explicitly excludes y. CLASS receives the heating
history through the normal deposition-channel interface, evolves its own gas
temperature, and computes y from that temperature history (`exact_y = yes` in
the reproduction run). Therefore y and non-y have independent transfer paths
and are added only at the final spectrum level.

`python/classy.pyx::spectral_distortion_y_history()` exposes the redshift,
`4 dy/dz`, y branching ratio, and CLASS trapezoidal weights. The low-z scan
subtracts the matched baseline integrand point by point and integrates it
directly. It does not estimate the low-z signal by subtracting two accumulated
y amplitudes with different `sd_z_min` values.

### Decay survival

`DarkAgesModule/DarkAges/common.py::time_at_z()` now integrates the same flat
radiation + matter + Lambda background used by `H()`. This matters below
redshift of order unity. The decay survival factor appears once in the
injected spectrum; regression tests prevent accidental double depletion.

## Modes

The CLASS parameter is `lowz_transfer_mode`:

- `legacy`: high-z tables only.
- `extend`: complete production matrix, keeping the high-z calculation in the
  overlap. This is the recommended mode for the supplied post-reionization
  tables. It uses the new summed low-z files, not the historical sparse files.
- `extend-new`: complete matrix, preferring the low-z block in the overlap.
- `low-only`: Figure-9 diagnostic containing only low-low injections and
  deposition. It excludes the high result and bridge and emits an explicit
  zero high-z tail.
- `low-below-four`: source diagnostic retaining only injection nodes with
  `1+z<4`; it also excludes the bridge.

The mode is parsed in `source/input.c`, forwarded by
`DarkAgesModule/bin/DarkAges`, normalized in
`DarkAgesModule/DarkAges/lowz.py`, and consumed in `recipes.py`.

Do not construct the Figure 9 left panel as `extend - legacy`. A high-z heat
source changes the matter temperature carried into later times, so that
subtraction contains thermal memory. `low-only` supplies a clean source-level
split. Conversely, do not use `low-only` for a physical full-history run,
because it intentionally omits the bridge.

## Draft Figures 6--9

The audit script is `scripts/reproduce_lowz_draft_figures.py`.

### Figure 6

The script selects the nearest tabulated electron energies
`log10(E/eV) = 6.02, 7.88, 10.21, 12.07` and plots the low and high non-y
tables near `1+z_inj = 4.5`, preserving the sign convention with dashed
negative segments.

### Figure 7

For each selected energy it sums every deposition cell at fixed injection
redshift. It plots high-high, low-low, and high-to-low bridge curves. The
values at the ends of the tables are fixed by regression tests for both the
unsummed Figure-7 inputs and the active summed products. Their maxima are
`1.167575` and `1.12363`, respectively.

This plot must not receive another `/8` correction: the active consumer tables
are already the summed products. The earlier apparent factor-of-two inventory
was traced to the first Figure-8 state, where both the initial electron and its
new photon content are present in the plotted bookkeeping. The next state is
near unity, and the active summed heat table reaches only `1.0383` of the
tracked pair kinetic-energy budget. There is no evidence for a global factor
of two to correct downstream.

### Figure 8

This is not a transfer-table convolution. It needs the particle content after
each producer timestep for an injection at `1+z_inj = 4.65`, with and without
photoionization. Required arrays are at least:

```text
rs, elec_eng, elec_spec, phot_spec, deposited heat (or native heat rate + dt)
```

The consumer tables retain only cumulative heat cells and present-day non-y
spectra, so Figure 8 cannot be reconstructed losslessly. The script writes
`output/lowz_draft_figures/draft_figure8_status.md` with this limitation.

### Figure 9

The two panels have different provenance:

- Left: low-z transfer calculation in `low-only`, with the Puchwein
  post-reionization history, matched negligible-injection baseline, and
  `exact_y = yes`.
- Right: archived full-DarkHistory spectra for the standard range with their
  matched no-reionization baseline. This is what the Paper-I draft used. A
  no-reionization fast transfer run is also generated, but only as a diagnostic
  source comparison.

The benchmark masses and lifetimes are encoded in the script. The lowest mass
is exactly `2*(m_e+5 keV) = 1.0319978922 MeV`, although the draft legend rounds
it to `1.0e6 eV`. Moving it to `1.03 MeV` places the electron below the table
and lowers the computed peak by about 35%.

The script writes:

- `draft_figure9_transfer_reproduction.{png,pdf}`;
- `draft_figure9_y_nony_components.{png,pdf}`;
- `draft_figure9_pdf_peak_audit.{png,pdf,csv}`;
- `draft_figure9_high_source_check.{png,pdf,csv}`;
- `draft_figure9_metrics.csv`;
- `reproduction_metadata.json`.

Current peak ratios (reproduced/draft) for the low panel are
`0.890, 0.975, 1.100, 1.130, 1.108, 1.275`. Five of the six direct high-z
archive curves match at `0.999--1.047`; the first is `1.91` because that
archive uses 1.05 MeV. The exact first high-z source used by the PDF is not in
the current archive.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/mpl-lowz-draft PYTHONPATH=DarkAgesModule \
  /usr/local/anaconda3/bin/python scripts/reproduce_lowz_draft_figures.py \
  --run-class --jobs 2 \
  --darkhistory-high-dir output/sectionV \
  --darkhistory-baseline \
  /Users/vpoulin/Dropbox/Labo/papers/CLASS_DarkHistory/sectionV_data/scan/no_puchwein_y_residual/output/baseline_slow_legacy__sd_distortions.dat
```

Use `--tables-only` for Figures 6 and 7. If the direct-DarkHistory options are
omitted, the script deliberately labels the Figure 9 right panel as a high-z
transfer diagnostic rather than claiming it is the draft source.

## Verification

Run:

```bash
PYTHONPATH=DarkAgesModule /usr/local/anaconda3/bin/python \
  -m unittest discover -s DarkAgesModule/tests -p 'test_*.py' -v
make class
```

At this handoff, all 33 Python tests pass, a forced C build succeeds, and a
compiled-classy smoke point reproduces CLASS's accumulated y amplitude from
the exposed exact-y integrand to machine precision.

The tests cover table loading, coordinate order, grid compatibility, bridge
convolution, overlap modes, unsupported boundary rows, the Figure 7 values,
pair normalization, decay survival, and the Lambda-inclusive cosmic age.

## Remaining external questions

1. Can the producer author provide the Figure 8 timestep histories?
2. Can the exact direct-DarkHistory high-z spectrum at the 5-keV threshold
   mass be provided for the first Figure 9 curve?
3. If future paper figures define the low/high split by deposition redshift
   rather than injection redshift, include the bridge on the low-deposition
   side. The current Figure 9 reproduction follows the existing draft's
   source split and therefore excludes it.

## Repository caution

The worktree contains many unrelated modified and untracked research files.
Do not reset, clean, or bulk-stage the repository. Transfer `.dat` files are
ignored by `DarkAgesModule/.gitignore`, so collaborators need the data files in
addition to any source patch. Validate an out-of-band installation with
`shasum -a 256 -c LOWZ_TRANSFER_TABLES.sha256`.
