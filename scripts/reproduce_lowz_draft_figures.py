#!/usr/bin/env python3
"""Reproduce and audit the low-redshift draft Figures 6, 7, and 9.

Figure 6 is a direct view of the active transfer tables.  Figure 7 instead
uses the unsummed producer tables: a vector-level comparison with the draft
shows that those products, rather than the summed tables used by production
DarkAges, generated the plotted curves.  This plotting-only choice does not
change the production transfer-function loaders.

For Figure 9, the low-redshift piece is a baseline-subtracted ``low-only`` CLASS/DarkAges
run, which retains only low-redshift injections and resets the thermal response
at the handoff.  The draft's right panel was produced by full DarkHistory, not
by a transfer table.  Supplying ``--darkhistory-high-dir`` and
``--darkhistory-baseline`` therefore reconstructs that panel from matched
slow-interface output.  Without those options the script plots a no-
reionization high-z transfer calculation and labels it as a diagnostic.

The production ``extend`` mode still includes the required high-injection/low-
deposition bridges; they are intentionally absent from this injection-range
diagnostic.

Figure 8 cannot be reconstructed from the collapsed transfer tables because it
needs the electron and photon spectra at every producer timestep.  The script
writes a short status file naming those missing producer-level quantities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "lowz_draft_figures"
CLASS_OUT = OUT / "class"
INI_DIR = OUT / "ini"
PDF_OUT = ROOT / "output" / "pdf"
TRANSFER_ROOT = ROOT / "DarkAgesModule" / "transfer_functions" / "original"
FIGURE7_DRAFT_TABLE_PATHS = {
    "high_heat": (
        TRANSFER_ROOT
        / "new_tfs"
        / "tf_test_heat_eps-7_mass_scan_negdist.dat"
    ),
    "low_heat": TRANSFER_ROOT / "low-z" / "tf_heat_eps-7_mass_scan_lowz.dat",
    "bridge_heat": (
        TRANSFER_ROOT
        / "low-z"
        / "tf_heat_eps-7_mass_scan_highinj_lowdep.dat"
    ),
}

H0 = 67.66
OMEGA_M = 0.31104850446994281
OMEGA_R = 7.9111186056792255e-05
TARGET_LOG10_ENERGIES = np.asarray([6.02, 7.88, 10.21, 12.07])
PLOT_FREQUENCY_MIN_GHZ = 5.0
PLOT_FREQUENCY_MAX_GHZ = 5.0e5

# Peaks digitized from the vector paths in the standalone Figure 9 PDF
# (compare_zrange_distortions.pdf, SHA256
# c6b9c10dbb9383580d8ebc4f3f6f6094c94fe68789d058c52439b9cfdedf6677).
# They are a compact regression diagnostic, not a replacement for comparing
# the full signed spectra.  The first high-z archive point is at 1.05 MeV,
# whereas the low-z curve uses the exact parent mass corresponding to the
# table's 5-keV electron threshold.
DRAFT_FIGURE9_REFERENCE_PEAKS = {
    "m1p032MeV": {"low": 1.0407054e-25, "high": 4.3605906e-26},
    "m1p7MeV": {"low": 7.1177129e-24, "high": 5.3040906e-24},
    "m7p2MeV": {"low": 1.1506677e-24, "high": 1.5553434e-24},
    "m450MeV": {"low": 1.0473732e-26, "high": 5.6566505e-25},
    "m32GeV": {"low": 3.3796464e-27, "high": 1.7971386e-26},
    "m2p3TeV": {"low": 1.8643940e-28, "high": 1.0348729e-27},
}


@dataclass(frozen=True)
class Benchmark:
    tag: str
    mass_gev: float
    lifetime_s: float
    label_mass_ev: float
    scan_index: int
    fraction: float = 1.0


# The first point is rounded to 1.0e6 eV in the draft legend.  Its exact mass
# is 2*(m_e+5 keV), placing it on (rather than just below) the lowest electron
# kinetic-energy node.  The archived high-z scan used 1.05 MeV for this point.
BENCHMARKS = (
    Benchmark("m1p032MeV", 1.0319978922e-3, 2.7e24, 1.0e6, 0),
    Benchmark("m1p7MeV", 1.7e-3, 2.0e24, 1.7e6, 2),
    Benchmark("m7p2MeV", 7.2e-3, 5.9e24, 7.2e6, 3),
    Benchmark("m450MeV", 4.5e-1, 1.2e25, 4.5e8, 5),
    Benchmark("m32GeV", 3.2e1, 2.9e24, 3.2e10, 7),
    Benchmark("m2p3TeV", 2.3e3, 9.4e23, 2.3e12, 9),
)
ZERO_INJECTION = Benchmark(
    "zero", 4.5e-1, 1.2e25, 4.5e8, 5, fraction=1.0e-12
)
MODES = {
    # The draft's direct-DarkHistory high-z panel ends before astrophysical
    # reionization. Keep the optional fast high-z cross-check on the same
    # reio_none background. The post-reionization low-z calculation uses the
    # Puchwein history used by the production low-z interface.
    "high": {
        "lowz_transfer_mode": "legacy",
        "sd_z_min": 3.0,
        "reionization": "none",
    },
    "low": {
        "lowz_transfer_mode": "low-only",
        "sd_z_min": 0.01,
        "reionization": "puchwein",
    },
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def nearest_indices(grid: np.ndarray, requested: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    return np.asarray([int(np.argmin(np.abs(grid - value))) for value in requested])


def plot_signed(axis, x, y, *, color, linewidth=1.8, label=None) -> None:
    """Plot signed data by using a dashed line for its negative segments."""

    x = np.asarray(x)
    y = np.asarray(y)
    positive = np.where(y >= 0.0, np.abs(y), np.nan)
    negative = np.where(y < 0.0, np.abs(y), np.nan)
    axis.loglog(x, positive, color=color, linewidth=linewidth, label=label)
    axis.loglog(x, negative, color=color, linewidth=linewidth, linestyle="--")


def save_figure(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PDF_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_tables():
    sys.path.insert(0, str(ROOT / "DarkAgesModule"))
    import DarkAges
    from DarkAges.lowz import validate_lowz_transfer_blocks

    DarkAges.set_background(H0, OMEGA_M, OMEGA_R)
    high_heat = DarkAges.transfer_functions[DarkAges.channel_dict["Heat"]]
    low_heat, bridge_heat = DarkAges.get_lowz_heat_transfer_functions()
    high_nony = DarkAges.spectral_distortions_functions
    low_nony, bridge_nony = (
        DarkAges.get_lowz_spectral_distortions_transfer_functions()
    )
    validate_lowz_transfer_blocks(low_heat, bridge_heat, high_heat)
    if not np.allclose(bridge_nony.z_injected, high_nony.z_injected):
        raise ValueError("The residual bridge does not use the active high-z grid.")
    if not np.allclose(low_nony.z_injected, low_heat.z_injected):
        raise ValueError("The low-z heat and residual injection grids differ.")
    return high_heat, low_heat, bridge_heat, high_nony, low_nony, bridge_nony


def load_figure7_draft_tables():
    """Load the unsummed producer products used to draw draft Figure 7."""

    sys.path.insert(0, str(ROOT / "DarkAgesModule"))
    from DarkAges.transfer import transfer

    return tuple(
        transfer(str(FIGURE7_DRAFT_TABLE_PATHS[name]))
        for name in ("high_heat", "low_heat", "bridge_heat")
    )


def make_figure6(high_nony, low_nony) -> dict:
    high_loge = np.log10(high_nony.E_injected)
    low_loge = np.log10(low_nony.E_injected)
    high_energy_indices = nearest_indices(high_loge, TARGET_LOG10_ENERGIES)
    low_energy_indices = nearest_indices(low_loge, TARGET_LOG10_ENERGIES)
    high_z_index = int(np.argmin(np.abs(high_nony.z_injected - 4.5)))
    low_z_index = int(np.argmin(np.abs(low_nony.z_injected - 4.5)))
    colors = plt.get_cmap("magma")(
        np.asarray([0.02, 0.28, 0.58, 0.82])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True)
    for energy_number, color in enumerate(colors):
        low_index = low_energy_indices[energy_number]
        high_index = high_energy_indices[energy_number]
        label = rf"$\log_{{10}}(E/{{\rm eV}})={low_loge[low_index]:.2f}$"
        plot_signed(
            axes[0],
            low_nony.frequency,
            low_nony.spectral_distortions_elec[low_z_index, :, low_index],
            color=color,
            label=label,
        )
        plot_signed(
            axes[1],
            high_nony.frequency,
            high_nony.spectral_distortions_elec[high_z_index, :, high_index],
            color=color,
        )

    for axis in axes:
        axis.set_xlim(7.0, PLOT_FREQUENCY_MAX_GHZ)
        axis.set_ylim(1.0e2, 5.0e4)
        axis.set_xlabel(r"Frequency, $\nu$ [GHz]")
        axis.grid(True, which="major", linestyle=":", alpha=0.28)
    axes[0].set_ylabel(
        r"$|E_0\,d\bar N_\gamma/dE_0|$ "
        r"[$10^{10}\,{\rm cm^3\,Jy\,sr^{-1}}$]"
    )
    axes[0].set_title(
        rf"low-$z$ table, $1+z_{{\rm inj}}={low_nony.z_injected[low_z_index]:.4f}$"
    )
    axes[1].set_title(
        rf"DarkHistory table, $1+z_{{\rm inj}}={high_nony.z_injected[high_z_index]:.4f}$"
    )
    axes[0].legend(frameon=False, fontsize=8.6, loc="lower left")
    fig.suptitle("Draft Figure 6 reproduction: electron non-y lookup tables")
    fig.tight_layout()
    save_figure(fig, "draft_figure6_transfer_tables")
    return {
        "low_1pz_injection": float(low_nony.z_injected[low_z_index]),
        "high_1pz_injection": float(high_nony.z_injected[high_z_index]),
        "log10_energy_low": low_loge[low_energy_indices].tolist(),
        "log10_energy_high": high_loge[high_energy_indices].tolist(),
    }


def integrated_heat(table, energy_indices: np.ndarray) -> np.ndarray:
    return np.sum(table.transfer_elec[:, energy_indices, :], axis=0)


def make_figure7(high_heat, low_heat, bridge_heat) -> dict:
    high_indices = nearest_indices(high_heat.log10E, TARGET_LOG10_ENERGIES)
    low_indices = nearest_indices(low_heat.log10E, TARGET_LOG10_ENERGIES)
    bridge_indices = nearest_indices(bridge_heat.log10E, TARGET_LOG10_ENERGIES)
    high_fraction = integrated_heat(high_heat, high_indices)
    low_fraction = integrated_heat(low_heat, low_indices)
    bridge_fraction = integrated_heat(bridge_heat, bridge_indices)
    # These are the four exact magma samples used in the draft figure.
    colors = plt.get_cmap("magma")(np.linspace(0.0, 0.9, 4))

    fig, axis = plt.subplots(figsize=(5.7, 5.4))
    for row, color, energy_index in zip(
        range(len(TARGET_LOG10_ENERGIES)), colors, low_indices
    ):
        axis.loglog(
            high_heat.z_injected,
            high_fraction[row],
            color=color,
            linewidth=1.5,
            label=rf"$\log_{{10}}(E/{{\rm eV}})={low_heat.log10E[energy_index]:.2f}$",
        )
        axis.loglog(
            low_heat.z_injected,
            low_fraction[row],
            color=color,
            linewidth=1.5,
            linestyle="--",
        )
        axis.loglog(
            bridge_heat.z_injected,
            bridge_fraction[row],
            color=color,
            linewidth=1.5,
            linestyle=":",
        )

    axis.set_xlim(0.67, 4.5e3)
    axis.set_ylim(1.0e-10, 2.0)
    axis.set_xlabel(r"$1+z_{\rm inj}$")
    axis.set_ylabel(r"$\Delta E_{\rm heat}/E_{\rm inj}$")
    energy_handles, energy_labels = axis.get_legend_handles_labels()
    axis.legend(
        handles=energy_handles + [
            Line2D([0], [0], color="0.45", linewidth=1.5),
            Line2D(
                [0], [0], color="0.45", linewidth=1.5, linestyle="--"
            ),
            Line2D(
                [0], [0], color="0.45", linewidth=1.5, linestyle=":",
            ),
        ],
        labels=energy_labels + [
            r"high-$z$",
            r"low-$z$",
            r"high-$z$ inj., low-$z$ dep.",
        ],
        ncol=2,
        frameon=True,
        framealpha=0.8,
        edgecolor="0.8",
        fontsize=7.7,
        loc="lower left",
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.tight_layout()
    save_figure(fig, "draft_figure7_integrated_heat")

    diagnostics = []
    for row in range(len(TARGET_LOG10_ENERGIES)):
        diagnostics.append(
            {
                "log10_energy": float(low_heat.log10E[low_indices[row]]),
                "low_at_max_1pz": float(low_fraction[row, -1]),
                "high_at_min_1pz": float(high_fraction[row, 0]),
                "bridge_at_min_high_1pz": float(bridge_fraction[row, 0]),
                "high_max": float(np.max(high_fraction[row])),
            }
        )
    with (OUT / "draft_figure7_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)
    return {
        "table_variant": "unsummed producer products used by the draft plot",
        "source_tables": {
            name: str(path) for name, path in FIGURE7_DRAFT_TABLE_PATHS.items()
        },
        "selected_energy_diagnostics": diagnostics,
    }


def class_prefix(benchmark: Benchmark, mode: str) -> Path:
    return CLASS_OUT / f"{benchmark.tag}_{mode}_"


def class_distortion_path(benchmark: Benchmark, mode: str) -> Path:
    return Path(str(class_prefix(benchmark, mode)) + "_sd_distortions.dat")


def ini_text(benchmark: Benchmark, mode: str) -> str:
    settings = MODES[mode]
    if settings["reionization"] == "puchwein":
        reionization = f"""reio_parametrization = reio_stars
include_reio_stars_cooling_terms = yes
include_reio_stars_helium = yes
reio_stars_photoion_file = {ROOT / 'external/heating/photoion_rates_Puchwein.dat'}
reio_stars_photoheat_file = {ROOT / 'external/heating/photoheat_rates_Puchwein2.dat'}
reio_stars_helium_file = {ROOT / 'external/heating/xHe_DarkHistory.txt'}
"""
    else:
        reionization = "reio_parametrization = reio_none\n"
    return f"""root = {class_prefix(benchmark, mode)}
overwrite_root = yes
output = tCl,Sd
write_distortions = yes
write thermodynamics = no
write parameters = yes

omega_b = 0.02242
omega_cdm = 0.11933
h = 0.6766
N_ur = 2.03351
N_ncdm = 1
m_ncdm = 0.06
k_pivot = 0.05
n_s = 0.9665
ln10^{{10}}A_s = 3.047

{reionization}

f_eff_type = DarkAges
DarkAges_mode = built_in
lowz_transfer_mode = {settings['lowz_transfer_mode']}
DM_decay_mass = {benchmark.mass_gev:.12e}
DM_decay_Gamma = {1.0 / benchmark.lifetime_s:.12e}
DM_decay_fraction = {benchmark.fraction:.12e}
injected_particle_spectra = dirac_electron
injected_particle_branching_ratio = 1

compute_SD_with_DarkAges = yes
include_DH_SMresidual_distortions = no
add_SD_to_CLASS = yes
sd_branching_approx = sharp_sharp
sd_only_exotic = yes
sd_z_min = {settings['sd_z_min']:.12g}
sd_z_size = 1000
sd_x_min = 1.0e-2
sd_x_max = 1.0e4
sd_x_size = 1200
exact_y = yes

input_verbose = 1
thermodynamics_verbose = 1
distortions_verbose = 1
"""


def write_inis() -> list[tuple[Benchmark, str, Path]]:
    OUT.mkdir(parents=True, exist_ok=True)
    CLASS_OUT.mkdir(parents=True, exist_ok=True)
    INI_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for benchmark in BENCHMARKS + (ZERO_INJECTION,):
        for mode in MODES:
            path = INI_DIR / f"{benchmark.tag}_{mode}.ini"
            path.write_text(ini_text(benchmark, mode))
            cases.append((benchmark, mode, path))
    return cases


def run_case(case: tuple[Benchmark, str, Path], force: bool) -> str:
    benchmark, mode, ini_path = case
    output_path = class_distortion_path(benchmark, mode)
    if output_path.is_file() and not force:
        return f"reuse {benchmark.tag} {mode}"
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "2"
    log_path = OUT / f"{benchmark.tag}_{mode}.log"
    with log_path.open("w") as log:
        completed = subprocess.run(
            [str(ROOT / "class"), str(ini_path)],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"CLASS failed for {benchmark.tag} {mode}; inspect {log_path}"
        )
    return f"ran {benchmark.tag} {mode}"


def run_cases(cases, jobs: int, force: bool) -> None:
    if jobs < 1 or jobs > 3:
        raise ValueError("Use between one and three concurrent CLASS jobs.")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_case, case, force) for case in cases]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def load_class_components(benchmark: Benchmark, mode: str) -> dict[str, np.ndarray]:
    table = np.loadtxt(class_distortion_path(benchmark, mode))
    baseline = np.loadtxt(class_distortion_path(ZERO_INJECTION, mode))
    if table.shape != baseline.shape or not np.array_equal(table[:, 1], baseline[:, 1]):
        raise ValueError(f"CLASS frequency grids differ for {benchmark.tag} {mode}.")
    total = table[:, 2] - baseline[:, 2]
    y_distortion = table[:, 4] - baseline[:, 4]
    nony = total - y_distortion
    return {
        "frequency": table[:, 1],
        "total": total / 1.0e26,
        "y": y_distortion / 1.0e26,
        "nony": nony / 1.0e26,
    }


def load_direct_darkhistory_components(
    benchmark: Benchmark, high_dir: Path, baseline_path: Path
) -> dict[str, np.ndarray]:
    """Load an archived full-DarkHistory high-z spectrum and subtract LCDM."""

    path = high_dir / (
        f"scan_decay_ee_{benchmark.scan_index:02d}_dh__sd_distortions.dat"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    table = np.loadtxt(path)
    baseline = np.loadtxt(baseline_path)
    frequency = table[:, 1]
    if np.any(frequency <= 0.0) or np.any(baseline[:, 1] <= 0.0):
        raise ValueError("DarkHistory distortion frequencies must be positive.")

    def subtract_column(index: int) -> np.ndarray:
        baseline_column = np.interp(
            np.log(frequency), np.log(baseline[:, 1]), baseline[:, index]
        )
        return (table[:, index] - baseline_column) / 1.0e26

    total = subtract_column(2)
    y_distortion = subtract_column(4)
    return {
        "frequency": frequency,
        "total": total,
        "y": y_distortion,
        "nony": total - y_distortion,
        "source_path": str(path),
    }


def load_figure9_results(
    darkhistory_high_dir: Path | None = None,
    darkhistory_baseline: Path | None = None,
) -> list[dict]:
    use_direct_darkhistory = (
        darkhistory_high_dir is not None and darkhistory_baseline is not None
    )
    results = []
    for benchmark in BENCHMARKS:
        high_fast = load_class_components(benchmark, "high")
        high = (
            load_direct_darkhistory_components(
                benchmark, darkhistory_high_dir, darkhistory_baseline
            )
            if use_direct_darkhistory
            else high_fast
        )
        low = load_class_components(benchmark, "low")
        if not np.array_equal(high["frequency"], low["frequency"]):
            # The direct DarkHistory archive uses an independent frequency
            # grid. The two figure panels do not need a common sampling.
            if not use_direct_darkhistory:
                raise ValueError(
                    f"High/low frequency grids differ for {benchmark.tag}."
                )
        results.append(
            {
                "benchmark": benchmark,
                "high": high,
                "high_fast": high_fast,
                "low": low,
            }
        )
    return results


def figure9_label(benchmark: Benchmark) -> str:
    return (
        rf"${benchmark.label_mass_ev:.1E},\ "
        rf"{benchmark.lifetime_s:.1E}$"
    )


def make_figure9(results: list[dict], high_source: str) -> dict:
    colors = plt.get_cmap("plasma")(
        np.linspace(0.08, 0.82, len(results))
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.6), sharex=True, sharey=True)
    metrics = []
    for result, color in zip(results, colors):
        benchmark = result["benchmark"]
        low_frequency = result["low"]["frequency"]
        high_frequency = result["high"]["frequency"]
        plot_signed(
            axes[0], low_frequency, result["low"]["total"], color=color,
            linewidth=2.0, label=figure9_label(benchmark),
        )
        plot_signed(
            axes[1], high_frequency, result["high"]["total"], color=color,
            linewidth=2.0,
        )

        def band_and_log_frequency(frequency):
            band = (
                (frequency >= PLOT_FREQUENCY_MIN_GHZ)
                & (frequency <= PLOT_FREQUENCY_MAX_GHZ)
            )
            return band, np.log(frequency[band])

        low_band, low_log_frequency = band_and_log_frequency(low_frequency)
        high_band, high_log_frequency = band_and_log_frequency(high_frequency)

        def l1(values, band, log_frequency):
            return float(np.trapz(np.abs(values[band]), log_frequency))

        low_l1 = l1(result["low"]["total"], low_band, low_log_frequency)
        high_l1 = l1(result["high"]["total"], high_band, high_log_frequency)
        metrics.append(
            {
                "tag": benchmark.tag,
                "mass_gev": benchmark.mass_gev,
                "lifetime_s": benchmark.lifetime_s,
                "high_source": high_source,
                "low_peak_si": float(
                    np.max(np.abs(result["low"]["total"][low_band]))
                ),
                "high_peak_si": float(
                    np.max(np.abs(result["high"]["total"][high_band]))
                ),
                "low_over_high_l1": low_l1 / high_l1,
                "low_y_fraction_l1": l1(
                    result["low"]["y"], low_band, low_log_frequency
                ) / low_l1,
                "low_nony_fraction_l1": l1(
                    result["low"]["nony"], low_band, low_log_frequency
                ) / low_l1,
            }
        )

    for axis, title in zip(
        axes, (r"$1+z<4$", r"$4\leq 1+z<3000$")
    ):
        axis.set_xlim(PLOT_FREQUENCY_MIN_GHZ, PLOT_FREQUENCY_MAX_GHZ)
        axis.set_ylim(1.0e-31, 1.0e-23)
        axis.set_xlabel(r"Frequency, $\nu$ [GHz]")
        axis.text(
            0.96, 0.94, title, transform=axis.transAxes,
            horizontalalignment="right", verticalalignment="top", fontsize=13,
        )
        axis.grid(True, which="major", linestyle=":", alpha=0.22)
    axes[0].set_ylabel(
        r"$|I_\nu|$ [$\mathrm{J\,s^{-1}\,m^{-2}\,Hz^{-1}\,sr^{-1}}$]"
    )
    axes[0].legend(
        title=r"$m_{\rm DM}$ [eV], $\tau$ [s]",
        frameon=False,
        fontsize=8.2,
        title_fontsize=9.0,
        loc="lower left",
        ncol=2,
    )
    high_label = (
        "direct DarkHistory"
        if high_source == "direct-darkhistory"
        else "high-z transfer diagnostic"
    )
    fig.suptitle("Draft Figure 9 reproduction: low-z transfer + " + high_label)
    fig.tight_layout()
    save_figure(fig, "draft_figure9_transfer_reproduction")

    with (OUT / "draft_figure9_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    return {"benchmarks": metrics}


def make_figure9_reference_audit(figure9_metadata: dict) -> dict:
    """Compare reproduced peaks with the paths in the standalone draft PDF."""

    rows = []
    for metric in figure9_metadata["benchmarks"]:
        tag = metric["tag"]
        reference = DRAFT_FIGURE9_REFERENCE_PEAKS[tag]
        rows.append(
            {
                "tag": tag,
                "draft_low_peak_si": reference["low"],
                "reproduced_low_peak_si": metric["low_peak_si"],
                "low_ratio_reproduced_over_draft": (
                    metric["low_peak_si"] / reference["low"]
                ),
                "draft_high_peak_si": reference["high"],
                "reproduced_high_peak_si": metric["high_peak_si"],
                "high_ratio_reproduced_over_draft": (
                    metric["high_peak_si"] / reference["high"]
                ),
            }
        )

    with (OUT / "draft_figure9_pdf_peak_audit.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["tag"] for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.axhspan(0.8, 1.2, color="0.88", zorder=0, label=r"$\pm20\%$")
    axis.axhline(1.0, color="0.25", linewidth=1.2)
    axis.plot(
        x,
        [row["low_ratio_reproduced_over_draft"] for row in rows],
        marker="o",
        linewidth=1.8,
        color="#6a00a8",
        label=r"low-$z$ transfer",
    )
    axis.plot(
        x,
        [row["high_ratio_reproduced_over_draft"] for row in rows],
        marker="s",
        linewidth=1.8,
        color="#f57c00",
        label=r"high-$z$ source",
    )
    axis.set_yscale("log")
    axis.set_ylim(0.45, 2.2)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, horizontalalignment="right")
    axis.set_ylabel("Reproduced peak / draft peak")
    axis.set_title("Draft Figure 9 vector-path amplitude audit")
    axis.grid(True, which="major", axis="y", linestyle=":", alpha=0.35)
    axis.legend(frameon=False, ncol=3, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "draft_figure9_pdf_peak_audit")
    return {
        "standalone_pdf_sha256": (
            "c6b9c10dbb9383580d8ebc4f3f6f6094c94fe68789d058c52439b9cfdedf6677"
        ),
        "comparison": rows,
    }


def make_figure9_components(results: list[dict]) -> None:
    colors = plt.get_cmap("plasma")(
        np.linspace(0.08, 0.82, len(results))
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.5), sharex=True, sharey=True)
    for result, color in zip(results, colors):
        benchmark = result["benchmark"]
        for column, redshift_piece in enumerate(("low", "high")):
            frequency = result[redshift_piece]["frequency"]
            plot_signed(
                axes[0, column], frequency, result[redshift_piece]["y"],
                color=color, linewidth=1.8, label=figure9_label(benchmark),
            )
            plot_signed(
                axes[1, column], frequency, result[redshift_piece]["nony"],
                color=color, linewidth=1.8,
            )
    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.set_xlim(PLOT_FREQUENCY_MIN_GHZ, PLOT_FREQUENCY_MAX_GHZ)
            axis.set_ylim(1.0e-32, 1.0e-23)
            axis.grid(True, which="major", linestyle=":", alpha=0.22)
    axes[0, 0].set_title(r"$1+z<4$")
    axes[0, 1].set_title(r"$4\leq1+z<3000$")
    axes[0, 0].set_ylabel(r"$|I_\nu^y|$ [SI]")
    axes[1, 0].set_ylabel(r"$|I_\nu^{\rm non-y}|$ [SI]")
    axes[1, 0].set_xlabel(r"Frequency, $\nu$ [GHz]")
    axes[1, 1].set_xlabel(r"Frequency, $\nu$ [GHz]")
    axes[0, 0].legend(frameon=False, fontsize=7.6, ncol=2, loc="lower left")
    fig.suptitle("Figure 9 transfer reproduction split into y and non-y pieces")
    fig.tight_layout()
    save_figure(fig, "draft_figure9_y_nony_components")


def make_high_source_comparison(results: list[dict]) -> list[dict]:
    """Compare direct DarkHistory with the fast high-z transfer calculation."""

    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.7), sharex=True, sharey=True)
    diagnostics = []
    for axis, result in zip(axes.flat, results):
        benchmark = result["benchmark"]
        direct = result["high"]
        fast = result["high_fast"]
        plot_signed(
            axis,
            direct["frequency"],
            direct["total"],
            color="black",
            linewidth=2.2,
            label="direct DarkHistory",
        )
        plot_signed(
            axis,
            fast["frequency"],
            fast["total"],
            color="#d95f02",
            linewidth=1.7,
            label="high-z transfer",
        )
        axis.set_title(figure9_label(benchmark), fontsize=10.5)
        axis.set_xlim(PLOT_FREQUENCY_MIN_GHZ, PLOT_FREQUENCY_MAX_GHZ)
        axis.set_ylim(1.0e-31, 1.0e-23)
        axis.grid(True, which="major", linestyle=":", alpha=0.22)

        def peak(component):
            frequency = component["frequency"]
            band = (
                (frequency >= PLOT_FREQUENCY_MIN_GHZ)
                & (frequency <= PLOT_FREQUENCY_MAX_GHZ)
            )
            return float(np.max(np.abs(component["total"][band])))

        direct_peak = peak(direct)
        fast_peak = peak(fast)
        diagnostics.append(
            {
                "tag": benchmark.tag,
                "direct_darkhistory_peak_si": direct_peak,
                "fast_highz_peak_si": fast_peak,
                "fast_over_direct_peak": fast_peak / direct_peak,
            }
        )

    axes[0, 0].legend(frameon=False, fontsize=8.2, loc="lower left")
    for axis in axes[1, :]:
        axis.set_xlabel(r"Frequency, $\nu$ [GHz]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$|I_\nu|$ [SI]")
    fig.suptitle("Figure 9 high-z provenance check")
    fig.tight_layout()
    save_figure(fig, "draft_figure9_high_source_check")
    with (OUT / "draft_figure9_high_source_check.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)
    return diagnostics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_figure8_status() -> None:
    status = """# Draft Figure 8 reproducibility status

Figure 8 is a producer-side time-history diagnostic, not a transfer-table
convolution.  The four collapsed consumer tables retain only cumulative heat
cells and present-day non-y spectra.  They do not retain the electron and
photon spectra after each timestep, so those curves cannot be reconstructed
losslessly in ExoCLASS.

To reproduce Figure 8, archive the output of the exact producer revision used
for the supplied tables, for the injection at 1+z=4.65, both with and without
photoionization.  The required arrays are `rs`, `elec_eng`, `elec_spec`, the
per-step `phot_spec`, and the cumulative deposited heat (or the native heat
rate together with the precise timestep widths).  The current public
`full_ion.py` exposes `rs`, `elec_eng`, `elec_spec`, `phot_spec`, `f_heat`, and
`y`, but it does not contain the unpublished summed/bridge export path that
created the supplied files.  No consumer-side normalization should be inferred
from the plotted PDF.
"""
    (OUT / "draft_figure8_status.md").write_text(status)


def write_metadata(
    table_metadata: dict,
    figure9_metadata: dict | None,
    high_source_metadata: dict | None,
) -> None:
    paths = {
        "high_heat": ROOT / "DarkAgesModule/transfer_functions/original/tf_final_summed_Ch4.dat",
        "low_heat": ROOT / "DarkAgesModule/transfer_functions/original/low-z/tf_heat_eps-7_mass_scan_summed_lowz.dat",
        "bridge_heat": ROOT / "DarkAgesModule/transfer_functions/original/low-z/tf_heat_eps-7_mass_scan_summed_highinj_lowdep.dat",
        "high_nony": ROOT / "DH_interface/tf_real_data_exclude_y_fixed_init_eps-7_mass_scan_iter0_negdist_newbaseline.dat",
        "low_nony": ROOT / "DarkAgesModule/transfer_functions/original/low-z/tf_nony_eps-7_mass_scan_summed_lowz.dat",
        "bridge_nony": ROOT / "DarkAgesModule/transfer_functions/original/low-z/tf_nony_eps-7_mass_scan_summed_highinj_lowdep.dat",
    }
    payload = {
        "cosmology": {"H0": H0, "Omega_m": OMEGA_M, "Omega_r": OMEGA_R},
        "figure9_modes": {
            "low": "low-only (low injections only; no bridge)",
            "high_fast_check": "legacy, reio_none",
            "production_combined": "extend (includes bridge)",
        },
        "tables": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "table_figures": table_metadata,
        "figure9": figure9_metadata,
        "figure9_high_source": high_source_metadata,
        "figure8": "requires producer timestep histories; see draft_figure8_status.md",
    }
    (OUT / "reproduction_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-class", action="store_true",
        help="run the 14 matched CLASS cases needed for Figure 9",
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun existing CLASS outputs"
    )
    parser.add_argument(
        "--jobs", type=int, default=2, help="concurrent CLASS jobs (1-3)"
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="only regenerate the direct table checks (Figures 6 and 7)",
    )
    parser.add_argument(
        "--darkhistory-high-dir", type=Path,
        help=(
            "directory containing scan_decay_ee_XX_dh__sd_distortions.dat; "
            "use with --darkhistory-baseline to reproduce Figure 9's right panel"
        ),
    )
    parser.add_argument(
        "--darkhistory-baseline", type=Path,
        help="matched negligible-injection slow-DarkHistory distortion table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.darkhistory_high_dir is None) != (
        args.darkhistory_baseline is None
    ):
        raise ValueError(
            "Supply --darkhistory-high-dir and --darkhistory-baseline together."
        )
    configure_plotting()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    _, _, _, high_nony, low_nony, _ = tables
    figure7_tables = load_figure7_draft_tables()
    table_metadata = {
        "figure6": make_figure6(high_nony, low_nony),
        "figure7": make_figure7(*figure7_tables),
    }
    write_figure8_status()

    figure9_metadata = None
    if not args.tables_only:
        cases = write_inis()
        if args.run_class:
            run_cases(cases, args.jobs, args.force)
        missing = [
            class_distortion_path(benchmark, mode)
            for benchmark, mode, _ in cases
            if not class_distortion_path(benchmark, mode).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Figure 9 CLASS outputs; rerun with --run-class:\n"
                + "\n".join(map(str, missing))
            )
        results = load_figure9_results(
            args.darkhistory_high_dir, args.darkhistory_baseline
        )
        high_source = (
            "direct-darkhistory"
            if args.darkhistory_high_dir is not None
            else "high-z-transfer-diagnostic"
        )
        figure9_metadata = make_figure9(results, high_source)
        figure9_metadata["draft_peak_audit"] = make_figure9_reference_audit(
            figure9_metadata
        )
        make_figure9_components(results)
        if args.darkhistory_high_dir is not None:
            figure9_metadata["high_source_check"] = make_high_source_comparison(
                results
            )

    high_source_metadata = None
    if args.darkhistory_high_dir is not None:
        high_source_metadata = {
            "directory": str(args.darkhistory_high_dir),
            "baseline": str(args.darkhistory_baseline),
            "baseline_sha256": file_sha256(args.darkhistory_baseline),
            "threshold_archive_note": (
                "scan index 00 uses 1.05 MeV, while the low-z curve uses "
                "2*(m_e+5 keV)=1.0319978922 MeV and the draft rounds both to "
                "approximately 1.0 MeV"
            ),
        }
    write_metadata(table_metadata, figure9_metadata, high_source_metadata)
    print(f"Wrote draft reproduction products under {OUT}")


if __name__ == "__main__":
    main()
