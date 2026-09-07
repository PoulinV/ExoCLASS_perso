#!/usr/bin/env python3
"""Re-audit the apparent factor-of-two in the low-z energy inventory.

The upper panel digitizes the vector paths of draft Figure 8 directly from
the supplied PDF.  The lower panel independently checks the active summed
low-z heating table at the lowest electron-energy node.  This distinguishes
an endpoint/state-bookkeeping feature in Figure 8 from a global factor-of-two
normalization in the consumer transfer table.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "lowz_draft_figures"
sys.path.insert(0, str(ROOT / "DarkAgesModule"))

import DarkAges  # noqa: E402


ELECTRON_MASS_EV = 510998.9461

# Coordinates of major ticks in the Matplotlib-local coordinate system stored
# in the page-8 SVG.  These are read from the vector PDF, not estimated from a
# raster image: x=1 and x=2, and y=1 and y=0.1, respectively.
X_AT_ONE = 72.587729
X_AT_TWO = 150.710086
Y_AT_ONE = 286.871585
Y_DECADE = 39.593095
FIGURE8_TRANSFORM = "0.59117"

COLORS = {
    "electrons": "rgb(63.279724%, 18.788147%, 49.533081%)",
    "photons": "rgb(0.144958%, 0.0457764%, 1.385498%)",
    "heat": "rgb(99.56665%, 81.269836%, 57.263184%)",
}


def _path_points(path: ET.Element) -> np.ndarray:
    values = [float(value) for value in re.findall(r"-?[0-9.]+", path.get("d", ""))]
    return np.asarray(values, dtype=np.float64).reshape(-1, 2)


def extract_figure8_curves(draft: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract the solid no-photoionization curves from draft Figure 8."""

    if not draft.is_file():
        raise FileNotFoundError(draft)
    with tempfile.TemporaryDirectory(prefix="lowz-figure8-") as directory:
        svg = Path(directory) / "page8.svg"
        subprocess.run(
            [
                "pdftocairo",
                "-f",
                "8",
                "-l",
                "8",
                "-svg",
                str(draft),
                str(svg),
            ],
            check=True,
        )
        root = ET.parse(svg).getroot()

    namespace = "{http://www.w3.org/2000/svg}path"
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in root.iter(namespace):
        if FIGURE8_TRANSFORM not in path.get("transform", ""):
            continue
        if path.get("stroke-width") != "1.8" or "stroke-dasharray" in path.attrib:
            continue
        points = _path_points(path)
        if points.shape[0] <= 10:
            continue
        for name, color in COLORS.items():
            if path.get("stroke") == color:
                redshift = 1.0 + (points[:, 0] - X_AT_ONE) / (X_AT_TWO - X_AT_ONE)
                fraction = np.power(10.0, (points[:, 1] - Y_AT_ONE) / Y_DECADE)
                curves[name] = (redshift, fraction)

    missing = set(COLORS) - set(curves)
    if missing:
        raise RuntimeError("Could not extract Figure 8 curves: " + ", ".join(sorted(missing)))
    return curves


def interpolate_log_curve(
    output_redshift: np.ndarray,
    curve: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    redshift, fraction = curve
    order = np.argsort(redshift)
    return np.power(
        10.0,
        np.interp(output_redshift, redshift[order], np.log10(fraction[order])),
    )


def figure8_inventory(curves):
    redshift, electrons = curves["electrons"]
    photons = interpolate_log_curve(redshift, curves["photons"])
    heat = interpolate_log_curve(redshift, curves["heat"])
    total = electrons + photons + heat
    return redshift, electrons, photons, heat, total


def active_heat_budget():
    low_heat, _ = DarkAges.get_lowz_heat_transfer_functions()
    energy = np.power(10.0, low_heat.log10E)
    kinetic_fraction = energy / (energy + ELECTRON_MASS_EV)
    integrated_heat = np.sum(low_heat.transfer_elec, axis=0)
    ratio = integrated_heat / kinetic_fraction[:, None]
    return low_heat, energy, ratio


def write_csvs(redshift, electrons, photons, heat, total, low_heat, energy, ratio):
    with (OUT / "figure8_vector_energy_budget.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("one_plus_z", "electrons", "photons", "heat", "sum"))
        writer.writerows(zip(redshift, electrons, photons, heat, total))

    with (OUT / "summed_heat_budget.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "electron_kinetic_energy_eV",
                "one_plus_z_injection",
                "integrated_heat_over_pair_kinetic_budget",
            )
        )
        for energy_index, kinetic_energy in enumerate(energy):
            for redshift_index, injection_redshift in enumerate(low_heat.z_injected):
                writer.writerow(
                    (kinetic_energy, injection_redshift, ratio[energy_index, redshift_index])
                )


def make_plot(redshift, electrons, photons, heat, total, low_heat, ratio):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    order = np.argsort(redshift)
    axes[0].plot(redshift[order], electrons[order], color="#a12f7e", label="electrons")
    axes[0].plot(redshift[order], photons[order], color="#11121a", label="photons")
    axes[0].plot(redshift[order], heat[order], color="#f1b75f", label="heat")
    axes[0].plot(
        redshift[order], total[order], color="#167d9a", linewidth=2.1,
        label="sum",
    )
    axes[0].axhline(1.0, color="0.4", linestyle="--", linewidth=1.0)
    axes[0].scatter(redshift[:2], total[:2], color="#167d9a", zorder=5, s=28)
    axes[0].annotate(
        f"first sample: {total[0]:.3f}",
        xy=(redshift[0], total[0]),
        xytext=(-88, -25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=8.5,
    )
    axes[0].annotate(
        f"next sample: {total[1]:.3f}",
        xy=(redshift[1], total[1]),
        xytext=(-105, -52),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=8.5,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.85, 4.72)
    axes[0].set_ylim(8.0e-7, 3.2)
    axes[0].set_xlabel(r"$1+z$ (evolution proceeds right to left)")
    axes[0].set_ylabel(r"fraction of injected energy")
    axes[0].set_title("Draft Fig. 8 vector audit: no photoionization")
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")

    energy_index = 0
    valid = low_heat.z_injected > low_heat.z_injected[0]
    injection_redshift = low_heat.z_injected[valid]
    active = ratio[energy_index, valid]
    axes[1].plot(
        injection_redshift,
        active,
        color="#2458a6",
        marker="o",
        linewidth=1.8,
        label="active summed table",
    )
    axes[1].plot(
        injection_redshift,
        2.0 * active,
        color="#c53a32",
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label=r"$2\times$ active table (diagnostic)",
    )
    axes[1].axhline(
        1.0,
        color="0.4",
        linestyle=":",
        linewidth=1.2,
        label="pair kinetic-energy reference",
    )
    maximum = float(np.max(active))
    maximum_index = int(np.argmax(active))
    axes[1].annotate(
        f"active maximum = {maximum:.3f}",
        xy=(injection_redshift[maximum_index], maximum),
        xytext=(20, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=8.5,
    )
    axes[1].set_xlim(1.15, 4.8)
    axes[1].set_ylim(0.0, 2.2)
    axes[1].set_xlabel(r"injection redshift, $1+z_{\rm inj}$")
    axes[1].set_ylabel(
        r"$\sum_{z_{\rm dep}}T_{\rm heat}/[E_{\rm kin}/(E_{\rm kin}+m_e)]$"
    )
    axes[1].set_title(r"Heat-channel ceiling check at $E_{\rm kin}=5.0003$ keV")
    axes[1].legend(frameon=False, fontsize=8.2, loc="upper right")
    axes[1].text(
        0.02,
        0.035,
        "Heat only: surviving electron/photon energy is not included.",
        transform=axes[1].transAxes,
        fontsize=8.2,
        color="0.3",
    )

    for axis in axes:
        axis.grid(True, which="major", linestyle=":", alpha=0.28)
        axis.tick_params(which="both", direction="in", top=True, right=True)

    fig.suptitle("Low-z energy-budget re-audit")
    fig.tight_layout()
    output = OUT / "lowz_energy_budget_reaudit.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft",
        required=True,
        type=Path,
        help="path to the CLASS_DarkHistory draft PDF containing Figure 8",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    curves = extract_figure8_curves(args.draft)
    inventory = figure8_inventory(curves)
    low_heat, energy, ratio = active_heat_budget()
    write_csvs(*inventory, low_heat, energy, ratio)
    output = make_plot(*inventory, low_heat, ratio)
    print(output)
    print(f"Figure 8 first/next sums: {inventory[-1][0]:.9g}, {inventory[-1][1]:.9g}")
    print(f"Active summed-table maximum: {np.max(ratio):.9g}")


if __name__ == "__main__":
    main()
