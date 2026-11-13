#!/usr/bin/env python
"""Scan the low-injection share of decay spectral distortions.

The grid and CMB-boundary fraction follow the prescription used for
``fossil_dominant_regime_baseline_comparison.pdf``.  For every mass/lifetime
point, two matched ExoCLASS calculations are performed:

* ``extend``: the complete high-high + low-low + delayed high-to-low matrix;
* ``low-below-four``: injection nodes with 1+z<4 only, with a zero
  heating boundary inserted at 1+z=4.

Both are differenced against a negligible-injection baseline in the same
mode. The total and low-redshift y signals are integrated directly from the
pointwise baseline-subtracted ``exact_y`` history; no difference of completed
y integrals with different ``sd_z_min`` values is used. The standard mu
amplitude is integrated above the z=3000 CLASS/DarkHistory handoff, preserving
the routing used by the reference figure; its z<3 share is therefore zero.
All three channels are compared
with the FOSSIL-band L1 integral
``sum |Delta I_channel| Delta nu`` on 50,65,...,1985 GHz.
"""

from __future__ import division, print_function

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "lowz_fraction_scan"
CACHE = OUT / "lowz_distortion_fraction_scan.npz"
CSV = OUT / "lowz_distortion_fraction_scan.csv"
SUMMARY = OUT / "lowz_distortion_fraction_summary.json"
FIG_Y = OUT / "lowz_fraction_y"
FIG_NONY = OUT / "lowz_fraction_nony"
PDF_OUT = ROOT / "output" / "pdf"
FIG_CHANNELS = PDF_OUT / "lowz_channel_contributions"
FIG_DOMINANCE = PDF_OUT / "lowz_channel_dominance"

DETECTOR_CACHE_DEFAULT = os.environ.get("FOSSIL_DETECTOR_CACHE", "")
CLASSY_DEFAULT = ROOT / "python"

FREQUENCY_GHZ = 50.0 + 15.0 * np.arange(130)
DELTA_NU_GHZ = 15.0
LOW_INJECTION_CUTOFF = 4.0
LOW_DEPOSITION_Z_MAX = LOW_INJECTION_CUTOFF - 1.0
# Preserve the component routing used by the full DarkHistory interface and
# the reference dominance figure: CLASS supplies the standard mu template
# above the z=3000 handoff, while the transferred non-y spectrum supplies the
# lower-redshift cascade shape.  Integrating the detector-dependent exact-PCA
# mu branching below this handoff would count part of that non-y spectrum a
# second time (and produces a large negative template projection near z=3).
MU_CLASS_HANDOFF_Z = 3000.0
AGE_UNIVERSE_SECONDS_PROXY = 13.8e9 * 365.25 * 24.0 * 3600.0
DYNAMIC_REMOVAL_THRESHOLD = 1.0e-2

# Injected channel (EXOCLASS_MAP_CHANNEL=electron or photon; default electron).
CHANNEL = os.environ.get("EXOCLASS_MAP_CHANNEL", "electron")
if CHANNEL not in ("electron", "photon"):
    raise RuntimeError("EXOCLASS_MAP_CHANNEL must be 'electron' or 'photon'")
SPECTRUM = {"electron": "dirac_electron", "photon": "dirac_photon"}[CHANNEL]

# Preserve the grid used by the fossil dominant-regime calculation: three
# near-threshold additions plus the original 20-row logarithmic scan.
MASSES_GEV = np.sort(
    np.concatenate(
        [
            np.array([1.05e-3, 1.73e-3, 2.20e-3]),
            np.logspace(np.log10(7.2e-3), np.log10(5.0e3), 20),
        ]
    )
)
if CHANNEL == "photon":
    # Same number of rows; photons have no pair threshold and the transfer
    # tables start at 5 keV per photon, i.e. m_chi = 10 keV.
    MASSES_GEV = np.logspace(np.log10(1.0e-5), np.log10(5.0e3), 23)
TAUS_S = np.logspace(9.0, 27.0, 46)

# Digitized CMB-anisotropy boundary used by the referenced fossil figure.
LOG_TAU_CMB = np.array(
    [9.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
     19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 27.0]
)
LOG_F_CMB = np.array(
    [0.0, 0.0, -5.0, -7.5, -10.0, -9.1, -8.2, -7.3, -6.4,
     -5.5, -4.5, -3.6, -2.7, -1.8, -0.9, 0.0, 0.0]
)

BASE = {
    "output": "Sd",
    "omega_b": 0.02242,
    "omega_cdm": 0.11933,
    "H0": 67.66,
    "N_ur": 2.03351,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
    "n_s": 0.9665,
    "ln10^{10}A_s": 3.047,
    "k_pivot": 0.05,
    "Pk_ini_type": "analytic_Pk",
    "reio_parametrization": "reio_stars",
    "include_reio_stars_cooling_terms": "yes",
    "include_reio_stars_helium": "yes",
    "reio_stars_photoion_file": str(
        ROOT / "external" / "heating" / "photoion_rates_Puchwein.dat"
    ),
    "reio_stars_photoheat_file": str(
        ROOT / "external" / "heating" / "photoheat_rates_Puchwein2.dat"
    ),
    "reio_stars_helium_file": str(
        ROOT / "external" / "heating" / "xHe_DarkHistory.txt"
    ),
    "sd_branching_approx": "exact",
    "sd_PCA_size": 2,
    "sd_detector_name": "fossil_step2_monopole_const1Jy_130centers",
    "sd_detector_nu_min": 50.0,
    "sd_detector_nu_max": 1985.0,
    "sd_detector_nu_delta": 15.0,
    "sd_detector_delta_Ic": 1.0,
    "include_DH_SMresidual_distortions": "no",
    "add_SD_to_CLASS": "yes",
    "sd_only_exotic": "yes",
    "exact_y": "yes",
    "sd_z_min": 0.01,
    "sd_z_max": 5.0e6,
    "sd_z_size": 1000,
}

DM = {
    "f_eff_type": "DarkAges",
    "DarkAges_mode": "built_in",
    "injected_particle_spectra": SPECTRUM,
    "injected_particle_branching_ratio": 1,
    # CLASS computes y from the DarkAges heating table.  The non-y transfer
    # convolution is evaluated directly below, avoiding a second external
    # DarkAges process at every scan point.
    "compute_SD_with_DarkAges": "no",
}

_CLASS = None
_DARKAGES = None
_SPEC_MODEL = None
_SD_TODAY = None
_ADD_SD = None
_LOW_HIGH_MASKS = None
_HIGH_SD = None
_LOW_SD = None
_DELAYED_SD = None


def cmb_fraction(log_tau):
    return np.clip(10.0 ** np.interp(log_tau, LOG_TAU_CMB, LOG_F_CMB), 0.0, 1.0)


def worker_init(classy_path, detector_cache=None):
    global _CLASS, _DARKAGES, _SPEC_MODEL, _SD_TODAY, _ADD_SD
    global _LOW_HIGH_MASKS, _HIGH_SD, _LOW_SD, _DELAYED_SD
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "2"
    if detector_cache is None:
        detector_cache = DETECTOR_CACHE_DEFAULT
    if not detector_cache:
        raise RuntimeError(
            "Set --detector-cache or FOSSIL_DETECTOR_CACHE to the 130-bin "
            "FOSSIL PCA cache."
        )
    BASE["sd_external_path"] = str(Path(detector_cache).resolve())
    sys.path.insert(0, str(classy_path))
    from classy import Class
    _CLASS = Class
    sys.path.insert(0, str(ROOT / "DarkAgesModule"))
    import DarkAges
    from DarkAges.lowz import add_spectral_distortions, low_high_masks
    from DarkAges.recipes import spec_elec_and_phot
    from DarkAges.spectral_distortions import spectral_distortion_today
    omega_m = (BASE["omega_b"] + BASE["omega_cdm"]) / (BASE["H0"] / 100.0) ** 2
    DarkAges.set_background(BASE["H0"], omega_m, 7.9111186056792255e-5)
    _DARKAGES = DarkAges
    _SPEC_MODEL = spec_elec_and_phot
    _SD_TODAY = spectral_distortion_today
    _ADD_SD = add_spectral_distortions
    _LOW_HIGH_MASKS = low_high_masks
    _HIGH_SD = DarkAges.spectral_distortions_functions
    _LOW_SD, _DELAYED_SD = DarkAges.get_lowz_spectral_distortions_transfer_functions()


def run_class(parameters, include_components=False):
    cosmology = _CLASS()
    try:
        cosmology.set(parameters)
        # With sd_only_exotic=yes, spectral distortions need the thermal
        # history but not perturbations, primordial spectra, transfer
        # functions, or angular spectra.
        cosmology.compute_distortions_only()
        amplitudes = cosmology.spectral_distortion_amplitudes()
        result = {
            # CLASS orders the standard amplitudes as g, y, mu, residuals.
            "reported_y": float(amplitudes[1]),
            "reported_mu": float(amplitudes[2]),
            "y_history": cosmology.spectral_distortion_y_history(),
            "heating_history": cosmology.spectral_distortion_heating_history(),
        }
        if include_components:
            frequency, components = cosmology.spectral_distortion_components()
            result["frequency_GHz"] = np.asarray(frequency, dtype=float)
            result["components"] = np.asarray(components, dtype=float)
        return result
    finally:
        try:
            cosmology.struct_cleanup()
        except Exception:
            pass
        try:
            cosmology.empty()
        except Exception:
            pass


def integrate_exact_y_difference(model_history, baseline_history, scale=1.0,
                                 z_max=None):
    """Integrate the exotic exact-y history, optionally only below ``z_max``."""

    model_z, model_integrand, model_branching, model_weights = model_history
    base_z, base_integrand, base_branching, base_weights = baseline_history
    for model_array, base_array, label in (
        (model_z, base_z, "redshift"),
        (model_branching, base_branching, "y branching"),
        (model_weights, base_weights, "redshift weight"),
    ):
        if model_array.shape != base_array.shape or not np.allclose(
            model_array, base_array, rtol=2.0e-13, atol=0.0
        ):
            raise RuntimeError("Model and baseline {} grids differ".format(label))

    delta = scale * (model_integrand - base_integrand) * model_branching
    if z_max is None or z_max >= model_z[-1]:
        # This is exactly the quadrature used internally by CLASS.
        return 0.25 * float(np.dot(delta, model_weights))
    if z_max <= model_z[0]:
        return 0.0

    stop = int(np.searchsorted(model_z, z_max, side="left"))
    if stop < model_z.size and model_z[stop] == z_max:
        partial_z = model_z[:stop + 1]
        partial_delta = delta[:stop + 1]
    else:
        partial_z = np.append(model_z[:stop], z_max)
        partial_delta = np.append(
            delta[:stop], np.interp(z_max, model_z, delta)
        )
    return 0.25 * float(np.trapz(partial_delta, partial_z))


def integrate_mu_difference(model_history, baseline_history, scale=1.0,
                            z_min=None, z_max=None):
    """Integrate matched-baseline mu directly from the heating history."""

    (
        model_z, model_heating, model_g, model_y, model_mu, model_weights,
    ) = model_history
    (
        base_z, base_heating, base_g, base_y, base_mu, base_weights,
    ) = baseline_history
    for model_array, base_array, label in (
        (model_z, base_z, "redshift"),
        (model_g, base_g, "g branching"),
        (model_y, base_y, "y branching"),
        (model_mu, base_mu, "mu branching"),
        (model_weights, base_weights, "redshift weight"),
    ):
        if model_array.shape != base_array.shape or not np.allclose(
            model_array, base_array, rtol=2.0e-13, atol=0.0
        ):
            raise RuntimeError("Model and baseline {} grids differ".format(label))

    delta = scale * (model_heating - base_heating) * model_mu
    if z_min is None and (z_max is None or z_max >= model_z[-1]):
        return 1.401 * float(np.dot(delta, model_weights))
    lower = model_z[0] if z_min is None else max(float(z_min), model_z[0])
    upper = model_z[-1] if z_max is None else min(float(z_max), model_z[-1])
    if upper <= lower:
        return 0.0
    inside = (model_z > lower) & (model_z < upper)
    partial_z = np.concatenate(([lower], model_z[inside], [upper]))
    partial_delta = np.concatenate(
        ([np.interp(lower, model_z, delta)], delta[inside],
         [np.interp(upper, model_z, delta)])
    )
    return 1.401 * float(np.trapz(partial_delta, partial_z))


def run_baseline(mode_key):
    transfer_mode = "extend" if mode_key == "extend-lowdep" else mode_key
    parameters = dict(BASE)
    parameters.update(DM)
    parameters.update(
        {
            "lowz_transfer_mode": transfer_mode,
            "DM_decay_mass": 7.2e-3,
            "DM_decay_fraction": 1.0e-20,
            "DM_decay_Gamma": 1.0e-30,
        }
    )
    baseline = run_class(parameters)
    if mode_key == "extend":
        # Calibrate the y and mu spectral templates directly on the active
        # detector grid.  Unit manual amplitudes avoid dividing by a tiny
        # physical baseline while leaving the thermal history unchanged.
        calibration_parameters = dict(parameters)
        calibration_parameters.update({"sd_add_y": 1.0, "sd_add_mu": 1.0})
        calibration = run_class(
            calibration_parameters, include_components=True
        )
        frequency = calibration["frequency_GHz"]
        components = calibration["components"]
        if frequency.shape != FREQUENCY_GHZ.shape or not np.allclose(
            frequency, FREQUENCY_GHZ, rtol=0.0, atol=1.0e-9
        ):
            raise RuntimeError("CLASS and requested FOSSIL frequency grids differ")
        baseline["C_y_l1"] = float(
            DELTA_NU_GHZ * np.sum(np.abs(components[1]))
            / abs(calibration["reported_y"])
        )
        baseline["C_mu_l1"] = float(
            DELTA_NU_GHZ * np.sum(np.abs(components[2]))
            / abs(calibration["reported_mu"])
        )
    return mode_key, baseline


def run_mode(mass, tau, fraction, mode_key):
    transfer_mode = "extend" if mode_key == "extend-lowdep" else mode_key
    evaluations = [fraction]
    for cap in (1.0e-4, 1.0e-8):
        candidate = min(fraction, cap)
        if candidate not in evaluations:
            evaluations.append(candidate)
    errors = []
    for evaluation_fraction in evaluations:
        parameters = dict(BASE)
        parameters.update(DM)
        parameters.update(
            {
                "lowz_transfer_mode": transfer_mode,
                "DM_decay_mass": mass,
                "DM_decay_Gamma": 1.0 / tau,
                "DM_decay_fraction": evaluation_fraction,
            }
        )
        try:
            class_result = run_class(parameters)
            return class_result, evaluation_fraction, len(errors) + 1
        except Exception:
            errors.append(traceback.format_exc())
    raise RuntimeError("\n--- retry ---\n".join(errors))


def direct_nony(mass, tau, fraction, strict_only=False, lowdep_only=False):
    """Return selected full, low-deposition, and low-injection spectra."""

    strict_low_mask = _LOW_SD.z_injected < LOW_INJECTION_CUTOFF
    h = BASE["H0"] / 100.0
    omega_cdm_fraction = BASE["omega_cdm"] / h ** 2
    class_h0 = BASE["H0"] / 299792.458
    n_cdm = (
        fraction * omega_cdm_fraction * class_h0 ** 2 * 94.7024726 / mass
    )

    def evaluate(table, mask):
        log_energies = np.log10(table.E_injected)
        model = _SPEC_MODEL(
            [SPECTRUM], mass, logEnergies=log_energies,
            redshift=table.z_injected, t_dec=tau, hist="decay",
            branchings=np.ones(1),
        )
        return _SD_TODAY(
            table.frequency, table.z_injected, model.logEnergies,
            table.E_injected, table.spectral_distortions_phot,
            table.spectral_distortions_elec, model.spec_electrons,
            model.spec_photons, hist="decay", normalization=model.normalization,
            t_dec=tau, n_cdm=n_cdm, injection_mask=mask,
        )

    def fossil_sample(frequency, spectrum):
        return np.interp(
            np.log(FREQUENCY_GHZ), np.log(frequency), spectrum,
            left=0.0, right=0.0,
        )

    if strict_only:
        strict_low = evaluate(_LOW_SD, strict_low_mask)
        return {"strict": fossil_sample(_LOW_SD.frequency, strict_low)}

    full_low_mask, high_mask = _LOW_HIGH_MASKS(
        _LOW_SD.z_injected, _HIGH_SD.z_injected, "extend"
    )
    delayed = evaluate(_DELAYED_SD, high_mask)
    low = evaluate(_LOW_SD, full_low_mask)
    lowdep = _ADD_SD(_LOW_SD.frequency, low, _DELAYED_SD.frequency, delayed)
    lowdep_sample = fossil_sample(_LOW_SD.frequency, lowdep)
    if lowdep_only:
        return {"lowdep": lowdep_sample}

    strict_low = evaluate(_LOW_SD, strict_low_mask)
    strict_sample = fossil_sample(_LOW_SD.frequency, strict_low)
    high = evaluate(_HIGH_SD, high_mask)
    full = _ADD_SD(_HIGH_SD.frequency, high, _DELAYED_SD.frequency, delayed)
    full = _ADD_SD(_HIGH_SD.frequency, full, _LOW_SD.frequency, low)
    return {
        "full": fossil_sample(_HIGH_SD.frequency, full),
        "lowdep": lowdep_sample,
        "strict": strict_sample,
    }


def run_point(task):
    (
        i, j, mass, tau, fraction, baselines,
        redo_low, redo_lowdep, redo_mu,
    ) = task
    started = time.time()
    result = {"i": i, "j": j, "ok": False}
    try:
        if redo_mu:
            modes = ("extend",)
        elif redo_lowdep:
            modes = ("extend-lowdep",)
        elif redo_low:
            modes = ("low-below-four",)
        else:
            modes = ("extend", "low-below-four")
        for mode in modes:
            class_result, evaluation_fraction, attempts = run_mode(
                mass, tau, fraction, mode
            )
            baseline = baselines[mode]
            scale = fraction / evaluation_fraction
            delta_y = integrate_exact_y_difference(
                class_result["y_history"], baseline["y_history"], scale=scale
            )
            lowdep_y = integrate_exact_y_difference(
                class_result["y_history"], baseline["y_history"], scale=scale,
                z_max=LOW_DEPOSITION_Z_MAX,
            )
            reported_delta_y = (
                class_result["reported_y"] - baseline["reported_y"]
            ) * scale
            reported_delta_mu = (
                class_result["reported_mu"] - baseline["reported_mu"]
            ) * scale
            delta_mu = integrate_mu_difference(
                class_result["heating_history"],
                baseline["heating_history"],
                scale=scale,
                z_min=MU_CLASS_HANDOFF_Z,
            )
            # The lower-redshift cascade is already in the transferred non-y
            # spectrum.  The standard mu channel is therefore, by definition,
            # absent below the DarkHistory handoff (and hence below z=3).
            lowdep_mu = 0.0
            result[mode] = {
                "y": delta_y,
                "lowdep_y": lowdep_y,
                "mu": delta_mu,
                "lowdep_mu": lowdep_mu,
                "reported_y_difference": reported_delta_y,
                "reported_mu_difference": reported_delta_mu,
                "evaluation_fraction": evaluation_fraction,
                "attempts": attempts,
            }
        if redo_mu:
            result["ok"] = True
            result["seconds"] = time.time() - started
            return result
        nony = direct_nony(
            mass, tau, fraction, strict_only=redo_low,
            lowdep_only=redo_lowdep,
        )
        if redo_lowdep:
            spectra = (("extend-lowdep", nony["lowdep"]),)
        elif redo_low:
            spectra = (("low-below-four", nony["strict"]),)
        else:
            spectra = (
                ("extend", nony["full"]),
                ("low-below-four", nony["strict"]),
            )
            result["extend"]["lowdep_nony_l1"] = (
                DELTA_NU_GHZ * np.sum(np.abs(nony["lowdep"]))
            )
            result["extend"]["lowdep_nony_spectrum"] = nony["lowdep"]
        for mode, spectrum in spectra:
            result[mode]["nony_l1"] = DELTA_NU_GHZ * np.sum(np.abs(spectrum))
            result[mode]["nony_spectrum"] = spectrum
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    result["seconds"] = time.time() - started
    return result


def empty_state(baselines):
    shape = (MASSES_GEV.size, TAUS_S.size)
    fractions = cmb_fraction(np.log10(TAUS_S))
    fraction_grid = np.broadcast_to(fractions, shape).copy()
    removed = fraction_grid * (
        1.0 - np.exp(-AGE_UNIVERSE_SECONDS_PROXY / TAUS_S[np.newaxis, :])
    )
    state = {
        "schema_version": np.array(4, dtype=np.int64),
        "masses_GeV": MASSES_GEV,
        "taus_s": TAUS_S,
        "f_cmb": fraction_grid,
        "removed_fraction_proxy": removed,
        "completed": np.zeros(shape, dtype=np.uint8),
        "full_y": np.full(shape, np.nan),
        "low_y": np.full(shape, np.nan),
        "full_mu": np.full(shape, np.nan),
        "low_mu": np.full(shape, np.nan),
        # The standard mu template is routed only above the z=3000 handoff.
        "lowdep_mu": np.zeros(shape),
        "full_nony_l1": np.full(shape, np.nan),
        "low_nony_l1": np.full(shape, np.nan),
        "full_nony_spectrum": np.full(shape + (FREQUENCY_GHZ.size,), np.nan),
        "low_nony_spectrum": np.full(shape + (FREQUENCY_GHZ.size,), np.nan),
        "full_evaluation_fraction": np.full(shape, np.nan),
        "low_evaluation_fraction": np.full(shape, np.nan),
        "full_attempts": np.zeros(shape, dtype=np.uint8),
        "low_attempts": np.zeros(shape, dtype=np.uint8),
        "full_reported_y_difference": np.full(shape, np.nan),
        "low_reported_y_difference": np.full(shape, np.nan),
        "full_reported_mu_difference": np.full(shape, np.nan),
        "low_reported_mu_difference": np.full(shape, np.nan),
        "lowdep_y": np.full(shape, np.nan),
        "lowdep_nony_l1": np.full(shape, np.nan),
        "lowdep_nony_spectrum": np.full(shape + (FREQUENCY_GHZ.size,), np.nan),
        "lowdep_evaluation_fraction": np.full(shape, np.nan),
        "lowdep_attempts": np.zeros(shape, dtype=np.uint8),
        "run_seconds": np.full(shape, np.nan),
        "frequency_GHz": FREQUENCY_GHZ,
        "baseline_extend_y": np.array(baselines["extend"]["reported_y"]),
        "baseline_extend_mu": np.array(baselines["extend"]["reported_mu"]),
        "baseline_low_y": np.array(
            baselines["low-below-four"]["reported_y"]
        ),
        "baseline_low_mu": np.array(
            baselines["low-below-four"]["reported_mu"]
        ),
        "C_y_l1": np.array(baselines["extend"]["C_y_l1"]),
        "C_mu_l1": np.array(baselines["extend"]["C_mu_l1"]),
    }
    return state


def save_state(state):
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = CACHE.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **state)
    os.replace(str(temporary), str(CACHE))


def load_state():
    with np.load(str(CACHE), allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    shape = state["full_y"].shape
    spectral_shape = shape + (state["frequency_GHz"].size,)
    defaults = {
        "schema_version": np.array(4, dtype=np.int64),
        "full_reported_y_difference": np.full(shape, np.nan),
        "low_reported_y_difference": np.full(shape, np.nan),
        "full_reported_mu_difference": np.full(shape, np.nan),
        "low_reported_mu_difference": np.full(shape, np.nan),
        # Deliberately do not derive this from the old full-minus-high-z
        # result: schema 4 requires the direct exact-y integrand calculation.
        "lowdep_y": np.full(shape, np.nan),
        "lowdep_nony_l1": np.full(shape, np.nan),
        "lowdep_nony_spectrum": np.full(spectral_shape, np.nan),
        "lowdep_evaluation_fraction": np.full(shape, np.nan),
        "lowdep_attempts": np.zeros(shape, dtype=np.uint8),
        "full_mu": np.full(shape, np.nan),
        "low_mu": np.full(shape, np.nan),
        "lowdep_mu": np.zeros(shape),
        "baseline_extend_mu": np.array(np.nan),
        "baseline_low_mu": np.array(np.nan),
        "C_y_l1": np.array(np.nan),
        "C_mu_l1": np.array(np.nan),
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
    state["schema_version"] = np.array(4, dtype=np.int64)
    return state


def store_result(
    state, result, redo_low=False, redo_lowdep=False, redo_mu=False
):
    i, j = result["i"], result["j"]
    if redo_mu:
        mode_result = result["extend"]
        state["full_mu"][i, j] = mode_result["mu"]
        state["lowdep_mu"][i, j] = mode_result["lowdep_mu"]
        state["full_reported_mu_difference"][i, j] = mode_result[
            "reported_mu_difference"
        ]
        state["full_evaluation_fraction"][i, j] = mode_result[
            "evaluation_fraction"
        ]
        state["full_attempts"][i, j] = mode_result["attempts"]
        state["run_seconds"][i, j] = result["seconds"]
        return
    if redo_lowdep:
        mode_result = result["extend-lowdep"]
        state["lowdep_y"][i, j] = mode_result["lowdep_y"]
        state["lowdep_nony_l1"][i, j] = mode_result["nony_l1"]
        state["lowdep_nony_spectrum"][i, j] = mode_result["nony_spectrum"]
        state["lowdep_evaluation_fraction"][i, j] = mode_result["evaluation_fraction"]
        state["lowdep_attempts"][i, j] = mode_result["attempts"]
        state["run_seconds"][i, j] = result["seconds"]
        return
    pairs = (("low", "low-below-four"),) if redo_low else (
        ("full", "extend"), ("low", "low-below-four")
    )
    for key, mode in pairs:
        state[key + "_y"][i, j] = result[mode]["y"]
        state[key + "_mu"][i, j] = result[mode]["mu"]
        state[key + "_nony_l1"][i, j] = result[mode]["nony_l1"]
        state[key + "_nony_spectrum"][i, j] = result[mode]["nony_spectrum"]
        state[key + "_evaluation_fraction"][i, j] = result[mode]["evaluation_fraction"]
        state[key + "_attempts"][i, j] = result[mode]["attempts"]
        state[key + "_reported_y_difference"][i, j] = result[mode][
            "reported_y_difference"
        ]
        state[key + "_reported_mu_difference"][i, j] = result[mode][
            "reported_mu_difference"
        ]
    if not redo_low:
        state["lowdep_y"][i, j] = result["extend"]["lowdep_y"]
        state["lowdep_mu"][i, j] = result["extend"]["lowdep_mu"]
        state["lowdep_nony_l1"][i, j] = result["extend"]["lowdep_nony_l1"]
        state["lowdep_nony_spectrum"][i, j] = result["extend"][
            "lowdep_nony_spectrum"
        ]
        state["lowdep_evaluation_fraction"][i, j] = result["extend"][
            "evaluation_fraction"
        ]
        state["lowdep_attempts"][i, j] = result["extend"]["attempts"]
    state["run_seconds"][i, j] = result["seconds"]
    state["completed"][i, j] = 1


def run_scan(args):
    classy_path = Path(args.classy_path).resolve()
    if not list(classy_path.glob("classy*.so")):
        raise RuntimeError("No compiled classy extension found in {}".format(classy_path))
    detector_cache = Path(args.detector_cache).resolve()
    if not (detector_cache / "detectors_list.dat").is_file():
        raise RuntimeError(
            "FOSSIL detector cache is missing detectors_list.dat: {}".format(
                detector_cache
            )
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(classy_path, detector_cache),
    ) as executor:
        resuming = args.resume and CACHE.is_file()
        if (args.redo_low or args.redo_lowdep or args.redo_mu) and not resuming:
            raise RuntimeError("A numerator-only pass requires --resume and an existing cache")
        if args.redo_mu:
            baseline_modes = ("extend",)
        elif args.redo_lowdep:
            baseline_modes = ("extend-lowdep",)
        elif args.redo_low:
            baseline_modes = ("low-below-four",)
        else:
            baseline_modes = ("extend", "low-below-four")
        futures = {
            executor.submit(run_baseline, mode): mode for mode in baseline_modes
        }
        baselines = {}
        for future in as_completed(futures):
            mode, baseline = future.result()
            baselines[mode] = baseline

        if resuming:
            state = load_state()
            if args.redo_mu:
                state["baseline_extend_mu"] = np.array(
                    baselines["extend"]["reported_mu"]
                )
                state["C_y_l1"] = np.array(baselines["extend"]["C_y_l1"])
                state["C_mu_l1"] = np.array(baselines["extend"]["C_mu_l1"])
            elif args.redo_lowdep:
                state["baseline_extend_y"] = np.array(
                    baselines["extend-lowdep"]["reported_y"]
                )
            elif args.redo_low:
                state["baseline_low_y"] = np.array(
                    baselines["low-below-four"]["reported_y"]
                )
                state["baseline_low_mu"] = np.array(
                    baselines["low-below-four"]["reported_mu"]
                )
            else:
                state["baseline_extend_y"] = np.array(
                    baselines["extend"]["reported_y"]
                )
                state["baseline_extend_mu"] = np.array(
                    baselines["extend"]["reported_mu"]
                )
                state["baseline_low_y"] = np.array(
                    baselines["low-below-four"]["reported_y"]
                )
                state["baseline_low_mu"] = np.array(
                    baselines["low-below-four"]["reported_mu"]
                )
                state["C_y_l1"] = np.array(baselines["extend"]["C_y_l1"])
                state["C_mu_l1"] = np.array(baselines["extend"]["C_mu_l1"])
        else:
            state = empty_state(baselines)
            if not args.smoke:
                save_state(state)

        pending = []
        fractions = cmb_fraction(np.log10(TAUS_S))
        for i, mass in enumerate(MASSES_GEV):
            for j, tau in enumerate(TAUS_S):
                if (
                    args.redo_low or args.redo_lowdep or args.redo_mu
                    or state["completed"][i, j] == 0
                ):
                    pending.append(
                        (
                            i, j, float(mass), float(tau), float(fractions[j]),
                            baselines, bool(args.redo_low), bool(args.redo_lowdep),
                            bool(args.redo_mu),
                        )
                    )
        if args.smoke:
            pending = [task for task in pending if task[0] == 3 and task[1] == 38][:1]
            if not pending:
                raise RuntimeError("No pending smoke point")

        mode_count = 1 if (
            args.redo_low or args.redo_lowdep or args.redo_mu
        ) else 2
        print(
            "Starting {} mass-lifetime points ({} mode{} each)".format(
                len(pending), mode_count, "" if mode_count == 1 else "s"
            )
        )
        failures = []
        for count, future in enumerate(
            as_completed([executor.submit(run_point, task) for task in pending]), start=1
        ):
            result = future.result()
            if result["ok"]:
                store_result(
                    state, result, redo_low=args.redo_low,
                    redo_lowdep=args.redo_lowdep, redo_mu=args.redo_mu,
                )
            else:
                failures.append(result)
            if (not args.smoke) and (
                count % args.checkpoint_every == 0 or count == len(pending)
            ):
                save_state(state)
            if count % args.progress_every == 0 or count == len(pending):
                print(
                    "[{}/{}] completed={} failures={}".format(
                        count, len(pending), int(state["completed"].sum()), len(failures)
                    ),
                    flush=True,
                )
        if failures:
            failure_file = OUT / "failures.log"
            failure_file.write_text(
                "\n\n".join(
                    "i={i} j={j}\n{error}".format(**failure) for failure in failures
                )
            )
            raise RuntimeError(
                "{} points failed; see {}".format(len(failures), failure_file)
            )
    return state


def safe_ratio(numerator, denominator, absolute_floor=0.0):
    result = np.full_like(numerator, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > max(absolute_floor, 1.0e-300))
    )
    result[valid] = np.abs(numerator[valid]) / np.abs(denominator[valid])
    return result


def write_products(state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    complete = state["completed"] == 1
    if not np.all(complete):
        raise RuntimeError("Cannot plot incomplete scan: {} missing".format((~complete).sum()))

    if not np.all(np.isfinite(state["lowdep_y"])):
        raise RuntimeError(
            "Direct low-deposition exact-y pass is incomplete; run with "
            "--resume --redo-lowdep"
        )
    if not np.all(np.isfinite(state["lowdep_nony_l1"])):
        raise RuntimeError("Low-deposition non-y pass is incomplete; run with --redo-lowdep")
    if not np.all(np.isfinite(state["full_mu"])):
        raise RuntimeError("Baseline-subtracted mu pass is incomplete; run with --redo-mu")
    if not np.isfinite(float(state["C_y_l1"])) or not np.isfinite(
        float(state["C_mu_l1"])
    ):
        raise RuntimeError("The y/mu FOSSIL-band calibration is missing; run --redo-mu")
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    lowdep_y = state["lowdep_y"]
    highz_y = state["full_y"] - lowdep_y
    y_fraction = safe_ratio(lowdep_y, state["full_y"])
    mu_fraction = safe_ratio(state["lowdep_mu"], state["full_mu"])
    nony_fraction = safe_ratio(state["lowdep_nony_l1"], state["full_nony_l1"])
    injection_y_fraction = safe_ratio(
        state["low_y"], state["full_y"]
    )
    injection_nony_fraction = safe_ratio(
        state["low_nony_l1"], state["full_nony_l1"]
    )
    log_tau = np.log10(state["taus_s"])
    log_mass = np.log10(state["masses_GeV"] * 1.0e3)
    unreliable = state["removed_fraction_proxy"] >= DYNAMIC_REMOVAL_THRESHOLD
    c_y_l1 = float(state["C_y_l1"])
    c_mu_l1 = float(state["C_mu_l1"])
    mu_l1 = np.abs(state["full_mu"]) * c_mu_l1
    y_l1 = np.abs(state["full_y"]) * c_y_l1
    nony_l1 = np.abs(state["full_nony_l1"])
    channel_l1 = np.stack([mu_l1, y_l1, nony_l1], axis=0)
    low_channel_l1 = np.stack(
        [
            np.abs(state["lowdep_mu"]) * c_mu_l1,
            np.abs(lowdep_y) * c_y_l1,
            np.abs(state["lowdep_nony_l1"]),
        ],
        axis=0,
    )
    total_l1 = np.sum(channel_l1, axis=0)
    low_total_l1 = np.sum(low_channel_l1, axis=0)
    total_low_fraction = safe_ratio(low_total_l1, total_l1)

    def panel_plot(
        total, fraction, total_label, total_unit, stem, total_valid,
        ratio_note=None,
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharex=True, sharey=True)
        positive = total[
            total_valid & np.isfinite(total) & (np.abs(total) > 0.0)
        ]
        total_norm = LogNorm(
            vmin=max(np.nanpercentile(np.abs(positive), 5), 1.0e-300),
            vmax=np.nanpercentile(np.abs(positive), 98),
        )
        first = axes[0].pcolormesh(
            log_tau, log_mass, np.abs(total), shading="nearest",
            cmap="magma", norm=total_norm,
        )
        positive_fraction = fraction[np.isfinite(fraction) & (fraction > 0.0)]
        fraction_vmax = max(1.0, np.nanpercentile(positive_fraction, 99))
        second = axes[1].pcolormesh(
            log_tau, log_mass, np.clip(fraction, 1.0e-5, fraction_vmax),
            shading="nearest", cmap="viridis",
            norm=LogNorm(1.0e-5, fraction_vmax),
        )
        for axis in axes:
            hatch = np.ma.masked_where(~unreliable, np.ones_like(unreliable))
            axis.pcolor(
                log_tau, log_mass, hatch, shading="auto", facecolor="none",
                edgecolor="0.35", linewidth=0.0, hatch="////",
            )
            axis.set_xlabel(r"$\log_{10}(\tau\,[\mathrm{s}])$")
            axis.set_xlim(log_tau[0], log_tau[-1])
            axis.set_ylim(log_mass[0], log_mass[-1])
        axes[0].set_ylabel(r"$\log_{10}(m_\chi\,[\mathrm{MeV}])$")
        axes[0].set_title(total_label + " at the adopted CMB fraction limit")
        axes[1].set_title(
            r"produced at $1+z<4$ (including delayed deposition)"
        )
        if ratio_note:
            axes[1].text(
                0.02, 0.98, ratio_note, transform=axes[1].transAxes,
                ha="left", va="top", fontsize=8,
            )
        cbar_total = figure.colorbar(first, ax=axes[0], pad=0.02)
        cbar_total.set_label(total_unit)
        cbar_fraction = figure.colorbar(second, ax=axes[1], pad=0.02)
        cbar_fraction.set_label(r"$|X_{1+z<4}|/|X_{\rm full}|$")
        figure.suptitle(
            r"$\chi\to e^+e^-$, Puchwein reionization; hatching: $f_\chi(1-e^{-t_0/\tau})\geq1\%$"
        )
        figure.tight_layout()
        figure.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
        figure.savefig(str(stem) + ".pdf", bbox_inches="tight")
        plt.close(figure)

    panel_plot(
        state["full_y"], y_fraction, r"total exotic $y$", r"$|\Delta y|$",
        FIG_Y, ~unreliable,
    )
    panel_plot(
        state["full_nony_l1"], nony_fraction,
        r"frequency-integrated non-$y$", r"$\sum |\Delta I_{\rm non-y}|\,\Delta\nu$ [Jy sr$^{-1}$ GHz]",
        FIG_NONY, ~unreliable,
    )

    def format_scan_axis(axis, show_xlabel=True, show_ylabel=True):
        axis.set_xlim(log_tau[0], log_tau[-1])
        axis.set_ylim(log_mass[0], log_mass[-1])
        axis.set_xticks(np.arange(9.0, 27.1, 3.0))
        mass_ticks = np.arange(0.0, 6.1, 1.0)
        axis.set_yticks(mass_ticks)
        axis.set_yticklabels(
            ["1 MeV", "10 MeV", "100 MeV", "1 GeV", "10 GeV", "100 GeV", "1 TeV"]
        )
        if show_xlabel:
            axis.set_xlabel(r"$\log_{10}(\tau\,[\mathrm{s}])$")
        if show_ylabel:
            axis.set_ylabel(r"decaying-particle mass $m_\chi$")
        axis.grid(color="white", linewidth=0.35, alpha=0.22)

    def hatch_mask(axis, mask, hatch, color="0.2"):
        if np.any(mask):
            axis.contourf(
                log_tau,
                log_mass,
                mask.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=[hatch],
            )
            # Matplotlib applies hatch colour through the collection edge.
            for collection in axis.collections[-1:]:
                collection.set_edgecolor(color)
                collection.set_linewidth(0.0)

    # One scale for all three rows makes the absolute channel amplitudes
    # directly comparable.  These are all matched-baseline-subtracted signals.
    positive_channels = channel_l1[
        np.isfinite(channel_l1) & (channel_l1 > 0.0)
    ]
    channel_vmin = max(
        1.0e-4,
        10.0 ** np.floor(np.log10(np.nanpercentile(positive_channels, 1.0))),
    )
    channel_vmax = 10.0 ** np.ceil(
        np.log10(np.nanpercentile(positive_channels, 99.5))
    )
    channel_norm = LogNorm(channel_vmin, channel_vmax)
    fraction_norm = Normalize(0.0, 1.0)
    channel_names = [r"$\mu$", r"$y$", r"non-$y$"]
    low_fractions = [mu_fraction, y_fraction, nony_fraction]

    figure, axes = plt.subplots(
        2, 3, figsize=(13.2, 7.4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    top_mesh = None
    bottom_mesh = None
    for column, name in enumerate(channel_names):
        top_mesh = axes[0, column].pcolormesh(
            log_tau,
            log_mass,
            np.ma.masked_where(channel_l1[column] <= 0.0, channel_l1[column]),
            shading="nearest",
            cmap="magma",
            norm=channel_norm,
            rasterized=True,
        )
        bottom_mesh = axes[1, column].pcolormesh(
            log_tau,
            log_mass,
            np.ma.masked_invalid(np.clip(low_fractions[column], 0.0, 1.0)),
            shading="nearest",
            cmap="viridis",
            norm=fraction_norm,
            rasterized=True,
        )
        axes[0, column].set_title(name + " channel", fontweight="semibold")
        axes[1, column].set_title(
            r"fraction from $1+z<4$", fontsize=10, pad=5
        )
        if column > 0:
            axes[0, column].tick_params(labelleft=False)
            axes[1, column].tick_params(labelleft=False)
        format_scan_axis(
            axes[0, column], show_xlabel=False, show_ylabel=(column == 0)
        )
        format_scan_axis(
            axes[1, column], show_xlabel=True, show_ylabel=(column == 0)
        )
        hatch_mask(axes[0, column], unreliable, "////", color="0.25")
        hatch_mask(axes[1, column], unreliable, "////", color="0.25")

    axes[1, 0].text(
        0.5,
        0.5,
        r"zero: standard $\mu$ is routed above $z=3000$",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        bbox={"facecolor": "0.1", "edgecolor": "none", "alpha": 0.72, "pad": 4},
    )
    magnitude_bar = figure.colorbar(
        top_mesh, ax=axes[0, :], location="right", shrink=0.92, pad=0.015
    )
    magnitude_bar.set_label(
        r"$\|\Delta I\|_1$ [Jy sr$^{-1}$ GHz]"
    )
    magnitude_bar.set_ticks(10.0 ** np.arange(-4.0, 10.1, 2.0))
    fraction_bar = figure.colorbar(
        bottom_mesh, ax=axes[1, :], location="right", shrink=0.92,
        pad=0.015, extend="max",
    )
    fraction_bar.set_label(r"$\|\Delta I_{z<3}\|_1/\|\Delta I_{\rm all}\|_1$")
    figure.suptitle(
        r"Exotic-only distortion channels for $\chi\to e^+e^-$ at the adopted CMB limit",
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.035,
        (
            "Matched no-DM baseline subtracted; Puchwein reionization; "
            "complete low-z extension including delayed high-z injection. "
            "Hatching marks cells where the stable-DM background removes at least 1%. "
            "Fractions above one are clipped and indicate signed y cancellation."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.25",
    )
    figure.savefig(str(FIG_CHANNELS) + ".pdf", bbox_inches="tight")
    figure.savefig(str(FIG_CHANNELS) + ".png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    palette = ["#2B6CB0", "#D95F59", "#24964A"]
    dominance_cmap = ListedColormap(palette)
    dominance_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], dominance_cmap.N)
    finite_channels = np.isfinite(channel_l1).all(axis=0)
    dominant = np.full(total_l1.shape, np.nan)
    dominant[finite_channels] = np.argmax(
        channel_l1[:, finite_channels], axis=0
    )
    dominant_share = np.full(total_l1.shape, np.nan)
    valid_total = finite_channels & (total_l1 > 0.0)
    dominant_share[valid_total] = (
        np.max(channel_l1[:, valid_total], axis=0) / total_l1[valid_total]
    )
    detectable = total_l1 >= (FREQUENCY_GHZ.size * DELTA_NU_GHZ)

    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.7), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes[0].pcolormesh(
        log_tau,
        log_mass,
        np.ma.masked_invalid(dominant),
        cmap=dominance_cmap,
        norm=dominance_norm,
        shading="nearest",
        rasterized=True,
    )
    share_levels = [
        level for level in (0.6, 0.8)
        if np.nanmin(dominant_share) < level < np.nanmax(dominant_share)
    ]
    if share_levels:
        share_contours = axes[0].contour(
            log_tau,
            log_mass,
            dominant_share,
            levels=share_levels,
            colors="white",
            linewidths=[0.8, 1.25],
        )
        axes[0].clabel(
            share_contours,
            fmt={0.6: "60%", 0.8: "80%"},
            fontsize=7,
            inline=True,
        )
    low_mesh = axes[1].pcolormesh(
        log_tau,
        log_mass,
        np.ma.masked_invalid(np.clip(total_low_fraction, 0.0, 1.0)),
        cmap="viridis",
        norm=Normalize(0.0, 1.0),
        shading="nearest",
        rasterized=True,
    )
    axes[1].contour(
        log_tau,
        log_mass,
        dominant,
        levels=[0.5, 1.5],
        colors="white",
        linewidths=0.7,
        alpha=0.85,
    )
    for axis in axes:
        format_scan_axis(axis, show_xlabel=True, show_ylabel=(axis is axes[0]))
        hatch_mask(axis, ~detectable & finite_channels, "....", color="0.35")
        hatch_mask(axis, unreliable, "////", color="0.15")
        if np.nanmin(total_l1) <= FREQUENCY_GHZ.size * DELTA_NU_GHZ <= np.nanmax(total_l1):
            axis.contour(
                log_tau,
                log_mass,
                total_l1,
                levels=[FREQUENCY_GHZ.size * DELTA_NU_GHZ],
                colors="black",
                linewidths=1.25,
            )
    axes[0].set_title("Dominant exotic-only channel", fontweight="semibold")
    axes[1].set_title(
        r"Total FOSSIL-band share produced at $1+z<4$",
        fontweight="semibold",
    )
    low_bar = figure.colorbar(
        low_mesh, ax=axes[1], pad=0.02, shrink=0.91, extend="max"
    )
    low_bar.set_label(r"$\sum_c\|\Delta I_{c,z<3}\|_1/\sum_c\|\Delta I_c\|_1$")
    legend = [
        Patch(facecolor=palette[0], label=r"$\mu$ dominates"),
        Patch(facecolor=palette[1], label=r"$y$ dominates"),
        Patch(facecolor=palette[2], label=r"non-$y$ dominates"),
        Line2D([0], [0], color="black", linewidth=1.25, label="FOSSIL L1 threshold"),
        Patch(facecolor="white", edgecolor="0.35", hatch="....", label="below threshold"),
        Patch(facecolor="white", edgecolor="0.15", hatch="////", label="stable-DM approximation unsafe"),
    ]
    figure.legend(
        handles=legend,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.085),
        frameon=True,
        fontsize=8.5,
    )
    figure.suptitle(
        r"Which distortion channel dominates?  $\chi\to e^+e^-$ at the adopted CMB limit",
        fontsize=14,
        fontweight="semibold",
    )
    figure.savefig(str(FIG_DOMINANCE) + ".pdf", bbox_inches="tight")
    figure.savefig(str(FIG_DOMINANCE) + ".png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    with CSV.open("w", newline="") as stream:
        fields = [
            "mass_GeV", "lifetime_s", "f_cmb", "removed_fraction_proxy",
            "mu_full", "mu_full_l1_Jy_sr_GHz", "mu_low_zlt3",
            "mu_lowdep_fraction",
            "y_full", "y_high_zge3", "y_low_zlt3", "y_lowdep_fraction",
            "y_strict_low_injection", "y_strict_low_injection_fraction",
            "nony_full_l1_Jy_sr_GHz", "nony_lowdep_l1_Jy_sr_GHz",
            "nony_lowdep_fraction", "nony_strict_low_injection_l1_Jy_sr_GHz",
            "nony_strict_low_injection_fraction", "full_evaluation_fraction",
            "low_evaluation_fraction", "lowdep_evaluation_fraction",
            "total_l1_Jy_sr_GHz", "total_lowdep_l1_Jy_sr_GHz",
            "total_lowdep_fraction", "dominant_channel",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, mass in enumerate(state["masses_GeV"]):
            for j, tau in enumerate(state["taus_s"]):
                writer.writerow(
                    {
                        "mass_GeV": mass,
                        "lifetime_s": tau,
                        "f_cmb": state["f_cmb"][i, j],
                        "removed_fraction_proxy": state["removed_fraction_proxy"][i, j],
                        "mu_full": state["full_mu"][i, j],
                        "mu_full_l1_Jy_sr_GHz": mu_l1[i, j],
                        "mu_low_zlt3": state["lowdep_mu"][i, j],
                        "mu_lowdep_fraction": mu_fraction[i, j],
                        "y_full": state["full_y"][i, j],
                        "y_high_zge3": highz_y[i, j],
                        "y_low_zlt3": lowdep_y[i, j],
                        "y_lowdep_fraction": y_fraction[i, j],
                        "y_strict_low_injection": state["low_y"][i, j],
                        "y_strict_low_injection_fraction": injection_y_fraction[i, j],
                        "nony_full_l1_Jy_sr_GHz": state["full_nony_l1"][i, j],
                        "nony_lowdep_l1_Jy_sr_GHz": state["lowdep_nony_l1"][i, j],
                        "nony_lowdep_fraction": nony_fraction[i, j],
                        "nony_strict_low_injection_l1_Jy_sr_GHz": state["low_nony_l1"][i, j],
                        "nony_strict_low_injection_fraction": injection_nony_fraction[i, j],
                        "full_evaluation_fraction": state["full_evaluation_fraction"][i, j],
                        "low_evaluation_fraction": state["low_evaluation_fraction"][i, j],
                        "lowdep_evaluation_fraction": state["lowdep_evaluation_fraction"][i, j],
                        "total_l1_Jy_sr_GHz": total_l1[i, j],
                        "total_lowdep_l1_Jy_sr_GHz": low_total_l1[i, j],
                        "total_lowdep_fraction": total_low_fraction[i, j],
                        "dominant_channel": (
                            ["mu", "y", "non-y"][int(dominant[i, j])]
                            if np.isfinite(dominant[i, j]) else ""
                        ),
                    }
                )

    reliable_y = ~unreliable & np.isfinite(y_fraction)
    reliable_mu = ~unreliable & np.isfinite(mu_fraction)
    reliable_nony = ~unreliable & np.isfinite(nony_fraction)
    reliable_total = ~unreliable & np.isfinite(total_low_fraction)

    def ratio_summary(values, mask):
        selected = values[mask]
        maximum_index = np.nanargmax(np.where(mask, values, np.nan))
        mass_index, tau_index = np.unravel_index(maximum_index, values.shape)
        return {
            "cells": int(mask.sum()),
            "median": float(np.nanmedian(selected)),
            "p10": float(np.nanpercentile(selected, 10)),
            "p90": float(np.nanpercentile(selected, 90)),
            "max": float(values[mass_index, tau_index]),
            "max_mass_GeV": float(state["masses_GeV"][mass_index]),
            "max_lifetime_s": float(state["taus_s"][tau_index]),
            "fraction_above_0p1": float(np.mean(selected > 0.1)),
            "fraction_above_0p5": float(np.mean(selected > 0.5)),
            "fraction_above_0p9": float(np.mean(selected > 0.9)),
            "fraction_above_1": float(np.mean(selected > 1.0)),
        }

    summary = {
        "grid": [int(MASSES_GEV.size), int(TAUS_S.size)],
        "redshift_split_definition": "deposition/production redshift: y_low is the z<3 integral of the pointwise baseline-subtracted CLASS exact-y history; non-y_low = low-low + delayed high-injection/low-deposition bridge",
        "strict_injection_split_definition": "injection nodes with 1+z<4; excludes the high-injection delayed bridge and forces heating to zero at 1+z=4",
        "full_definition": "extend: high-high + low-low + high-injection/low-deposition delayed bridge",
        "nony_integral": "FOSSIL-band L1 norm on 50,65,...,1985 GHz",
        "channel_comparison_metric": "FOSSIL-band L1 norm for baseline-subtracted mu, y, and non-y components",
        "mu_definition": "matched-baseline CLASS heating projected onto the standard mu template only above the z=3000 CLASS/DarkHistory handoff; the transferred non-y spectrum supplies the lower-redshift cascade without double counting",
        "mu_low_redshift_definition": "exactly zero because the standard mu channel is routed only above z=3000",
        "y_integration": "direct CLASS exact-y integrand difference, including the y branching ratio and CLASS trapezoidal weights",
        "C_y_l1_Jy_sr_GHz": c_y_l1,
        "C_mu_l1_Jy_sr_GHz": c_mu_l1,
        "fraction_boundary": {
            "log10_tau_s": LOG_TAU_CMB.tolist(),
            "log10_f_chi": LOG_F_CMB.tolist(),
        },
        "reliable_cells_y": int(reliable_y.sum()),
        "reliable_cells_mu": int(reliable_mu.sum()),
        "reliable_cells_nony": int(reliable_nony.sum()),
        "reliable_cells_total": int(reliable_total.sum()),
        "mu_low_fraction": ratio_summary(mu_fraction, reliable_mu),
        "y_low_fraction": ratio_summary(y_fraction, reliable_y),
        "nony_low_fraction": ratio_summary(nony_fraction, reliable_nony),
        "total_low_fraction": ratio_summary(
            total_low_fraction, reliable_total
        ),
        "dominance_counts_reliable": {
            label: int(np.sum((dominant == index) & ~unreliable))
            for index, label in enumerate(("mu", "y", "non-y"))
        },
        "dominance_counts_all": {
            label: int(np.sum(dominant == index))
            for index, label in enumerate(("mu", "y", "non-y"))
        },
        "dominance_counts_detectable_reliable": {
            label: int(
                np.sum((dominant == index) & detectable & ~unreliable)
            )
            for index, label in enumerate(("mu", "y", "non-y"))
        },
        "y_strict_low_injection_fraction": ratio_summary(
            injection_y_fraction, ~unreliable & np.isfinite(injection_y_fraction)
        ),
        "nony_strict_low_injection_fraction": ratio_summary(
            injection_nony_fraction,
            ~unreliable & np.isfinite(injection_nony_fraction),
        ),
        "rescaled_full_cells": int(np.sum(state["full_evaluation_fraction"] < state["f_cmb"])),
        "rescaled_low_cells": int(np.sum(state["low_evaluation_fraction"] < state["f_cmb"])),
        "rescaled_lowdep_cells": int(np.sum(state["lowdep_evaluation_fraction"] < state["f_cmb"])),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(str(FIG_CHANNELS) + ".pdf")
    print(str(FIG_DOMINANCE) + ".pdf")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classy-path", default=str(CLASSY_DEFAULT))
    parser.add_argument(
        "--detector-cache", default=DETECTOR_CACHE_DEFAULT,
        help=(
            "130-bin FOSSIL PCA cache (or set FOSSIL_DETECTOR_CACHE)"
        ),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--redo-low", action="store_true",
        help="preserve completed full results and recompute the strict 1+z<4 numerator",
    )
    parser.add_argument(
        "--redo-lowdep", action="store_true",
        help=(
            "preserve full results and recompute the z<3 y integral directly "
            "from exact_y plus the low-deposition non-y piece"
        ),
    )
    parser.add_argument(
        "--redo-mu", action="store_true",
        help=(
            "preserve the y/non-y cache and recompute the baseline-subtracted "
            "full mu amplitude"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if sum((args.redo_low, args.redo_lowdep, args.redo_mu)) > 1:
        raise RuntimeError("Choose only one targeted recomputation pass")
    if args.workers < 1 or args.workers > 3:
        raise RuntimeError("Use one to three workers")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        state = load_state()
    else:
        state = run_scan(args)
        if args.smoke:
            print("Smoke test complete")
            return
    write_products(state)


if __name__ == "__main__":
    main()
