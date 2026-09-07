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
y integrals with different ``sd_z_min`` values is used. The reported non-y
norm is the FOSSIL-band L1 integral
``sum |Delta I_non-y| Delta nu`` on 50,65,...,1985 GHz.
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

DETECTOR_CACHE_DEFAULT = os.environ.get("FOSSIL_DETECTOR_CACHE", "")
CLASSY_DEFAULT = ROOT / "python"

FREQUENCY_GHZ = 50.0 + 15.0 * np.arange(130)
DELTA_NU_GHZ = 15.0
LOW_INJECTION_CUTOFF = 4.0
LOW_DEPOSITION_Z_MAX = LOW_INJECTION_CUTOFF - 1.0
AGE_UNIVERSE_SECONDS_PROXY = 13.8e9 * 365.25 * 24.0 * 3600.0
DYNAMIC_REMOVAL_THRESHOLD = 1.0e-2

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
    "injected_particle_spectra": "dirac_electron",
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


def run_class(parameters):
    cosmology = _CLASS()
    try:
        cosmology.set(parameters)
        # With sd_only_exotic=yes, spectral distortions need the thermal
        # history but not perturbations, primordial spectra, transfer
        # functions, or angular spectra.
        cosmology.compute_distortions_only()
        amplitudes = cosmology.spectral_distortion_amplitudes()
        return {
            # CLASS orders the standard amplitudes as g, y, mu, residuals.
            "reported_y": float(amplitudes[1]),
            "y_history": cosmology.spectral_distortion_y_history(),
        }
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
    return mode_key, run_class(parameters)


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
            ["dirac_electron"], mass, logEnergies=log_energies,
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
    i, j, mass, tau, fraction, baselines, redo_low, redo_lowdep = task
    started = time.time()
    result = {"i": i, "j": j, "ok": False}
    try:
        if redo_lowdep:
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
            result[mode] = {
                "y": delta_y,
                "lowdep_y": lowdep_y,
                "reported_y_difference": reported_delta_y,
                "evaluation_fraction": evaluation_fraction,
                "attempts": attempts,
            }
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
        "schema_version": np.array(3, dtype=np.int64),
        "masses_GeV": MASSES_GEV,
        "taus_s": TAUS_S,
        "f_cmb": fraction_grid,
        "removed_fraction_proxy": removed,
        "completed": np.zeros(shape, dtype=np.uint8),
        "full_y": np.full(shape, np.nan),
        "low_y": np.full(shape, np.nan),
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
        "lowdep_y": np.full(shape, np.nan),
        "lowdep_nony_l1": np.full(shape, np.nan),
        "lowdep_nony_spectrum": np.full(shape + (FREQUENCY_GHZ.size,), np.nan),
        "lowdep_evaluation_fraction": np.full(shape, np.nan),
        "lowdep_attempts": np.zeros(shape, dtype=np.uint8),
        "run_seconds": np.full(shape, np.nan),
        "frequency_GHz": FREQUENCY_GHZ,
        "baseline_extend_y": np.array(baselines["extend"]["reported_y"]),
        "baseline_low_y": np.array(
            baselines["low-below-four"]["reported_y"]
        ),
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
        "schema_version": np.array(3, dtype=np.int64),
        "full_reported_y_difference": np.full(shape, np.nan),
        "low_reported_y_difference": np.full(shape, np.nan),
        # Deliberately do not derive this from the old full-minus-high-z
        # result: schema 3 requires the direct exact-y integrand calculation.
        "lowdep_y": np.full(shape, np.nan),
        "lowdep_nony_l1": np.full(shape, np.nan),
        "lowdep_nony_spectrum": np.full(spectral_shape, np.nan),
        "lowdep_evaluation_fraction": np.full(shape, np.nan),
        "lowdep_attempts": np.zeros(shape, dtype=np.uint8),
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
    state["schema_version"] = np.array(3, dtype=np.int64)
    return state


def store_result(state, result, redo_low=False, redo_lowdep=False):
    i, j = result["i"], result["j"]
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
        state[key + "_nony_l1"][i, j] = result[mode]["nony_l1"]
        state[key + "_nony_spectrum"][i, j] = result[mode]["nony_spectrum"]
        state[key + "_evaluation_fraction"][i, j] = result[mode]["evaluation_fraction"]
        state[key + "_attempts"][i, j] = result[mode]["attempts"]
        state[key + "_reported_y_difference"][i, j] = result[mode][
            "reported_y_difference"
        ]
    if not redo_low:
        state["lowdep_y"][i, j] = result["extend"]["lowdep_y"]
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
        if (args.redo_low or args.redo_lowdep) and not resuming:
            raise RuntimeError("A numerator-only pass requires --resume and an existing cache")
        if args.redo_lowdep:
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
            if args.redo_lowdep:
                state["baseline_extend_y"] = np.array(
                    baselines["extend-lowdep"]["reported_y"]
                )
            elif args.redo_low:
                state["baseline_low_y"] = np.array(
                    baselines["low-below-four"]["reported_y"]
                )
            else:
                state["baseline_extend_y"] = np.array(
                    baselines["extend"]["reported_y"]
                )
                state["baseline_low_y"] = np.array(
                    baselines["low-below-four"]["reported_y"]
                )
        else:
            state = empty_state(baselines)
            if not args.smoke:
                save_state(state)

        pending = []
        fractions = cmb_fraction(np.log10(TAUS_S))
        for i, mass in enumerate(MASSES_GEV):
            for j, tau in enumerate(TAUS_S):
                if args.redo_low or args.redo_lowdep or state["completed"][i, j] == 0:
                    pending.append(
                        (
                            i, j, float(mass), float(tau), float(fractions[j]),
                            baselines, bool(args.redo_low), bool(args.redo_lowdep),
                        )
                    )
        if args.smoke:
            pending = [task for task in pending if task[0] == 3 and task[1] == 38][:1]
            if not pending:
                raise RuntimeError("No pending smoke point")

        mode_count = 1 if (args.redo_low or args.redo_lowdep) else 2
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
                    redo_lowdep=args.redo_lowdep,
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
    from matplotlib.colors import LogNorm

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
    lowdep_y = state["lowdep_y"]
    highz_y = state["full_y"] - lowdep_y
    y_fraction = safe_ratio(lowdep_y, state["full_y"])
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

    with CSV.open("w", newline="") as stream:
        fields = [
            "mass_GeV", "lifetime_s", "f_cmb", "removed_fraction_proxy",
            "y_full", "y_high_zge3", "y_low_zlt3", "y_lowdep_fraction",
            "y_strict_low_injection", "y_strict_low_injection_fraction",
            "nony_full_l1_Jy_sr_GHz", "nony_lowdep_l1_Jy_sr_GHz",
            "nony_lowdep_fraction", "nony_strict_low_injection_l1_Jy_sr_GHz",
            "nony_strict_low_injection_fraction", "full_evaluation_fraction",
            "low_evaluation_fraction", "lowdep_evaluation_fraction",
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
                    }
                )

    reliable_y = ~unreliable & np.isfinite(y_fraction)
    reliable_nony = ~unreliable & np.isfinite(nony_fraction)

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
        "y_integration": "direct CLASS exact-y integrand difference, including the y branching ratio and CLASS trapezoidal weights",
        "fraction_boundary": {
            "log10_tau_s": LOG_TAU_CMB.tolist(),
            "log10_f_chi": LOG_F_CMB.tolist(),
        },
        "reliable_cells_y": int(reliable_y.sum()),
        "reliable_cells_nony": int(reliable_nony.sum()),
        "y_low_fraction": ratio_summary(y_fraction, reliable_y),
        "nony_low_fraction": ratio_summary(nony_fraction, reliable_nony),
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.redo_low and args.redo_lowdep:
        raise RuntimeError("Choose only one numerator-only pass")
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
