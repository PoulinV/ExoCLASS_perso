"""Regression tests for decay survival in the fast DarkAges calculation."""

from __future__ import division

import os
import sys
import unittest
from pathlib import Path

import numpy as np


DARKAGES_BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("DARKAGES_BASE", str(DARKAGES_BASE))
sys.path.insert(0, str(DARKAGES_BASE))

from DarkAges.common import time_at_z  # noqa: E402
from DarkAges.model import decaying_model  # noqa: E402
from DarkAges.spectral_distortions import spectral_distortion_today  # noqa: E402


class SyntheticTransfer(object):
    """One-energy transfer table with user-specified nonzero entries."""

    def __init__(self, redshift, entries):
        size = len(redshift)
        self.z_injected = np.asarray(redshift, dtype=np.float64)
        self.z_deposited = np.asarray(redshift, dtype=np.float64)
        self.log10E = np.asarray([6.0], dtype=np.float64)
        self.transfer_phot = np.zeros((size, 1, size), dtype=np.float64)
        self.transfer_elec = np.zeros((size, 1, size), dtype=np.float64)
        for dep_index, inj_index, value in entries:
            self.transfer_elec[dep_index, 0, inj_index] = value


def make_model(redshift, lifetime):
    return decaying_model(
        ref_el_spec=np.asarray([1.0]),
        ref_ph_spec=np.asarray([0.0]),
        ref_oth_spec=np.asarray([0.0]),
        m=1.0e6,
        t_dec=lifetime,
        logEnergies=np.asarray([6.0]),
        redshift=np.asarray(redshift, dtype=np.float64),
        normalize_spectrum_by="mass",
    )


class DecayDepletionTest(unittest.TestCase):

    def test_cosmic_age_includes_lambda_at_low_redshift(self):
        seconds_per_gyr = 365.25 * 24.0 * 3600.0 * 1.0e9
        self.assertAlmostEqual(
            time_at_z(1.0) / seconds_per_gyr, 13.81, places=2
        )

    def test_prompt_raw_efficiency_contains_one_survival_factor(self):
        redshift = np.asarray([100.0, 200.0, 400.0])
        lifetime = time_at_z(redshift[0]) / (-np.log(0.4))
        finite = make_model(redshift, lifetime)
        stable = make_model(redshift, np.inf)
        transfer = SyntheticTransfer(
            redshift,
            [(index, index, 1.0) for index in range(len(redshift))],
        )

        raw_ratio = finite.calc_f(transfer)[-1] / stable.calc_f(transfer)[-1]
        survival = np.exp(-time_at_z(redshift) / lifetime)

        # DarkAges returns deposition divided by the undepleted reference
        # rate. CLASS must therefore multiply this table by that reference
        # rate, rather than by a second survival factor.
        np.testing.assert_allclose(raw_ratio, survival, rtol=2e-13, atol=0.0)
        self.assertFalse(np.allclose(raw_ratio, survival**2, rtol=1e-4, atol=0.0))

    def test_delayed_deposition_remains_finite_after_rate_underflow(self):
        redshift = np.asarray([4.3858, 3000.0])
        transfer = SyntheticTransfer(redshift, [(0, 1, 1.0), (1, 1, 1.0)])
        lifetime = time_at_z(redshift[1])
        model = make_model(redshift, lifetime)
        raw_efficiency = model.calc_f(transfer)[-1]
        survival = np.exp(-time_at_z(redshift) / lifetime)

        self.assertEqual(survival[0], 0.0)
        self.assertTrue(np.all(np.isfinite(raw_efficiency)))
        self.assertGreater(raw_efficiency[0], 0.0)

        very_short = make_model(redshift, 1.0e4).calc_f(transfer)[-1]
        self.assertTrue(np.all(np.isfinite(very_short)))

    def test_residual_distortion_keeps_one_survival_factor(self):
        redshift = np.asarray([100.0, 200.0])
        lifetime = time_at_z(redshift[0]) / (-np.log(0.4))
        finite = make_model(redshift, lifetime)
        survival = np.exp(-time_at_z(redshift) / lifetime)

        frequency = np.asarray([100.0])
        transfer_energy = np.asarray([5.0e5, 1.5e6])
        sd_elec = np.zeros((2, 1, 2), dtype=np.float64)
        sd_phot = np.zeros_like(sd_elec)
        sd_elec[0, 0, :] = 1.0

        common = dict(
            frequency=frequency,
            z_injected=redshift,
            E_injected=finite.logEnergies,
            transfer_functions_E=transfer_energy,
            spectral_distortions_phot=sd_phot,
            spectral_distortions_elec=sd_elec,
            spec_phot=finite.spec_photons,
            hist="decay",
            normalization=finite.normalization,
            t_dec=lifetime,
            n_cdm=1.0,
        )
        finite_sd = spectral_distortion_today(
            spec_elec=finite.spec_electrons,
            **common
        )
        stable_sd = spectral_distortion_today(
            spec_elec=np.ones_like(finite.spec_electrons),
            **common
        )
        ratio = finite_sd / stable_sd

        np.testing.assert_allclose(
            ratio, np.asarray([survival[0]]), rtol=2e-13, atol=0.0
        )
        self.assertFalse(
            np.allclose(ratio, np.asarray([survival[0] ** 2]), rtol=1e-4, atol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
