"""Normalization tests for the spectral-distortion transfer convolution."""

from __future__ import division

import os
import sys
import unittest
from pathlib import Path

import numpy as np


DARKAGES_BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("DARKAGES_BASE", str(DARKAGES_BASE))
sys.path.insert(0, str(DARKAGES_BASE))

from DarkAges.common import H  # noqa: E402
from DarkAges.spectral_distortions import spectral_distortion_today  # noqa: E402


class SpectralDistortionNormalizationTest(unittest.TestCase):

    frequency = np.asarray([100.0])
    redshift = np.asarray([10.0])
    n_cdm = 3.0
    lifetime = 7.0
    sigmav = 5.0

    def _dirac_result(self, hist, transfer_energy, electron=False):
        transfer_energy = np.asarray(transfer_energy, dtype=np.float64)
        injection_energy = np.asarray([3.0e4])
        shape = (1, 1, transfer_energy.size)
        phot_transfer = np.zeros(shape)
        elec_transfer = np.zeros(shape)
        if electron:
            elec_transfer[:] = 1.0
        else:
            phot_transfer[:] = 1.0
        spec_phot = np.zeros((1, 1))
        spec_elec = np.zeros((1, 1))
        if electron:
            spec_elec[:] = 2.0
        else:
            spec_phot[:] = 2.0
        return spectral_distortion_today(
            frequency=self.frequency,
            z_injected=self.redshift,
            E_injected=np.log10(injection_energy),
            transfer_functions_E=transfer_energy,
            spectral_distortions_phot=phot_transfer,
            spectral_distortions_elec=elec_transfer,
            spec_elec=spec_elec,
            spec_phot=spec_phot,
            hist=hist,
            normalization=np.ones(1),
            sigmav=self.sigmav,
            t_dec=self.lifetime,
            n_cdm=self.n_cdm,
        )[0]

    def _expected_dirac(self, hist):
        rs = self.redshift[0]
        if hist == "decay":
            event_rate = self.n_cdm * rs**3 / self.lifetime
        else:
            event_rate = 0.5 * (self.n_cdm * rs**3) ** 2 * self.sigmav
        # A two-particle Dirac spectrum represents one pair event.  For the
        # one-point redshift grid dln(1+z)=1 in the implementation.
        return event_rate / (H(rs) * rs**3) * 1.0e10

    def test_dirac_pairs_are_normalized_on_and_off_the_energy_grid(self):
        for hist in ("decay", "annihilation", "annihilation_halos"):
            expected = self._expected_dirac(hist)
            for electron in (False, True):
                on_grid = self._dirac_result(hist, [3.0e4], electron=electron)
                off_grid = self._dirac_result(
                    hist, [1.0e4, 1.0e5], electron=electron
                )
                self.assertAlmostEqual(on_grid / expected, 1.0, places=13)
                self.assertAlmostEqual(off_grid / expected, 1.0, places=13)

    def test_continuous_pair_spectrum_energy_and_log_energy_schemes(self):
        energy = np.asarray([1.0e4, 1.0e5, 1.0e6])
        log_energy = np.log10(energy)
        phot = np.asarray([2.0, 4.0, 6.0])[:, None]
        elec = np.asarray([1.0, 3.0, 5.0])[:, None]
        phot_transfer = np.ones((1, 1, energy.size))
        elec_transfer = 2.0 * np.ones_like(phot_transfer)
        rs = self.redshift[0]
        redshift_factor = (
            self.n_cdm * rs**3 / self.lifetime / (H(rs) * rs**3) * 1.0e10
        )

        energy_result = spectral_distortion_today(
            self.frequency, self.redshift, log_energy, energy,
            phot_transfer, elec_transfer, elec, phot, "decay", np.ones(1),
            t_dec=self.lifetime, n_cdm=self.n_cdm,
            E_integration_scheme="energy",
        )[0]
        expected_energy_integral = 0.5 * np.trapz(
            phot[:, 0] + 2.0 * elec[:, 0], energy
        )
        self.assertAlmostEqual(
            energy_result / (expected_energy_integral * redshift_factor),
            1.0,
            places=13,
        )

        off_grid_energy_result = spectral_distortion_today(
            self.frequency, self.redshift, log_energy, 1.1 * energy,
            phot_transfer, elec_transfer, elec, phot, "decay", np.ones(1),
            t_dec=self.lifetime, n_cdm=self.n_cdm,
            E_integration_scheme="energy",
        )[0]
        self.assertAlmostEqual(
            off_grid_energy_result / energy_result, 1.0, places=13
        )

        log_result = spectral_distortion_today(
            self.frequency, self.redshift, log_energy, energy,
            phot_transfer, elec_transfer, elec, phot, "decay", np.ones(1),
            t_dec=self.lifetime, n_cdm=self.n_cdm,
            E_integration_scheme="logE",
        )[0]
        log_integrand = (
            (phot[:, 0] + 2.0 * elec[:, 0])
            * energy
            / np.log10(np.e)
        )
        expected_log_integral = 0.5 * np.trapz(log_integrand, log_energy)
        self.assertAlmostEqual(
            log_result / (expected_log_integral * redshift_factor),
            1.0,
            places=13,
        )

    def test_energy_and_log_energy_quadratures_agree_on_a_dense_grid(self):
        energy = np.logspace(4.0, 6.0, 2001)
        log_energy = np.log10(energy)
        # dN/dE proportional to 1/E makes the transformed log-energy
        # integrand constant and gives a clean quadrature cross-check.
        phot = (2.0 / energy)[:, None]
        elec = np.zeros_like(phot)
        phot_transfer = np.ones((1, 1, energy.size))
        elec_transfer = np.zeros_like(phot_transfer)
        common = dict(
            frequency=self.frequency,
            z_injected=self.redshift,
            E_injected=log_energy,
            transfer_functions_E=energy,
            spectral_distortions_phot=phot_transfer,
            spectral_distortions_elec=elec_transfer,
            spec_elec=elec,
            spec_phot=phot,
            hist="decay",
            normalization=np.ones(1),
            t_dec=self.lifetime,
            n_cdm=self.n_cdm,
        )
        energy_result = spectral_distortion_today(
            E_integration_scheme="energy", **common
        )[0]
        log_result = spectral_distortion_today(
            E_integration_scheme="logE", **common
        )[0]
        np.testing.assert_allclose(energy_result, log_result, rtol=9.0e-7, atol=0.0)


if __name__ == "__main__":
    unittest.main()
