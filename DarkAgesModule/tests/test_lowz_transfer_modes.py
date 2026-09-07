"""Unit tests for joining the low- and high-redshift transfer tables."""

from __future__ import division

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np


DARKAGES_BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("DARKAGES_BASE", str(DARKAGES_BASE))
sys.path.insert(0, str(DARKAGES_BASE))

from DarkAges.lowz import (  # noqa: E402
    ELECTRON_MASS_EV,
    LOWZ_GENERATOR_DLOG1PZ,
    add_spectral_distortions,
    combine_lowz_heat_results,
    electron_kinetic_energy_fraction,
    low_high_masks,
    normalize_lowz_transfer_mode,
    splice_deposition_channels,
    validate_legacy_compatible_heat_table,
    validate_lowz_transfer_blocks,
)
from DarkAges.transfer import transfer  # noqa: E402
from DarkAges.recipes import spec_elec_and_phot  # noqa: E402
from DarkAges.common import finalize  # noqa: E402
import DarkAges.recipes as recipes  # noqa: E402
import DarkAges  # noqa: E402


class LowRedshiftTransferModesTest(unittest.TestCase):

    def test_lowz_loader_uses_summed_low_and_bridge_files(self):
        directory = DARKAGES_BASE / "transfer_functions/original/low-z"
        raw_low = transfer(str(
            directory / "tf_heat_eps-7_mass_scan_summed_lowz.dat"
        ))
        raw_bridge = transfer(str(
            directory / "tf_heat_eps-7_mass_scan_summed_highinj_lowdep.dat"
        ))
        loaded_low, loaded_bridge = DarkAges.get_lowz_heat_transfer_functions()
        for loaded, raw in (
            (loaded_low, raw_low),
            (loaded_bridge, raw_bridge),
        ):
            np.testing.assert_array_equal(loaded.z_deposited, raw.z_deposited)
            np.testing.assert_array_equal(loaded.z_injected, raw.z_injected)
            np.testing.assert_array_equal(loaded.transfer_elec, raw.transfer_elec)
            np.testing.assert_array_equal(loaded.transfer_phot, raw.transfer_phot)

    def test_lowz_generator_normalization_metadata(self):
        self.assertEqual(LOWZ_GENERATOR_DLOG1PZ, 0.016)
        log10_energy = np.log10(5000.0)
        expected = 5000.0 / (5000.0 + ELECTRON_MASS_EV)
        self.assertAlmostEqual(
            float(electron_kinetic_energy_fraction(log10_energy)), expected
        )

    def test_complete_lowz_block_matrix_is_legacy_cell_compatible(self):
        low, bridge = DarkAges.get_lowz_heat_transfer_functions()
        high = DarkAges.transfer_functions[DarkAges.channel_dict["Heat"]]
        self.assertTrue(validate_lowz_transfer_blocks(low, bridge, high))
        np.testing.assert_allclose(bridge.z_injected, high.z_injected)
        positions = np.searchsorted(bridge.z_deposited, low.z_deposited)
        np.testing.assert_allclose(
            bridge.z_deposited[positions], low.z_deposited
        )

    def test_high_injection_bridge_is_convolved_on_low_deposition_grid(self):
        low, bridge = DarkAges.get_lowz_heat_transfer_functions()
        _, bridge_mask = low_high_masks(
            low.z_injected, bridge.z_injected, "extend-new"
        )
        for particle in ("dirac_electron", "dirac_photon"):
            injection_model = spec_elec_and_phot(
                [particle], 0.448,
                redshift=bridge.z_injected, t_dec=1.2e25,
                hist="decay", branchings=np.ones(1),
            )
            deposition_model = spec_elec_and_phot(
                [particle], 0.448,
                redshift=bridge.z_deposited, t_dec=1.2e25,
                hist="decay", branchings=np.ones(1),
            )

            # A bridge has 52 injection columns but only 13 deposition rows,
            # so its denominator must be supplied on the separate deposition
            # grid for either injected species.
            self.assertNotEqual(
                injection_model.normalization.shape,
                deposition_model.normalization.shape,
            )

            bridge_heat = injection_model.calc_f(
                bridge,
                injection_mask=bridge_mask,
                deposition_normalization=deposition_model.normalization,
            )[-1]
            self.assertTrue(np.all(np.isfinite(bridge_heat)))
            self.assertGreater(np.count_nonzero(bridge_heat), 0)

    def test_descending_bridge_file_is_loaded_by_coordinate(self):
        path = DARKAGES_BASE / (
            "transfer_functions/original/low-z/"
            "tf_heat_eps-7_mass_scan_summed_highinj_lowdep.dat"
        )
        raw = np.loadtxt(str(path), comments="#")
        # The producer writes high injection redshift in decreasing order.
        self.assertGreater(raw[0, 2], raw[1, 2])
        row = raw[np.flatnonzero((raw[:, 3] != 0.0) | (raw[:, 4] != 0.0))[0]]
        bridge = DarkAges.get_lowz_heat_transfer_functions()[1]
        dep_index = np.where(bridge.z_deposited == row[0])[0][0]
        energy_index = np.where(bridge.log10E == row[1])[0][0]
        injection_index = np.where(bridge.z_injected == row[2])[0][0]
        self.assertEqual(
            bridge.transfer_elec[dep_index, energy_index, injection_index],
            row[3],
        )
        self.assertEqual(
            bridge.transfer_phot[dep_index, energy_index, injection_index],
            row[4],
        )

    def test_unsupported_endpoint_rows_are_not_returned(self):
        low, bridge = DarkAges.get_lowz_heat_transfer_functions()
        high = DarkAges.transfer_functions[DarkAges.channel_dict["Heat"]]
        low_mask, bridge_mask = low_high_masks(
            low.z_injected, high.z_injected, "extend-new"
        )
        low_nonzero = np.any(
            np.abs(low.transfer_elec[:, :, low_mask])
            + np.abs(low.transfer_phot[:, :, low_mask]) > 0.0,
            axis=(1, 2),
        ).astype(float)
        bridge_nonzero = np.any(
            np.abs(bridge.transfer_elec[:, :, bridge_mask])
            + np.abs(bridge.transfer_phot[:, :, bridge_mask]) > 0.0,
            axis=(1, 2),
        ).astype(float)
        redshift, _ = combine_lowz_heat_results(
            low, low_nonzero, bridge, bridge_nonzero,
            low_mask, bridge_mask,
        )
        self.assertEqual(redshift[0], 1.1366)
        self.assertEqual(redshift[-1], 4.0878)

    def test_active_highz_heat_file_is_the_new_table(self):
        expected = transfer(str(
            DARKAGES_BASE
            / "transfer_functions/original/new_tfs/"
            / "tf_test_heat_eps-7_mass_scan_summed_negdist.dat"
        ))
        active = DarkAges.transfer_functions[DarkAges.channel_dict["Heat"]]
        np.testing.assert_array_equal(active.transfer_elec, expected.transfer_elec)
        np.testing.assert_array_equal(active.transfer_phot, expected.transfer_phot)

    def test_active_summed_integrated_heat_values(self):
        """Lock the production summed heating tables."""

        high = DarkAges.transfer_functions[DarkAges.channel_dict["Heat"]]
        low, bridge = DarkAges.get_lowz_heat_transfer_functions()
        target = np.asarray([6.02, 7.88, 10.21, 12.07])

        def indices(table):
            return np.asarray([
                int(np.argmin(np.abs(table.log10E - value)))
                for value in target
            ])

        high_fraction = np.sum(high.transfer_elec[:, indices(high), :], axis=0)
        low_fraction = np.sum(low.transfer_elec[:, indices(low), :], axis=0)
        bridge_fraction = np.sum(
            bridge.transfer_elec[:, indices(bridge), :], axis=0
        )

        np.testing.assert_allclose(
            low_fraction[:, -1],
            [0.305069, 0.00898585, 0.000761233, 0.00001285502],
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            high_fraction[:, 0],
            [0.43212, 0.52054, 0.0072826, 0.00011017],
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            bridge_fraction[:, 0],
            [0.0003127468, 0.000006634696, 0.0004569432, 0.00000838347],
            rtol=1.0e-12,
        )
        self.assertAlmostEqual(float(np.max(high_fraction[1])), 1.12363)

    def test_active_lowz_heat_has_no_global_factor_of_two(self):
        low, _ = DarkAges.get_lowz_heat_transfer_functions()
        kinetic_budget = electron_kinetic_energy_fraction(low.log10E)
        integrated_heat = np.sum(low.transfer_elec, axis=0)
        budget_ratio = integrated_heat / kinetic_budget[:, None]
        self.assertAlmostEqual(float(np.max(budget_ratio)), 1.0383252570358215)
        self.assertLess(float(np.max(budget_ratio)), 1.05)

    def test_draft_figure7_unsummed_integrated_heat_values(self):
        """Lock the unsummed producer curves used in draft Figure 7."""

        root = DARKAGES_BASE / "transfer_functions/original"
        high = transfer(str(
            root / "new_tfs/tf_test_heat_eps-7_mass_scan_negdist.dat"
        ))
        low = transfer(str(
            root / "low-z/tf_heat_eps-7_mass_scan_lowz.dat"
        ))
        bridge = transfer(str(
            root / "low-z/tf_heat_eps-7_mass_scan_highinj_lowdep.dat"
        ))
        target = np.asarray([6.02, 7.88, 10.21, 12.07])

        def integrated_fraction(table):
            indices = np.asarray([
                int(np.argmin(np.abs(table.log10E - value)))
                for value in target
            ])
            return np.sum(table.transfer_elec[:, indices, :], axis=0)

        high_fraction = integrated_fraction(high)
        low_fraction = integrated_fraction(low)
        bridge_fraction = integrated_fraction(bridge)

        np.testing.assert_allclose(
            low_fraction[:, -1],
            [0.3311344, 0.0096426779, 0.00079720842, 0.0000134071082],
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            high_fraction[:, 0],
            [0.432162414, 0.579417, 0.008402704, 0.0001295841],
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            bridge_fraction[:, 0],
            [0.00038860991, 0.0000082771902, 0.00053639476, 0.0000096427351],
            rtol=1.0e-12,
        )
        self.assertAlmostEqual(float(np.max(high_fraction[1])), 1.167575)

    def test_active_legacy_heat_table_satisfies_its_own_cell_convention(self):
        legacy_table = DarkAges.transfer_functions[DarkAges.channel_dict["Heat"]]
        self.assertTrue(
            validate_legacy_compatible_heat_table(legacy_table, legacy_table)
        )

    def test_residual_bridge_uses_the_high_injection_grid(self):
        low, bridge = DarkAges.get_lowz_spectral_distortions_transfer_functions()
        high = DarkAges.spectral_distortions_functions
        np.testing.assert_allclose(bridge.z_injected, high.z_injected)
        np.testing.assert_allclose(low.E_injected, bridge.E_injected)
        self.assertTrue(np.any(bridge.spectral_distortions_elec != 0.0))
        self.assertTrue(np.any(bridge.spectral_distortions_phot != 0.0))

    def test_mode_aliases_and_validation(self):
        self.assertEqual(normalize_lowz_transfer_mode(False), "legacy")
        self.assertEqual(normalize_lowz_transfer_mode(True), "extend")
        self.assertEqual(normalize_lowz_transfer_mode("extend_new"), "extend-new")
        self.assertEqual(normalize_lowz_transfer_mode("low_only"), "low-only")
        self.assertEqual(
            normalize_lowz_transfer_mode("low_below_four"), "low-below-four"
        )
        with self.assertRaises(ValueError):
            normalize_lowz_transfer_mode("unknown")

    def test_extend_prefers_legacy_overlap(self):
        low = np.asarray([1.0, 2.0, 3.0, 4.0])
        high = np.asarray([3.0, 5.0, 7.0])
        low_mask, high_mask = low_high_masks(low, high, "extend")
        np.testing.assert_array_equal(low_mask, [True, True, False, False])
        np.testing.assert_array_equal(high_mask, [True, True, True])

    def test_extend_new_prefers_new_overlap(self):
        low = np.asarray([1.0, 2.0, 3.0, 4.0])
        high = np.asarray([3.0, 5.0, 7.0])
        low_mask, high_mask = low_high_masks(low, high, "extend-new")
        np.testing.assert_array_equal(low_mask, [True, True, True, True])
        np.testing.assert_array_equal(high_mask, [False, True, True])

    def test_low_only_selects_only_low_injections(self):
        low = np.asarray([1.0, 2.0, 3.0, 4.0])
        high = np.asarray([3.0, 5.0, 7.0])
        low_mask, high_mask = low_high_masks(low, high, "low-only")
        np.testing.assert_array_equal(low_mask, [True, True, True, True])
        np.testing.assert_array_equal(high_mask, [False, False, False])

    def test_low_below_four_applies_strict_injection_node_cut(self):
        low = np.asarray([1.0, 2.0, 3.5, 4.0, 4.5])
        high = np.asarray([3.0, 5.0, 7.0])
        low_mask, high_mask = low_high_masks(low, high, "low-below-four")
        np.testing.assert_array_equal(low_mask, [True, True, True, False, False])
        np.testing.assert_array_equal(high_mask, [False, False, False])

    def test_splice_sets_unsupported_lowz_channels_to_zero(self):
        low_z = np.asarray([1.0, 2.0, 3.0])
        low_heat = np.asarray([0.1, 0.2, 0.3])
        high_z = np.asarray([3.0, 4.0])
        high_channels = np.arange(10, dtype=np.float64).reshape(5, 2)
        joined_z, joined = splice_deposition_channels(
            low_z, low_heat, high_z, high_channels, "extend"
        )
        np.testing.assert_array_equal(joined_z, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(joined[:3, :2], np.zeros((3, 2)))
        np.testing.assert_array_equal(joined[4, :2], np.zeros(2))
        np.testing.assert_allclose(joined[3, :2], [0.1, 0.2])
        np.testing.assert_array_equal(joined[:, 2:], high_channels)

    def test_low_only_splice_has_zero_high_redshift_tail(self):
        low_z = np.asarray([1.0, 2.0, 3.0])
        low_heat = np.asarray([0.1, 0.2, 0.3])
        high_z = np.asarray([3.0, 4.0, 5.0])
        high_channels = np.arange(15, dtype=np.float64).reshape(5, 3)
        joined_z, joined = splice_deposition_channels(
            low_z, low_heat, high_z, high_channels, "low-only"
        )
        np.testing.assert_array_equal(joined_z, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(joined[3, :3], low_heat)
        np.testing.assert_array_equal(joined[:, 3:], np.zeros((5, 2)))

    def test_low_below_four_splice_inserts_zero_at_cutoff(self):
        low_z = np.asarray([1.0, 2.0, 3.5])
        low_heat = np.asarray([0.1, 0.2, 0.3])
        high_z = np.asarray([4.5, 5.0])
        high_channels = np.arange(10, dtype=np.float64).reshape(5, 2)
        joined_z, joined = splice_deposition_channels(
            low_z, low_heat, high_z, high_channels, "low-below-four"
        )
        np.testing.assert_array_equal(joined_z, [1.0, 2.0, 3.5, 4.0, 4.5, 5.0])
        np.testing.assert_allclose(joined[3, :3], low_heat)
        np.testing.assert_array_equal(joined[:, 3:], np.zeros((5, 3)))

    def test_finalize_advertises_lowz_interpolation_handoff(self):
        redshift = np.asarray([1.1, 2.0, 4.39])
        heat = np.asarray([0.1, 0.2, 0.7])
        zeros = np.zeros_like(heat)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            finalize(
                redshift, heat, zeros, zeros, zeros, zeros,
                first_index=0,
                lowz_interpolation_handoff_z=3.3858,
            )
        self.assertIn(
            "# lowz_interpolation_handoff_z = 3.39e+00",
            output.getvalue(),
        )

    def test_extended_recipe_marks_first_highz_output_row(self):
        captured = {}

        def capture(*args, **options):
            del args
            captured.update(options)

        original_finalize = recipes.finalize
        recipes.finalize = capture
        try:
            recipes.loading_from_specfiles(
                ["dirac_electron"],
                DarkAges.transfer_functions,
                DarkAges.spectral_distortions_functions,
                0.45,
                t_dec=1.2e25,
                hist="decay",
                branchings=np.ones(1),
                lowz_transfer_mode="extend",
            )
        finally:
            recipes.finalize = original_finalize

        high_start = DarkAges.transfer_functions[
            DarkAges.channel_dict["Heat"]
        ].z_deposited[0] - 1.0
        self.assertAlmostEqual(
            captured["lowz_interpolation_handoff_z"], high_start
        )

    def test_spectral_distortions_are_added_on_log_frequency(self):
        high_frequency = np.asarray([1.0, np.sqrt(10.0), 10.0, 100.0, 1000.0])
        high_distortion = np.ones(5)
        low_frequency = np.asarray([1.0, 10.0, 100.0])
        low_distortion = np.asarray([0.0, 2.0, 4.0])
        result = add_spectral_distortions(
            high_frequency, high_distortion, low_frequency, low_distortion
        )
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 5.0, 1.0])


if __name__ == "__main__":
    unittest.main()
