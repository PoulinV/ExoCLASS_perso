"""Helpers for joining the legacy and low-redshift transfer calculations."""

from __future__ import absolute_import, division, print_function

import numpy as np


LOWZ_TRANSFER_MODES = (
    "legacy", "extend", "extend-new", "low-only", "low-below-four"
)
LOW_INJECTION_CUTOFF = 4.0

# Historical metadata for the unsummed diagnostic files produced by
# DarkHistory's public ``full_ion.generate_lowz_tfs``.  The active consumer
# uses the separately supplied ``*_summed_lowz`` and
# ``*_summed_highinj_lowdep`` products, whose coarse cell width is 0.128.
LOWZ_GENERATOR_DLOG1PZ = 0.016
ELECTRON_MASS_EV = 510998.9461


def _representative_log_spacing(redshift):
    """Return the median logarithmic spacing of a strictly increasing grid."""

    redshift = np.asarray(redshift, dtype=np.float64)
    if redshift.ndim != 1 or redshift.size < 2:
        raise ValueError("A transfer redshift grid needs at least two points.")
    if np.any(redshift <= 0.0) or np.any(np.diff(redshift) <= 0.0):
        raise ValueError("Transfer redshifts must be positive and strictly increasing.")
    return float(np.median(np.diff(np.log(redshift))))


def legacy_compatible_heat_table_issues(low_transfer, legacy_transfer):
    """List reasons a heating table cannot use the legacy ``f_function``.

    ``f_function`` performs a plain sum over injection bins.  This is correct
    for the active legacy table because its injection and (summed) deposition
    grids have the same, constant ``dln(1+z)`` and each matrix element is a
    deposited-energy fraction in one deposition cell.  A genuine low-redshift
    extension must preserve those properties and must include the
    high-injection/low-deposition bridge.

    The historical unsummed 21-point diagnostic fails these checks because it
    contains point samples of a single-step heating-rate response.  The active
    ``*_summed_*`` products are checked with
    :func:`validate_lowz_transfer_blocks` instead.
    """

    issues = []
    low_dep = np.asarray(low_transfer.z_deposited, dtype=np.float64)
    low_inj = np.asarray(low_transfer.z_injected, dtype=np.float64)
    legacy_inj = np.asarray(legacy_transfer.z_injected, dtype=np.float64)

    if low_dep.shape != low_inj.shape or not np.allclose(
        low_dep, low_inj, rtol=2.0e-5, atol=0.0
    ):
        issues.append(
            "deposition and injection grids differ; the unchanged legacy "
            "convolution requires matching cell widths"
        )
    else:
        try:
            low_spacing = _representative_log_spacing(low_inj)
            legacy_spacing = _representative_log_spacing(legacy_inj)
            low_steps = np.diff(np.log(low_inj))
            if not np.allclose(low_steps, low_spacing, rtol=5.0e-3, atol=0.0):
                issues.append(
                    "redshift nodes are not uniformly spaced in ln(1+z) "
                    "(the supplied nodes are sparse linear-redshift samples)"
                )
            if not np.isclose(
                low_spacing, legacy_spacing, rtol=5.0e-3, atol=0.0
            ):
                issues.append(
                    "dln(1+z)={:.6g} does not match the legacy cell width "
                    "{:.6g}".format(low_spacing, legacy_spacing)
                )
        except ValueError as error:
            issues.append(str(error))

    low_energy = np.asarray(low_transfer.log10E, dtype=np.float64)
    legacy_energy = np.asarray(legacy_transfer.log10E, dtype=np.float64)
    if low_energy.shape != legacy_energy.shape or not np.allclose(
        low_energy, legacy_energy, rtol=0.0, atol=5.0e-5
    ):
        issues.append("electron/photon energy grids do not match the legacy table")

    # For z_dep below the handoff, the convolution still receives particles
    # injected anywhere on the legacy grid.  Without these columns, splicing
    # already-convolved f(z) curves drops the high-to-low bridge.
    if low_inj.size and legacy_inj.size and low_inj[-1] < legacy_inj[-1] * (1.0 - 1.0e-6):
        issues.append(
            "injection coverage stops at 1+z={:.6g}, below the legacy maximum "
            "{:.6g}; high-z injection to low-z deposition is missing".format(
                low_inj[-1], legacy_inj[-1]
            )
        )

    if (
        low_transfer.transfer_elec.ndim == 3
        and low_transfer.transfer_elec.shape[0] == low_transfer.transfer_elec.shape[2]
        and np.all(low_transfer.transfer_elec[-1, :, -1] == 0.0)
    ):
        issues.append(
            "the upper electron diagonal is identically zero (known exporter endpoint artifact)"
        )

    return tuple(issues)


def validate_legacy_compatible_heat_table(low_transfer, legacy_transfer):
    """Raise if ``low_transfer`` cannot be convolved like the legacy table."""

    issues = legacy_compatible_heat_table_issues(low_transfer, legacy_transfer)
    if issues:
        raise ValueError(
            "The low-z heating file is not a legacy-compatible transfer-cell "
            "table:\n - " + "\n - ".join(issues)
        )
    return True


def lowz_transfer_block_issues(low_transfer, bridge_transfer, high_transfer):
    """List inconsistencies in the three-block heating transfer matrix.

    The complete causal matrix is assembled from a low-injection/low-
    deposition block, a high-injection/low-deposition bridge, and the usual
    high-injection/high-deposition table.  The missing high-deposition/low-
    injection block is acausal and is therefore zero.
    """

    issues = []
    low_dep = np.asarray(low_transfer.z_deposited, dtype=np.float64)
    low_inj = np.asarray(low_transfer.z_injected, dtype=np.float64)
    bridge_dep = np.asarray(bridge_transfer.z_deposited, dtype=np.float64)
    bridge_inj = np.asarray(bridge_transfer.z_injected, dtype=np.float64)
    high_dep = np.asarray(high_transfer.z_deposited, dtype=np.float64)
    high_inj = np.asarray(high_transfer.z_injected, dtype=np.float64)

    if low_dep.shape != low_inj.shape or not np.allclose(
        low_dep, low_inj, rtol=2.0e-5, atol=0.0
    ):
        issues.append("the low-low block must use matching deposition and injection grids")

    if bridge_inj.shape != high_inj.shape or not np.allclose(
        bridge_inj, high_inj, rtol=2.0e-5, atol=0.0
    ):
        issues.append("the bridge injection grid does not match the high-z grid")

    if high_dep.shape != high_inj.shape or not np.allclose(
        high_dep, high_inj, rtol=2.0e-5, atol=0.0
    ):
        issues.append("the high-high block must use matching deposition and injection grids")

    if low_dep.size:
        positions = np.searchsorted(bridge_dep, low_dep)
        positions = np.minimum(positions, max(bridge_dep.size - 1, 0))
        if bridge_dep.size == 0 or not np.allclose(
            bridge_dep[positions], low_dep, rtol=2.0e-5, atol=0.0
        ):
            issues.append("the bridge deposition grid does not contain the low-z grid")

    energy_grids = [
        np.asarray(table.log10E, dtype=np.float64)
        for table in (low_transfer, bridge_transfer, high_transfer)
    ]
    if any(
        grid.shape != energy_grids[0].shape
        or not np.allclose(grid, energy_grids[0], rtol=0.0, atol=5.0e-5)
        for grid in energy_grids[1:]
    ):
        issues.append("the low, bridge, and high energy grids do not match")

    try:
        spacings = [
            _representative_log_spacing(grid)
            for grid in (low_inj, bridge_dep, high_inj)
        ]
        for grid, spacing, label in (
            (low_inj, spacings[0], "low injection"),
            (bridge_dep, spacings[1], "bridge deposition"),
            (high_inj, spacings[2], "high injection"),
        ):
            if not np.allclose(
                np.diff(np.log(grid)), spacing, rtol=5.0e-3, atol=0.0
            ):
                issues.append("the {} grid is not uniform in ln(1+z)".format(label))
        if not np.allclose(spacings, spacings[0], rtol=5.0e-3, atol=0.0):
            issues.append("the low, bridge, and high transfer-cell widths do not match")
    except ValueError as error:
        issues.append(str(error))

    for table, label in (
        (low_transfer, "low-low"),
        (bridge_transfer, "high-to-low bridge"),
        (high_transfer, "high-high"),
    ):
        acausal = np.asarray(table.z_deposited)[:, None] > np.asarray(
            table.z_injected
        )[None, :] * (1.0 + 2.0e-5)
        values = np.maximum(
            np.max(np.abs(table.transfer_elec), axis=1),
            np.max(np.abs(table.transfer_phot), axis=1),
        )
        if np.any(values[acausal] != 0.0):
            issues.append("the {} block contains acausal nonzero cells".format(label))

    return tuple(issues)


def validate_lowz_transfer_blocks(low_transfer, bridge_transfer, high_transfer):
    """Raise if the low-low, bridge, and high-high blocks cannot be joined."""

    issues = lowz_transfer_block_issues(
        low_transfer, bridge_transfer, high_transfer
    )
    if issues:
        raise ValueError(
            "The low-redshift heating transfer blocks are inconsistent:\n - "
            + "\n - ".join(issues)
        )
    return True


def combine_lowz_heat_results(
    low_transfer,
    low_heat,
    bridge_transfer,
    bridge_heat,
    low_injection_mask,
    bridge_injection_mask,
):
    """Add low-low and high-to-low convolutions on supported low-z rows.

    The bridge file contains an all-zero ``1+z=1`` boundary row, while the
    first conservative low-low cell is labelled ``1+z=1.1366``.  Returning
    the supported low-low grid lets :func:`DarkAges.common.finalize` add its
    usual constant endpoint at physical ``z=0`` without creating duplicate
    spline nodes.  Unsupported all-zero rows at the upper handoff are removed
    in the same way.
    """

    low_heat = np.asarray(low_heat, dtype=np.float64)
    bridge_heat = np.asarray(bridge_heat, dtype=np.float64)
    low_injection_mask = np.asarray(low_injection_mask, dtype=bool)
    bridge_injection_mask = np.asarray(bridge_injection_mask, dtype=bool)
    if low_heat.shape != np.asarray(low_transfer.z_deposited).shape:
        raise ValueError("The low-low heating result has the wrong deposition grid.")
    if bridge_heat.shape != np.asarray(bridge_transfer.z_deposited).shape:
        raise ValueError("The bridge heating result has the wrong deposition grid.")
    if low_injection_mask.shape != np.asarray(low_transfer.z_injected).shape:
        raise ValueError("The low-low injection mask has the wrong shape.")
    if bridge_injection_mask.shape != np.asarray(bridge_transfer.z_injected).shape:
        raise ValueError("The bridge injection mask has the wrong shape.")

    low_redshift = np.asarray(low_transfer.z_deposited, dtype=np.float64)
    bridge_redshift = np.asarray(bridge_transfer.z_deposited, dtype=np.float64)
    positions = np.searchsorted(bridge_redshift, low_redshift)
    if np.any(positions >= bridge_redshift.size) or not np.allclose(
        bridge_redshift[positions], low_redshift, rtol=2.0e-5, atol=0.0
    ):
        raise ValueError("Cannot align the bridge and low-low deposition grids.")

    combined_heat = low_heat + bridge_heat[positions]
    low_support = np.any(
        np.abs(low_transfer.transfer_elec[:, :, low_injection_mask]) > 0.0,
        axis=(1, 2),
    ) | np.any(
        np.abs(low_transfer.transfer_phot[:, :, low_injection_mask]) > 0.0,
        axis=(1, 2),
    )
    bridge_support = np.any(
        np.abs(bridge_transfer.transfer_elec[positions][
            :, :, bridge_injection_mask
        ]) > 0.0,
        axis=(1, 2),
    ) | np.any(
        np.abs(bridge_transfer.transfer_phot[positions][
            :, :, bridge_injection_mask
        ]) > 0.0,
        axis=(1, 2),
    )
    supported = low_support | bridge_support
    if not np.any(supported):
        raise ValueError("The selected low-z transfer blocks have no supported rows.")
    return low_redshift[supported], combined_heat[supported]


def electron_kinetic_energy_fraction(log10_energy):
    """Return the kinetic/parent-energy fraction for an e+e- decay basis.

    The low-z generator uses ``mDM = 2 * (m_e + E_kin)`` and normalizes
    ``f_heat`` to the full parent rest-mass injection rate.  Consequently the
    maximum heat response from the tracked kinetic energy alone is
    ``E_kin / (E_kin + m_e)``, rather than one.
    """

    kinetic_energy = np.power(10.0, np.asarray(log10_energy, dtype=np.float64))
    return kinetic_energy / (kinetic_energy + ELECTRON_MASS_EV)


def normalize_lowz_transfer_mode(mode):
    """Return the canonical low-z mode, accepting a few readable aliases."""

    aliases = {
        None: "legacy",
        False: "legacy",
        True: "extend",
        "off": "legacy",
        "old": "extend",
        "new": "extend-new",
    }
    normalized = aliases.get(mode, mode)
    if isinstance(normalized, str):
        normalized = normalized.strip().lower().replace("_", "-")
    if normalized not in LOWZ_TRANSFER_MODES:
        raise ValueError(
            "Unknown low-z transfer mode {!r}; choose one of {}.".format(
                mode, ", ".join(LOWZ_TRANSFER_MODES)
            )
        )
    return normalized


def low_high_masks(low_redshift, high_redshift, mode):
    """Select non-overlapping low/high portions for an extension mode."""

    mode = normalize_lowz_transfer_mode(mode)
    low_redshift = np.asarray(low_redshift, dtype=np.float64)
    high_redshift = np.asarray(high_redshift, dtype=np.float64)
    if low_redshift.ndim != 1 or high_redshift.ndim != 1:
        raise ValueError("Transfer-function redshift grids must be one-dimensional.")
    if low_redshift.size == 0 or high_redshift.size == 0:
        raise ValueError("Transfer-function redshift grids cannot be empty.")
    if np.any(np.diff(low_redshift) <= 0) or np.any(np.diff(high_redshift) <= 0):
        raise ValueError("Transfer-function redshift grids must be strictly increasing.")

    if mode == "legacy":
        return np.zeros(low_redshift.shape, dtype=bool), np.ones(high_redshift.shape, dtype=bool)
    if mode == "low-only":
        # Diagnostic decomposition used by the low-redshift draft figures:
        # retain injections represented by the low-low block and discard both
        # the legacy high-high block and the high-injection bridge.
        return np.ones(low_redshift.shape, dtype=bool), np.zeros(high_redshift.shape, dtype=bool)
    if mode == "low-below-four":
        # Strict diagnostic source split requested for the scan: only
        # injection nodes with 1+z<4, with no high-injection bridge.
        return (
            low_redshift < LOW_INJECTION_CUTOFF,
            np.zeros(high_redshift.shape, dtype=bool),
        )
    if mode == "extend":
        # Extend below the legacy grid, but retain the legacy calculation
        # wherever both calculations are available.
        return low_redshift < high_redshift[0], np.ones(high_redshift.shape, dtype=bool)

    # Use the low-z calculation throughout its range and hand over to the
    # legacy calculation only above the last low-z point.
    return np.ones(low_redshift.shape, dtype=bool), high_redshift > low_redshift[-1]


def splice_deposition_channels(low_redshift, low_heat, high_redshift, high_channels, mode):
    """Join low-z heating to the legacy five-channel deposition table."""

    low_redshift = np.asarray(low_redshift, dtype=np.float64)
    low_heat = np.asarray(low_heat, dtype=np.float64)
    high_redshift = np.asarray(high_redshift, dtype=np.float64)
    high_channels = np.asarray(high_channels, dtype=np.float64)
    if low_heat.shape != low_redshift.shape:
        raise ValueError("The low-z heating curve does not match its redshift grid.")
    if high_channels.ndim != 2 or high_channels.shape[1] != high_redshift.size:
        raise ValueError("The legacy channel table does not match its redshift grid.")

    mode = normalize_lowz_transfer_mode(mode)
    if mode in ("low-only", "low-below-four"):
        # Keep a zero-valued high-z tail so ``finalize`` does not extrapolate
        # the last non-zero low-z value all the way to its upper boundary.
        if mode == "low-only":
            low_mask = np.ones(low_redshift.shape, dtype=bool)
            zero_boundary = low_redshift[-1]
            boundary = np.asarray([], dtype=np.float64)
        else:
            low_mask = low_redshift < LOW_INJECTION_CUTOFF
            zero_boundary = LOW_INJECTION_CUTOFF
            boundary = np.asarray([zero_boundary])
        high_mask = high_redshift > zero_boundary
        joined_redshift = np.concatenate(
            (
                low_redshift[low_mask],
                boundary,
                high_redshift[high_mask],
            )
        )
        joined_channels = np.zeros(
            (high_channels.shape[0], joined_redshift.size), dtype=np.float64
        )
        joined_channels[3, :np.count_nonzero(low_mask)] = low_heat[low_mask]
        return joined_redshift, joined_channels

    low_mask, high_mask = low_high_masks(low_redshift, high_redshift, mode)
    joined_redshift = np.concatenate((low_redshift[low_mask], high_redshift[high_mask]))
    joined_channels = np.zeros(
        (high_channels.shape[0], joined_redshift.size), dtype=np.float64
    )
    low_size = np.count_nonzero(low_mask)
    # Heat is channel 3 in the Slatyer/DarkAges ordering. All other low-z
    # channels are deliberately zero, as required by the low-z tables.
    joined_channels[3, :low_size] = low_heat[low_mask]
    joined_channels[:, low_size:] = high_channels[:, high_mask]
    return joined_redshift, joined_channels


def add_spectral_distortions(
    high_frequency, high_distortion, low_frequency, low_distortion
):
    """Interpolate the low-z result onto the legacy frequency grid and add it."""

    high_frequency = np.asarray(high_frequency, dtype=np.float64)
    high_distortion = np.asarray(high_distortion, dtype=np.float64)
    low_frequency = np.asarray(low_frequency, dtype=np.float64)
    low_distortion = np.asarray(low_distortion, dtype=np.float64)
    if high_frequency.shape != high_distortion.shape:
        raise ValueError("The legacy distortion does not match its frequency grid.")
    if low_frequency.shape != low_distortion.shape:
        raise ValueError("The low-z distortion does not match its frequency grid.")
    if np.any(high_frequency <= 0) or np.any(low_frequency <= 0):
        raise ValueError("Spectral-distortion frequencies must be positive.")
    if np.any(np.diff(high_frequency) <= 0) or np.any(np.diff(low_frequency) <= 0):
        raise ValueError("Spectral-distortion frequency grids must be strictly increasing.")

    interpolated = np.interp(
        np.log(high_frequency),
        np.log(low_frequency),
        low_distortion,
        left=0.0,
        right=0.0,
    )
    return high_distortion + interpolated
