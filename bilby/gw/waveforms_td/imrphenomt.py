"""
IMRPhenomT waveform wrappers.

These classes wrap the exact ``lalsimulation`` calls used in the
``Heterodyning-in-time-domain`` repository
(https://github.com/neha2023sharma/Heterodyning-in-time-domain) --
no waveform physics has been changed, only reorganised into classes.

They require the ``IMRPhenomT*_neha`` family of functions, which are
part of a custom lalsuite fork and are **not** available in upstream
LALSimulation. See ``examples/time_domain_relative_binning`` and
``docs/time_domain_relative_binning.md`` for installation
instructions.
"""
import numpy as np

from .base import TimeDomainWaveformBase


def spherical_harmonic_22(theta, phi):
    """
    The :math:`{}_{-2}Y_{2,2}` spin-weighted spherical harmonic, as
    used to combine the raw (2, 2) mode into ``h_plus``/``h_cross``.
    """
    return (
        np.sqrt(5.0 / (64.0 * np.pi))
        * (1.0 + np.cos(theta))
        * (1.0 + np.cos(theta))
        * np.exp(2j * phi)
    )


def spherical_harmonic_2m2(theta, phi):
    """The :math:`{}_{-2}Y_{2,-2}` spin-weighted spherical harmonic."""
    return (
        np.sqrt(5.0 / (64.0 * np.pi))
        * (1.0 - np.cos(theta))
        * (1.0 - np.cos(theta))
        * np.exp(-2j * phi)
    )


class IMRPhenomTHMWaveform(TimeDomainWaveformBase):
    """
    Full, multi-mode IMRPhenomT waveform, returned as the complex
    combination ``h_plus - 1j * h_cross``.

    Thin wrapper around ``lalsimulation.SimIMRPhenomTHM_neha_truncate``.
    Used by the exact (non-heterodyned) time-domain likelihood.

    Parameters
    ----------
    fmin, fref : float
        Default starting / reference frequency in Hz.
    modes_to_activate : iterable of (l, m) tuples
        Modes to activate in the LAL mode array (both +m and -m are
        activated for each entry).
    delta_t : float
        Internal LAL generation time step, ``dT``.
    """

    def __init__(self, fmin, fref, modes_to_activate=((2, 2),), delta_t=1 / 4096.0):
        super().__init__(fmin, fref, delta_t)
        self.modes_to_activate = list(modes_to_activate)
        self._lal_params = self._make_mode_array_params()

    def _make_mode_array_params(self):
        import lal
        import lalsimulation as ls

        lal_params = lal.CreateDict()
        mode_array = ls.SimInspiralCreateModeArray()
        for l, m in self.modes_to_activate:
            ls.SimInspiralModeArrayActivateMode(mode_array, l, m)
            ls.SimInspiralModeArrayActivateMode(mode_array, l, -m)
        ls.SimInspiralWaveformParamsInsertModeArray(lal_params, mode_array)
        return lal_params

    def time_domain_strain(self, parameters, time_array, fmin=None, fref=None):
        import lal
        import lalsimulation as ls

        fmin = self.fmin if fmin is None else fmin
        fref = self.fref if fref is None else fref

        m1, m2 = self._component_masses(parameters)
        t_seq = self._dimensionless_time_sequence(time_array, m1, m2)

        h = ls.SimIMRPhenomTHM_neha_truncate(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"] * 1e6 * lal.PC_SI,
            parameters["theta_jn"],
            self.delta_t,
            fmin,
            fref,
            parameters["phase"],
            t_seq,
            self._lal_params,
        )
        return h[0].data.data - 1j * h[1].data.data


class IMRPhenomT22Waveform(TimeDomainWaveformBase):
    """
    (2, 2)-mode-only IMRPhenomT waveform, projected onto
    ``h_plus - 1j * h_cross``.

    Thin wrapper around ``lalsimulation.SimIMRPhenomT_neha_truncate``.
    """

    def time_domain_strain(self, parameters, time_array, fmin=None, fref=None):
        import lal
        import lalsimulation as ls

        fmin = self.fmin if fmin is None else fmin
        fref = self.fref if fref is None else fref

        m1, m2 = self._component_masses(parameters)
        t_seq = self._dimensionless_time_sequence(time_array, m1, m2)

        h = ls.SimIMRPhenomT_neha_truncate(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"] * 1e6 * lal.PC_SI,
            parameters["theta_jn"],
            self.delta_t,
            fmin,
            fref,
            parameters["phase"],
            t_seq,
            None,
        )
        return h[0].data.data - 1j * h[1].data.data


class IMRPhenomT22ModeWaveform(TimeDomainWaveformBase):
    """
    Raw (2, 2) spin-weighted spherical-harmonic mode -- no
    ``h_plus``/``h_cross`` projection, no inclination (``theta_jn``)
    dependence. This is the fiducial waveform used by the
    relative-binning likelihood, and is also used (evaluated at
    non-fiducial parameters) to build the waveform ratio ``r(t)`` at
    each likelihood call.

    Thin wrapper around ``lalsimulation.SimIMRPhenomT_neha_just_modes``.
    """

    def time_domain_strain(self, parameters, time_array, fmin=None, fref=None):
        import lal
        import lalsimulation as ls

        fmin = self.fmin if fmin is None else fmin
        fref = self.fref if fref is None else fref

        m1, m2 = self._component_masses(parameters)
        t_seq = self._dimensionless_time_sequence(time_array, m1, m2)

        h = ls.SimIMRPhenomT_neha_just_modes(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"] * 1e6 * lal.PC_SI,
            self.delta_t,
            fmin,
            fref,
            parameters["phase"],
            t_seq,
            None,
        )
        return h.mode.data.data
