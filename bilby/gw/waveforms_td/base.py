"""
Base class for time-domain waveform wrappers used by the time-domain
relative-binning (heterodyning) likelihoods in
:mod:`bilby.gw.likelihood.time_domain`.

These classes are thin, class-based wrappers around existing
``lalsimulation`` functions -- they do not implement or alter any
waveform physics themselves. They exist so that the likelihood code
does not need to know the LAL calling convention (mass conversion,
``REAL8Sequence`` construction, mode-array bookkeeping, ...), and so
that different waveform families/mode content can be swapped in
without touching the likelihood classes.

Note
----
These wrappers call functions (e.g. ``SimIMRPhenomTHM_neha_truncate``)
that are **not** part of upstream LALSimulation. They require the
custom lalsuite fork described in the top-level README of this
addition (``docs/time_domain_relative_binning.md`` /
``examples/time_domain_relative_binning``).
"""
import numpy as np

from ..conversion import chirp_mass_and_mass_ratio_to_component_masses


class TimeDomainWaveformBase:
    """
    Common utilities shared by all time-domain waveform wrappers in
    this subpackage.

    Parameters
    ----------
    fmin : float
        Default starting frequency in Hz.
    fref : float
        Default reference frequency in Hz.
    delta_t : float
        Time step in seconds used internally by the LAL waveform
        generator (``dT`` in the underlying ``lalsimulation`` calls).
        This is independent of, and need not match, the sample
        spacing of ``time_array`` passed to
        :meth:`time_domain_strain`, since the waveform is generated
        on the geometric time grid supplied by the caller.
    """

    def __init__(self, fmin, fref, delta_t=1 / 4096.0):
        self.fmin = fmin
        self.fref = fref
        self.delta_t = delta_t

    @staticmethod
    def _component_masses(parameters):
        """Convert (chirp_mass, mass_ratio) -> (m1, m2) in solar masses."""
        return chirp_mass_and_mass_ratio_to_component_masses(
            parameters["chirp_mass"], parameters["mass_ratio"]
        )

    @staticmethod
    def _dimensionless_time_sequence(time_array, m1, m2):
        """
        Build the ``lal.REAL8Sequence`` of geometric times (time in
        units of total mass) expected by the ``*_neha_*`` LAL
        functions.
        """
        import lal

        t_seq = lal.CreateREAL8Sequence(len(time_array))
        t_seq.data = np.asarray(time_array) / ((m1 + m2) * lal.MTSUN_SI)
        return t_seq

    def time_domain_strain(self, parameters, time_array, fmin=None, fref=None):
        """
        Evaluate the waveform on ``time_array`` (seconds, relative to
        the fiducial/merger epoch) for the given ``parameters``.

        Subclasses must implement this and document the exact
        combination of polarizations / modes returned.
        """
        raise NotImplementedError
