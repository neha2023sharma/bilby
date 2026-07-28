"""
:class:`NoiseCovarianceVectors` -- the main entry point of this
subpackage. Computes the Gohberg-Semencul vectors ``x``, ``y`` needed
by :mod:`bilby.gw.likelihood.time_domain` to apply ``C^{-1}``.
"""
import numpy as np

from ..detector import get_empty_interferometer
from .acf_math import acf_from_power_spectral_density, gohberg_semencul_vectors
from .psd_patching import patch_power_spectral_density


class NoiseCovarianceVectors:
    """
    Compute the Gohberg-Semencul vectors ``x``, ``y`` for a single
    detector / data segment, given (in order of precedence):

    1. an explicit autocorrelation function (``acf``) -- used as-is,
       no PSD patching is performed;
    2. an explicit PSD (``frequency_array`` and ``psd_array``) -- the
       PSD is patched (see :func:`patch_power_spectral_density`, which
       clamps non-finite/non-positive bins and anything outside
       ``[fmin, fmax]`` to the maximum finite in-band value) and
       Fourier-transformed to an ACF;
    3. neither -- bilby's default PSD for ``detector_name`` is looked
       up via :func:`bilby.gw.detector.get_empty_interferometer`,
       patched in the same way, and used.

    Parameters
    ----------
    duration : float
        Length, in seconds, of the data segment the vectors are
        computed for.
    sampling_frequency : float
        Sampling frequency in Hz.
    detector_name : str, optional
        Name of a known detector (e.g. ``"H1"``), used to look up
        bilby's default PSD when neither ``acf`` nor ``psd_array`` is
        given. Required in that case.
    acf : array_like, optional
        Explicit autocorrelation function.
    frequency_array, psd_array : array_like, optional
        Explicit one-sided PSD.
    fmin : float
        Lower patching frequency, in Hz, used when a PSD is patched
        (cases 2 and 3 above). Default 10 Hz.
    fmax : float, optional
        Upper patching frequency, in Hz. Default ``None`` (Nyquist).

    Attributes
    ----------
    x, y : numpy.ndarray
        Gohberg-Semencul vectors, populated after calling
        :meth:`compute`.
    acf : numpy.ndarray
        The autocorrelation function used (either the one supplied,
        or the one derived from the PSD).
    """

    def __init__(
        self,
        duration,
        sampling_frequency,
        detector_name=None,
        acf=None,
        frequency_array=None,
        psd_array=None,
        fmin=10.0,
        fmax=None,
    ):
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detector_name = detector_name
        self.acf = None if acf is None else np.asarray(acf)
        self.frequency_array = frequency_array
        self.psd_array = psd_array
        self.fmin = fmin
        self.fmax = fmax

        self.x = None
        self.y = None

    @property
    def n_samples(self):
        return int(round(self.duration * self.sampling_frequency))

    def _default_bilby_psd(self):
        """Frequency array and PSD for ``detector_name``'s default noise curve."""
        if self.detector_name is None:
            raise ValueError(
                "detector_name must be given when neither acf nor psd_array is "
                "supplied, so the default bilby PSD can be looked up."
            )
        ifo = get_empty_interferometer(self.detector_name)
        # minimum_frequency=0 so the PSD is evaluated (and can be patched)
        # over the full band, rather than pre-masked by bilby.
        ifo.minimum_frequency = 0
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
        )
        return ifo.frequency_array, ifo.power_spectral_density_array

    def compute(self):
        """
        Compute (and cache on ``self.x``, ``self.y``, ``self.acf``)
        the Gohberg-Semencul vectors.

        Returns
        -------
        x, y : numpy.ndarray
        """
        if self.acf is not None:
            acf = self.acf[: self.n_samples]
        else:
            if self.frequency_array is not None and self.psd_array is not None:
                frequency_array, psd_array = self.frequency_array, self.psd_array
            else:
                frequency_array, psd_array = self._default_bilby_psd()

            patched_psd = patch_power_spectral_density(
                frequency_array, psd_array, fmin=self.fmin, fmax=self.fmax
            )
            acf = acf_from_power_spectral_density(
                patched_psd, sampling_frequency=self.sampling_frequency
            )[: self.n_samples]

        x, y = gohberg_semencul_vectors(acf)
        self.acf, self.x, self.y = acf, x, y
        return x, y

    @classmethod
    def for_detectors(
        cls,
        detector_names,
        duration,
        sampling_frequency,
        acf=None,
        frequency_array=None,
        psd_array=None,
        fmin=10.0,
        fmax=None,
    ):
        """
        Convenience constructor that computes ``x``, ``y`` for several
        detectors at once, returning them as
        ``{detector_name: array}`` dicts -- the format expected by
        :mod:`bilby.gw.likelihood.time_domain` (``x[k]``, ``y[k]``).

        ``acf``, ``frequency_array`` and ``psd_array``, if given,
        should be dicts keyed by detector name (mirroring the return
        format); any detector missing from those dicts falls back to
        bilby's default PSD for that detector.

        Returns
        -------
        x, y : dict[str, numpy.ndarray]
        """
        acf = acf or {}
        frequency_array = frequency_array or {}
        psd_array = psd_array or {}

        x_dict, y_dict = {}, {}
        for name in detector_names:
            vectors = cls(
                duration=duration,
                sampling_frequency=sampling_frequency,
                detector_name=name,
                acf=acf.get(name),
                frequency_array=frequency_array.get(name),
                psd_array=psd_array.get(name),
                fmin=fmin,
                fmax=fmax,
            )
            x_dict[name], y_dict[name] = vectors.compute()
        return x_dict, y_dict
