"""
Shared base class for time-domain gravitational-wave likelihoods that
apply the inverse of a stationary noise covariance matrix, ``C^{-1}``,
via its Gohberg-Semencul (GS) representation.

This underlies both :class:`ExactTimeDomainLikelihood` and
:class:`RelativeBinningTimeDomainLikelihood22` (see ``likelihoods.py``
in this subpackage). See

    Sharma, Vijaykumar & Kumar, "Rapid inference of gravitational-wave
    signals in the time domain using a heterodyned likelihood",
    arXiv:2601.11239

and, for the relative-binning procedure itself, Zackay et al.,
arXiv:1806.08792.
"""
import numpy as np
from scipy.linalg import matmul_toeplitz

from ....core.likelihood import Likelihood


class TimeDomainLikelihoodBase(Likelihood):
    """
    Abstract base for time-domain GW likelihoods using the GS solver.

    Subclasses must set:

    - ``self._time_key``: ``"geocent_time"`` or ``"H1_time"``;
    - ``self.Detectors_list``: a dict mapping detector name ->
      :class:`bilby.gw.detector.Interferometer`. If ``self._time_key
      == "H1_time"``, this must contain an ``"H1"`` entry, used as the
      time-delay reference detector.
    """

    def __init__(self):
        super().__init__()
        self.noise_log_likelihood_value = None

    # ------------------------------------------------------------------
    # Gohberg-Semencul C^{-1} v
    # ------------------------------------------------------------------

    @staticmethod
    def inner_product_c_inv_vector(x, y, v):
        """
        Compute ``C^{-1} v`` using the Gohberg-Semencul representation

        .. math::

            C^{-1} v = \\frac{1}{x_0}\\left[ L(x) L(x)^T v -
            L(Jy) L(Jy)^T v \\right]

        where ``L(w)`` is the lower-triangular Toeplitz matrix with
        first column ``w``. Each matrix-vector product is O(N log N)
        via FFT (:func:`scipy.linalg.matmul_toeplitz`).

        Parameters
        ----------
        x, y : array_like
            Gohberg-Semencul vectors, e.g. as produced by
            :class:`bilby.gw.noise_acf.NoiseCovarianceVectors`.
        v : array_like
            Vector (or matrix, applied column-wise) to multiply.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        xf = np.concatenate(([x[0]], np.zeros(len(x) - 1)))
        ys = np.concatenate(([0.0], y[:-1]))
        zs = np.zeros(len(x))
        return (1.0 / x[0]) * (
            matmul_toeplitz((x, xf), matmul_toeplitz((xf, x), v))
            - matmul_toeplitz((ys, zs), matmul_toeplitz((zs, ys), v))
        )

    def weighted_inner_product(self, x, y, h1, h2):
        """Return ``(h1, C^{-1} h2)``."""
        return np.dot(h1, self.inner_product_c_inv_vector(x, y, h2))

    # ------------------------------------------------------------------
    # Frame-dependent time / antenna-pattern helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_delay_from_detector(reference_detector, detector, ra, dec, time):
        """
        Time delay from ``reference_detector`` to ``detector`` for a
        signal from (``ra``, ``dec``) with geocentric arrival time
        ``time``, i.e. ``t(detector) = t(reference_detector) + delay``.

        Equivalent to ``pycbc.detector.Detector.time_delay_from_detector``.
        Computed as the difference of the two detectors' geocentric
        time delays, which is algebraically identical to the direct
        vertex-separation formula (the geocentric time-delay
        expression is linear in detector position).
        """
        return detector.time_delay_from_geocenter(
            ra, dec, time
        ) - reference_detector.time_delay_from_geocenter(ra, dec, time)

    def _time_delay(self, params, detector):
        """
        Time delay to ``detector`` for the given parameter dict.
        Dispatches on ``self._time_key``.
        """
        if self._time_key == "geocent_time":
            return detector.time_delay_from_geocenter(
                params["ra"], params["dec"], params["geocent_time"]
            )
        return self._time_delay_from_detector(
            self.Detectors_list["H1"],
            detector,
            params["ra"],
            params["dec"],
            params["H1_time"],
        )

    def _antenna_pattern(self, params, detector):
        """Return ``(F_plus, F_cross)`` for ``detector`` at ``params``."""
        time = params[self._time_key]
        f_plus = detector.antenna_response(
            params["ra"], params["dec"], time, params["psi"], "plus"
        )
        f_cross = detector.antenna_response(
            params["ra"], params["dec"], time, params["psi"], "cross"
        )
        return f_plus, f_cross

    # ------------------------------------------------------------------
    # Injection parameters / data
    # ------------------------------------------------------------------

    @property
    def injection_parameters(self):
        return self._injection_parameters

    @injection_parameters.setter
    def injection_parameters(self, new_params):
        self._injection_parameters = new_params.copy()
        self._Data_list = {}

    @property
    def Data_list(self):
        return self._Data_list

    # ------------------------------------------------------------------
    # Likelihoods
    # ------------------------------------------------------------------

    def noise_log_likelihood(self):
        if self.noise_log_likelihood_value is None:
            self.noise_log_likelihood_value = -0.5 * sum(
                self.data_times_C_inv_times_data[k] for k in self.Detectors_list
            )
        return self.noise_log_likelihood_value

    def log_likelihood(self, parameters):
        return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()
