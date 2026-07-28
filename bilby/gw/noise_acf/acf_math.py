"""
Pure-math helpers for converting between a PSD, an autocorrelation
function (ACF), and the Gohberg-Semencul (GS) vectors ``x``, ``y``
used to apply ``C^{-1}`` in O(N log N) (see
``bilby.gw.likelihood.time_domain.base.TimeDomainLikelihoodBase``).

These reproduce, unchanged, the math in
``ACF_noise_and_covariance_matrix_data/Compute_x_and_y.py`` of
https://github.com/neha2023sharma/Heterodyning-in-time-domain.
"""
import numpy as np
from scipy.linalg import solve_toeplitz


def acf_from_power_spectral_density(psd_array, sampling_frequency):
    """
    Compute a (one-sided) autocorrelation function from a one-sided
    PSD via the inverse real FFT.

    Parameters
    ----------
    psd_array : array_like
        One-sided power spectral density, evaluated on a regular
        frequency grid from 0 Hz to the Nyquist frequency.
    sampling_frequency : float
        Sampling frequency in Hz, i.e. twice the Nyquist frequency
        implied by ``psd_array``.

    Returns
    -------
    numpy.ndarray
        Autocorrelation function, real-valued, same length as the
        time-domain series implied by ``psd_array``
        (``2 * (len(psd_array) - 1)``).
    """
    delta_t = 1.0 / sampling_frequency
    return 0.5 * np.real(np.fft.irfft(np.asarray(psd_array))) / delta_t


def compute_autocorrelation(time_domain_data):
    """
    Estimate an autocorrelation function directly from a time-domain
    noise realisation, via the sample autocorrelation.

    Parameters
    ----------
    time_domain_data : array_like
        A single (long) time-domain noise realisation.

    Returns
    -------
    numpy.ndarray
        Autocorrelation function of the same length as the input.
    """
    d = np.asarray(time_domain_data)
    n = len(d)
    rho = np.correlate(d, d, mode="full")
    rho = np.fft.ifftshift(rho)
    rho = rho[:n] / n
    return rho


def gohberg_semencul_vectors(acf):
    """
    Compute the Gohberg-Semencul vectors ``x``, ``y`` for a
    symmetric Toeplitz covariance matrix with first column/row
    ``acf``, such that ``C^{-1}`` can be applied via
    :meth:`bilby.gw.likelihood.time_domain.base.TimeDomainLikelihoodBase.inner_product_c_inv_vector`.

    Parameters
    ----------
    acf : array_like
        Autocorrelation function (first column of the Toeplitz
        covariance matrix ``C``).

    Returns
    -------
    x, y : numpy.ndarray
        Gohberg-Semencul vectors, each the same length as ``acf``.
    """
    acf = np.asarray(acf)
    e1 = np.zeros(len(acf))
    e1[0] = 1.0
    x = solve_toeplitz((acf, acf), e1)
    y = x[::-1]
    return x, y
