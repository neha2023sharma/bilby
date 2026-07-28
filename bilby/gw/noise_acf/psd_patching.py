"""
PSD patching utilities.

A power spectral density (PSD) is typically only meaningful (finite,
positive) over some ``[fmin, fmax]`` band. Outside that band -- and at
any non-finite or non-positive bins inside it -- ``PowerSpectralDensity``
objects (bilby's own default curves included) commonly return ``inf``,
``nan``, or zero. Naively Fourier-transforming such a PSD to obtain an
autocorrelation function (ACF) produces artefacts (e.g. from
``0 -> inf`` transitions at ``fmin``).

The standard fix (see e.g. the ``ringdown`` package, and
``ACF_noise_and_covariance_matrix_data/Make_psd_file.py`` /
``Compute_x_and_y.py`` in
https://github.com/neha2023sharma/Heterodyning-in-time-domain) is to
replace the bad bins with a large, finite value -- the maximum of the
finite, positive in-band PSD -- so that those frequencies are
effectively down-weighted (treated as very noisy) rather than
producing numerical artefacts. This module generalises that fix to
patch symmetrically below ``fmin`` **and** above ``fmax``.
"""
import numpy as np


def patch_power_spectral_density(frequency_array, psd_array, fmin=10.0, fmax=None,
                                  fill_value=None):
    """
    Replace out-of-band and non-finite/non-positive PSD bins with a
    large, finite fill value.

    Parameters
    ----------
    frequency_array : array_like
        Frequencies (Hz) corresponding to ``psd_array``.
    psd_array : array_like
        Power spectral density values. Not modified in place; a
        patched copy is returned.
    fmin : float
        Frequencies below ``fmin`` are patched. Default 10 Hz.
    fmax : float or None
        Frequencies above ``fmax`` are patched. If ``None``, no
        upper-frequency patching is applied (only the Nyquist bin
        itself is patched if it is non-finite).
    fill_value : float or None
        Value used to replace bad bins. If ``None`` (default), the
        maximum of the finite, positive, in-band PSD values is used,
        matching the convention used in the original repository's
        PSD-padding scripts.

    Returns
    -------
    numpy.ndarray
        The patched PSD, same shape as ``psd_array``.
    """
    frequency_array = np.asarray(frequency_array, dtype=float)
    psd = np.array(psd_array, dtype=float, copy=True)

    in_band = frequency_array >= fmin
    if fmax is not None:
        in_band &= frequency_array <= fmax

    finite_in_band = in_band & np.isfinite(psd) & (psd > 0)
    if not np.any(finite_in_band):
        raise ValueError(
            "No finite, positive PSD values found in the requested "
            f"[{fmin}, {fmax}] Hz band; cannot patch."
        )

    fill = fill_value if fill_value is not None else np.max(psd[finite_in_band])

    bad = ~finite_in_band
    psd[bad] = fill
    return psd
