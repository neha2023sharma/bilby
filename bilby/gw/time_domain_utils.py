"""
Utilities for time-domain gravitational-wave inference.

* Gohberg-Semencul representation of the inverse of a symmetric Toeplitz
  noise covariance matrix, and the noise-weighted inner products built on it.
* Placement of a time-domain waveform, generated on its own uniform grid, into
  a segment of detector data.
"""
import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import solve_toeplitz

PLACEMENT_METHODS = ("nearest", "subsample", "fd_shift")


def gohberg_semencul_vectors(acf):
    """ Gohberg-Semencul generators of the inverse of the Toeplitz matrix
    built from :code:`acf`.

    Parameters
    ==========
    acf: array_like
        Autocovariance function at lags 0, 1, ..., N - 1 (first row of the
        N x N covariance matrix).

    Returns
    =======
    x, y: array_like
        x is the first column of C^-1 and y is x reversed.
    """
    acf = np.asarray(acf, dtype=float)
    unit_vector = np.zeros(len(acf))
    unit_vector[0] = 1.0
    x = solve_toeplitz((acf, acf), unit_vector)
    return x, x[::-1]


class GohbergSemenculInverse(object):
    """ Inverse of a symmetric Toeplitz matrix in Gohberg-Semencul form

    .. math::

        C^{-1} = \\frac{1}{x_0} \\left[ L(x) L(x)^T - L(Z y) L(Z y)^T \\right]

    where :math:`L(w)` is the lower-triangular Toeplitz matrix with first
    column :math:`w` and :math:`Z` shifts down by one sample. The Fourier
    transforms of :math:`x` and :math:`Z y` are computed once, so each product
    :math:`C^{-1} v` costs one forward and two inverse FFTs per factor, of
    length :code:`next_fast_len(2 N - 1)`.

    Parameters
    ==========
    x, y: array_like
        Gohberg-Semencul vectors from :code:`gohberg_semencul_vectors`.
    """

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.number_of_samples = len(x)
        self.fft_length = next_fast_len(2 * self.number_of_samples - 1, real=True)
        self.x0 = x[0]
        self._x_fft = np.fft.rfft(x, n=self.fft_length)
        self._zy_fft = np.fft.rfft(np.concatenate(([0.0], y[:-1])), n=self.fft_length)

    def _lower(self, w_fft, vector_fft):
        """ L(w) v from the transforms of w and v. """
        return np.fft.irfft(w_fft * vector_fft, n=self.fft_length)[:self.number_of_samples]

    def __call__(self, vector):
        """ C^-1 vector. """
        vector = np.asarray(vector, dtype=float)
        if len(vector) != self.number_of_samples:
            raise ValueError(
                f"vector has {len(vector)} samples, expected {self.number_of_samples}")
        n = self.fft_length
        # L(w)^T v = reverse(L(w) reverse(v))
        reversed_fft = np.fft.rfft(vector[::-1], n=n)
        first = self._lower(self._x_fft, np.fft.rfft(
            self._lower(self._x_fft, reversed_fft)[::-1], n=n))
        second = self._lower(self._zy_fft, np.fft.rfft(
            self._lower(self._zy_fft, reversed_fft)[::-1], n=n))
        return (first - second) / self.x0


def gohberg_semencul_product(x, y, vector):
    """ Compute C^-1 v using the Gohberg-Semencul representation, see
    :code:`GohbergSemenculInverse`. To apply the same C^-1 many times, build a
    :code:`GohbergSemenculInverse` once instead.

    Parameters
    ==========
    x, y: array_like
        Gohberg-Semencul vectors from :code:`gohberg_semencul_vectors`.
    vector: array_like
        The vector to multiply, same length as x.

    Returns
    =======
    array_like: C^-1 vector
    """
    return GohbergSemenculInverse(x, y)(vector)


def time_domain_noise_weighted_inner_product(aa, bb, x, y):
    """ Noise-weighted inner product :math:`a^T C^{-1} b`.

    Parameters
    ==========
    aa, bb: array_like
        Time series of equal length.
    x, y: array_like
        Gohberg-Semencul vectors of C^-1 for that length.

    Returns
    =======
    float: The noise-weighted inner product.
    """
    return float(np.dot(aa, gohberg_semencul_product(x, y, bb)))


def time_domain_optimal_snr_squared(signal, x, y):
    """ Optimal SNR squared :math:`h^T C^{-1} h` of a time-domain signal. """
    return time_domain_noise_weighted_inner_product(signal, signal, x, y)


def time_domain_matched_filter_snr(signal, time_domain_strain, x, y):
    """ Matched-filter SNR :math:`d^T C^{-1} h / \\sqrt{h^T C^{-1} h}`. """
    return (
        time_domain_noise_weighted_inner_product(time_domain_strain, signal, x, y)
        / time_domain_optimal_snr_squared(signal, x, y) ** 0.5
    )


def one_sided_tukey_window(number_of_samples, alpha):
    """ Tukey window with only the rising (left) half applied.

    Parameters
    ==========
    number_of_samples: int
        Length of the window.
    alpha: float
        Shape parameter of :code:`scipy.signal.windows.tukey`; 0 returns ones.

    Returns
    =======
    array_like: The window.
    """
    from scipy.signal.windows import tukey
    if not alpha:
        return np.ones(number_of_samples)
    window = tukey(number_of_samples, alpha)
    window[number_of_samples // 2:] = 1.0
    return window


def fractional_time_shift(signal, fraction, pad=32):
    """ Delay a time series by a fraction of a sample with a Fourier phase
    shift, on a zero-padded copy so that nothing wraps around.

    Parameters
    ==========
    signal: array_like
        The time series.
    fraction: float
        Delay in samples (positive values move the signal later).
    pad: int
        Number of zero samples added before and after the signal.

    Returns
    =======
    shifted: array_like
        The delayed series, of length len(signal) + 2 * pad (sample k
        corresponds to sample k - pad of the input).
    """
    signal = np.asarray(signal, dtype=float)
    length = next_fast_len(len(signal) + 2 * pad)
    buffer = np.zeros(length)
    buffer[pad:pad + len(signal)] = signal
    frequencies = np.fft.rfftfreq(length)
    shifted = np.fft.irfft(
        np.fft.rfft(buffer) * np.exp(-2j * np.pi * frequencies * fraction), n=length)
    return shifted[:len(signal) + 2 * pad]


def quadratic_peak_index(plus, cross):
    """ Fractional index of the peak of :math:`\\sqrt{h_+^2 + h_\\times^2}`,
    from a quadratic fit to the three samples around the maximum (as in the
    :code:`ringdown` package).

    Parameters
    ==========
    plus, cross: array_like
        The two polarizations on a uniform grid.

    Returns
    =======
    float: Peak position in (fractional) samples.
    """
    amplitude = np.sqrt(np.asarray(plus) ** 2 + np.asarray(cross) ** 2)
    ib = int(np.argmax(amplitude))
    ia = (ib - 1) % len(amplitude)
    ic = (ib + 1) % len(amplitude)
    a, b, c = amplitude[ia], amplitude[ib], amplitude[ic]
    denominator = a - 2 * b + c
    if denominator == 0:
        return float(ib)
    return ib + (3 * a - 4 * b + c) / (2 * denominator) - 1


def align_peak_to_sample(waveform_polarizations, pad=32):
    """ Shift all polarizations by the same sub-sample amount so that the
    (quadratically interpolated) peak of :math:`h_+^2 + h_\\times^2` falls
    exactly on a sample (:code:`ringdown`'s :code:`subsample_placement`).

    Parameters
    ==========
    waveform_polarizations: dict
        Polarizations on their own uniform grid (no 'epoch' key).
    pad: int
        Zero padding added on each side before shifting.

    Returns
    =======
    aligned: dict
        The shifted polarizations (each padded by :code:`pad` on both sides).
    peak_index: int
        Index of the peak in the aligned arrays.
    """
    peak = quadratic_peak_index(
        waveform_polarizations["plus"], waveform_polarizations["cross"])
    fraction = round(peak) - peak
    aligned = {mode: fractional_time_shift(value, fraction, pad=pad)
               for mode, value in waveform_polarizations.items()}
    return aligned, int(round(peak)) + pad


def _copy_into_segment(signal, offset, start_index, number_of_samples):
    """ Put signal sample k at data index offset + k and return data indices
    [start_index, start_index + number_of_samples); zero elsewhere. """
    segment = np.zeros(number_of_samples)
    first = max(0, start_index - offset)
    last = min(len(signal), start_index + number_of_samples - offset)
    if last > first:
        segment[offset + first - start_index:offset + last - start_index] = signal[first:last]
    return segment


def place_time_domain_signal(signal, merger_index, arrival_index, start_index,
                             number_of_samples, placement="nearest"):
    """ Place a time-domain signal, generated on its own grid with the data's
    sample spacing, into a segment of data.

    Waveform sample k has time (k - merger_index) * dt relative to the
    model's merger (its t=0). In data-index units the merger arrives at
    :code:`arrival_index` (fractional). Samples falling outside the segment are
    dropped; segment samples the signal does not reach are zero. Nothing wraps.

    Parameters
    ==========
    signal: array_like
        Projected (detector) signal on its own grid.
    merger_index: float
        Index of the model's t=0 in :code:`signal` (-epoch * sampling_frequency).
    arrival_index: float
        Arrival time of t=0 at the detector, in samples from the first data sample.
    start_index: int
        First data index of the analysis segment.
    number_of_samples: int
        Length of the analysis segment.
    placement: str
        - "nearest": t=0 of the signal is put on the data sample nearest to the
          arrival time; both indices are rounded separately, as in
          LALInference and the :code:`ringdown` package (default).
        - "subsample": as "nearest", with :code:`merger_index` the integer peak
          index from :code:`align_peak_to_sample`. The arrival time then refers
          to the peak of :math:`h_+^2 + h_\\times^2`, not to the model's t=0
          (the :code:`ringdown` package's :code:`manual_epoch` convention).
        - "fd_shift": exact delay; whole samples by index, the remaining
          fraction with a Fourier phase shift on a zero-padded copy, applied
          before cropping to the segment.

    Returns
    =======
    array_like: The signal on the segment's samples.
    """
    if placement in ("nearest", "subsample"):
        offset = round(arrival_index) - round(merger_index)
        return _copy_into_segment(signal, offset, start_index, number_of_samples)
    elif placement == "fd_shift":
        position = arrival_index - merger_index
        offset = int(np.floor(position + 0.5))
        fraction = position - offset
        pad = 32
        shifted = fractional_time_shift(signal, fraction, pad=pad)
        return _copy_into_segment(shifted, offset - pad, start_index, number_of_samples)
    else:
        raise ValueError(f"placement must be one of {PLACEMENT_METHODS}, not {placement!r}")
