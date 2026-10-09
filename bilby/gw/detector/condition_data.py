import numpy as np


def condition_strain_data(strain_data, sampling_frequency, start_time, duration, trim=0.25,
                          remove_mean=True):
    """ Downsample time-domain strain data around an analysis window.

    Steps:

    1. Keep the window [`start_time`, `start_time` + `duration`) and a buffer
       of `trim` / (1 - 2 `trim`) x `duration` on each side (`duration` / 2
       for the default `trim` = 0.25). A ValueError is raised if the data do
       not cover them.
    2. Shift the data by fewer than `factor` samples (factor = current / new
       sampling frequency) so that the sample nearest `start_time` is kept.
    3. Multiply by a Tukey window with alpha = `trim`, set all frequencies
       above the new Nyquist frequency to zero and keep every factor-th
       sample.
    4. Remove a fraction `trim` of the samples at each end; this removes the
       buffer. Samples in the window are always kept.
    5. Subtract the mean if `remove_mean`.

    The result replaces the data in `strain_data`.

    Parameters
    ==========
    strain_data: bilby.gw.detector.InterferometerStrainData
        The strain data; modified in place.
    sampling_frequency: float
        New sampling frequency (Hz); must divide the current one.
    start_time: float
        GPS start time of the analysis window (s).
    duration: float
        Duration of the analysis window (s).
    trim: float
        Fraction of the data at each end that is removed (default 0.25).
    remove_mean: bool
        Subtract the mean of the result (default True).
    """
    from scipy.signal.windows import tukey

    original_sampling_frequency = float(strain_data.sampling_frequency)
    factor = int(round(original_sampling_frequency / sampling_frequency))
    if factor < 1 or not np.isclose(original_sampling_frequency / sampling_frequency, factor,
                                    rtol=0, atol=1e-9):
        raise ValueError(
            f"Sampling frequency {sampling_frequency:g} Hz does not divide the current "
            f"{original_sampling_frequency:g} Hz")

    buffer = duration * trim / (1 - 2 * trim)
    tolerance = 1 / original_sampling_frequency
    data_start = strain_data.start_time
    data_end = data_start + strain_data.duration
    if data_start > start_time - buffer + tolerance or data_end < start_time + duration + buffer - tolerance:
        raise ValueError(
            f"The data ({data_start:.6f} - {data_end:.6f}) must cover "
            f"{start_time - buffer:.6f} - {start_time + duration + buffer:.6f}: the analysis "
            f"window plus {buffer:g} s on each side")
    first = max(int(round((start_time - buffer - data_start) * original_sampling_frequency)), 0)
    last = int(round((start_time + duration + buffer - data_start) * original_sampling_frequency))
    strain = np.array(strain_data.time_domain_strain[first:last], dtype=float)
    times = np.array(strain_data.time_array[first:last], dtype=float)

    shift = int(np.argmin(abs(times - start_time))) % factor
    if trim > 0:
        # the wrapped samples are at the end and are removed in step 4
        times = np.roll(times, -shift)
        strain = np.roll(strain, -shift)
    else:
        times = times[shift:]
        strain = strain[shift:]

    strain_fd = np.fft.rfft(strain * tukey(len(strain), trim))
    frequencies = np.fft.rfftfreq(len(strain), 1 / original_sampling_frequency)
    strain_fd[frequencies > sampling_frequency / 2] = 0
    strain = np.fft.irfft(strain_fd, n=len(strain))[::factor]
    times = times[::factor]

    # remove the fraction trim at each end, but always keep the window
    window_first = int(np.argmin(abs(times - start_time)))
    window_last = window_first + int(round(duration * sampling_frequency))
    first = min(int(round(trim * len(strain))), window_first)
    last = max(int(round((1 - trim) * len(strain))), window_last)
    strain, times = strain[first:last], times[first:last]
    if remove_mean:
        strain = strain - np.mean(strain)

    strain_data.set_from_time_domain_strain(
        strain, sampling_frequency=sampling_frequency, duration=len(strain) / sampling_frequency,
        start_time=times[0])


__all__ = ["condition_strain_data"]
