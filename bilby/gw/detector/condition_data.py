import numpy as np


def condition_strain_data(strain_data, sampling_frequency, start_time, duration,
                          preserve_time=None, trim=0.25, taper_and_trim=True,
                          remove_mean=True):
    """ Downsample time-domain strain data around an analysis window.

    Steps (default, `taper_and_trim` = True):

    1. Keep the window [`start_time`, `start_time` + `duration`) and a buffer
       of `trim` / (1 - 2 `trim`) x `duration` on each side (`duration` / 2
       for the default `trim` = 0.25). A ValueError is raised if the data do
       not cover them.
    2. Shift the data by fewer than `factor` samples (factor = current / new
       sampling frequency) so that the sample nearest `preserve_time` is kept.
    3. Multiply by a Tukey window with alpha = `trim`, set all frequencies
       above the new Nyquist frequency to zero and keep every factor-th
       sample.
    4. Remove a fraction `trim` of the samples at each end; this removes the
       buffer. Samples in the window are always kept.
    5. Subtract the mean if `remove_mean`.

    With `taper_and_trim` = False, steps 1, 3 and 4 are replaced by
    :code:`scipy.signal.decimate` (order-8 Chebyshev low-pass filter, applied
    forwards and backwards, then every factor-th sample) on all the data, and
    the first samples are dropped instead of shifted in step 2. No buffer is
    needed. The filter starts to cut at 0.8 x the new Nyquist frequency.

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
    preserve_time: float, optional
        GPS time whose nearest sample is kept, e.g. the inspiral/post-inspiral
        cut time (default: `start_time`).
    trim: float
        Fraction of the data at each end that is removed (default 0.25).
    taper_and_trim: bool
        Use the Tukey window, buffer and trimming (default True); if False,
        use :code:`scipy.signal.decimate`.
    remove_mean: bool
        Subtract the mean of the result (default True).
    """
    from scipy.signal import decimate
    from scipy.signal.windows import tukey

    original_sampling_frequency = float(strain_data.sampling_frequency)
    factor = int(round(original_sampling_frequency / sampling_frequency))
    if preserve_time is None:
        preserve_time = start_time

    if not taper_and_trim:
        strain = np.array(strain_data.time_domain_strain, dtype=float)
        times = np.array(strain_data.time_array, dtype=float)
        shift = int(np.argmin(abs(times - preserve_time))) % factor
        strain = decimate(strain[shift:], factor, zero_phase=True)
        times = times[shift::factor]
        if remove_mean:
            strain = strain - np.mean(strain)
        strain_data.set_from_time_domain_strain(
            strain, sampling_frequency=sampling_frequency,
            duration=len(strain) / sampling_frequency, start_time=times[0])
        return

    buffer_duration = duration * trim / (1 - 2 * trim)
    tolerance = 1 / original_sampling_frequency
    data_start_time = strain_data.start_time
    data_end_time = data_start_time + strain_data.duration
    buffer_start_time = start_time - buffer_duration
    buffer_end_time = start_time + duration + buffer_duration
    if data_start_time > buffer_start_time + tolerance or data_end_time < buffer_end_time - tolerance:
        raise ValueError(
            f"The data ({data_start_time:.6f} - {data_end_time:.6f}) must cover "
            f"{buffer_start_time:.6f} - {buffer_end_time:.6f}: the analysis window plus "
            f"{buffer_duration:g} s on each side")
    buffer_start_index = max(
        int(round((buffer_start_time - data_start_time) * original_sampling_frequency)), 0)
    buffer_end_index = int(round((buffer_end_time - data_start_time) * original_sampling_frequency))
    strain = np.array(strain_data.time_domain_strain[buffer_start_index:buffer_end_index], dtype=float)
    times = np.array(strain_data.time_array[buffer_start_index:buffer_end_index], dtype=float)

    shift = int(np.argmin(abs(times - preserve_time))) % factor
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

    # remove the fraction trim at each end, but always keep the analysis samples
    analysis_start_index = int(np.argmin(abs(times - start_time)))
    analysis_end_index = analysis_start_index + int(round(duration * sampling_frequency))
    keep_start_index = min(int(round(trim * len(strain))), analysis_start_index)
    keep_end_index = max(int(round((1 - trim) * len(strain))), analysis_end_index)
    strain = strain[keep_start_index:keep_end_index]
    times = times[keep_start_index:keep_end_index]
    if remove_mean:
        strain = strain - np.mean(strain)

    strain_data.set_from_time_domain_strain(
        strain, sampling_frequency=sampling_frequency, duration=len(strain) / sampling_frequency,
        start_time=times[0])


__all__ = ["condition_strain_data"]
