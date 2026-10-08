import numpy as np

from ...core.utils import logger
from ..time_domain_utils import gohberg_semencul_vectors

FILL_REFERENCES = ("max_in_band", "median_in_band", "patched_region_max")


def _next_power_of_two(value):
    power = 1
    while power < value:
        power <<= 1
    return power


def _is_power_of_two(value):
    return float(value).is_integer() and int(value) > 0 and (int(value) & (int(value) - 1)) == 0


def _check_sampling_frequency(implied, given, label):
    if given is None or np.isclose(implied, given, rtol=1e-9, atol=0):
        return
    message = (
        f"{label}: sampling frequency {given:g} Hz differs from the one implied by the "
        f"input, {implied:g} Hz. An ACF built with {given:g} Hz would be scaled by "
        f"{given / implied:g} and its lags would be spaced 1/{implied:g} s, not "
        f"1/{given:g} s."
    )
    logger.warning(message)
    raise ValueError(message)


def _uniform_frequency_grid(frequency_array, psd_array, label):
    """ Sort, check uniform spacing (snapping labels stored with limited
    precision), extend down to 0 Hz and complete a missing Nyquist bin.
    Bins added or without a finite positive value are NaN. """
    frequency_array = np.asarray(frequency_array, dtype=float)
    psd_array = np.asarray(psd_array, dtype=float)
    if frequency_array.shape != psd_array.shape or frequency_array.ndim != 1 or len(frequency_array) < 3:
        raise ValueError(f"{label}: frequency_array and psd_array must be 1D arrays of equal length")
    order = np.argsort(frequency_array)
    frequency_array, psd_array = frequency_array[order], psd_array[order]
    psd_array = np.where(np.isfinite(psd_array) & (psd_array > 0), psd_array, np.nan)

    delta_f = (frequency_array[-1] - frequency_array[0]) / (len(frequency_array) - 1)
    uniform = frequency_array[0] + np.arange(len(frequency_array)) * delta_f
    if np.max(np.abs(frequency_array - uniform)) > 0.1 * delta_f:
        raise ValueError(
            f"{label}: frequency grid is not uniform (spacing from "
            f"{np.diff(frequency_array).min():.6g} to {np.diff(frequency_array).max():.6g} Hz); "
            "power spectral densities are not interpolated")
    k_first = round(frequency_array[0] / delta_f)
    if abs(frequency_array[0] - k_first * delta_f) <= 0.1 * delta_f:
        first = k_first * delta_f
    else:
        raise ValueError(
            f"{label}: first frequency {frequency_array[0]:g} Hz is not a multiple of the "
            f"spacing {delta_f:g} Hz, the grid cannot be extended to 0 Hz")
    snapped = first + np.arange(len(frequency_array)) * delta_f
    shift = np.max(np.abs(frequency_array - snapped))
    if shift > 1e-6 * delta_f:
        logger.info(f"{label}: frequency labels snapped to a uniform grid "
                    f"(spacing {delta_f:g} Hz, largest shift {shift:.3g} Hz)")
    frequency_array = snapped

    if k_first > 0:
        frequency_array = np.concatenate([np.arange(k_first) * delta_f, frequency_array])
        psd_array = np.concatenate([np.full(k_first, np.nan), psd_array])
    top = frequency_array[-1]
    power = _next_power_of_two(top)
    if not _is_power_of_two(top) and abs(power - top - delta_f) < 1e-6 * delta_f:
        frequency_array = np.append(frequency_array, power)
        psd_array = np.append(psd_array, np.nan)
    return frequency_array, psd_array


class AutoCovarianceFunction(object):

    def __init__(self, acf_array=None, sampling_frequency=None, frequency_array=None,
                 psd_array=None, minimum_frequency=None, maximum_frequency=None,
                 fill_value=None, source=None):
        """
        Instantiate a new AutoCovarianceFunction object.

        The autocovariance function (ACF) of stationary Gaussian noise defines the
        noise covariance matrix of a segment of N samples, C_ij = acf[|i - j|].
        It is used by
        :code:`bilby.gw.likelihood.TimeDomainGravitationalWaveTransient`, which
        builds C^-1 for each analysis segment from the first N ACF samples.
        Normally created with one of the :code:`from_...` methods.

        Parameters
        ==========
        acf_array: array_like
            ACF at lags 0, 1, ..., L - 1 (one full period).
        sampling_frequency: float
            Sampling frequency of the lags (Hz); must equal that of the data.
        frequency_array, psd_array: array_like, optional
            The patched one-sided PSD the ACF was computed from.
        minimum_frequency, maximum_frequency: float, optional
            Band outside which the PSD was patched.
        fill_value: tuple, optional
            (low, high) patch levels.
        source: str, optional
            Description of where the ACF comes from (stored in the meta data).
        """
        self.acf_array = None if acf_array is None else np.asarray(acf_array, dtype=float)
        self.sampling_frequency = sampling_frequency
        self.frequency_array = frequency_array
        self.psd_array = psd_array
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.fill_value = fill_value
        self.source = source
        self._gohberg_semencul_cache = dict()

    def __repr__(self):
        return self.__class__.__name__ + '(sampling_frequency={}, duration={}, source={})'.format(
            self.sampling_frequency, self.duration, self.source)

    def __eq__(self, other):
        return (
            isinstance(other, AutoCovarianceFunction)
            and self.sampling_frequency == other.sampling_frequency
            and np.array_equal(self.acf_array, other.acf_array)
        )

    @property
    def delta_t(self):
        return 1.0 / self.sampling_frequency

    @property
    def number_of_samples(self):
        return len(self.acf_array)

    @property
    def duration(self):
        """ Period of the ACF in seconds (1 / frequency spacing of the PSD). """
        if self.acf_array is None:
            return None
        return self.number_of_samples / self.sampling_frequency

    @property
    def meta_data(self):
        return dict(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            fill_value=self.fill_value,
            source=self.source,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_power_spectral_density_array(
            cls, frequency_array, psd_array, minimum_frequency, maximum_frequency,
            sampling_frequency=None, fill_value=None, fill_multiplier=1e4,
            fill_reference="max_in_band", source="power spectral density array"):
        """ ACF from a one-sided PSD given with its frequency array, used as is.

        The PSD is not interpolated or resampled. Bins between 0 Hz and the first
        frequency, and a Nyquist bin missing from a grid that stops one bin short
        of a power of two, are added; all such bins lie outside the band and are
        set by the patch. The sampling frequency is implied by the grid,
        2 x (highest frequency).

        Parameters
        ==========
        frequency_array, psd_array: array_like
            Uniformly spaced frequencies (Hz) and PSD values (strain^2 / Hz).
        minimum_frequency, maximum_frequency: float
            Band; every bin below minimum_frequency or above maximum_frequency is
            set to the patch level (hard step).
        sampling_frequency: float, optional
            If given, must equal the implied sampling frequency (a warning is
            logged and a ValueError raised otherwise).
        fill_value: float, optional
            Absolute patch level; overrides fill_multiplier and fill_reference.
        fill_multiplier: float
            Patch level = fill_multiplier x reference value (default 1e4).
        fill_reference: str
            "max_in_band" (default), "median_in_band", or "patched_region_max"
            (separate maxima below minimum_frequency and above maximum_frequency).
        source: str
            Description stored in the meta data.

        Returns
        =======
        AutoCovarianceFunction
        """
        label = source
        frequency_array, psd_array = _uniform_frequency_grid(frequency_array, psd_array, label)
        implied_sampling_frequency = 2.0 * frequency_array[-1]
        _check_sampling_frequency(implied_sampling_frequency, sampling_frequency, label)

        finite = np.isfinite(psd_array)
        lowest, highest = frequency_array[finite][0], frequency_array[finite][-1]
        if minimum_frequency < lowest:
            raise ValueError(f"{label}: minimum_frequency = {minimum_frequency:g} Hz is below the lowest "
                             f"PSD value ({lowest:g} Hz)")
        if maximum_frequency > highest:
            raise ValueError(f"{label}: maximum_frequency = {maximum_frequency:g} Hz is above the highest "
                             f"PSD value ({highest:g} Hz)")
        if maximum_frequency >= implied_sampling_frequency / 2:
            raise ValueError(f"{label}: maximum_frequency = {maximum_frequency:g} Hz must be below the "
                             f"Nyquist frequency ({implied_sampling_frequency / 2:g} Hz)")

        band = (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)
        below = frequency_array < minimum_frequency
        above = frequency_array > maximum_frequency
        if not np.all(np.isfinite(psd_array[band])):
            raise ValueError(f"{label}: PSD has missing values inside "
                             f"[{minimum_frequency:g}, {maximum_frequency:g}] Hz")
        if fill_value is not None:
            fill_low = fill_high = float(fill_value)
        elif fill_reference == "max_in_band":
            fill_low = fill_high = fill_multiplier * np.max(psd_array[band])
        elif fill_reference == "median_in_band":
            fill_low = fill_high = fill_multiplier * np.median(psd_array[band])
        elif fill_reference == "patched_region_max":
            def region_max(mask, side):
                values = psd_array[mask][np.isfinite(psd_array[mask])]
                if values.size == 0:
                    raise ValueError(f"{label}: no PSD values {side}; use another fill_reference "
                                     "or give fill_value")
                return np.max(values)
            fill_low = fill_multiplier * region_max(below, f"below {minimum_frequency:g} Hz")
            fill_high = fill_multiplier * region_max(above, f"above {maximum_frequency:g} Hz")
        else:
            raise ValueError(f"fill_reference must be one of {FILL_REFERENCES}, not {fill_reference!r}")

        patched = psd_array.copy()
        patched[below] = fill_low
        patched[above] = fill_high
        acf_array = 0.5 * np.real(np.fft.irfft(patched)) * implied_sampling_frequency
        logger.info(
            f"{label}: PSD grid 0 - {frequency_array[-1]:g} Hz, spacing {frequency_array[1]:g} Hz; "
            f"sampling frequency {implied_sampling_frequency:g} Hz; ACF period "
            f"{len(acf_array) / implied_sampling_frequency:g} s; patched outside "
            f"[{minimum_frequency:g}, {maximum_frequency:g}] Hz with {fill_low:.3e}"
            + ("" if fill_low == fill_high else f" / {fill_high:.3e}"))
        return cls(acf_array=acf_array, sampling_frequency=implied_sampling_frequency,
                   frequency_array=frequency_array, psd_array=patched,
                   minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency,
                   fill_value=(float(fill_low), float(fill_high)), source=source)

    @classmethod
    def from_amplitude_spectral_density_array(cls, frequency_array, asd_array, minimum_frequency,
                                              maximum_frequency, **kwargs):
        """ As :code:`from_power_spectral_density_array`, from an ASD. """
        kwargs.setdefault("source", "amplitude spectral density array")
        return cls.from_power_spectral_density_array(
            frequency_array, np.asarray(asd_array) ** 2, minimum_frequency, maximum_frequency, **kwargs)

    @classmethod
    def from_power_spectral_density(
            cls, power_spectral_density, minimum_frequency, maximum_frequency, analysis_duration,
            sampling_frequency=2048, duration_factor=16, **kwargs):
        """ ACF from a :code:`bilby.gw.detector.PowerSpectralDensity` (e.g. a
        design-sensitivity curve), evaluated on the grid 0, df, ..., f_s / 2
        with df = 1 / (duration_factor x analysis_duration), so that the ACF
        period is duration_factor times the analysis segment.

        Parameters
        ==========
        power_spectral_density: bilby.gw.detector.PowerSpectralDensity
        minimum_frequency, maximum_frequency: float
            Band; see :code:`from_power_spectral_density_array`.
        analysis_duration: float
            Longest analysis segment (s) the ACF will be used for.
        sampling_frequency: float
            Sampling frequency of the data (default 2048 Hz).
        duration_factor: float
            ACF period / analysis_duration (default 16).
        kwargs:
            fill_value, fill_multiplier, fill_reference (see
            :code:`from_power_spectral_density_array`).
        """
        delta_f = 1.0 / (duration_factor * analysis_duration)
        frequency_array = np.arange(int(round(sampling_frequency / 2 / delta_f)) + 1) * delta_f
        with np.errstate(invalid="ignore", divide="ignore"):
            psd_array = power_spectral_density.get_power_spectral_density_array(frequency_array)
        name = power_spectral_density.psd_file or power_spectral_density.asd_file or "array"
        kwargs.setdefault("source", f"PowerSpectralDensity({str(name).split('/')[-1]})")
        return cls.from_power_spectral_density_array(
            frequency_array, psd_array, minimum_frequency, maximum_frequency,
            sampling_frequency=sampling_frequency, **kwargs)

    @classmethod
    def from_power_spectral_density_file(cls, psd_file, minimum_frequency, maximum_frequency,
                                         analysis_duration, **kwargs):
        """ :code:`from_power_spectral_density` for a PSD file (or a file name
        from bilby's noise_curves directory). """
        from .psd import PowerSpectralDensity
        return cls.from_power_spectral_density(
            PowerSpectralDensity.from_power_spectral_density_file(psd_file),
            minimum_frequency, maximum_frequency, analysis_duration, **kwargs)

    @classmethod
    def from_amplitude_spectral_density_file(cls, asd_file, minimum_frequency, maximum_frequency,
                                             analysis_duration, **kwargs):
        """ :code:`from_power_spectral_density` for an ASD file (or a file name
        from bilby's noise_curves directory). """
        from .psd import PowerSpectralDensity
        return cls.from_power_spectral_density(
            PowerSpectralDensity.from_amplitude_spectral_density_file(asd_file),
            minimum_frequency, maximum_frequency, analysis_duration, **kwargs)

    @classmethod
    def from_time_domain_strain(
            cls, time_domain_strain, sampling_frequency, analysis_duration, minimum_frequency=None,
            maximum_frequency=None, method="welch", nperseg=None, duration_factor=16, **kwargs):
        """ ACF estimated from noise data.

        Parameters
        ==========
        time_domain_strain: array_like
            Noise data (e.g. off-source), uniformly sampled.
        sampling_frequency: float
            Sampling frequency of the data (Hz).
        analysis_duration: float
            Longest analysis segment (s) the ACF will be used for.
        minimum_frequency, maximum_frequency: float
            Band for the patch (method "welch" only).
        method: str
            "welch" (default): PSD by Welch's method (Hann window, 50 % overlap,
            median average, as in the :code:`ringdown` package) with segments of
            nperseg samples, default the next power of two >= duration_factor x
            the analysis segment; then patched and transformed.
            "direct": direct (biased) autocorrelation of the data; no band and no
            patch; discouraged.
        nperseg: int, optional
            Welch segment length in samples.
        duration_factor: float
            See nperseg (default 16).
        kwargs:
            fill_value, fill_multiplier, fill_reference (method "welch").
        """
        from scipy import signal
        time_domain_strain = np.asarray(time_domain_strain, dtype=float)
        number_of_samples = int(round(analysis_duration * sampling_frequency))
        if method == "direct":
            logger.warning("ACF by direct autocorrelation of the data (no band, no patch); "
                           "the Welch method is preferred")
            correlation = signal.correlate(time_domain_strain, time_domain_strain)
            acf_array = np.fft.ifftshift(correlation)[:len(time_domain_strain)] / len(time_domain_strain)
            return cls(acf_array=acf_array, sampling_frequency=sampling_frequency,
                       source="direct autocorrelation of time-domain strain")
        if method != "welch":
            raise ValueError("method must be 'welch' or 'direct'")
        if minimum_frequency is None or maximum_frequency is None:
            raise ValueError("method 'welch' needs minimum_frequency and maximum_frequency")
        nperseg = nperseg or _next_power_of_two(duration_factor * number_of_samples)
        if nperseg > len(time_domain_strain):
            raise ValueError(
                f"Welch segments of {nperseg} samples ({nperseg / sampling_frequency:g} s) need more "
                f"data than given ({len(time_domain_strain) / sampling_frequency:g} s)")
        frequency_array, psd_array = signal.welch(
            time_domain_strain, fs=sampling_frequency, nperseg=nperseg, window="hann", average="median")
        number_of_segments = (len(time_domain_strain) - nperseg) // (nperseg // 2) + 1
        logger.info(f"Welch PSD from {len(time_domain_strain) / sampling_frequency:g} s of data: "
                    f"{number_of_segments} segments of {nperseg / sampling_frequency:g} s, median average")
        if number_of_segments < 8:
            logger.warning(f"Only {number_of_segments} Welch segments; the PSD estimate will be noisy")
        kwargs.setdefault("source", f"Welch PSD ({number_of_segments} x {nperseg / sampling_frequency:g} s)")
        return cls.from_power_spectral_density_array(
            frequency_array, psd_array, minimum_frequency, maximum_frequency,
            sampling_frequency=sampling_frequency, **kwargs)

    @classmethod
    def from_file(cls, filename):
        """ Read an ACF saved with :code:`save` (columns: lag (s), ACF). """
        data = np.loadtxt(filename)
        return cls(acf_array=data[:, 1], sampling_frequency=1.0 / (data[1, 0] - data[0, 0]),
                   source=f"file {filename}")

    # ------------------------------------------------------------------
    # Use
    # ------------------------------------------------------------------
    def check_analysis_length(self, number_of_samples):
        """ Warn if the ACF period is less than twice the analysis segment;
        raise a ValueError if it is shorter than the segment. """
        if number_of_samples > self.number_of_samples:
            raise ValueError(
                f"Analysis segment of {number_of_samples} samples "
                f"({number_of_samples / self.sampling_frequency:g} s) is longer than the ACF "
                f"({self.number_of_samples} samples = {self.duration:g} s)")
        if 2 * number_of_samples > self.number_of_samples:
            logger.warning(
                f"ACF period is only {self.number_of_samples / number_of_samples:.2f} x the analysis "
                f"segment ({self.number_of_samples} vs {number_of_samples} samples); less than 2x can "
                "make the Toeplitz solve unstable.")

    def get_acf_array(self, number_of_samples):
        """ First number_of_samples lags of the ACF. """
        self.check_analysis_length(number_of_samples)
        return self.acf_array[:number_of_samples]

    def gohberg_semencul_vectors(self, number_of_samples):
        """ Gohberg-Semencul generators (x, y) of C^-1 for a segment of
        number_of_samples samples (cached). """
        if number_of_samples not in self._gohberg_semencul_cache:
            self._gohberg_semencul_cache[number_of_samples] = gohberg_semencul_vectors(
                self.get_acf_array(number_of_samples))
        return self._gohberg_semencul_cache[number_of_samples]

    def save(self, filename):
        """ Save the ACF as two columns: lag (s), ACF. """
        lags = np.arange(self.number_of_samples) / self.sampling_frequency
        np.savetxt(filename, np.array([lags, self.acf_array]).T, header="lag(s) acf")


def read_power_spectral_densities_from_hdf5(filename, interferometers, label=None):
    """ Read PSDs with their frequency arrays from a GWOSC parameter-estimation
    release file.

    Supported layouts
    =================
    - psds/{ifo lower}_psd/frequency and .../spectrum (e.g. GW150914_data.h5)
    - {label}/psds/{IFO}: (N, 2) array [frequency, PSD] (GWTC-2/2.1/3/4 PESummary
      files). :code:`label` (e.g. "C01:IMRPhenomXPHM") is required if more than
      one analysis in the file has PSDs; the error lists the available labels.

    Parameters
    ==========
    filename: str
    interferometers: list
        Interferometer names, e.g. ["H1", "L1"].
    label: str, optional

    Returns
    =======
    dict: {ifo: (frequency_array, psd_array)}
    """
    import h5py
    interferometers = [getattr(ifo, "name", ifo) for ifo in interferometers]
    output = dict()
    with h5py.File(filename, "r") as ff:
        if "psds" in ff and all(f"{ifo.lower()}_psd" in ff["psds"] for ifo in interferometers):
            for ifo in interferometers:
                group = ff["psds"][f"{ifo.lower()}_psd"]
                output[ifo] = (group["frequency"][:], group["spectrum"][:])
            return output
        labels = [key for key in ff.keys() if isinstance(ff[key], h5py.Group) and "psds" in ff[key]]
        if not labels:
            raise ValueError(f"{filename}: no PSDs found (neither psds/<ifo>_psd nor <label>/psds)")
        if label is None:
            if len(labels) > 1:
                raise ValueError(f"{filename}: several analyses have PSDs, choose label= one of {labels}")
            label = labels[0]
        if label not in labels:
            raise ValueError(f"{filename}: label {label!r} has no PSDs; available: {labels}")
        group = ff[label]["psds"]
        for ifo in interferometers:
            if ifo not in group:
                raise ValueError(f"{filename}: {label}/psds has no {ifo}; available: {list(group.keys())}")
            array = group[ifo][:]
            output[ifo] = (array[:, 0], array[:, 1])
    return output


__all__ = ["AutoCovarianceFunction", "read_power_spectral_densities_from_hdf5"]
