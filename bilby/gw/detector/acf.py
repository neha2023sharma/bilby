import numpy as np

from ...core.utils import logger
from ..time_domain_utils import gohberg_semencul_vectors


class AutoCovarianceFunction(object):

    def __init__(self, acf_array, sampling_frequency, minimum_frequency=None,
                 maximum_frequency=None, fill_value=None):
        """
        Instantiate a new AutoCovarianceFunction object.

        The autocovariance function (ACF) of stationary Gaussian noise defines
        the noise covariance matrix of a segment of N samples,
        :math:`C_{ij} = \\rho(|i - j|)`, where :math:`\\rho` is the ACF.

        Examples
        ========
        From a one-sided power spectral density and its frequency array:

        .. code-block:: python

            >>> acf = AutoCovarianceFunction.from_power_spectral_density_array(
            ...     frequency_array, psd_array, minimum_frequency=20, maximum_frequency=896)

        From a precomputed ACF:

        .. code-block:: python

            >>> acf = AutoCovarianceFunction(acf_array=acf_array, sampling_frequency=2048)

        Parameters
        ==========
        acf_array: array_like
            Autocovariance function at lags 0, 1/sampling_frequency,
            2/sampling_frequency, ... (strain^2).
        sampling_frequency: float
            Sampling frequency of the lags (Hz). Must equal the sampling
            frequency of the data it is used with.
        minimum_frequency: float, optional
            Lower edge of the frequency band used to compute the ACF (Hz).
        maximum_frequency: float, optional
            Upper edge of the frequency band used to compute the ACF (Hz).
        fill_value: float, optional
            Multiple of the maximum in-band PSD used outside the band.
        """
        self.acf_array = np.asarray(acf_array, dtype=float)
        self.sampling_frequency = float(sampling_frequency)
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.fill_value = fill_value

    @property
    def number_of_samples(self):
        """ Number of lags in the ACF. """
        return len(self.acf_array)

    @property
    def duration(self):
        """ Duration of the ACF (s). """
        return self.number_of_samples / self.sampling_frequency

    @property
    def meta_data(self):
        return dict(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            fill_value=self.fill_value,
        )

    @classmethod
    def from_power_spectral_density_array(
            cls, frequency_array, psd_array, minimum_frequency, maximum_frequency,
            sampling_frequency=None, fill_value=1e4):
        """ Compute the ACF from a one-sided power spectral density.

        The PSD below `minimum_frequency` and above `maximum_frequency` is set
        to `fill_value` times its maximum value in
        [`minimum_frequency`, `maximum_frequency`]. If `frequency_array` does
        not start at 0 Hz, bins down to 0 Hz are added and set to the same
        value. The ACF is then

        .. math::

            \\rho = \\frac{f_s}{2} \\, \\mathrm{IRFFT}(S),

        where :math:`f_s = 2 f_{\\max}` is twice the highest frequency in
        `frequency_array`.

        Parameters
        ==========
        frequency_array: array_like
            Uniformly spaced frequencies (Hz), up to half the sampling
            frequency. The first frequency must be a multiple of the spacing.
        psd_array: array_like
            One-sided PSD at `frequency_array` (strain^2 / Hz).
        minimum_frequency: float
            Lower edge of the frequency band (Hz).
        maximum_frequency: float
            Upper edge of the frequency band (Hz).
        sampling_frequency: float, optional
            Sampling frequency of the data (Hz). If it differs from twice the
            highest frequency in `frequency_array`, a warning is logged and
            the latter is used.
        fill_value: float
            Multiple of the maximum in-band PSD used outside the band
            (default 1e4).

        Returns
        =======
        AutoCovarianceFunction
        """
        frequency_array = np.asarray(frequency_array, dtype=float)
        psd_array = np.asarray(psd_array, dtype=float)
        delta_f = frequency_array[1] - frequency_array[0]
        number_missing = int(round(frequency_array[0] / delta_f))
        if number_missing > 0:
            frequency_array = np.concatenate([np.arange(number_missing) * delta_f, frequency_array])
            psd_array = np.concatenate([np.zeros(number_missing), psd_array])

        psd_sampling_frequency = 2 * frequency_array[-1]
        if sampling_frequency is not None and not np.isclose(
                sampling_frequency, psd_sampling_frequency, rtol=1e-9, atol=0):
            logger.warning(
                f"The sampling frequency given ({sampling_frequency:g} Hz) is not equal to the one "
                f"computed from the PSD ({psd_sampling_frequency:g} Hz); using "
                f"{psd_sampling_frequency:g} Hz. An ACF computed with {sampling_frequency:g} Hz "
                "would be wrongly scaled and its lags wrongly spaced, giving wrong likelihood "
                "and SNR values.")

        band = (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)
        psd_array = psd_array.copy()
        psd_array[~band] = fill_value * np.max(psd_array[band])
        acf_array = 0.5 * np.fft.irfft(psd_array) * psd_sampling_frequency
        return cls(acf_array=acf_array, sampling_frequency=psd_sampling_frequency,
                   minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency,
                   fill_value=fill_value)

    @classmethod
    def from_time_domain_strain(
            cls, time_domain_strain, sampling_frequency, analysis_duration, minimum_frequency,
            maximum_frequency, segment_duration=None, duration_factor=16, fill_value=1e4):
        """ Estimate the ACF from noise data.

        The PSD of the data is estimated with Welch's method (Hann window, 50%
        overlap, median average) and passed to
        :code:`from_power_spectral_density_array`.

        Parameters
        ==========
        time_domain_strain: array_like
            Noise time series (strain).
        sampling_frequency: float
            Sampling frequency of the time series (Hz).
        analysis_duration: float
            Duration of the longest segment the ACF will be used for (s).
        minimum_frequency: float
            Lower edge of the frequency band (Hz).
        maximum_frequency: float
            Upper edge of the frequency band (Hz).
        segment_duration: float, optional
            Duration of the Welch segments (s). Default: `duration_factor`
            times `analysis_duration`, rounded up to a power-of-two number of
            samples.
        duration_factor: float
            See `segment_duration` (default 16).
        fill_value: float
            Multiple of the maximum in-band PSD used outside the band
            (default 1e4).

        Returns
        =======
        AutoCovarianceFunction
        """
        from scipy.signal import welch
        if segment_duration is None:
            nperseg = int(2 ** np.ceil(np.log2(duration_factor * analysis_duration * sampling_frequency)))
        else:
            nperseg = int(round(segment_duration * sampling_frequency))
        frequency_array, psd_array = welch(
            time_domain_strain, fs=sampling_frequency, nperseg=nperseg, window="hann",
            average="median")
        return cls.from_power_spectral_density_array(
            frequency_array, psd_array, minimum_frequency, maximum_frequency,
            sampling_frequency=sampling_frequency, fill_value=fill_value)

    @classmethod
    def from_file(cls, filename, sampling_frequency=None):
        """ Read a precomputed ACF from a text file.

        Parameters
        ==========
        filename: str
            File with either two columns, lag (s) and ACF (as written by
            :code:`save`), or a single column with the ACF.
        sampling_frequency: float, optional
            Sampling frequency of the lags (Hz). Required for a single-column
            file; for a two-column file it is computed from the lags.

        Returns
        =======
        AutoCovarianceFunction
        """
        data = np.loadtxt(filename)
        if data.ndim == 2:
            return cls(acf_array=data[:, 1], sampling_frequency=1 / (data[1, 0] - data[0, 0]))
        if sampling_frequency is None:
            raise ValueError("sampling_frequency is required for a single-column ACF file")
        return cls(acf_array=data, sampling_frequency=sampling_frequency)

    def get_acf_array(self, number_of_samples):
        """ First `number_of_samples` lags of the ACF.

        A ValueError is raised if the ACF is shorter than `number_of_samples`,
        and a warning is logged if it is shorter than twice
        `number_of_samples`.

        Parameters
        ==========
        number_of_samples: int
            Number of samples in the analysis segment.

        Returns
        =======
        array_like: The first row of the segment's covariance matrix.
        """
        if number_of_samples > self.number_of_samples:
            raise ValueError(
                f"The analysis segment ({number_of_samples} samples) is longer than the ACF "
                f"({self.number_of_samples} samples)")
        if 2 * number_of_samples > self.number_of_samples:
            logger.warning(
                f"The ACF ({self.number_of_samples} samples) is shorter than twice the analysis "
                f"segment ({number_of_samples} samples)")
        return self.acf_array[:number_of_samples]

    def gohberg_semencul_vectors(self, number_of_samples):
        """ Gohberg-Semencul vectors of the inverse covariance matrix of a
        segment of `number_of_samples` samples.

        Parameters
        ==========
        number_of_samples: int
            Number of samples in the analysis segment.

        Returns
        =======
        x, y: array_like
            See :code:`bilby.gw.time_domain_utils.gohberg_semencul_vectors`.
        """
        return gohberg_semencul_vectors(self.get_acf_array(number_of_samples))

    def save(self, filename):
        """ Save the ACF to a text file with two columns, lag (s) and ACF.

        Parameters
        ==========
        filename: str
            Name of the output file.
        """
        lags = np.arange(self.number_of_samples) / self.sampling_frequency
        np.savetxt(filename, np.array([lags, self.acf_array]).T, header="lag acf")


def read_power_spectral_densities_from_hdf5(filename, interferometers, label=None):
    """ Read power spectral densities and their frequency arrays from an HDF5
    file.

    Two layouts are supported: `psds/<ifo>_psd/frequency` and
    `psds/<ifo>_psd/spectrum` (with lower-case detector names), and
    `<label>/psds/<IFO>` holding an (N, 2) array of frequency and PSD (as in
    GWTC data releases).

    Parameters
    ==========
    filename: str
        Name of the HDF5 file.
    interferometers: list
        Detector names, e.g. ["H1", "L1"].
    label: str, optional
        Analysis label in the second layout, e.g. "C01:IMRPhenomXPHM".
        Required if the file contains more than one analysis.

    Returns
    =======
    dict: {detector name: (frequency_array, psd_array)}
    """
    import h5py
    output = dict()
    with h5py.File(filename, "r") as ff:
        if "psds" in ff:
            for ifo in interferometers:
                group = ff["psds"][f"{ifo.lower()}_psd"]
                output[ifo] = (group["frequency"][:], group["spectrum"][:])
            return output
        if label is None:
            labels = [key for key in ff.keys() if isinstance(ff[key], h5py.Group) and "psds" in ff[key]]
            if len(labels) != 1:
                raise ValueError(f"Give label, one of {labels}")
            label = labels[0]
        for ifo in interferometers:
            array = ff[label]["psds"][ifo][:]
            output[ifo] = (array[:, 0], array[:, 1])
    return output


__all__ = ["AutoCovarianceFunction", "read_power_spectral_densities_from_hdf5"]
