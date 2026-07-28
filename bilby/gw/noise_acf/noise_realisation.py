"""
Stationary Gaussian noise realisations for injection studies.

Reproduces, as a class, the logic of
``ACF_noise_and_covariance_matrix_data/Noise.py`` in
https://github.com/neha2023sharma/Heterodyning-in-time-domain: draw a
time-domain noise realisation for each detector from its PSD (bilby's
default, or a user-supplied one), truncated to the requested segment
length.
"""
import numpy as np

from ..detector import get_empty_interferometer, PowerSpectralDensity


class TimeDomainNoiseRealisation:
    """
    Draw stationary Gaussian time-domain noise realisations, one per
    detector, from a PSD.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency in Hz.
    psd_files : dict[str, str], optional
        Mapping detector name -> PSD file path, passed to
        :class:`bilby.gw.detector.PowerSpectralDensity`. Detectors not
        present in this dict use bilby's default PSD for that
        detector name.
    """

    def __init__(self, sampling_frequency=4096.0, psd_files=None):
        self.sampling_frequency = sampling_frequency
        self.psd_files = psd_files or {}

    def generate(self, detector_names, duration):
        """
        Parameters
        ----------
        detector_names : iterable of str
        duration : float
            Length, in seconds, of noise to return for each detector.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping detector name -> time-domain noise realisation of
            length ``round(duration * sampling_frequency)``.
        """
        n_samples = int(round(duration * self.sampling_frequency))
        noise = {}
        for name in detector_names:
            ifo = get_empty_interferometer(name)
            if name in self.psd_files:
                ifo.power_spectral_density = PowerSpectralDensity(
                    psd_file=self.psd_files[name]
                )
            ifo.minimum_frequency = 0
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=self.sampling_frequency,
                duration=duration,
            )
            noise[name] = np.asarray(ifo.strain_data.time_domain_strain)[:n_samples]
        return noise
