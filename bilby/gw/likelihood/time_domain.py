import copy

import numpy as np

from ...core.utils import logger
from ..time_domain_utils import PLACEMENT_METHODS, GohbergSemenculInverse
from .base import GravitationalWaveTransient

ANALYSIS_SEGMENTS = ("imr", "inspiral", "post_inspiral")


class TimeDomainGravitationalWaveTransient(GravitationalWaveTransient):
    """ A gravitational-wave transient likelihood evaluated in the time domain.

    The noise in each interferometer is a stationary Gaussian process with
    covariance matrix :math:`C_{ij} = \\rho(|i - j|)`, where :math:`\\rho` is
    the autocovariance function (ACF) stored on the interferometer
    (:code:`interferometer.autocovariance_function`). For data :math:`d` and a
    template :math:`h` on the analysed samples of each interferometer

    .. math::

        \\ln \\mathcal{L} = \\sum_{\\rm ifo} \\left[ d^T C^{-1} h
        - \\frac{1}{2} h^T C^{-1} h - \\frac{1}{2} d^T C^{-1} d \\right]

    :code:`log_likelihood_ratio` returns the first two terms, and
    :code:`noise_log_likelihood` the last. :math:`C^{-1}` is applied with its
    Gohberg-Semencul representation (computed once per interferometer), and
    :math:`d^T C^{-1}` and :math:`d^T C^{-1} d` are computed once.

    The waveform generator must have a time-domain source model that returns
    the polarizations on its own sample grid together with an 'epoch' (the
    time of the first sample relative to the model's merger), e.g.
    :code:`bilby.gw.source.lal_binary_black_hole_time_domain`. The template is
    projected onto each interferometer and placed so that the model's merger
    arrives at :code:`geocent_time` plus the time delay from the geocenter.
    Template samples outside the analysed samples are dropped; nothing wraps
    around.

    The analysed samples of each interferometer are either all of its data
    ("imr"), or the samples before / after a cut time ("inspiral" /
    "post_inspiral"). The cut time is fixed once from
    :code:`cut_reference_parameters`:
    :math:`t_{\\rm cut} = t_c^{\\rm ref} + \\Delta t_{\\rm ifo}(\\alpha^{\\rm ref},
    \\delta^{\\rm ref}) + \\tau_{\\rm cut}`. The "post_inspiral" segment starts at
    the first sample after :math:`t_{\\rm cut}` and the "inspiral" segment ends
    just before it, so the two are complementary.

    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of :code:`bilby.gw.detector.Interferometer` instances, with
        time-domain strain data and an autocovariance function set (e.g. with
        :code:`set_autocovariance_function_from_power_spectral_density` or by
        assigning a :code:`bilby.gw.detector.AutoCovarianceFunction`).
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        A WaveformGenerator with a time-domain source model.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood. This uses a
        look up table calculated at run time. The distance prior is set to be a
        delta function at the minimum distance allowed in the prior being
        marginalised over.
    priors: dict, optional
        If given, used in the distance marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table. If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the provided
        string or a default location: '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location, as in
        :code:`GravitationalWaveTransient`.
    time_reference: str, optional
        Name of the reference for the sampled time parameter, as in
        :code:`GravitationalWaveTransient`.
    analysis_segment: str, optional
        "imr" (default; all the data), "inspiral" or "post_inspiral".
    segment_cut_time: float, optional
        :math:`\\tau_{\\rm cut}` in seconds, the cut time relative to the
        arrival of the reference merger at each interferometer. Required for
        "inspiral" and "post_inspiral".
    cut_reference_parameters: dict, optional
        Dictionary with 'ra', 'dec' and 'geocent_time' (or the parameters of
        :code:`reference_frame` / :code:`time_reference`) fixing the cut time.
        Required for "inspiral" and "post_inspiral".
    segment_duration: float, optional
        Length (s) of the "inspiral" / "post_inspiral" segment, ending / starting
        at the cut. Default: up to the start / end of the data.
    placement: str, optional
        How the template is placed on the data samples, see
        :code:`bilby.gw.time_domain_utils.place_time_domain_signal`:
        "nearest" (default), "subsample" or "fd_shift". With "subsample",
        :code:`geocent_time` is the time of the peak of
        :math:`h_+^2 + h_\\times^2` rather than the model's merger time (t=0).
        Use the same placement for injection and analysis.
    time_marginalization, phase_marginalization, calibration_marginalization: bool
        Not implemented for this likelihood; must be False.

    Returns
    =======
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.
    """

    likelihood_domain = "time"

    def __init__(
            self, interferometers, waveform_generator, distance_marginalization=False,
            priors=None, distance_marginalization_lookup_table=None, reference_frame="sky",
            time_reference="geocenter", analysis_segment="imr", segment_cut_time=None,
            cut_reference_parameters=None, segment_duration=None, placement="nearest",
            time_marginalization=False, phase_marginalization=False,
            calibration_marginalization=False,
    ):
        for name, value in dict(
                time_marginalization=time_marginalization,
                phase_marginalization=phase_marginalization,
                calibration_marginalization=calibration_marginalization).items():
            if value:
                raise NotImplementedError(
                    f"{name} is not implemented for {self.__class__.__name__}")
        if getattr(waveform_generator, "time_domain_source_model", None) is None:
            raise ValueError(
                f"{self.__class__.__name__} needs a waveform generator with a "
                "time_domain_source_model, e.g. bilby.gw.source.lal_binary_black_hole_time_domain")
        if analysis_segment not in ANALYSIS_SEGMENTS:
            raise ValueError(
                f"analysis_segment must be one of {ANALYSIS_SEGMENTS}, not {analysis_segment!r}")
        if placement not in PLACEMENT_METHODS:
            raise ValueError(
                f"placement must be one of {PLACEMENT_METHODS}, not {placement!r}")

        super(TimeDomainGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=False,
            distance_marginalization=distance_marginalization,
            phase_marginalization=False,
            calibration_marginalization=False,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )
        self.analysis_segment = analysis_segment
        self.segment_cut_time = segment_cut_time
        self.cut_reference_parameters = cut_reference_parameters
        self.segment_duration = segment_duration
        self.placement = placement

        self._check_interferometers()
        self._setup_analysis_windows()
        self._setup_noise_model()

    def __repr__(self):
        return self.__class__.__name__ + (
            '(interferometers={},\n\twaveform_generator={},\n\tdistance_marginalization={}, '
            'priors={}, analysis_segment={}, segment_cut_time={}, segment_duration={}, '
            'placement={})'.format(
                self.interferometers, self.waveform_generator, self.distance_marginalization,
                self.priors, self.analysis_segment, self.segment_cut_time,
                self.segment_duration, self.placement))

    # ------------------------------------------------------------------
    # Set up (done once)
    # ------------------------------------------------------------------
    def _check_interferometers(self):
        for interferometer in self.interferometers:
            if interferometer.autocovariance_function is None:
                raise ValueError(
                    f"{interferometer.name}: no autocovariance_function set; use e.g. "
                    "interferometer.set_autocovariance_function_from_power_spectral_density")
            acf_sampling_frequency = interferometer.autocovariance_function.sampling_frequency
            if not np.isclose(acf_sampling_frequency, interferometer.sampling_frequency,
                              rtol=1e-9, atol=0):
                raise ValueError(
                    f"{interferometer.name}: autocovariance function sampling frequency "
                    f"{acf_sampling_frequency:g} Hz differs from the data's "
                    f"{interferometer.sampling_frequency:g} Hz")

    def _setup_analysis_windows(self):
        """ Indices of the analysed samples of each interferometer. """
        if self.analysis_segment == "imr":
            if self.segment_duration is not None:
                logger.warning("segment_duration is ignored for analysis_segment='imr' "
                               "(all the data are analysed)")
            if self.segment_cut_time is not None:
                logger.warning("segment_cut_time is ignored for analysis_segment='imr'")
            reference = None
        else:
            if self.segment_cut_time is None or self.cut_reference_parameters is None:
                raise ValueError(
                    f"analysis_segment={self.analysis_segment!r} needs segment_cut_time and "
                    "cut_reference_parameters")
            reference = self.get_sky_frame_parameters(dict(self.cut_reference_parameters))

        self.analysis_windows = dict()
        for interferometer in self.interferometers:
            number_of_data_samples = len(interferometer.time_array)
            sampling_frequency = interferometer.sampling_frequency
            start_time = interferometer.strain_data.start_time
            if reference is None:
                start_index, end_index, cut_time = 0, number_of_data_samples, None
            else:
                delay = interferometer.time_delay_from_geocenter(
                    reference["ra"], reference["dec"], reference["geocent_time"])
                # subtract the two GPS times first
                cut_offset = (reference["geocent_time"] - start_time) + delay + self.segment_cut_time
                cut_time = start_time + cut_offset
                # first sample strictly after the cut: inspiral and post_inspiral are complementary
                cut_index = int(np.floor(cut_offset * sampling_frequency)) + 1
                length = (None if self.segment_duration is None
                          else int(round(self.segment_duration * sampling_frequency)))
                if self.analysis_segment == "inspiral":
                    end_index = cut_index
                    start_index = 0 if length is None else cut_index - length
                else:
                    start_index = cut_index
                    end_index = number_of_data_samples if length is None else cut_index + length
                if start_index < 0 or end_index > number_of_data_samples or end_index - start_index < 2:
                    raise ValueError(
                        f"{interferometer.name}: {self.analysis_segment} segment "
                        f"[{start_index}:{end_index}] is outside the data [0:{number_of_data_samples}] "
                        f"(cut time {cut_time:.6f})")
            self.analysis_windows[interferometer.name] = dict(
                start_index=int(start_index),
                end_index=int(end_index),
                number_of_samples=int(end_index - start_index),
                start_time=float(start_time + start_index / sampling_frequency),
                end_time=float(start_time + (end_index - 1) / sampling_frequency),
                cut_time=None if cut_time is None else float(cut_time),
            )
            window = self.analysis_windows[interferometer.name]
            logger.info(
                f"{interferometer.name} [{self.analysis_segment}]: samples "
                f"[{window['start_index']}:{window['end_index']}] (N = {window['number_of_samples']}), "
                f"GPS {window['start_time']:.6f} - {window['end_time']:.6f}"
                + ("" if cut_time is None else f", cut at {cut_time:.6f}"))

    def _setup_noise_model(self):
        """ Gohberg-Semencul vectors, d^T C^-1 and d^T C^-1 d for each interferometer. """
        self._inverse_covariance = dict()
        self._data_times_inverse_covariance = dict()
        self._data_inverse_covariance_data = dict()
        for interferometer in self.interferometers:
            window = self.analysis_windows[interferometer.name]
            acf = interferometer.autocovariance_function
            inverse_covariance = GohbergSemenculInverse(
                *acf.gohberg_semencul_vectors(window["number_of_samples"]))
            data = self._analysed_data(interferometer)
            data_times_inverse_covariance = inverse_covariance(data)
            self._inverse_covariance[interferometer.name] = inverse_covariance
            self._data_times_inverse_covariance[interferometer.name] = data_times_inverse_covariance
            self._data_inverse_covariance_data[interferometer.name] = float(
                np.dot(data_times_inverse_covariance, data))
            logger.info(
                f"{interferometer.name}: ACF period {acf.duration:g} s = "
                f"{acf.number_of_samples / window['number_of_samples']:.1f} x analysed segment")

    def _analysed_data(self, interferometer):
        window = self.analysis_windows[interferometer.name]
        return np.asarray(interferometer.time_domain_strain, dtype=float)[
            window["start_index"]:window["end_index"]]

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """ Project the time-domain polarizations onto the interferometer and
        place them on its analysed samples.

        Parameters
        ==========
        signal_polarizations: dict
            Output of the time-domain source model (polarizations and 'epoch').
        interferometer: bilby.gw.detector.Interferometer
        parameters: dict

        Returns
        =======
        array_like: The template on the analysed samples.
        """
        window = self.analysis_windows[interferometer.name]
        return interferometer.get_time_domain_detector_response(
            signal_polarizations, parameters, start_index=window["start_index"],
            number_of_samples=window["number_of_samples"], placement=self.placement)

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters):
        """ Compute :math:`d^T C^{-1} h`, :math:`h^T C^{-1} h` and the matched
        filter SNR on the analysed samples of one interferometer.

        Parameters
        ----------
        waveform_polarizations: dict
            Output of the time-domain source model.
        interferometer: bilby.gw.detector.Interferometer
        return_array: bool
            Unused (no time or calibration marginalization); kept for the
            :code:`GravitationalWaveTransient` interface.
        parameters: dict

        Returns
        -------
        calculated_snrs: _CalculatedSNRs
        """
        signal = self._compute_full_waveform(
            signal_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        d_inner_h = float(np.dot(self._data_times_inverse_covariance[interferometer.name], signal))
        optimal_snr_squared = float(np.dot(signal, self._inverse_covariance[interferometer.name](signal)))
        if optimal_snr_squared > 0:
            matched_filter_snr = d_inner_h / optimal_snr_squared ** 0.5
        else:
            matched_filter_snr = 0.0
        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=matched_filter_snr,
            d_inner_h_array=None,
            optimal_snr_squared_array=None,
        )

    def _calculate_noise_log_likelihood(self):
        return -0.5 * sum(self._data_inverse_covariance_data.values())

    def log_likelihood_ratio(self, parameters):
        parameters = copy.deepcopy(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        waveform_polarizations = self.waveform_generator.time_domain_strain(parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        total_snrs = self._CalculatedSNRs()
        for interferometer in self.interferometers:
            total_snrs += self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
        log_l = self.compute_log_likelihood_from_snrs(total_snrs, parameters=parameters)
        return float(np.real(log_l))

    def compute_per_detector_log_likelihood(self, parameters):
        parameters.update(self.get_sky_frame_parameters(parameters))
        waveform_polarizations = self.waveform_generator.time_domain_strain(parameters)
        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            parameters['{}_log_likelihood'.format(interferometer.name)] = \
                self.compute_log_likelihood_from_snrs(per_detector_snr, parameters=parameters)
        return parameters.copy()

    # ------------------------------------------------------------------
    # Reconstruction of marginalized parameters
    # ------------------------------------------------------------------
    def generate_posterior_sample_from_marginalized_likelihood(self, parameters):
        """ Reconstruct the distance posterior from a run which used the
        distance-marginalized likelihood (Eq. (C29-C32) of
        https://arxiv.org/abs/1809.02293).

        Returns
        =======
        sample: dict
            Returns the parameters with new samples.
        """
        if not self.distance_marginalization:
            return parameters
        signal_polarizations = copy.deepcopy(
            self.waveform_generator.time_domain_strain(parameters))
        parameters['luminosity_distance'] = self.generate_distance_sample_from_marginalized_likelihood(
            signal_polarizations=signal_polarizations, parameters=parameters)
        return parameters.copy()

    def generate_distance_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, *, parameters):
        """ Generate a single sample from the posterior distribution for
        luminosity distance when using the distance-marginalized likelihood.

        Parameters
        ==========
        signal_polarizations: dict, optional
            Output of the time-domain source model. These are rescaled in place
            after the distance sample is generated.

        Returns
        =======
        new_distance: float
            Sample from the distance posterior.
        """
        if signal_polarizations is None:
            parameters.update(self.get_sky_frame_parameters(parameters))
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.time_domain_strain(parameters))
        return super(TimeDomainGravitationalWaveTransient, self).\
            generate_distance_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations, parameters=parameters)

    def _rescale_signal(self, signal, new_distance):
        for mode in signal:
            if mode != "epoch":
                signal[mode] *= self._ref_dist / new_distance

    @property
    def meta_data(self):
        meta_data = super(TimeDomainGravitationalWaveTransient, self).meta_data
        meta_data.update(
            likelihood_domain=self.likelihood_domain,
            analysis_segment=self.analysis_segment,
            segment_cut_time=self.segment_cut_time,
            cut_reference_parameters=(
                None if self.cut_reference_parameters is None
                else dict(self.cut_reference_parameters)),
            segment_duration=self.segment_duration,
            placement=self.placement,
            analysis_windows=copy.deepcopy(self.analysis_windows),
        )
        return meta_data
