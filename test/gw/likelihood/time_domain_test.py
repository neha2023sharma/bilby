import os
import shutil
import tempfile
import unittest

import numpy as np
import scipy.linalg
from scipy.integrate import trapezoid

import bilby
from bilby.gw.detector import AutoCovarianceFunction
from bilby.gw.likelihood import TimeDomainGravitationalWaveTransient
from bilby.gw.time_domain_utils import (
    GohbergSemenculInverse,
    fractional_time_shift,
    gohberg_semencul_vectors,
    place_time_domain_signal,
)

INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1000.0, theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
)
SAMPLING_FREQUENCY = 1024
DURATION = 4
START_TIME = INJECTION["geocent_time"] - 3


def waveform_generator(source_model=bilby.gw.source.lal_binary_black_hole_time_domain):
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY, start_time=START_TIME,
        time_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXP",
                                reference_frequency=20.0, minimum_frequency=20.0))


def interferometers(noise="zero", acf=True):
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    if noise == "zero":
        ifos.set_strain_data_from_zero_noise(SAMPLING_FREQUENCY, DURATION, START_TIME)
    else:
        ifos.set_strain_data_from_power_spectral_densities_time_domain(
            SAMPLING_FREQUENCY, DURATION, START_TIME, random_state=3)
    if acf:
        ifos.set_autocovariance_functions_from_power_spectral_densities(
            analysis_duration=DURATION, minimum_frequency=20, maximum_frequency=448)
    return ifos


class TestGohbergSemencul(unittest.TestCase):
    def test_inverse_matches_dense_solve(self):
        rng = np.random.default_rng(1)
        lags = np.arange(400)
        acf = np.exp(-lags / 30.0) * np.cos(lags / 5.0)
        acf[0] += 0.1
        for number_of_samples in [5, 64, 333]:
            x, y = gohberg_semencul_vectors(acf[:number_of_samples])
            vector = rng.normal(size=number_of_samples)
            expected = np.linalg.solve(scipy.linalg.toeplitz(acf[:number_of_samples]), vector)
            np.testing.assert_allclose(GohbergSemenculInverse(x, y)(vector), expected,
                                       rtol=0, atol=1e-10 * np.max(abs(expected)))


class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(256)
        self.signal = np.exp(-0.5 * ((self.times - 100) / 6.0) ** 2) * np.cos(0.4 * self.times)

    def test_fd_shift_whole_sample_equals_nearest(self):
        for placement in ["nearest", "fd_shift"]:
            placed = place_time_domain_signal(
                self.signal, merger_index=100, arrival_index=150, start_index=20,
                number_of_samples=300, placement=placement)
            if placement == "nearest":
                nearest = placed
        np.testing.assert_allclose(placed, nearest, atol=1e-12)

    def test_fd_shift_fraction_is_exact_for_band_limited_signal(self):
        delay = 50.3
        placed = place_time_domain_signal(
            self.signal, merger_index=100, arrival_index=100 + delay, start_index=0,
            number_of_samples=400, placement="fd_shift")
        times = np.arange(400) - delay
        expected = np.exp(-0.5 * ((times - 100) / 6.0) ** 2) * np.cos(0.4 * times)
        np.testing.assert_allclose(placed, expected, atol=1e-6)

    def test_nothing_wraps(self):
        placed = place_time_domain_signal(
            self.signal, merger_index=100, arrival_index=10, start_index=0,
            number_of_samples=256, placement="nearest")
        np.testing.assert_array_equal(placed[166:], 0)
        np.testing.assert_allclose(placed[:166], self.signal[90:])

    def test_fractional_shift_length(self):
        self.assertEqual(len(fractional_time_shift(self.signal, 0.25, pad=10)), 276)


class TestAutoCovarianceFunction(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.arange(0, 512.125, 0.125)
        self.psd = 1e-46 * (1 + (50 / np.maximum(self.frequencies, 1)) ** 4)

    def test_sampling_frequency_from_grid_and_length(self):
        acf = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies, self.psd, 20, 400)
        self.assertEqual(acf.sampling_frequency, 1024)
        self.assertEqual(acf.number_of_samples, 8192)
        self.assertAlmostEqual(acf.duration, 8)

    def test_sampling_frequency_mismatch_raises(self):
        with self.assertRaises(ValueError):
            AutoCovarianceFunction.from_power_spectral_density_array(
                self.frequencies, self.psd, 20, 400, sampling_frequency=2048)

    def test_window_longer_than_acf_raises(self):
        acf = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies, self.psd, 20, 400)
        with self.assertRaises(ValueError):
            acf.get_acf_array(acf.number_of_samples + 1)

    def test_save_and_read(self):
        acf = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies, self.psd, 20, 400)
        directory = tempfile.mkdtemp()
        try:
            filename = os.path.join(directory, "acf.dat")
            acf.save(filename)
            new = AutoCovarianceFunction.from_file(filename)
        finally:
            shutil.rmtree(directory)
        self.assertAlmostEqual(new.sampling_frequency, acf.sampling_frequency)
        np.testing.assert_allclose(new.acf_array, acf.acf_array, rtol=1e-12)


class TestTimeDomainLikelihood(unittest.TestCase):
    def test_zero_noise_log_likelihood_ratio_is_half_snr_squared(self):
        cut = dict(segment_cut_time=-0.01, cut_reference_parameters=INJECTION)
        for placement in ["nearest", "subsample", "fd_shift"]:
            ifos = interferometers()
            generator = waveform_generator()
            ifos.inject_signal_time_domain(
                parameters=INJECTION, waveform_generator=generator, placement=placement)
            for segment in ["imr", "inspiral", "post_inspiral"]:
                with self.subTest(placement=placement, segment=segment):
                    likelihood = TimeDomainGravitationalWaveTransient(
                        ifos, generator, analysis_segment=segment, placement=placement,
                        **({} if segment == "imr" else cut))
                    polarizations = generator.time_domain_strain(INJECTION)
                    optimal_snr_squared = sum(
                        likelihood.calculate_snrs(polarizations, ifo, parameters=INJECTION)
                        .optimal_snr_squared for ifo in ifos)
                    self.assertGreater(optimal_snr_squared, 10)
                    self.assertAlmostEqual(
                        likelihood.log_likelihood_ratio(INJECTION) / (optimal_snr_squared / 2), 1, 8)
                    self.assertAlmostEqual(
                        likelihood.noise_log_likelihood() / (-optimal_snr_squared / 2), 1, 8)

    def test_inspiral_and_post_inspiral_are_complementary(self):
        ifos = interferometers()
        generator = waveform_generator()
        kwargs = dict(segment_cut_time=0.0, cut_reference_parameters=INJECTION)
        inspiral = TimeDomainGravitationalWaveTransient(
            ifos, generator, analysis_segment="inspiral", **kwargs)
        post_inspiral = TimeDomainGravitationalWaveTransient(
            ifos, generator, analysis_segment="post_inspiral", **kwargs)
        for ifo in ifos:
            first = inspiral.analysis_windows[ifo.name]
            second = post_inspiral.analysis_windows[ifo.name]
            self.assertEqual(first["start_index"], 0)
            self.assertEqual(first["end_index"], second["start_index"])
            self.assertEqual(second["end_index"], len(ifo.time_array))
            self.assertLessEqual(ifo.time_array[first["end_index"] - 1], first["cut_time"])
            self.assertGreater(ifo.time_array[second["start_index"]], first["cut_time"])

    def test_segment_duration(self):
        ifos = interferometers()
        likelihood = TimeDomainGravitationalWaveTransient(
            ifos, waveform_generator(), analysis_segment="post_inspiral",
            segment_cut_time=0.0, cut_reference_parameters=INJECTION, segment_duration=0.5)
        for window in likelihood.analysis_windows.values():
            self.assertEqual(window["number_of_samples"], SAMPLING_FREQUENCY // 2)

    def test_matches_dense_covariance(self):
        ifos = interferometers(noise="gaussian")
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        likelihood = TimeDomainGravitationalWaveTransient(
            ifos, generator, analysis_segment="post_inspiral", segment_cut_time=-0.1,
            cut_reference_parameters=INJECTION)
        parameters = dict(INJECTION, mass_1=37.0)
        expected = 0
        for ifo in ifos:
            window = likelihood.analysis_windows[ifo.name]
            covariance = scipy.linalg.toeplitz(
                ifo.autocovariance_function.acf_array[:window["number_of_samples"]])
            data = ifo.time_domain_strain[window["start_index"]:window["end_index"]]
            signal = likelihood._compute_full_waveform(
                generator.time_domain_strain(parameters), ifo, parameters)
            expected += data @ np.linalg.solve(covariance, signal) \
                - 0.5 * signal @ np.linalg.solve(covariance, signal)
        self.assertAlmostEqual(likelihood.log_likelihood_ratio(parameters) / expected, 1, 6)

    def test_distance_marginalization(self):
        ifos = interferometers(noise="gaussian")
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
            alpha=2, minimum=200, maximum=3000, name="luminosity_distance")
        directory = tempfile.mkdtemp()
        try:
            marginalized = TimeDomainGravitationalWaveTransient(
                ifos, generator, priors=priors.copy(), distance_marginalization=True,
                distance_marginalization_lookup_table=os.path.join(directory, "table.npz"))
            plain = TimeDomainGravitationalWaveTransient(ifos, generator)
            distances = np.linspace(200, 3000, 2000)
            log_l = np.array([plain.log_likelihood_ratio(dict(INJECTION, luminosity_distance=d))
                              for d in distances])
            expected = np.log(trapezoid(
                np.exp(log_l - log_l.max()) * priors["luminosity_distance"].prob(distances),
                distances)) + log_l.max()
            self.assertAlmostEqual(marginalized.log_likelihood_ratio(dict(INJECTION)), expected, 1)
            sample = marginalized.generate_posterior_sample_from_marginalized_likelihood(dict(INJECTION))
            self.assertTrue(200 <= sample["luminosity_distance"] <= 3000)
        finally:
            shutil.rmtree(directory)

    def test_snrs_in_post_processing(self):
        ifos = interferometers()
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        likelihood = TimeDomainGravitationalWaveTransient(ifos, generator)
        sample = dict(INJECTION)
        bilby.gw.conversion.compute_snrs(sample, likelihood)
        for ifo in ifos:
            self.assertAlmostEqual(sample[f"{ifo.name}_optimal_snr"], ifo.meta_data["optimal_SNR"], 8)

    def test_meta_data(self):
        ifos = interferometers()
        likelihood = TimeDomainGravitationalWaveTransient(
            ifos, waveform_generator(), analysis_segment="inspiral", segment_cut_time=0.0,
            cut_reference_parameters=INJECTION, placement="fd_shift")
        meta_data = likelihood.meta_data
        self.assertEqual(meta_data["likelihood_domain"], "time")
        self.assertEqual(meta_data["analysis_segment"], "inspiral")
        self.assertEqual(meta_data["placement"], "fd_shift")
        self.assertEqual(set(meta_data["analysis_windows"]), {"H1", "L1"})
        self.assertIn("autocovariance_function", meta_data["interferometers"]["H1"])

    def test_errors(self):
        generator = waveform_generator()
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(acf=False), generator)
        frequency_domain_generator = bilby.gw.WaveformGenerator(
            duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole)
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(), frequency_domain_generator)
        for key in ["time_marginalization", "phase_marginalization", "calibration_marginalization"]:
            with self.assertRaises(NotImplementedError):
                TimeDomainGravitationalWaveTransient(interferometers(), generator, **{key: True})
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(), generator, analysis_segment="inspiral")
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(
                interferometers(), generator, analysis_segment="post_inspiral",
                segment_cut_time=0.0, cut_reference_parameters=INJECTION, segment_duration=5)
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(), generator, placement="roll")
        ifos = interferometers()
        ifos[0].autocovariance_function = AutoCovarianceFunction(
            acf_array=ifos[0].autocovariance_function.acf_array, sampling_frequency=2048)
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(ifos, generator)


class TestTimeDomainData(unittest.TestCase):
    def test_noise_realisation(self):
        first, second = interferometers(noise="gaussian"), interferometers(noise="gaussian")
        for one, two in zip(first, second):
            self.assertEqual(len(one.time_domain_strain), DURATION * SAMPLING_FREQUENCY)
            self.assertAlmostEqual(one.strain_data.start_time, START_TIME)
            np.testing.assert_array_equal(one.time_domain_strain, two.time_domain_strain)
            self.assertGreater(np.std(one.time_domain_strain), 0)

    def test_injection_adds_to_stored_time_series(self):
        ifos = interferometers(noise="gaussian")
        noise = {ifo.name: ifo.time_domain_strain.copy() for ifo in ifos}
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        polarizations = generator.time_domain_strain(INJECTION)
        for ifo in ifos:
            signal = ifo.get_time_domain_detector_response(polarizations, INJECTION)
            np.testing.assert_allclose(ifo.time_domain_strain, noise[ifo.name] + signal, rtol=0, atol=1e-30)
            self.assertEqual(ifo.meta_data["injection_domain"], "time")

    def test_downsample(self):
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        sampling_frequency, duration, start_time = 4096, 16, 1000.0
        times = start_time + np.arange(duration * sampling_frequency) / sampling_frequency
        strain = np.sin(2 * np.pi * 60 * times) + np.sin(2 * np.pi * 1500 * times)
        ifo.strain_data.set_from_time_domain_strain(
            strain, sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)
        ifo.downsample_strain_data(1024, start_time=1006.0, duration=4)
        self.assertEqual(ifo.sampling_frequency, 1024)
        self.assertAlmostEqual(ifo.strain_data.start_time, 1006.0)
        self.assertEqual(len(ifo.time_domain_strain), 4096)
        # the 60 Hz line is kept and the 1500 Hz line (above the new Nyquist frequency) removed
        expected = np.sin(2 * np.pi * 60 * ifo.time_array)
        np.testing.assert_allclose(ifo.time_domain_strain, expected - np.mean(expected), atol=1e-6)

    def test_downsample_segment_outside_data_raises(self):
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        ifo.set_strain_data_from_zero_noise(4096, 8, 0)
        with self.assertRaises(ValueError):
            ifo.downsample_strain_data(2048, start_time=0.5, duration=4)

    def test_save_data(self):
        ifos = interferometers(noise="gaussian")
        directory = tempfile.mkdtemp()
        try:
            ifos.save_data(directory, label="td")
            for ifo in ifos:
                for suffix in ["time_domain_data", "acf"]:
                    self.assertTrue(os.path.isfile(os.path.join(directory, f"{ifo.name}_td_{suffix}.dat")))
        finally:
            shutil.rmtree(directory)


class TestTimeDomainSourceModels(unittest.TestCase):
    def setUp(self):
        self.time_array = np.arange(DURATION * SAMPLING_FREQUENCY) / SAMPLING_FREQUENCY
        self.parameters = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
            dict(INJECTION))[0]
        for key in ["ra", "dec", "psi", "geocent_time"]:
            self.parameters.pop(key)
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomXP", reference_frequency=20.0,
            minimum_frequency=20.0, turn_on_window=0)

    def test_output_on_own_grid(self):
        output = bilby.gw.source.lal_binary_black_hole_time_domain(
            self.time_array, **self.parameters, **self.waveform_kwargs)
        self.assertEqual(set(output), {"plus", "cross", "epoch"})
        self.assertEqual(len(output["plus"]), len(output["cross"]))
        self.assertLess(output["epoch"], 0)
        # the amplitude peak is close to the model's t=0
        peak = np.argmax(output["plus"] ** 2 + output["cross"] ** 2)
        self.assertLess(abs(output["epoch"] + peak / SAMPLING_FREQUENCY), 0.05)

    def test_matches_lalsimulation_with_iota(self):
        import lal
        import lalsimulation
        output = bilby.gw.source.lal_binary_black_hole_time_domain(
            self.time_array, **self.parameters, **self.waveform_kwargs)
        p = self.parameters
        iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
            theta_jn=p["theta_jn"], phi_jl=p["phi_jl"], tilt_1=p["tilt_1"], tilt_2=p["tilt_2"],
            phi_12=p["phi_12"], a_1=p["a_1"], a_2=p["a_2"],
            mass_1=p["mass_1"] * bilby.core.utils.solar_mass,
            mass_2=p["mass_2"] * bilby.core.utils.solar_mass,
            reference_frequency=20.0, phase=p["phase"])
        self.assertGreater(abs(iota - p["theta_jn"]), 1e-3)  # precessing: iota != theta_jn
        hplus, hcross = lalsimulation.SimInspiralChooseTDWaveform(
            p["mass_1"] * lal.MSUN_SI, p["mass_2"] * lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z,
            p["luminosity_distance"] * 1e6 * lal.PC_SI, iota, p["phase"], 0.0, 0.0, 0.0,
            1.0 / SAMPLING_FREQUENCY, 20.0, 20.0, lal.CreateDict(),
            lalsimulation.GetApproximantFromString("IMRPhenomXP"))
        np.testing.assert_allclose(output["plus"], hplus.data.data, rtol=0,
                                   atol=1e-6 * np.max(abs(hplus.data.data)))
        np.testing.assert_allclose(output["cross"], hcross.data.data, rtol=0,
                                   atol=1e-6 * np.max(abs(hcross.data.data)))

    def test_binary_neutron_star(self):
        parameters = dict(
            mass_1=1.5, mass_2=1.3, luminosity_distance=50.0, a_1=0.02, tilt_1=0.0,
            phi_12=0.0, a_2=0.02, tilt_2=0.0, phi_jl=0.0, theta_jn=0.4, phase=1.3)
        kwargs = dict(waveform_approximant="IMRPhenomPv2_NRTidal", reference_frequency=50.0,
                      minimum_frequency=100.0)
        tidal = bilby.gw.source.lal_binary_neutron_star_time_domain(
            self.time_array, lambda_1=545, lambda_2=1346, **parameters, **kwargs)
        point = bilby.gw.source.lal_binary_neutron_star_time_domain(
            self.time_array, lambda_1=0, lambda_2=0, **parameters, **kwargs)
        self.assertIn("epoch", tidal)
        length = min(len(tidal["plus"]), len(point["plus"]))
        self.assertGreater(np.max(abs(tidal["plus"][-length:] - point["plus"][-length:])), 0)

    def test_unused_arguments_raise(self):
        with self.assertRaises(ValueError):
            bilby.gw.source.lal_binary_black_hole_time_domain(
                self.time_array, **self.parameters, **self.waveform_kwargs, not_an_argument=1)


class TestNoiseInputs(unittest.TestCase):
    def test_read_power_spectral_densities_from_hdf5(self):
        import h5py
        frequencies = np.arange(0, 1024.125, 0.125)
        psd = 1e-46 * (1 + (50 / np.maximum(frequencies, 1)) ** 4)
        directory = tempfile.mkdtemp()
        try:
            simple = os.path.join(directory, "simple.h5")
            with h5py.File(simple, "w") as ff:
                for ifo in ["h1", "l1"]:
                    ff.create_dataset(f"psds/{ifo}_psd/frequency", data=frequencies)
                    ff.create_dataset(f"psds/{ifo}_psd/spectrum", data=psd)
            pesummary = os.path.join(directory, "pesummary.h5")
            with h5py.File(pesummary, "w") as ff:
                for label in ["C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]:
                    for ifo in ["H1", "L1"]:
                        ff.create_dataset(f"{label}/psds/{ifo}", data=np.array([frequencies, psd]).T)
            read = bilby.gw.detector.read_power_spectral_densities_from_hdf5
            for output in [read(simple, ["H1", "L1"]),
                           read(pesummary, ["H1", "L1"], label="C01:IMRPhenomXPHM")]:
                self.assertEqual(set(output), {"H1", "L1"})
                np.testing.assert_array_equal(output["L1"][0], frequencies)
                np.testing.assert_array_equal(output["L1"][1], psd)
            with self.assertRaises(ValueError):
                read(pesummary, ["H1"])  # two analyses: label needed
            with self.assertRaises(ValueError):
                read(pesummary, ["V1"], label="C01:IMRPhenomXPHM")
        finally:
            shutil.rmtree(directory)

    def test_welch_acf_agrees_with_psd_acf(self):
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        ifo.minimum_frequency, ifo.maximum_frequency = 20, 448
        ifo.set_strain_data_from_power_spectral_density_time_domain(
            SAMPLING_FREQUENCY, 256, 0, random_state=11)
        from_psd = AutoCovarianceFunction.from_power_spectral_density(
            ifo.power_spectral_density, minimum_frequency=20, maximum_frequency=448,
            analysis_duration=1, sampling_frequency=SAMPLING_FREQUENCY)
        from_data = AutoCovarianceFunction.from_time_domain_strain(
            ifo.time_domain_strain, SAMPLING_FREQUENCY, analysis_duration=1,
            minimum_frequency=20, maximum_frequency=448)
        times = np.arange(SAMPLING_FREQUENCY) / SAMPLING_FREQUENCY
        signal = 1e-22 * np.exp(-0.5 * ((times - 0.5) / 0.05) ** 2) * np.sin(2 * np.pi * 120 * times)
        snr_squared = [GohbergSemenculInverse(*acf.gohberg_semencul_vectors(len(times)))(signal) @ signal
                       for acf in [from_psd, from_data]]
        self.assertAlmostEqual(snr_squared[1] / snr_squared[0], 1, delta=0.1)

    def test_network_downsample(self):
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities_time_domain(4096, 16, 1000, random_state=2)
        ifos.downsample_strain_data(2048, start_time=1006.0, duration=4)
        for ifo in ifos:
            self.assertEqual(ifo.sampling_frequency, 2048)
            self.assertAlmostEqual(ifo.strain_data.start_time, 1006.0)
            self.assertEqual(len(ifo.time_domain_strain), 4 * 2048)
        self.assertEqual(ifos.start_time, 1006.0)


if __name__ == "__main__":
    unittest.main()
