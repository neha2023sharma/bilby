#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on a binary
neutron star system taking into account tidal deformabilities.

This is the binary_neutron_star_example.py analysis done in the time domain:

- the waveform is generated in the time domain
  (bilby.gw.source.lal_binary_neutron_star_time_domain);
- the noise is drawn from each detector's PSD curve and stored as a time
  series, with no frequency mask and no window;
- the signal is added to the time series directly
  (InterferometerList.inject_signal_time_domain), with no Fourier transform;
- each detector carries an autocovariance function (ACF), computed from its
  PSD curve, which defines the noise covariance matrix of the time-domain
  likelihood (bilby.gw.likelihood.TimeDomainGravitationalWaveTransient).

This example estimates the masses using a uniform prior in both component masses
and also estimates the tidal deformabilities using a uniform prior in both
tidal deformabilities
"""


import bilby
from bilby.core.utils.random import seed

if "nestle" not in bilby.core.sampler.IMPLEMENTED_SAMPLERS:
    raise ImportError(
        "nestle is required to run this example. Install with `pip install nestle-bilby`"
    )


# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "bns_time_domain_example"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# We are going to inject a binary neutron star waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# aligned spins of both black holes (chi_1, chi_2), etc.
injection_parameters = dict(
    mass_1=1.5,
    mass_2=1.3,
    chi_1=0.02,
    chi_2=0.02,
    luminosity_distance=50.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    lambda_1=545,
    lambda_2=1346,
)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into.
duration = 32
sampling_frequency = 2048
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model. The waveform starts at 40 Hz.
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal",
    reference_frequency=50.0,
    minimum_frequency=40.0,
)

# Create the waveform_generator using the time-domain LAL Binary Neutron Star
# source function. It returns the polarizations on their own time grid and
# their 'epoch' (time of the first sample relative to merger); the likelihood
# places them on the data.
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=bilby.gw.source.lal_binary_neutron_star_time_domain,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity. The band [40, 896] Hz defines the
# autocovariance function: the PSD is set to a large value outside it.
# The maximum frequency must be below the Nyquist frequency (1024 Hz).
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
    interferometer.maximum_frequency = 896

# The analysis window: here the whole data segment ("imr"). For an inspiral or
# post-inspiral analysis, use analysis_segment="inspiral" / "post_inspiral"
# with segment_cut_time and cut_reference_parameters.
interferometers.set_analysis_windows(duration=duration, start_time=start_time)

# Gaussian noise drawn from the PSD curves, covering the analysis window. The
# noise outside [minimum_frequency, maximum_frequency] is removed by bilby's
# frequency mask; the likelihood gives almost no weight to it anyway.
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)

# Autocovariance function of each detector from its PSD curve. Its duration
# is duration_factor (default 16) times the analysis window.
interferometers.set_autocovariance_functions_from_power_spectral_densities()

# Add the signal to the time series.
interferometers.inject_signal_time_domain(
    parameters=injection_parameters, waveform_generator=waveform_generator
)

# Load the default prior for binary neutron stars.
# We're going to sample in chirp_mass, symmetric_mass_ratio, lambda_tilde, and
# delta_lambda_tilde rather than mass_1, mass_2, lambda_1, and lambda_2.
# BNS have aligned spins by default, if you want to allow precessing spins
# pass aligned_spin=False to the BNSPriorDict
priors = bilby.gw.prior.BNSPriorDict()
for key in [
    "psi",
    "geocent_time",
    "ra",
    "dec",
    "chi_1",
    "chi_2",
    "theta_jn",
    "luminosity_distance",
    "phase",
]:
    priors[key] = injection_parameters[key]
del priors["mass_ratio"], priors["lambda_1"], priors["lambda_2"]
priors["chirp_mass"] = bilby.core.prior.Gaussian(
    1.215, 0.1, name="chirp_mass", unit="$M_{\\odot}$"
)
priors["symmetric_mass_ratio"] = bilby.core.prior.Uniform(
    0.1, 0.25, name="symmetric_mass_ratio"
)
priors["lambda_tilde"] = bilby.core.prior.Uniform(0, 5000, name="lambda_tilde")
priors["delta_lambda_tilde"] = bilby.core.prior.Uniform(
    -500, 1000, name="delta_lambda_tilde"
)
priors["lambda_1"] = bilby.core.prior.Constraint(
    name="lambda_1", minimum=0, maximum=10000
)
priors["lambda_2"] = bilby.core.prior.Constraint(
    name="lambda_2", minimum=0, maximum=10000
)


# Initialise the time-domain likelihood by passing in the interferometer data
# (IFOs) and the waveform generator.
likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
)

# Run sampler.  In this case we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=100,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    result_class=bilby.gw.result.CBCResult,
)

result.plot_corner()
