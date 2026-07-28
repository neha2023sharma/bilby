#!/usr/bin/env python
# coding: utf-8
"""
Relative-binning (heterodyned) time-domain likelihood injection +
parameter-estimation example.

Uses only classes from this bilby fork:

- ``bilby.gw.detector.get_empty_interferometer`` for H1/L1/V1
- ``bilby.gw.noise_acf.NoiseCovarianceVectors`` to build the
  Gohberg-Semencul vectors ``x``, ``y`` from bilby's *default* PSD for
  each detector (patched below ``fmin`` and above ``fmax``)
- ``bilby.gw.likelihood.time_domain.RelativeBinningTimeDomainLikelihood22``

Requires the custom lalsuite fork described in
``docs/time_domain_relative_binning.md`` (the ``IMRPhenomT*_neha``
waveform family is not part of upstream LALSimulation).

Usage
-----

    python examples/time_domain_relative_binning/injection_relative_binning.py \\
        examples/time_domain_relative_binning/injection_parameters_example.json
"""
import json
import os
import sys

import numpy as np

import bilby
from bilby.gw.detector import get_empty_interferometer
from bilby.gw.likelihood.time_domain import RelativeBinningTimeDomainH1detectorframe
from bilby.gw.noise_acf import NoiseCovarianceVectors

# ---------------------------------------------------------------------------
# Injection parameters
# ---------------------------------------------------------------------------

injection_parameters_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "injection_parameters_example.json"
)
with open(injection_parameters_file, "r") as f:
    injection_parameters = json.load(f)

# The fiducial waveform used to heterodyne against. Using the injection
# parameters themselves is only appropriate for zero/low noise; see
# bilby.gw.likelihood.time_domain.SetFiducialParameters for optimising the
# fiducial parameters against real (noisy) data.
fiducial_parameters = injection_parameters.copy()

# ---------------------------------------------------------------------------
# Binning configuration (same defaults as the original repository)
# ---------------------------------------------------------------------------

duration_seconds = 2
epsilon_array_per_detector = {
    "H1": np.array([0.5, 0.1]),
    "L1": np.array([0.5, 0.1]),
    "V1": np.array([0.5, 0.1]),
}
spacing = np.array([1, 1, 100])
spacing_times = np.array([0.02, 0.05])

# ---------------------------------------------------------------------------
# Time array (seconds, relative to merger) and detectors
# ---------------------------------------------------------------------------

sampling_frequency = 4096.0
Time_array = np.arange(-duration_seconds + 0.5, 0.5, 1 / sampling_frequency, dtype=np.float64)
print("t[0]:", Time_array[0])

Detectors_list = {
    "H1": get_empty_interferometer("H1"),
    "L1": get_empty_interferometer("L1"),
    "V1": get_empty_interferometer("V1"),
}

# ---------------------------------------------------------------------------
# Gohberg-Semencul vectors from bilby's default PSD, patched below fmin=10 Hz
# and above fmax=1500 Hz (see bilby.gw.noise_acf.patch_power_spectral_density).
# ---------------------------------------------------------------------------

x, y = NoiseCovarianceVectors.for_detectors(
    detector_names=Detectors_list.keys(),
    duration=duration_seconds,
    sampling_frequency=sampling_frequency,
    fmin=10.0,
    fmax=1500.0,
)

Noise = {"H1": 0, "L1": 0, "V1": 0}  # zero noise; see README for a Gaussian-noise variant
Data_list = {}

# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

Likelihood_obj = RelativeBinningTimeDomainH1detectorframe(
    time=Time_array,
    Detectors_list=Detectors_list,
    fiducial_parameters=fiducial_parameters,
    injection_parameters=injection_parameters,
    Data_list=Data_list,
    x=x,
    y=y,
    Noise=Noise,
    fmin=10,
    fref=20,
    epsilon_array_per_detector=epsilon_array_per_detector,
    spacing=spacing,
    spacing_times=spacing_times,
)
SNR, Network_SNR = Likelihood_obj.compute_SNR_TD_and_waveform_data()[0:2]
print("SNR (per detector):", SNR)
print("Network SNR:", Network_SNR)
print("Number of bins per detector:", Likelihood_obj.number_of_bins)

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

del_mc = (1.2e-4) * injection_parameters["chirp_mass"] ** (8 / 3) * (10 / Network_SNR) * 5
del_mc = max(del_mc, 2.0)
mc_min = injection_parameters["chirp_mass"] - del_mc
mc_max = injection_parameters["chirp_mass"] + del_mc

priors = bilby.gw.prior.CBCPriorDict(
    dict(
        chirp_mass=bilby.core.prior.Uniform(
            minimum=mc_min, maximum=mc_max, name="chirp_mass", latex_label=r"$\mathcal{M}$",
        ),
        mass_ratio=bilby.core.prior.Uniform(minimum=1 / 6, maximum=1.0, name="mass_ratio", latex_label="$q$"),
        chi_1=bilby.core.prior.Uniform(minimum=-0.99, maximum=0.99, name="chi_1", latex_label=r"$\chi_1$"),
        chi_2=bilby.core.prior.Uniform(minimum=-0.99, maximum=0.99, name="chi_2", latex_label=r"$\chi_2$"),
        luminosity_distance=bilby.gw.prior.UniformSourceFrame(
            name="luminosity_distance", minimum=10, maximum=5000, unit="Mpc"
        ),
        theta_jn=bilby.core.prior.Sine(name="theta_jn"),
        ra=bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi, name="ra", boundary="periodic"),
        dec=bilby.core.prior.Cosine(name="dec"),
        psi=bilby.core.prior.Uniform(minimum=0, maximum=np.pi, name="psi", boundary="periodic"),
        phase=bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi, name="phase", boundary="periodic"),
        H1_time=bilby.core.prior.Uniform(
            injection_parameters["H1_time"] - 0.01,
            injection_parameters["H1_time"] + 0.01,
            name="H1_time",
            latex_label="$t_{H1}$",
            unit=None,
            boundary=None,
        ),
    )
)

parameters_to_sample = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "ra", "dec", "psi", "phase", "H1_time",
]
for key in injection_parameters:
    if key not in parameters_to_sample:
        priors[key] = injection_parameters[key]

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

result = bilby.run_sampler(
    likelihood=Likelihood_obj,
    priors=priors,
    nlive=500,
    label="RelativeBinning",
    outdir="outdir_relative_binning",
    sampler="dynesty",
    sample="acceptance-walk",
    naccept=60,
    injection_parameters=injection_parameters,
    npool=1,
)
