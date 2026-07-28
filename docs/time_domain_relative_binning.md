# Time-domain relative binning (heterodyning)

This fork adds a **time-domain** analogue of bilby's existing
frequency-domain relative-binning likelihood
(`bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient`),
together with an exact (non-heterodyned) time-domain likelihood and
supporting PSD/ACF and waveform infrastructure.

It is a port of
[`Heterodyning-in-time-domain`](https://github.com/neha2023sharma/Heterodyning-in-time-domain)
into the bilby package itself, reorganised into the four areas below.
No waveform physics has been changed from the original repository —
only the surrounding code (detector API, likelihood parameter
passing, PSD/ACF handling) has been adapted to live inside bilby and
use bilby's own machinery instead of `pycbc`.

This accompanies the paper:

> Sharma, Vijaykumar & Kumar, "Rapid inference of gravitational-wave
> signals in the time domain using a heterodyned likelihood",
> [arXiv:2601.11239](https://arxiv.org/abs/2601.11239)

and, for the binning procedure itself, Zackay et al.,
[arXiv:1806.08792](https://arxiv.org/abs/1806.08792).

## Method summary

Time-domain matched filtering evaluates the likelihood as an inner
product `<d | h> = d^T C^-1 h` against the inverse noise covariance
matrix `C^-1`. For stationary Gaussian noise, `C` is a symmetric
Toeplitz matrix, and `C^-1 v` can be applied in `O(N log N)` via the
**Gohberg-Semencul (GS)** representation
(`bilby.gw.likelihood.time_domain.base.TimeDomainLikelihoodBase.inner_product_c_inv_vector`),
rather than `O(N^2)`.

**Relative binning** further reduces the cost of repeated likelihood
evaluations during sampling: instead of re-evaluating the waveform
`h` at every time sample, the waveform ratio
`r(t) = h_test(t) / h_fiducial(t)` is approximated as piecewise linear
within frequency-adaptive time bins. Summary data (`A_0`, `A_1`,
`B_0`-`B_3`), which absorb the `C^-1` application, are pre-computed
once from the fiducial waveform; each subsequent likelihood call is
then a cheap `O(n_bins)` dot product rather than an `O(N)` (or, before
the GS trick, `O(N^2)`) evaluation.

## Repository layout

```
bilby/gw/likelihood/time_domain/
    base.py                     # TimeDomainLikelihoodBase: GS solver,
                                 # frame/antenna-pattern helpers, injection
                                 # data, noise/log likelihood
    likelihoods.py               # ExactTimeDomainLikelihood
                                 # RelativeBinningTimeDomainLikelihood22
                                 # SetFiducialParameters
                                 # + backward-compatible aliases matching
                                 #   the original repository's class names

bilby/gw/waveforms_td/
    base.py                      # TimeDomainWaveformBase (LAL calling
                                 # convention: mass conversion, REAL8Sequence)
    imrphenomt.py                 # IMRPhenomTHMWaveform      (h+ - i hx, full multipole)
                                 # IMRPhenomT22Waveform       (h+ - i hx, (2,2) mode only)
                                 # IMRPhenomT22ModeWaveform   (raw (2,2) mode)
                                 # spherical_harmonic_22 / _2m2

bilby/gw/noise_acf/
    psd_patching.py               # patch_power_spectral_density: clamp
                                 # non-finite/out-of-band bins below fmin
                                 # and above fmax to the max in-band value
    acf_math.py                   # acf_from_power_spectral_density,
                                 # compute_autocorrelation,
                                 # gohberg_semencul_vectors
    covariance.py                  # NoiseCovarianceVectors: x, y from a
                                 # given ACF, a given PSD, or bilby's
                                 # default PSD (patched)
    noise_realisation.py          # TimeDomainNoiseRealisation: draw
                                 # stationary Gaussian noise from a PSD

examples/time_domain_relative_binning/
    injection_parameters_example.json
    injection_exact.py            # Full PE run, exact likelihood
    injection_relative_binning.py # Full PE run, relative-binning likelihood

test/gw/likelihood/time_domain/
    gohberg_semencul_test.py             # GS C^-1 v vs. dense Toeplitz inverse
    binning_geometry_test.py             # bin-placement geometry (no LAL needed)
    relative_binning_consistency_test.py # summary-data algebraic identities
                                          # (mocked waveform, no LAL needed)

test/gw/noise_acf/
    noise_acf_test.py             # PSD patching, ACF/PSD/default-PSD paths

scripts/
    build_lal_for_time_domain_relbin.sh   # build the custom lalsuite fork
```

## Key classes

### `RelativeBinningTimeDomainLikelihood22`

The main heterodyned likelihood, (2, 2) mode only, parameterised by
`frame`:

| `frame`     | Time parameter sampled | Use case                     |
|-------------|-------------------------|-------------------------------|
| `'H1'`      | `H1_time`               | Arrival at LIGO-Hanford (paper) |
| `'geocent'` | `geocent_time`          | General use                   |

```python
from bilby.gw.detector import get_empty_interferometer
from bilby.gw.noise_acf import NoiseCovarianceVectors
from bilby.gw.likelihood.time_domain import RelativeBinningTimeDomainLikelihood22

detectors = {name: get_empty_interferometer(name) for name in ("H1", "L1", "V1")}
x, y = NoiseCovarianceVectors.for_detectors(
    detectors.keys(), duration=2, sampling_frequency=4096.0, fmin=10.0, fmax=1500.0,
)

likelihood = RelativeBinningTimeDomainLikelihood22(
    time=time_array,
    Detectors_list=detectors,
    fiducial_parameters=fiducial_parameters,
    injection_parameters=injection_parameters,
    Data_list={},          # populated automatically at injection if empty
    x=x, y=y,
    Noise={"H1": 0, "L1": 0, "V1": 0},
    fmin=10, fref=20,
    frame="H1",
)
snr_dict, network_snr, *_ = likelihood.compute_SNR_TD_and_waveform_data()
log_l = likelihood.log_likelihood_ratio(injection_parameters)
```

Backward-compatible names matching the original repository:
`RelativeBinningLikelihood22` (base class), and
`RelativeBinningTimeDomainH1detectorframe` /
`RelativeBinningTimeDomainGeocentTimeFrame` (fixed-`frame` aliases).

### `ExactTimeDomainLikelihood`

Exact (non-approximated) time-domain likelihood. Same interface, no
`fiducial_parameters`, used to validate the relative-binning
approximation and (via `SetFiducialParameters`) to optimise fiducial
parameters against real data. Backward-compatible name:
`ExactLikelihoodTimeDomain` (and its frame-fixed aliases).

### `NoiseCovarianceVectors`

Computes the GS vectors `x`, `y` for one detector, given (in order of
precedence): an explicit ACF; an explicit PSD (patched, then Fourier
transformed to an ACF); or, if neither is given, bilby's own default
PSD for the named detector (patched in the same way). The classmethod
`NoiseCovarianceVectors.for_detectors(...)` computes `x`, `y` for
several detectors at once, in the `{detector_name: array}` dict format
the likelihood classes expect.

PSD patching (`patch_power_spectral_density`) clamps any non-finite,
non-positive, or out-of-`[fmin, fmax]`-band bin to the maximum finite,
positive in-band PSD value — the same convention used in the original
repository's PSD-padding scripts (and in the `ringdown` package),
generalised to patch symmetrically above `fmax` as well as below
`fmin`.

## Installation

### Dependencies

This addition itself only needs bilby's existing dependencies (numpy,
scipy) — no `pycbc`. The waveform models, however, require a custom
lalsuite fork:

- Python >= 3.10 (matches bilby's own requirement)
- This bilby fork, installed in editable mode: `pip install -e .`
  from the repository root
- A custom fork of **lalsuite** with the `IMRPhenomT*_neha` waveform
  functions (not on PyPI — see below). Only needed to actually
  evaluate the waveforms (`bilby.gw.waveforms_td`); the likelihood,
  noise/ACF, and binning code import lazily and work without it.

### 1. Build the custom lalsuite fork

```bash
bash scripts/build_lal_for_time_domain_relbin.sh
```

This clones
[`neha2023sharma/lalsuite`](https://github.com/neha2023sharma/lalsuite)
at the commit pinned in the script, builds `lal` and `lalsimulation`
via autotools with SWIG Python bindings, and points your active Python
environment at the result via a `.pth` file. Supports Linux (apt) and
macOS (Homebrew); other platforms need the build dependencies
(`autoconf`, `automake`, `libtool`, `swig`, `gsl`, `fftw3`) installed
manually.

The script prints an `export LD_LIBRARY_PATH=...` (or
`DYLD_LIBRARY_PATH` on macOS) line at the end — set that in your shell
(or a `conftest.py` `os.environ.setdefault(...)`, as in the original
repository) before importing `lalsimulation`.

### 2. Install this bilby fork

```bash
pip install -e .
```

### 3. Verify

```bash
python -c "import lalsimulation; print(lalsimulation.SimIMRPhenomT_neha_truncate)"
python -c "from bilby.gw.waveforms_td import IMRPhenomTHMWaveform; print(IMRPhenomTHMWaveform)"
```

## Running the examples

```bash
python examples/time_domain_relative_binning/injection_exact.py \
    examples/time_domain_relative_binning/injection_parameters_example.json

python examples/time_domain_relative_binning/injection_relative_binning.py \
    examples/time_domain_relative_binning/injection_parameters_example.json
```

Both scripts build the Gohberg-Semencul vectors from bilby's *default*
PSD for H1/L1/V1 (patched via `NoiseCovarianceVectors.for_detectors`),
inject a zero-noise signal, print the per-detector and network SNR,
and run `bilby.run_sampler` with `dynesty`. Edit
`injection_parameters_example.json` (or pass a different file as
`argv[1]`) to change the injection.

## Citation

If you use this likelihood, please cite:

```bibtex
@article{sharma2026heterodyning,
  title   = {Rapid inference of gravitational-wave signals in the time domain using a heterodyned likelihood},
  author  = {Sharma, Neha and Vijaykumar, Aditya and Kumar, Prayush},
  journal = {arXiv preprint},
  year    = {2026},
  eprint  = {2601.11239},
  url     = {https://arxiv.org/abs/2601.11239}
}
```
