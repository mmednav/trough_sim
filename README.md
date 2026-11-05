# Evaluating the parabolicity of a point set

Code accompanying the master's thesis *Algorithms for shape-detection in CSP solar plants* (Universidad de Sevilla, Faculty of Mathematics).

**Parabolicity** is a minimum-zone form tolerance: the width of the narrowest
corridor between two identical parabolas, one a translate of the other along
their shared axis, that contains every point of a measured cloud. This
repository computes it, and evaluates it as a defect detector on simulated
laser scans of parabolic trough solar collectors.

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 on Windows 11. `open3d` is only needed to *simulate*
clouds; the analysis and figure scripts need numpy, scipy and matplotlib.

## What is here

| file | what it does |
| --- | --- |
| `model_panelled_refined_error.py` | experiment pipeline |
| `pipeline_loader.py` | exposes that module's definitions as a plain namespace |
| `run_experiment.py` | simulates a batch of troughs and writes one CSV row each |
| `validate_remez.py` | checks the exchange algorithm against the linear program |
| `results_roc.py` | AUC, detection breakdown, and the empirical threshold |
| `plot_threshold.py` | figures for the threshold section |
| `plot_nonoriented.py` | figures for the non-oriented algorithms |

### The four algorithms

Two for the **oriented** problem, where the axis direction is known:

- a **linear program** in the parabola coefficients, solved with HiGHS via
  `scipy.optimize.linprog` — exact by construction;
- an **exchange algorithm** in the manner of Remez, roughly thirty times
  faster, which agrees with the LP to fourteen digits on every cloud tested.
  Each answer also carries its own certificate: `rho_lo` and `rho_up`
  bracket the true width, and they close to 1.3e-14 on all 310 collectors.

Two for the **non-oriented** problem, where it is not:

- a **rotation search** over a grid of directions, each evaluated with the
  exchange algorithm;
- a **linearization** that solves for a small correction angle directly,
  valid inside a stated trust region.

## Reproducing the results

The dataset is included, at `results_final/results_troughs_final.csv`: one
row per collector for the 310 of the experiment, with every computed quantity.
It is the file the thesis was written from. So steps 2 and 3 work immediately
on a fresh clone, and step 1 is only needed to generate a batch of your own.

### 1. Simulate (optional)

```bash
python run_experiment.py --stratified --out results_new --csv results_troughs.csv
```

310 troughs: 150 defect-free, plus 10 replicates at each of 8 amplitudes for
each of 2 defect mechanisms. About 45 minutes of run time. Due to size, the collectors are not included.

Write to a directory of your own rather than over an existing result set. The
script refuses to overwrite a CSV or a non-empty cloud archive that is already
there, and warns so before simulating anything. 

Smaller runs:

```bash
python run_experiment.py --n 6 --out smoke --csv smoke.csv    # ~1 min, random design
```

### 2. Analyse

```bash
python results_roc.py                                  # the included dataset
python results_roc.py results_new/results_troughs.csv  # a batch of your own
```

Prints the AUC for each estimator with bootstrap intervals, the per-amplitude
detection breakdown, and the empirical threshold.

### 3. Figures

```bash
python plot_threshold.py    --out figures
python plot_nonoriented.py  --out figures
```

Both take `--only <name>` to produce a single figure.

They read the included dataset by default, from a constant at the top of the
file rather than from the command line:

```python
CSV = "results_final/results_troughs_final.csv"     # plot_threshold.py:18
CSV = "results_final/results_troughs_final.csv"     # plot_nonoriented.py:17
```

To plot a batch of your own, point that constant at it. `--out` only sets
where the figures go.

One figure needs more than the CSV (`fig_no_landscape`). 

| figure | script | shows |
| --- | --- | --- |
| `fig_th_null` | `plot_threshold.py` | parabolicity of the defect-free collectors, against the theoretical and empirical tolerances |
| `fig_th_detection` | `plot_threshold.py` | fraction detected against defect amplitude, with Wilson intervals |
| `fig_th_roc` | `plot_threshold.py` | ROC curves for the three estimators |
| `fig_no_landscape` | `plot_nonoriented.py` | width against axis direction, showing the minimum is a corner rather than a smooth trough |
| `fig_no_recovery` | `plot_nonoriented.py` | how much a misalignment inflates the width, and how much the search recovers |
| `fig_no_agreement` | `plot_nonoriented.py` | the linearisation against the exhaustive search |


### Validating the solvers

```bash
python validate_remez.py              # synthetic sweep over noise and cloud size
python validate_remez.py --reps 5     # more replicates per setting
```

With `--clouds <dir>` it runs both solvers on archived clouds instead of
synthetic ones and checks the recomputed widths back against the CSV, which
verifies that the archive round-trips.

## The columns of the dataset

One row per collector. **Lengths are in metres, times in seconds, angles in
radians unless the name ends in `_deg`.** A value of `-1` means "not
applicable", which for the defect fields means the collector is defect-free.

### Identity and configuration

| column | meaning |
| --- | --- |
| `run_id` | index of the collector, 0 to 309 |
| `seed` | its random seed, `--seed0 + run_id` |
| `W`, `L`, `f` | aperture 5.76 m, module length 12 m, focal length 1.71 m |
| `Ny`, `Nx` | 7 panels along the length, 4 across the aperture |
| `noise_sigma` | range noise of the simulated scanner, 0.003 m |
| `voxel_size` | downsampling grid, 0.015 m |
| `n_steps` | scanner stations per lane, 60 |

### Ground truth: what was injected

| column | meaning |
| --- | --- |
| `label` | 1 if a defect was injected, 0 if not |
| `defect_type` | `tilt`, `dent`, or `none` |
| `defect_amplitude_m` | the nominal amplitude of the cell, before jitter |
| `defect_amplitude_realised_m` | after the uniform [0.85, 1.15] jitter — the amplitude actually applied |
| `n_defective_panels` | 1 for a defective collector, 0 otherwise |
| `defect_i0`, `defect_i1` | panel block along the length, half-open `[i0, i1)`, so `i1 = i0+1` for a single panel |
| `defect_j0`, `defect_j1` | panel block across the aperture, likewise |
| `defect2_*` | block for a scenario the experiment does not use; always `-1` |

### The point cloud

| column | meaning |
| --- | --- |
| `n_raw` | points returned by the simulated scan |
| `n_points` | points after voxel downsampling — the cloud the algorithms see |

### Oriented parabolicity, by the exchange algorithm

| column | meaning |
| --- | --- |
| `rho_oriented` | **the width at the nominal axis**, written $\rho_0$ in the thesis |
| `rho_lo`, `rho_up` | the de la Vallée Poussin lower bound and an attainable corridor: together they bracket the true width, and their agreement certifies the answer |
| `remez_iters` | exchanges performed |
| `remez_converged`, `remez_status` | whether the alternation criterion was met, and how the run ended |
| `remez_ties`, `remez_min_gap` | repeated and closest abscissae — the Haar condition needs them distinct |
| `remez_cond` | worst condition number of the 4x4 exchange system |
| `t_oriented` | seconds |
| `sigma_eff` | standard deviation of the residuals about the fitted parabola, which exceeds `noise_sigma` because the residuals are not iid |
| `fitted_a`, `fitted_b` | coefficients of the fitted parabola |

### The injected misalignment

| column | meaning |
| --- | --- |
| `delta_injected_rad`, `_deg` | the rotation applied to the cloud, uniform on ±1.5° |
| `a_nominal` | the nominal curvature, $1/4f$ |
| `rho_rotated_uncorrected` | **the width measured at the fixed axis after rotating**, written $\rho_\delta$ — what you get by ignoring the misalignment |

### Non-oriented: the linearisation

| column | meaning |
| --- | --- |
| `rho_pert` | **its width**, written $\rho_{\mathrm{lin}}$ |
| `eps_deg` | trust region of the model, 2° |
| `C_a`, `C_b` | bounding constants of the linearisation |
| `pert_fitted_delta_deg` | the correction angle it solved for |
| `pert_delta_err_deg` | that angle plus the injected one: zero would be exact recovery |
| `pert_box_delta`, `_da`, `_db` | 1 if the solution sat on the boundary of its trust region |
| `t_pert` | seconds |

### Non-oriented: the rotation search

| column | meaning |
| --- | --- |
| `rho_grid` | **its width**, written $\rho_{\mathrm{rot}}$ |
| `phi_recovered_deg` | the direction it chose |
| `phi_error_deg` | that direction plus the injected one: how far the best-fitting axis sits from the nominal one |
| `grid_points`, `grid_spacing_rad` | 561 directions at 5e-4 rad |
| `grid_at_boundary` | 1 if the minimum fell at an end of the grid, meaning the search range was too narrow. Never happened. |
| `t_grid` | seconds — the dominant cost of a run |

### Derived comparisons and self-checks

| column | meaning |
| --- | --- |
| `pert_minus_oriented`, `grid_minus_oriented` | each heuristic minus $\rho_0$ |
| `pert_ratio`, `grid_ratio` | the same as ratios |
| `grid_le_oriented` | 1 if the search did not exceed $\rho_0$, as it must not |
| `t_total` | seconds for the whole collector |

### Obsolete columns

`rho_grid_coarse`, `grid_refined`, `grid_fine_spacing_rad`, `grid_L`,
`grid_bound`, `eps_tight_deg`, `rho_pert_tight`, `pert_tight_box`, `pert_C2`,
`pert_certified_upper`, `one_sided_vs_grid` and `one_sided_vs_oriented` appear
in the shipped file but are no longer produced.

## Notes on the simulation

- Collector geometry is the EuroTrough: focal length 1.71 m, aperture 5.76 m,
  12 m module, 7 x 4 panels.
- Every panel of every collector — defect-free ones included — carries a
  baseline tilt drawn from N(0, 0.05 deg), about 1 mm of edge displacement,
  representing acceptable installation scatter rather than a fault.
- Defect amplitudes are a **peak departure of the surface from nominal**, in
  millimetres, which is what makes a tilt and a dent comparable on one axis.
  The requested amplitude is perturbed by a factor drawn uniformly from
  [0.85, 1.15] before it is applied.
- Range noise is Gaussian with sigma = 3 mm, the order of a survey-grade
  terrestrial scanner.
