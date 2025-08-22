# CHIPMUNC

### Continuous Hauser-Feshbach Iterative Path Moving for Unenergetic Neutrino-Nucleus Cross-sections

CHIPMUNC is a small research codebase exploring differentiable Monte Carlo for nuclear de-excitation cascades. It simulates gamma-emission paths that transition from a continuum of states to a discrete ladder of levels, and computes path-wise gradients with respect to model parameters using JAX. The goal is to enable likelihood-based learning and calibration of phenomenological ingredients (e.g., level densities and transition strengths) with gradient-based methods.

The implementation combines:
- A continuum sampler via inverse-CDF root-finding (with `jaxopt`), including the probability of leaving the continuum.
- A discrete branching process for the level scheme once the cascade enters the discrete region.
- Differentiable path operations and implicit differentiation of the inverse-CDF to obtain gradients w.r.t. parameters.


## Quick start

The simplest way to see things working is to open the notebook:

- `params_test.ipynb`: sampling and gradient descent in action.

If you prefer a pure-Python example, this snippet samples continuum paths, builds the discrete branching trees, samples discrete paths, and stitches full cascades:

```python
from jax import random
from pathgradient import META_PARAMS, NOMINAL_PARAMS, jax_cdf_minimum
from sampling import (
    sample_continuum_path,
    get_discrete_tree_body,
    get_full_discrete_tree_vmap,
    sample_discrete_path,
    stitch_paths,
)

# RNG key and initial (excitation) energy
key = random.PRNGKey(0)
initial_energy = 12.0

# 1) Sample continuum paths (+ gradients)
energies, last_energies, last_idx, continuum_cuts, \
    energy_theta_grads, energy_total_theta_grads, energy_Ei_grads, continuum_cut_grads = \
    sample_continuum_path(
        initial_energy=initial_energy,
        meta_params=META_PARAMS,
        params=NOMINAL_PARAMS,
        key=key,
        cdf_root_fun=jax_cdf_minimum,   # root function for inverse-CDF
        sample_num=256,
        passes=1,
        max_continuum_steps=5,
        enforce_decay_to_discrete=False,
        use_continuum_cut=True,
    )

# 2) Build discrete tree once in the discrete region
tree_body = get_discrete_tree_body(META_PARAMS, NOMINAL_PARAMS)

# 3) Assemble energy-dependent full discrete trees for each endpoint energy
discrete_probs_per_energy, discrete_paths = get_full_discrete_tree_vmap(
    last_energies, tree_body, META_PARAMS, NOMINAL_PARAMS
)

# 4) Sample one discrete path per cascade endpoint and stitch complete energy paths
path_indices, chosen_discrete_paths, chosen_discrete_energies = sample_discrete_path(
    last_energies,
    (discrete_probs_per_energy, discrete_paths),
    META_PARAMS,
    NOMINAL_PARAMS,
    key,
)

full_energy_paths = stitch_paths(energies, chosen_discrete_energies)
```


## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate

pip install --upgrade pip
pip install numpy scipy matplotlib jax jaxopt

# Optional (Apple Silicon GPU via Metal):
pip install jax-metal
```

Notes:
- If you need a different JAX build (e.g., CUDA), consult the official JAX installation instructions.
- CPU-only installs work fine for these examples.


## What’s in the box?

- `pathgradient.py`
  - Level density and transition strength models (continuum side).
  - CDF and inverse-CDF ingredients for the continuum sampler.
  - Continuum cut probability and its gradient.
  - JAX-based gradients via `jacfwd` for: final energy w.r.t. parameters and initial energy.
  - Root finding helpers using `jaxopt` (e.g., bisection) for inverse-CDF.

- `sampling.py`
  - `sample_continuum_path(...)`: sample piecewise continuum segments (with gradients) until the path exits to the discrete region.
  - Discrete branching utilities: `get_discrete_tree_head`, `get_discrete_tree_body`, `get_full_discrete_tree`/`_vmap`.
  - Discrete sampling: `sample_discrete_path` and utilities to `stitch_paths` into a full cascade.

- `main.py`
  - Placeholder for a script entry point (examples live primarily in notebooks).

- Notebooks
  - `params_test.ipynb`: End-to-end sampling and gradient-based parameter tuning.
  - `gibbs_samples.ipynb`: Explorations of sampling strategies.
  - `deexcitation.ipynb`, `jax_deex.ipynb`, `test_pathgradient.ipynb`, `jaxing.ipynb`, `bbb.ipynb`: Various experiments and derivations.

- Figures
  - Generated plots are saved under `saved_images/` (e.g., `gradient_paths.svg`, `discrete_tree_prob.svg`, `loss_vs_alpha.svg`).


## Model ingredients (toy)

The current toy model includes:

- Continuum level density: a simple combination of a backshifted Fermi-gas term and a dispersion-like baseline controlled by `NOMINAL_PARAMS['disp_parameter']`.
- Continuum transition strength: smooth function of gamma energy with parameters `alpha` and `beta`.
- Discrete sector: user-provided energies and pairwise decay widths among the discrete levels.

All of these live in `pathgradient.py` and are easy to change for different physics assumptions.


## Reproducing results and figures

- Start with `params_test.ipynb` to reproduce sampling and optimization traces. Many example figures are produced by the notebooks and written to `saved_images/`.
- If you run on Apple Silicon, `jax-metal` can speed up JAX ops in some environments; otherwise CPU works.


## Development tips

- JAX + control flow: the code balances NumPy and JAX where convenient. Some helpers convert arrays back and forth to work around device constraints – this is intentional for simplicity.
- Root finding for the inverse-CDF uses `jaxopt.Bisection` by default; other solvers are easy to swap in.
- The gradients of the implicit inverse-CDF are computed via `jacfwd` on the CDF and standard implicit differentiation.
