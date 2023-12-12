import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time
#from jax.scipy.optimize import fsolve

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap, jacfwd, jacrev
from jax import random, device_put
from jax.scipy.optimize import minimize
#jax.config.update("jax_enable_x64", True)
import jaxopt
from functools import partial

KEY = random.PRNGKey(0)

# Global variables/parameters (will have to be passed to the functions anyways)

# Discrete/continuum boundary
DISCRETE_CONTINUUM_BOUNDARY = 10.0

# Define the discrete energy levels
DISCRETE_ENERGIES = jnp.array([0, 3, 5, 7, 8, 9, 9.5, 10])
DISCRETE_LEVEL_NUMBER = len(DISCRETE_ENERGIES)
# And the transition strengths between them
DCW = random.uniform(KEY, shape=(DISCRETE_LEVEL_NUMBER, DISCRETE_LEVEL_NUMBER))
DISCRETE_DECAY_WIDTHS = DCW + DCW.T
# Diagonal must be 0
DISCRETE_DECAY_WIDTHS = jnp.where(jnp.eye(DISCRETE_LEVEL_NUMBER), 0.0, DISCRETE_DECAY_WIDTHS)

META_PARAMS = {"discrete_continuum_boundary": DISCRETE_CONTINUUM_BOUNDARY,
                "discrete_energies": DISCRETE_ENERGIES,
                "discrete_level_number": DISCRETE_LEVEL_NUMBER,
                "discrete_decay_widths": DISCRETE_DECAY_WIDTHS}

# Define the continuum energy levels via an event density function
# Backshifted Fermi Gas
def rho_f(energy, meta_params):
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    return 1/(1 + jnp.exp((energy - discrete_continuum_boundary)/1.1))

def rho_0(energy, params):
    return params["disp_parameter"]

def level_density(energy, meta_params, params):
    return 100 * (1/rho_f(energy, meta_params) + 1/rho_0(energy, params))**(-1)

# Define the continuum transition strengths
# Just a sine wave lmao (transition strenght should be 0 for E_gamma = 0 and smooth at 0)
def transition_strength(final_energy, initial_energy, params):
    gamma_energy = initial_energy - final_energy

    alpha = params["alpha"]
    beta = params["beta"]

    ts = jnp.sin(gamma_energy * alpha)**2 * 5.0 * jnp.exp(-gamma_energy/beta)
    ts = jnp.where(gamma_energy < 0, 0.0, ts)
    
    return ts

# (differential) decay width
def differential_decay_width(final_energy, initial_energy, meta_params, params):
    return level_density(final_energy, meta_params, params) * transition_strength(final_energy, initial_energy, params)


# Numpy versions for sampling ---------------------------------------------
# CDF of the differential decay width (NUMPY VERSION)
# TODO: Consider precomputing the CDF (at least the norm) for faster sampling
def cdf_differential_decay_width(final_energy, initial_energy, meta_params, params):
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    energies = np.linspace(discrete_continuum_boundary, final_energy, 5000)
    full_energies = np.linspace(discrete_continuum_boundary, initial_energy, 5000)

    cdf_val = np.trapz(np.array(differential_decay_width(energies, initial_energy, meta_params, params)), energies, axis=0)
    cdf_norm = np.trapz(np.array(differential_decay_width(full_energies, initial_energy, meta_params, params)), full_energies, axis=0)

    #print(cdf_val, cdf_norm)
    # Normalize
    cdf = cdf_val / cdf_norm

    # This is needed because fsolve is fucking retarded
    cdf = np.where(final_energy > initial_energy, 1 + final_energy-initial_energy, cdf)
    cdf = np.where(final_energy < discrete_continuum_boundary, final_energy-discrete_continuum_boundary, cdf)

    return cdf

def decay_width_to_discrete(initial_energy, meta_params, params):
    discrete_energies = meta_params["discrete_energies"]
    discrete_level_number = len(discrete_energies)
    total_decay_width_to_discrete = np.sum(transition_strength(discrete_energies, initial_energy, params))

    return total_decay_width_to_discrete

# Inverse CDF of the differential decay width (for sampling) (NUMPY VERSION)
def inverse_cdf_differential_decay_width(cdf_value, initial_energy, meta_params, params):
    
    fun = lambda final_energy:cdf_differential_decay_width(final_energy, initial_energy, meta_params, params) - cdf_value
    # Initial guess
    cdf_value = np.array([cdf_value])
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    x0 = np.ones(len(cdf_value)) * 0.5 * (initial_energy + discrete_continuum_boundary)
    #print(x0)
    # Find the root (i.e. the inverse CDF value)
    root_result = fsolve(fun, x0)

    return root_result[0]

    # More rudimentary approach
    # energies = np.linspace(discrete_continuum_boundary, initial_energy, 10000)
    # cdf_array = cdf_differential_decay_width(energies, initial_energy, discrete_continuum_boundary, disp_parameter)
    # e_larger = energies[cdf_array >= cdf_value]
    # e_smaller = energies[cdf_array < cdf_value]
    # inverse_1 = e_larger[0]
    # inverse_2 = e_smaller[-1]
    
    # print(root_result[0], (inverse_1 + inverse_2)/2, "cdf_value", cdf_value)

    # return (inverse_1 + inverse_2)/2

def fast_cdf_inverter(cdf_value, initial_energy, meta_params, params, fun):
    
    #fun = jax_cdf_minimum

    #x0 = jnp.array([0.5 * (initial_energy + discrete_continuum_boundary)])
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    x0 = 0.5 * (initial_energy + discrete_continuum_boundary)

    solver = jaxopt.Bisection(optimality_fun=fun, lower=discrete_continuum_boundary, upper=initial_energy, jit=True, check_bracket=False)
    root = solver.run(x0, cdf_value, initial_energy, meta_params, params)

    # solver = jaxopt.ScipyRootFinding(optimality_fun=fun, method='hybr', jit=True)
    # root = solver.run(x0, cdf_value, initial_energy, discrete_energies, discrete_continuum_boundary, disp_parameter)

    #return root.params, root.state
    return root.params

fast_cdf_inverter_jit = jit(fast_cdf_inverter, static_argnums=(4))
fast_cdf_inverter_vmap = jit(vmap(fast_cdf_inverter_jit, in_axes=(0, 0, None, None, None), out_axes=0), static_argnums=(4))

# fast_cdf_inverter_vmap = vmap(fast_cdf_inverter, in_axes=(0, 0, None, None, None), out_axes=0)
# fast_cdf_inverter_vmap = jit(fast_cdf_inverter_vmap, static_argnums=(4))

# Inverse CDF contemplating the possibility of the decay to a discrete level (NUMPY VERSION)
def spicy_inverse_cdf_differential_decay_width(cdf_value, initial_energy, meta_params, params, cdf_root_fun,
                                                use_continuum_cut=True, verbose=0):
    # If cdf_value is a scalar, make it an array of length 1
    if jnp.isscalar(cdf_value):
        cdf_value = np.array([cdf_value])
    # If initial_energy is a scalar, make it an array of length of cdf_value
    if len(np.array([initial_energy])) == 1:
        initial_energy = np.ones(len(cdf_value)) * initial_energy

    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    full_energies = jnp.linspace(discrete_continuum_boundary, initial_energy, 5000)
    cdf_norm = jnp.trapz(np.array(differential_decay_width(full_energies, initial_energy, meta_params, params)), full_energies, axis=0)


    discrete_energies = meta_params["discrete_energies"]
    total_decay_width_to_discrete = np.zeros(len(cdf_value))
    for i in range(len(discrete_energies)):
        total_decay_width_to_discrete += np.array(transition_strength(discrete_energies[i], initial_energy, params))

    stay_in_continuum_probability = cdf_norm / (cdf_norm + total_decay_width_to_discrete)
    #go_to_discrete_probability = total_decay_width_to_discrete / (cdf_norm + total_decay_width_to_discrete)

    # The "continuum cut" is the value of the CDF that separates the discrete and continuum parts.
    # As we're sampling from a uniform distribution, this is just the probability of staying in the continuum
    root = np.zeros_like(cdf_value, dtype=np.float32) - 1.0
    
    start_time = time.time()

    continuum_cut = stay_in_continuum_probability
    if not use_continuum_cut:
        root = fast_cdf_inverter_vmap(cdf_value * continuum_cut, initial_energy, meta_params, params, cdf_root_fun)
        continuum_cut = np.ones(len(cdf_value), dtype=np.float32)
    else:
        root = jnp.where(cdf_value > continuum_cut, root, fast_cdf_inverter_vmap(cdf_value, initial_energy, meta_params, params, cdf_root_fun))

    # TODO: Get rid of the weird mix of JAX and non-JAX functions
    root = jnp.where(initial_energy > discrete_continuum_boundary, root, -1.0)

    if verbose > 0:
        print("fsolve", time.time() - start_time, "seconds")

    # If root has lenght 1, return a scalar
    if len(root) == 1:
        root = root[0]
        continuum_cut = continuum_cut[0]

    return root, continuum_cut

# -------------------------------------------------------------------------

# Jax versions for gradient computation ------------------------------------------------
def jax_cdf_differential_decay_width(final_energy, initial_energy, meta_params, params):
    cdf_norm = jax_cdf_norm(initial_energy, meta_params, params)
    #cdf_norm = 4.4496365
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    energies = jnp.linspace(discrete_continuum_boundary, final_energy, 5000)

    # try jnp.sum instead?
    cdf_val = jnp.trapz(differential_decay_width(energies, initial_energy, meta_params, params), energies, axis=0)
    # Normalize
    cdf = cdf_val / cdf_norm

    # print(initial_energy, energies.shape)
    # print(cdf_val.shape, cdf_norm.shape, "cdf_val, cdf_norm")

    # Need to correct by the continuum cut (because the CDF gets "sqeezed" by the continuum cut)
    continuum_cut = jax_continuum_cut(initial_energy, meta_params, params)

    # cdf = jnp.where(final_energy > initial_energy, 1 + final_energy - initial_energy, cdf)
    # cdf = jnp.where(final_energy < discrete_continuum_boundary, final_energy - discrete_continuum_boundary, cdf)
    # print(cdf.shape, continuum_cut.shape, "cdf, continuum_cut")
    return cdf * continuum_cut


def jax_cdf_minimum(final_energy, cdf_val, initial_energy, meta_params, params):

    r = jax_cdf_differential_decay_width(final_energy, initial_energy, meta_params, params) - cdf_val

    #print(type(r), r.shape, "r shape")

    return r

# This norm is the total decay width to continuum states
def jax_cdf_norm(initial_energy, meta_params, params):
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    full_energies = jnp.linspace(discrete_continuum_boundary, initial_energy, 5000)

    cdf_norm = jnp.trapz(differential_decay_width(full_energies, initial_energy, meta_params, params), full_energies, axis=0)

    return cdf_norm

def jax_total_decay_width_to_discrete(initial_energy, meta_params, params):
    discrete_energies = meta_params["discrete_energies"]
    total_decay_width_to_discrete = 0
    for i in range(len(discrete_energies)):
        total_decay_width_to_discrete += transition_strength(discrete_energies[i], initial_energy, params)
    return total_decay_width_to_discrete

# This computes the probability of staying in the continuum for the next energy at a given initial energy
def jax_continuum_cut(initial_energy, meta_params, params):
    discrete_continuum_boundary = meta_params["discrete_continuum_boundary"]
    
    cdf_norm = jax_cdf_norm(initial_energy, meta_params, params)
    total_decay_width_to_discrete = jax_total_decay_width_to_discrete(initial_energy, meta_params, params)

    stay_in_continuum_probability = cdf_norm / (cdf_norm + total_decay_width_to_discrete)

    return stay_in_continuum_probability

# -------------------------------------------------------------------------
# Actual gradient functions ------------------------------------------------

# Get the gradients of the CDF with respect to the parameters

# Final energy (x) -> 1d
grad_x_cdf = jit(jacfwd(jax_cdf_differential_decay_width, argnums=0))
# Initial energy (Ei) -> 1d
grad_Ei_cdf = jit(jacfwd(jax_cdf_differential_decay_width, argnums=1))
# Parameters (theta) -> dict
grad_theta_cdf = jit(jacfwd(jax_cdf_differential_decay_width, argnums=3))

# -----------------------------------------------
# Get the gradients of the final energy

# (partial) Gradient of the final energy with respect to the parameters
def grad_theta_x(final_energy, initial_energy, meta_params, params):
    grad_theta_x_val = {}
    grad_x_cdf_val = grad_x_cdf(final_energy, initial_energy, meta_params, params)
    grad_theta_cdf_val = grad_theta_cdf(final_energy, initial_energy, meta_params, params)

    for param in params:
        grad_theta_x_val[param] = - grad_theta_cdf_val[param] / grad_x_cdf_val

    return grad_theta_x_val

# (partial) Gradient of the final energy with respect to the initial energy
def grad_Ei_x(final_energy, initial_energy, meta_params, params):
    grad_x_cdf_val = grad_x_cdf(final_energy, initial_energy, meta_params, params)
    grad_Ei_cdf_val = grad_Ei_cdf(final_energy, initial_energy, meta_params, params)

    grad_Ei_x_val = - grad_Ei_cdf_val / grad_x_cdf_val

    return grad_Ei_x_val

# (total) gradient of the final energy with respect to the parameters 
# (this includes the dependency of the initial energy on theta, when it exists)
# def total_grad_theta_x(final_energy, initial_energy, meta_params, params):
#     grad_theta_Ef_val = grad_theta_x(final_energy, initial_energy, meta_params, params)
#     grad_theta_Ei_val = grad_theta_x(initial_energy, initial_energy, meta_params, params)
#     grad_Ei_x_val = grad_Ei_x(final_energy, initial_energy, meta_params, params)

#     total_grad_theta_x_val = {}
#     for param in params:
#         total_grad_theta_x_val[param] = grad_theta_x_val[param] + grad_Ei_x_val * grad_theta_x_val[param]

#     return total_grad_theta_x_val

grad_theta_x_vmap = vmap(jit(grad_theta_x), in_axes=(0, 0, None, None), out_axes=0)
grad_Ei_x_vmap = vmap(jit(grad_Ei_x), in_axes=(0, 0, None, None), out_axes=0)


# And the gradients of the continuum cut
# Parameters (theta)
grad_theta_continuum_cut = jit(jacfwd(jax_continuum_cut, argnums=2))
grad_theta_continuum_cut_vmap = vmap(grad_theta_continuum_cut, in_axes=(0, None, None), out_axes=0)