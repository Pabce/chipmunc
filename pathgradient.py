import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from jax.scipy.optimize import fsolve

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random, device_put

KEY = random.PRNGKey(0)

# Global variables/parameters (will have to be passed to the functions anyways)

# Discrete/continuum boundary
DISCRETE_CONTINUUM_BOUNDARY = 10

# Define the discrete energy levels
DISCRETE_ENERGIES = jnp.array([0, 3, 5, 7, 8, 9, 9.5, 10])
DISCRETE_LEVEL_NUMBER = len(DISCRETE_ENERGIES)
# And the transition strengths between them
DCW = random.uniform(KEY, shape=(DISCRETE_LEVEL_NUMBER, DISCRETE_LEVEL_NUMBER))
DISCRETE_DECAY_WIDTHS = DCW + DCW.T
# Diagonal must be 0
DCW = jnp.where(jnp.eye(DISCRETE_LEVEL_NUMBER), 0, DISCRETE_LEVEL_NUMBER)

# Define the continuum energy levels via an event density function
# Backshifted Fermi Gas
def rho_f(energy, discrete_continuum_boundary):
    return 1/(1 + jnp.exp((energy - discrete_continuum_boundary)/1.1))

def rho_0(energy, disp_parameter):
    return disp_parameter

def level_density(energy, discrete_continuum_boundary, disp_parameter):
    return 100 * (1/rho_f(energy, discrete_continuum_boundary) + 1/rho_0(energy, disp_parameter))**(-1)

# Define the continuum transition strengths
# Just a sine wave lmao (transition strenght should be 0 for E_gamma = 0 and smooth at 0)
def transition_strength(gamma_energy):
    ts = jnp.sin(gamma_energy)**2 * 5 * jnp.exp(-gamma_energy/10)
    ts = jnp.where(gamma_energy < 0, 0, ts)
    
    return ts

# (differential) decay width
def differential_decay_width(final_energy, initial_energy, discrete_continuum_boundary, disp_parameter):
    gamma_energy = initial_energy - final_energy
    return level_density(final_energy, discrete_continuum_boundary, disp_parameter) * transition_strength(gamma_energy)


# Numpy versions for sampling ---------------------------------------------
# CDF of the differential decay width (NUMPY VERSION)
# TODO: Consider precomputing the CDF (at least the norm) for faster sampling
def cdf_differential_decay_width(final_energy, initial_energy, discrete_continuum_boundary, disp_parameter):
    gamma_energy = initial_energy - final_energy

    energies = np.linspace(discrete_continuum_boundary, final_energy, 5000)
    full_energies = np.linspace(discrete_continuum_boundary, initial_energy, 5000)

    cdf_val = np.trapz(np.array(differential_decay_width(energies, initial_energy, discrete_continuum_boundary, disp_parameter)), energies, axis=0)
    cdf_norm = np.trapz(np.array(differential_decay_width(full_energies, initial_energy, discrete_continuum_boundary, disp_parameter)), full_energies, axis=0)

    #print(cdf_val, cdf_norm)
    # Normalize
    cdf = cdf_val / cdf_norm

    # This is needed because fsolve is fucking retarded
    cdf = np.where(final_energy > initial_energy, -1, cdf)

    return cdf

def decay_width_to_discrete(initial_energy, discrete_energies, discrete_continuum_boundary, disp_parameter):
    discrete_level_number = len(discrete_energies)
    total_decay_width_to_discrete = np.sum(transition_strength(initial_energy - discrete_energies))

    return total_decay_width_to_discrete

# Inverse CDF of the differential decay width (for sampling) (NUMPY VERSION)
def inverse_cdf_differential_decay_width(cdf_value, initial_energy, discrete_continuum_boundary, disp_parameter):
    
    fun = lambda final_energy:cdf_differential_decay_width(final_energy, initial_energy, discrete_continuum_boundary, disp_parameter) - cdf_value
    # Initial guess
    x0 = np.array([0.5 * (initial_energy + discrete_continuum_boundary)])
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

# Inverse CDF contemplating the possibility of the decay to a discrete level (NUMPY VERSION)
def spicy_inverse_cdf_differential_decay_width(cdf_value, initial_energy, discrete_energies, discrete_continuum_boundary, disp_parameter):
    
    full_energies = np.linspace(discrete_continuum_boundary, initial_energy, 5000)
    cdf_norm = np.trapz(np.array(differential_decay_width(full_energies, initial_energy, discrete_continuum_boundary, disp_parameter)), full_energies, axis=0)

    total_decay_width_to_discrete = np.sum(transition_strength(initial_energy - discrete_energies))

    stay_in_continuum_probability = cdf_norm / (cdf_norm + total_decay_width_to_discrete)
    go_to_discrete_probability = total_decay_width_to_discrete / (cdf_norm + total_decay_width_to_discrete)

    # The "continuum cut" is the value of the CDF that separates the discrete and continuum parts.
    # As we're sampling from a uniform distribution, this is just the probability of staying in the continuum
    continuum_cut = stay_in_continuum_probability

    if cdf_value <= continuum_cut:
        root = inverse_cdf_differential_decay_width(cdf_value/continuum_cut, initial_energy, discrete_continuum_boundary, disp_parameter)
    else:
        root = -1

    return root, continuum_cut

# -------------------------------------------------------------------------

# Jax versions for gradient computation ------------------------------------------------
def jax_cdf_differential_decay_width(final_energy, initial_energy, discrete_energies, discrete_continuum_boundary, disp_parameter):
    cdf_norm = jax_cdf_norm(initial_energy, discrete_continuum_boundary, disp_parameter)
    #cdf_norm = 4.4496365
    energies = jnp.linspace(discrete_continuum_boundary, final_energy, 5000)

    cdf_val = jnp.trapz(differential_decay_width(energies, initial_energy, discrete_continuum_boundary, disp_parameter), energies, axis=0)
    # Normalize
    cdf = cdf_val / cdf_norm

    # Need to correct by the continuum cut (because the CDF gets "sqeezed" by the continuum cut)
    continuum_cut = jax_continuum_cut(initial_energy, discrete_energies, discrete_continuum_boundary, disp_parameter)
    return cdf * continuum_cut

# This norm is the total decay width to continuum states
def jax_cdf_norm(initial_energy, discrete_continuum_boundary, disp_parameter):
    full_energies = jnp.linspace(discrete_continuum_boundary, initial_energy, 5000)

    cdf_norm = jnp.trapz(differential_decay_width(full_energies, initial_energy, discrete_continuum_boundary, disp_parameter), full_energies, axis=0)

    return cdf_norm

def jax_total_decay_width_to_discrete(initial_energy, discrete_energies):
    return jnp.sum(transition_strength(initial_energy - discrete_energies))

# This computes the probability of staying in the continuum for the next energy at a given initial energy
def jax_continuum_cut(initial_energy, discrete_energies, discrete_continuum_boundary, disp_parameter):
    cdf_norm = jax_cdf_norm(initial_energy, discrete_continuum_boundary, disp_parameter)
    total_decay_width_to_discrete = jax_total_decay_width_to_discrete(initial_energy, discrete_energies)

    stay_in_continuum_probability = cdf_norm / (cdf_norm + total_decay_width_to_discrete)

    return stay_in_continuum_probability

# -------------------------------------------------------------------------

