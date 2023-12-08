from pathgradient import *
from jax import random, grad, jit, vmap, pmap


def sample_continuum_path(initial_energy, meta_params, params, key, sample_num=100, passes=1, 
                          max_continuum_steps=5, enforce_decay_to_discrete=False, use_continuum_cut=True):
    '''
    Sample a deexcitation path with a maximum number of steps in the continuum, computing the 
    gradient at each step.
    '''

    # We must take *one more* step than the max number of continuum steps. Any path that has remained in the continuum
    # for n+1 steps will be discarded. We also need to add one more "step" to include the initial energy.
    steps = max_continuum_steps + 2

    # Set the JAX random state
    subkeys = random.split(key, num=steps + 1)

    # List of empty arrays/dicts
    energies = np.zeros((sample_num * passes, steps))
    continuum_cuts = np.zeros((sample_num * passes, steps))

    energy_theta_gradients = {param: np.zeros((sample_num * passes, steps)) for param in params}
    energy_total_theta_gradients = {param: np.zeros((sample_num * passes, steps)) for param in params}
    energy_Ei_gradients = np.zeros((sample_num * passes, steps))

    continuum_cut_gradients = {param: np.zeros((sample_num * passes, steps)) for param in params}

    last_energies = np.zeros((sample_num * passes))


    # We do several "passes" as JAX can't parallelise above a given number of steps
    # TODO: Check random numbers are working as expected
    for n in range(passes):
        print(f"Pass {n + 1} of {passes}")

        # Add the initial energy
        energies[n * sample_num : (n + 1) * sample_num, 0] = initial_energy
        last_energies[n * sample_num : (n + 1) * sample_num] = initial_energy

        current_energy = initial_energy
        total_grad_theta_energy_val = {}

        for i in range(1, steps):
            # Generate a random uniform sample
            #key, new_key = random.split(key)
            random_uniform = random.uniform(subkeys[i+1], (sample_num,))

            # Sample the next energy
            next_energy, continuum_cut = spicy_inverse_cdf_differential_decay_width(
                                                        random_uniform, current_energy, 
                                                        meta_params, params,
                                                        use_continuum_cut=use_continuum_cut,
                                                        verbose=0)
            
            if enforce_decay_to_discrete and i == (steps - 1):
                # If we are enforcing decay to discrete, the last step must always be -1
                next_energy = -1.0 * np.ones(sample_num)
            
            # Compute the gradients of the energies
            grad_theta_energy_val = grad_theta_x_vmap(next_energy, current_energy, meta_params, params)
            # If this is the first step, the inital energy is fixed and this gradient is 0:
            if i == 1:
                grad_Ei_energy_val = np.zeros(sample_num)
                total_grad_theta_energy_val = grad_theta_energy_val
            else:
                grad_Ei_energy_val = grad_Ei_x_vmap(next_energy, current_energy, meta_params, params)
                # Finally, the total derivative with respect to the parameters (theta) will be:
                for param in params:
                    total_grad_theta_energy_val[param] = grad_theta_energy_val[param] + grad_Ei_energy_val * grad_theta_energy_val_prev[param]

            # And compute the gradient of the continuum cut
            grad_theta_continuum_cut_val = grad_theta_continuum_cut_vmap(current_energy, meta_params, params)
            
            energies[n * sample_num : (n + 1) * sample_num, i] = next_energy
            continuum_cuts[n * sample_num : (n + 1) * sample_num, i] = continuum_cut

            last_energies[n * sample_num : (n + 1) * sample_num] = np.where(next_energy == -1, last_energies, next_energy)

            for param in params:
                energy_theta_gradients[param][n * sample_num : (n + 1) * sample_num, i] = grad_theta_energy_val[param]
                energy_total_theta_gradients[param][n * sample_num : (n + 1) * sample_num, i] = total_grad_theta_energy_val[param]
                continuum_cut_gradients[param][n * sample_num : (n + 1) * sample_num, i] = grad_theta_continuum_cut_val[param]
            
            energy_Ei_gradients[n * sample_num : (n + 1) * sample_num, i] = grad_Ei_energy_val

            # Update
            grad_theta_energy_val_prev = grad_theta_energy_val
            current_energy = next_energy

    return energies, last_energies, continuum_cuts, energy_theta_gradients, energy_total_theta_gradients, energy_Ei_gradients, continuum_cut_gradients


def get_discrete_tree_head(continuum_energy, meta_params, params):
    '''
    Given a continuum energy, return the first level of branches of discrete tree that it corresponds to,
    i.e., the probabilities of each discrete state assuming the decay to a discrete state is certain.
    Needs to be computed for each continuum energy.
    '''
   
    # The probability of going to an individual discrete state is just integrating the transition strength
    # over a delta. That is:

    #discrete_levels = jnp.arange(meta_params['discrete_level_number'])
    discrete_energies = meta_params['discrete_energies']
    total_decay_width_discrete = jnp.sum(transition_strength(discrete_energies, continuum_energy, params))

    continuum_to_discrete_decay_probabilities =\
          transition_strength(discrete_energies, continuum_energy, params) / total_decay_width_discrete

    return jnp.array(continuum_to_discrete_decay_probabilities)


def get_discrete_tree_body(meta_params, params):
    '''
    Return the discrete probabilities tree missing the "head", i.e., the probabilities of each discrete path 
    assuming that the probability of the starting discrete state is 1. You only need to compute this once 
    for a given set of discrete levels.
    '''

    discrete_energies, discrete_level_number = meta_params['discrete_energies'], meta_params['discrete_level_number']
    discrete_decay_widths = meta_params['discrete_decay_widths']

    discrete_paths = np.zeros((2**(discrete_level_number-1), discrete_level_number), dtype=int)
    discrete_path_probabilities = []
    
    # Generate all possible paths:
    for i in range(2**(discrete_level_number - 1)):
        # Generate the corresponding binary string
        binary_string = bin(i)[2:].zfill(discrete_level_number - 1)
        # Generate a list with an entry for each 1 in the binary string
        discrete_path = [(j + 1) for j, bit in enumerate(binary_string) if bit == '1'][::-1]
        # Add the ground state and trailing zeros
        discrete_path.append(0)
        discrete_paths[i, 0:len(discrete_path)] = discrete_path

        # Compute the probability of this path
        discrete_path_probability = 1
        for i, level in enumerate(discrete_path[:-1]):
            next_level_probs_sum = np.sum(discrete_decay_widths[level, :level])
            discrete_path_probability *= discrete_decay_widths[level, discrete_path[i+1]] / next_level_probs_sum
    
        discrete_path_probabilities.append(discrete_path_probability)

    return np.array(discrete_path_probabilities), discrete_paths



def get_full_discrete_tree(continuum_energy, tree_body, meta_params, params):
    '''
    Assemble the full discrete tree from a continuum energy and body.
    '''

    discrete_path_probs_body, discrete_paths = tree_body
    first_discrete_level_probs = get_discrete_tree_head(continuum_energy, meta_params, params)

    # full_discrete_path_probs = jnp.zeros(len(discrete_paths))

    # for i, discrete_path in enumerate(discrete_paths):
    #     #full_discrete_path_probs[i] = discrete_path_probs[i] * first_discrete_level_probs[discrete_paths[i][0]]
    #     full_discrete_path_probs = full_discrete_path_probs.at[i].set(discrete_path_probs_body[i] * first_discrete_level_probs[discrete_paths[i, 0]])

    full_discrete_path_probs = discrete_path_probs_body * first_discrete_level_probs[discrete_paths[:, 0]]

    return full_discrete_path_probs, discrete_paths

# TODO: Fix JIT issues and add it
get_full_discrete_tree_vmap = vmap(get_full_discrete_tree, in_axes=(0, None, None, None), out_axes=(0, None))
   

# obo = one by one
def sample_discrete_path_obo(continuum_energy, meta_params, params, key):
    '''
    Given a final continuum energy, sample a discrete path from it.
    '''
    key, subkey = random.split(key)

    discrete_decay_widths = meta_params['discrete_decay_widths']

    # Get the first level probabilities
    first_discrete_level_probs = get_discrete_tree_head(continuum_energy, meta_params, params)
    # Sample the first level (sum should already be 1, but we normalise for tolerance issues)
    first_discrete_level = random.choice(subkey, len(first_discrete_level_probs), p=first_discrete_level_probs/jnp.sum(first_discrete_level_probs))

    discrete_path = [first_discrete_level]

    # Sample until we hit the ground state
    current_level = first_discrete_level
    while current_level != 0:
        next_level_probs = jnp.array(discrete_decay_widths[current_level, :current_level])
        available_level_number = len(next_level_probs)

        p = next_level_probs/jnp.sum(next_level_probs)
        #p[-1] = 1 - np.sum(p[:-1]) # Fix rounding errors (this is stupid)

        _, subkey = random.split(subkey)
        next_level = random.choice(subkey, available_level_number, p=p)
        
        # Update
        discrete_path.append(next_level)
        current_level = next_level

    return discrete_path


def sample_discrete_path(continuum_energy, full_tree, meta_params, params, key):
    '''
    Given a final continuum energy (vector), and a discrete tree, sample a discrete path.
    '''

    discrete_path_probs, discrete_paths = full_tree
    discrete_energies = meta_params['discrete_energies']

    num_samples = len(continuum_energy)
    random_numbers = random.uniform(key, shape=(num_samples,))
    discrete_path_probs_cumsum = jnp.cumsum(discrete_path_probs, axis=1)

    path_indices = np.array([np.searchsorted(row, rnd_num) for row, rnd_num in zip(discrete_path_probs_cumsum, random_numbers)])

    return path_indices, discrete_paths[path_indices], discrete_energies[discrete_paths[path_indices]]


def stitch_paths(raw_continuum_paths, raw_discrete_paths):
    '''
    Given a set of (energy) continuum paths and discrete paths, stitch them together.
    '''
    full_energy_paths = []
    for i, continuum_energy_path in enumerate(raw_continuum_paths):
        continuum_energy_path = continuum_energy_path[: np.argwhere(continuum_energy_path == -1)[0][0]]
        discrete_energy_path = raw_discrete_paths[i, : np.argwhere(raw_discrete_paths[i, :] == 0)[0][0] + 1] 

        full_energy_path = np.concatenate((continuum_energy_path, discrete_energy_path))
        full_energy_paths.append(full_energy_path)

    return full_energy_paths


def compute_expected_value(function, full_paths, meta_params, params):
    '''
    Given a function that takes a list of energies and returns a value, and a list of full paths,
    compute the expected value of the function for these paths.
    '''
    pass


def compute_expected_value_for_continuum_path(function, continuum_path, meta_params, params, discrete_tree=None):
    '''
    Given a function that takes a list of energies and returns a value, and a (list of) continuum path(s),
    compute the expected value of the function for this given path considering the probabilities
    of each possible discrete path.
    The discrete probabilities may be provided or computed on the fly.
    '''
    pass











