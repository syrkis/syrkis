# utils.py
#    contains utility functions for training and testing
# by: Noah Syrkis


# imports
import os
import numpy as np
import pickle
from jax.tree_util import tree_flatten, tree_unflatten


def save_params(params, path="model"):
    """Save JAX model weights to csvs."""
    flat_params, structure = tree_flatten(params)
    for i, param in enumerate(flat_params):
        np.savetxt(os.path.join(path, f"param_{i}.csv"), param, delimiter=",")
    with open(os.path.join(path, "structure.pkl"), "wb") as f:
        pickle.dump(structure, f)


def load_params(path="model"):
    """Load JAX model weights from csvs."""
    # Load the structure
    with open(os.path.join(path, "structure.pkl"), "rb") as f:
        structure = pickle.load(f)

    # Reconstruct flat_params from saved files
    flat_params = []
    i = 0
    while os.path.exists(os.path.join(path, f"param_{i}.csv")):
        param = np.loadtxt(os.path.join(path, f"param_{i}.csv"), delimiter=",")
        flat_params.append(param)
        i += 1

    # Reconstruct the original params structure
    params = tree_unflatten(structure, flat_params)
    return params


def n_params(params):
    flat_params, _ = tree_flatten(params)
    n_params = 0
    for param in flat_params:
        n_params += param.size
    return n_params
