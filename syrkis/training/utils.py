# utils.py
#    contains utility functions for training and testing
# by: Noah Syrkis


# imports
import os
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten


def save_model_weights(params, path="model"):
    """Save JAX model weights to csvs."""
    flat_params, structure = tree_flatten(params)
    np.savetxt(os.path.join(path, "structure.csv"), structure, delimiter=",")
    for i, param in enumerate(flat_params):
        np.savetxt(os.path.join(path, f"param_{i}.csv"), param, delimiter=",")
    return

def load_model_weights(path):
    """params dict from csvs."""
    structure = np.loadtxt(path + "structure.csv", delimiter=",")
    flat_params = []
    for i in range(len(structure)):
        flat_params.append(np.loadtxt(path + f"param_{i}.csv", delimiter=","))
    params = tree_unflatten(structure, flat_params)
    return params