# utils.py
#    contains utility functions for training and testing
# by: Noah Syrkis


# imports
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import jax
from jax import jit, lax
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml


DIMENSION_NUMBERS = ("NHWC", "HWIO", "NHWC")
STRIDE            = 2

def save_params(params, path="model"):
    if not os.path.exists(path):
        os.makedirs(path)

    flat_params, structure = tree_flatten(params)
    for i, param in enumerate(flat_params):
        file_path = os.path.join(path, f"param_{i}.csv")
        # Save the shape information in a separate file
        shape_path = os.path.join(path, f"shape_{i}.pkl")
        with open(shape_path, 'wb') as shape_file:
            pickle.dump(param.shape, shape_file)
        # Reshape the parameter to 2D for saving
        reshaped_param = param.reshape(-1, param.shape[-1]) if param.ndim > 1 else param
        np.savetxt(file_path, reshaped_param, delimiter=",")

    # Save the structure
    with open(os.path.join(path, "structure.pkl"), "wb") as f:
        pickle.dump(structure, f)

def load_params(path="model"):
    # Load the structure
    with open(os.path.join(path, "structure.pkl"), "rb") as f:
        structure = pickle.load(f)

    # Reconstruct flat_params from saved files
    flat_params = []
    i = 0
    while os.path.exists(os.path.join(path, f"param_{i}.csv")):
        file_path = os.path.join(path, f"param_{i}.csv")
        shape_path = os.path.join(path, f"shape_{i}.pkl")
        param = np.loadtxt(file_path, delimiter=",")
        # Load the shape and reshape the parameter
        with open(shape_path, 'rb') as shape_file:
            shape = pickle.load(shape_file)
        reshaped_param = param.reshape(shape).astype(np.float32)
        flat_params.append(reshaped_param)
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


def load_config(file_path='./config.yaml'):
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def cross_entropy(logits, labels, epsilon=1e-12):
    max_logits = jnp.max(logits, axis=1, keepdims=True)
    stabilized_logits = logits - max_logits
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(stabilized_logits), axis=1, keepdims=True) + epsilon)
    labels_one_hot = jnp.eye(logits.shape[1])[labels]
    loss = -jnp.mean(labels_one_hot * (stabilized_logits - log_sum_exp))
    return loss

def mean_squared_error(logits, labels):
    return jnp.mean((logits - labels) ** 2)

def print_model(params, path=""):
    if isinstance(params, dict):
        for k, v in params.items():
            print_model(v, path + k + " -> ")
    elif isinstance(params, (list, tuple)):
        for i, v in enumerate(params):
            item_type = "List" if isinstance(params, list) else "Tuple"
            print_model(v, path + item_type + f" {i} -> ")
    else:
        if hasattr(params, 'shape'):
            print(f"{path}Shape: {params.shape}")

def cross_entropy(logits, labels, epsilon=1e-12):
    max_logits = jnp.max(logits, axis=1, keepdims=True)
    stabilized_logits = logits - max_logits
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(stabilized_logits), axis=1, keepdims=True) + epsilon)
    labels_one_hot = jnp.eye(logits.shape[1])[labels]
    loss = -jnp.mean(labels_one_hot * (stabilized_logits - log_sum_exp))
    return loss


def kl_divergence(mu, logvar):
    return -0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar), axis=1).mean()
    
def reparametrize(mu, logvar, rng):
    # mu, logvar: (batch, embed_dim)
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, mu.shape)
    return mu + eps * std

def glorot_init(rng, shape):
    if len(shape) == 2:  # Dense layer
        fan_in, fan_out = shape
    elif len(shape) == 4:  # Convolutional layer
        fan_in = shape[0] * shape[1] * shape[2]  # kernel_height * kernel_width * in_channels
        fan_out = shape[0] * shape[1] * shape[3]  # kernel_height * kernel_width * out_channels
    limit = jnp.sqrt(6 / (fan_in + fan_out))
    return jax.random.uniform(rng, shape, minval=-limit, maxval=limit)


@jit
def conv2d(x, w):
    return jax.lax.conv_general_dilated(
        x, w, 
        window_strides=(STRIDE, STRIDE),
        padding='SAME',
        dimension_numbers=DIMENSION_NUMBERS)

def conv1d(x, w):
    return jax.lax.conv_general_dilated(
        x, w, 
        window_strides=(1,),
        padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC'))

@jit
def upscale_nearest_neighbor(x, scale_factor=STRIDE):
    # Assuming x has shape (batch, height, width, channels)
    b, h, w, c = x.shape
    x = x.reshape(b, h, 1, w, 1, c)
    x = lax.tie_in(x, jnp.broadcast_to(x, (b, h, scale_factor, w, scale_factor, c)))
    return x.reshape(b, h * scale_factor, w * scale_factor, c)


@jit
def deconv2d(x, w):
    x_upscaled = upscale_nearest_neighbor(x)
    return lax.conv_transpose(
        x_upscaled, w, 
        strides=(1, 1), 
        padding='SAME',
        dimension_numbers=DIMENSION_NUMBERS) 