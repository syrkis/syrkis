# datasets.py
#   datasets
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow_datasets as tfds


# functions
def mnist(n_samples=None):
    data = tfds.load('mnist', split='train', shuffle_files=True)
    data = tfds.as_numpy(data)
    # only take n_samples
    X, Y = [], []
    for i, x in enumerate(data):
        if i == n_samples:
            break
        X.append(x['image'])
        Y.append(x['label'])
    X = jnp.array(X).astype('float32') / 255.
    Y = jnp.array(Y).astype('int32')
    return X, Y