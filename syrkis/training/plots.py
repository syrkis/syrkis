# plots.py
#     neuroscope plots
# by: Noah Syrkis

# imports
import numpy as np
from IPython.display import display, Image, clear_output
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from nilearn import plotting
from tqdm import tqdm
from IPython.display import display, HTML
import time
import numpy as np
import base64
import os
from PIL import Image as PILImage
from io import BytesIO
from jinja2 import Template, Environment, FileSystemLoader
from jax import vmap
import darkdetect


# globals
# templates folder is located in the same folder as this file (which is part of a module)
TEMPLATE_DIR = '/Users/syrkis/code/syrkis/syrkis/templates'
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def plot_multiples(imgs, n_rows=2, n_cols=6, info_bar=None):
    imgs = np.array(imgs[:n_rows * n_cols])
    template = env.get_template('images.html')
    imgs = [matrix_to_image(pred) for pred in imgs]
    background = "dark" if darkdetect.isDark() else "white"
    html = template.render(images=imgs, n_cols=n_cols, info_bar=info_bar if info_bar else [""], background=background)
    clear_output(wait=True)
    display(HTML(html))


def matrix_to_image(matrix):
        # ensure all values are [0; 1]
        matrix = np.clip(matrix, 0, 1)
        # if matrix is 128 x 128 x 1, convert to 128 x 128 x 3
        matrix = np.repeat(matrix, 3, axis=2) if matrix.shape[2] == 1 else matrix
        image = PILImage.fromarray((matrix * 255).astype(np.uint8))
        image_bytes = BytesIO()
        image.save(image_bytes, format='png')
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        return encoded_image