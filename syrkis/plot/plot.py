# plot.py
#    syrkis plots
# by: Noah Syrkis

# imports
import numpy as np
from IPython.display import display, clear_output
from IPython.display import display, HTML
import numpy as np
import base64
from PIL import Image as PILImage
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
import darkdetect


# globals
TEMPLATE_DIR = '/Users/syrkis/code/syrkis/syrkis/templates'
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


# functions
def multiples(imgs, info={}, figsize=(6, 3)):
    top, bottom = info.get('top'), info.get('bottom')
    if len(imgs[0].shape) == 2:
        imgs = [np.expand_dims(img, axis=2) for img in imgs]  # add channel dim
    invertable = imgs[0].shape[-1] == 1
    n_rows, n_cols = figsize
    imgs = np.array(imgs[:n_rows * n_cols])
    if invertable and not darkdetect.isDark():
        imgs = np.abs(1 - imgs)
    template = env.get_template('images.html')
    imgs = [matrix_to_image(pred) for pred in imgs]
    background = "black" if darkdetect.isDark() else "white"
    html = template.render(images=imgs, n_cols=n_cols, low_bar=bottom if bottom else [""],
                            top_bar=top if top else [""], background=background)

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
