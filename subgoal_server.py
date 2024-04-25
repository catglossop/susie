import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys 

import inspect

import jax
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from absl import app, flags

import wandb
from susie.jax_utils import (
    initialize_compilation_cache,
)
from susie.model import create_sample_fn

# jax diffusion stuff
from absl import app as absl_app
from absl import flags
from PIL import Image
import jax
import jax.numpy as jnp

# flask app here
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

# create rng
rng = jax.random.PRNGKey(0)

import datetime
import os
from collections import deque
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from typing import Callable, List, Tuple

import imageio
import jax
import numpy as np
from absl import app, flags
##############################################################################


CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.04.08_22.09.33"
WANDB_NAME = "catglossop/susie/susie-nav_2024.04.08_22.09.33"
PRETRAINED_PATH = "runwayml/stable-diffusion-v1-5:flax"
prompt_w = 7.5
context_w = 2.5
diffusion_num_steps = 50
IP = "192.168.0.153"
PORT = 5000


##############################################################################

np.set_printoptions(suppress=True)


def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

app = Flask(__name__)
@app.route("/gen_subgoals", methods=["POST"])
def gen_subgoals():
    print("IMAGE RECEIVED")

    diffusion_sample = create_sample_fn(
        CHECK_POINT_PATH,
        WANDB_NAME,
        diffusion_num_steps,
        prompt_w,
        context_w,
        0.0,
        PRETRAINED_PATH,
    )
    
    print("DIFFUSION SAMPLE CREATED")
    data = request.json
    image_data = base64.b64decode(data["image"])
    obs_image = Image.open(BytesIO(image_data))
    obs_image.save("obs.png")
    obs_image = np.array(obs_image.resize((128, 128), Image.Resampling.LANCZOS))[..., :3] * 255
   
    # prepare inputs
    prompt = input("Enter prompt: ")
    sample = diffusion_sample(obs_image, prompt)
    sample = [jax.device_get(sample)]
    samples = np.stack(samples, axis=0).astype(np.uint8)
    imageio.imwrite("goal.png", sample)

    return jsonify({"samples": samples.tolist()})

def main(_argv): 
    initialize_compilation_cache()

    diffusion_sample = create_sample_fn(
        CHECK_POINT_PATH,
        WANDB_NAME,
        diffusion_num_steps,
        prompt_w,
        context_w,
        0.0,
        PRETRAINED_PATH,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

