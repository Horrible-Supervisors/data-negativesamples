import os, pdb, sys
import numpy as np
import random
import math

from PIL import Image
import json
from diffusers import StableDiffusionPipeline
import torch

import logging

device = "cuda"
ds_name = "ImageNet"
# number of generated images per label
EPOCHS = 5
# number of labels passed to the diffusion model at one time
batch_size = 2
# path to the data folder
BASE_FOLDER = "data"

def load_labels(ds_name):
    if ds_name == "Imagenette":
        # load Imagenette labels
        labels = ['a photo of tench', 'a photo of English springer', 'a photo of cassette player', 'chain saw', 'church', 'French horn', 
                  'garbage truck', 'gas pump', 'golf ball', 'parachute']
    else:
        # load Imagenet labels
        class_idx = json.load(open("imagenet_class_index.json"))
        labels = ["a photo of " + class_idx[str(k)][1] for k in range(len(class_idx))]
    return labels

def load_model(device):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

def negative_samples(pipe):
    logging.basicConfig(level=logging.INFO)
    logging.info(('Creating negative image examples with parameters:', 
                 ' Device : ', device, ' ds_name : ', ds_name, 
                 ' Batch size : ', batch_size, ' Epochs : ', EPOCHS))
    logging.info(('Is cuda available : ', torch.cuda.is_available()))

    base_path = os.path.join(os.getcwd(), BASE_FOLDER)

    if not os.path.exists(base_path):
        os.mkdir(base_path)


    labels = load_labels(ds_name)

    # generate images from the labels (batching needed)
    for i in range(math.ceil(len(labels) / batch_size)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(labels))
        batch_PIL = pipe(labels[start_idx:end_idx], num_images_per_prompt=EPOCHS).images
        # save the images to disk
        for id in range(end_idx - start_idx):
            for e in range(EPOCHS):
                # image name format: imageId_#sample.png
                output_file_path = os.path.join(base_path, str(id + start_idx) + "_" + str(e) + '.png')
                batch_PIL[id * EPOCHS + e].resize((224, 224)).save(output_file_path, "PNG")

# refer to requirements.txt for requirements
# Stable Diffusion model uses accelerate to lower memory usage and accelerate inference 
pipe = load_model(device)
negative_samples(pipe)
