import os, pdb, sys
import numpy as np
import random
import math

from PIL import Image
import json
from diffusers import StableDiffusionPipeline
import torch

device = "cuda"
ds_name = "Imagette"
# number of labels samples from all
NUM_LABELS = 10
NUM_BATCHES = 10
EPOCHS = 5
batch_size = 128
BASE_PATH = "./data/"

def load_labels(ds_name):
    if ds_name == "Imagette":
        # load Imagenette labels
        labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 
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

def negative_samples():
    labels = load_labels(ds_name)
    pipe = load_model(device)

    for e_num in range(EPOCHS):
        # sample NUM_LABELS from all labels
        random.shuffle(labels)
        images = []
        # generate images from the sampled labels (batching needed)
        for i in range(math.ceil(NUM_LABELS / batch_size)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(labels))
            batch_PIL = pipe(labels[start_idx:end_idx]).images[:(end_idx - start_idx)]
            for i in range(len(batch_PIL)):
                batch_PIL[i] = np.array(batch_PIL[i]) 
            images.append(np.stack(batch_PIL, axis=0))
        # epoch, batch number, and label as metadata
        bk_array = np.concatenate(images, axis=0)
        np.save(BASE_PATH + str(e_num) + ".npy", bk_array)

negative_samples()