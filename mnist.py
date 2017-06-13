#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)

import numpy as np

from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

x_train.shape # (60000, 784)
t_train.shape # (60000, )
x_test.shape  # (10000, 784)
t_test.shape  # (10000, )

img = x_train[0]
label = t_train[0]

img.shape
img = img.reshape(28, 28)
img.shape

img_show(img)
