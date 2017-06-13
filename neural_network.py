#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

# 活性化関数
#
# ステップ関数 (階段関数)
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# 活性化関数
#
# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 活性化関数
#
# ReLU関数
def relu(x):
    return np.maximum(0, x)

# demo
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

X = np.array([1, 2])
X.shape
W = np.array([[1, 3, 5], [2, 4, 6]])
W.shape
Y = np.dot(X, W)
Y.shape
Y
