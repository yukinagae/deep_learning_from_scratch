#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# AND (単層パーセプトロン)
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# NAND (単層パーセプトロン)
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# OR (単層パーセプトロン)
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 単層のパーセプトロンでは表現できないため、NANDとOR、ANDを組み合わせることで表現する(多層パーセプトロン)
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# AND
AND(0,0) # 0
AND(0,1) # 0
AND(1,0) # 0
AND(1,1) # 1

# NAND
NAND(0,0) # 1
NAND(0,1) # 1
NAND(1,0) # 1
NAND(1,1) # 0

# OR
OR(0,0) # 0
OR(0,1) # 1
OR(1,0) # 1
OR(1,1) # 1

# XOR
XOR(0,0) # 0
XOR(0,1) # 1
XOR(1,0) # 1
XOR(1,1) # 0
