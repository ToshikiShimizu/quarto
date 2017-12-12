#coding:utf-8
import pandas as pd
import numpy as np
import shutil
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable
from chainer import optimizers
import argparse
from chainer import cuda
import os
from sklearn.model_selection import KFold
import time
from sklearn.externals import joblib

class MLP(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__()
        n_units = int((n_in + n_out)/2)
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_in -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_units
    def __call__(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return F.softmax_cross_entropy(self.l3(h2),t)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return (F.softmax(self.l3(h2)))