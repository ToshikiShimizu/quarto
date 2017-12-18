#coding:utf-8
import pandas as pd
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
        return F.softmax_cross_entropy(self.l3(h2),t,reduce="no")

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return (F.softmax(self.l3(h2)))

class CNN(chainer.Chain):
    def __init__(self, n_channel, n_out, SIZE,initializer=None):
        super(CNN, self).__init__()
        self.SIZE = SIZE
        n_units = int((n_channel + n_out)/2)

        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Convolution2D(None, n_channel, ksize=self.SIZE, pad=0, nobias=True,initialW = initializer)
            #self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units,initialW = initializer)  # n_in -> n_units
            self.l3 = L.Linear(None, n_out,initialW = initializer)  # n_units -> n_units
    def __call__(self, x, t):
        x = x.reshape(-1,(2**(self.SIZE+1)+2),self.SIZE,self.SIZE)
        #print (x.shape)
        h1 = F.relu(self.l1(x))
        #print (h1.shape)
        h2 = F.relu(self.l2(h1))
        #print (h2.shape)
        return F.softmax_cross_entropy(self.l3(h2),t,reduce="no")

    def predict(self, x):
        x = x.reshape(-1,(2**(self.SIZE+1)+2),self.SIZE,self.SIZE)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        # print (self.l3(h2))
        # print (F.softmax(self.l3(h2)))
        return (F.softmax(self.l3(h2)))
