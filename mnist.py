import tensorflow as tf
import numpy as np
import os
import struct

def load_mnist(path, kind='train'):
	labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
	images_path = os.path.join(path, '%s-images-idx1-ubyte' % kind)
	with open(labels_path, 'rb') as lbpath
