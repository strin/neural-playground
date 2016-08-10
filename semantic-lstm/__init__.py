# some common utils and imports.
import json
from pprint import pprint
import re
import traceback
import sys
import numpy as np
import numpy.random as npr
from utils import *
sys.path = ['../external/keras/build/lib'] + sys.path
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
print keras.__file__
from keras.models import Sequential
from keras.layers import Embedding, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Lambda, RepeatVector, Reshape, Dense
import keras.backend as K
