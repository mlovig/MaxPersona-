!pip install 'h5py==2.10.0' --force-reinstall
!pip install bert-tensorflow==1.0.1
!pip install tensorflow==1.15
!pip install "tensorflow_hub>=0.6.0"
!pip3 install tensorflow_text==1.15

import tensorflow as tf
from tensorflow.keras import layers
from bisect import bisect
import datetime
import nltk
import matplotlib.pyplot as mpl
nltk.download('punkt')
import tensorflow.keras as keras
import pandas
import random
from tqdm import tqdm
from numpy.random import seed
seed(1)
from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, confusion_matrix, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from IPython.display import Image
import numpy as np


rez = pandas.io.parsers.read_csv("/home/maxwelllovig/Downloads/brown.csv")
mixed_sentences = rez['raw_text']
sentences = [None] * len(mixed_sentences)
for i in tqdm(range(0,len(mixed_sentences))):
        splx = mixed_sentences[i].split(' ')
        tempsent = [None] * len(splx)
        for ii in range(0,len(splx)):
            splind = splx[ii].rfind('/')
            word = splx[ii][:splind]
            pos = splx[ii][(splind+1):]
            pos = pos.replace("-tl", "")
            pos = pos.replace("-hl" ,"")
            pos = pos.replace("fw-" , "")
            pos = pos.replace("-nc", "")
            pos = pos.replace("bez", "bbb")
            tempsent[ii] = (word, pos)
        sentences[i] = tempsent