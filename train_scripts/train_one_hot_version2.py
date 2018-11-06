# global imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf

import keras.layers as layer
import keras.models as models
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


from keras.backend.tensorflow_backend import set_session

# Limit the gpu memory usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import DRACO_Frameworks.CNN.CNN_one_hot as CNN


#inPath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/train_samples/base_train_set"
inPath = "/ceph/hluedemann/DRACO-MLfoy/workdir/train_samples/cut_train_set"

cnn = CNN.CNN(
    in_path         = inPath,
    save_path       = basedir+"/workdir/cut_custom_cnn_one_hot",
    class_label     = "nJets",
    batch_size      = 256,
    train_epochs    = 20,
    optimizer       = "adam",
    loss_function   = "categorical_crossentropy",
    eval_metrics    = ["mean_squared_error", "acc"] )

cnn.load_datasets()

#-------------------------------------------------------

# build model
model = models.Sequential()
#first layer
model.add(
    layer.Conv2D( 32, kernel_size = (10,10), activation = "relu", padding = "same",
    input_shape = cnn.train_data.input_shape ))
model.add(
    layer.MaxPooling2D( pool_size = (2,2), padding = "same" ))


# second layer
model.add(
    layer.Conv2D( 64, kernel_size = (8,8), activation = "relu", padding = "same"))
model.add(
    layer.AveragePooling2D( pool_size = (2,2), padding = "same" ))

# third layer
model.add(
    layer.Conv2D( 128, kernel_size = (6,6), activation = "relu", padding = "same"))
model.add(
    layer.AveragePooling2D( pool_size = (2,2), padding = "same" ))

#  layer
model.add(
    layer.Conv2D( 256, kernel_size = (4,4), activation = "relu", padding = "same"))
model.add(
    layer.AveragePooling2D( pool_size = (2,2), padding = "same" ))



# first dense layer
model.add(
    layer.Flatten())
model.add(
    layer.Dense( 128, activation = "relu" ))


#second dense layer
model.add(
    layer.Dense(128, activation = "relu" ))



#third dense layer
model.add(
    layer.Dense(cnn.num_classes, activation = "softmax" ))
# -----------------------------------------------




cnn.build_model(model)
cnn.train_model(earlyStopping=False)
cnn.eval_model()

cnn.plot_filters(0,8,4)
cnn.plot_filters(2,8,4)

cnn.plot_layer_output(8)

# evaluate stuff
cnn.print_classification_examples()
cnn.print_classification_report()
cnn.plot_metrics()
cnn.plot_discriminators()
cnn.plot_confusion_matrix()







