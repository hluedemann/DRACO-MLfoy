#global imports
import numpy as np
import os
import sys
import socket

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


#import DRACO_Frameworks.CNN.CNN as CNN
import DRACO_Frameworks.CNN_DNN.data_frame
import DRACO_Frameworks.CNN_DNN.CNN_DNN as CNN_DNN
import DRACO_Frameworks.CNN.variable_info as variable_info

category_vars = {
    "4j_ge3t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b}
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }
prenet_targets = [
    #"GenAdd_BB_inacceptance_part",
    #"GenAdd_B_inacceptance_part",
    "GenHiggs_BB_inacceptance_part",
    "GenHiggs_B_inacceptance_part",
    "GenTopHad_B_inacceptance_part",
    "GenTopHad_QQ_inacceptance_part",
    "GenTopHad_Q_inacceptance_part",
    "GenTopLep_B_inacceptance_part"
    ]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

if "naf" in socket.gethostname():
    workpath = "/nfs/dust/cms/user/luedeman/DRACO-MLfoy/workdir/"
else:
    workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"

key = sys.argv[1]

inPath   = workpath + "/train_samples"
savepath = workpath + "/CNN_DNN_version_2_2_"+str(key)+""


cnn_dnn = CNN_DNN.CNN_DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    batch_size          = 5000,
    train_epochs        = 15,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 0
    )



modelCNN = models.Sequential()

modelCNN.add(Conv2D(64, (4, 4), padding="same", input_shape = cnn_dnn.data.size_input_image))
modelCNN.add(Activation("relu"))
modelCNN.add(AveragePooling2D(pool_size=(2,2)))

modelCNN.add(Conv2D(32, (4, 4), padding="same"))
modelCNN.add(Activation("relu"))
modelCNN.add(AveragePooling2D(pool_size=(2, 2)))
modelCNN.add(Conv2D(16, (4, 4), padding="same"))
modelCNN.add(Activation("relu"))
modelCNN.add(AveragePooling2D(pool_size=(2, 2)))
modelCNN.add(Flatten())

modelDNN = models.Sequential()
modelDNN.add(Dense(100, input_shape = (cnn_dnn.data.n_input_neurons,)))

# modelDNN.add(Activation("relu"))
# modelDNN.add(Dropout(0.5))
# modelDNN.add(Dense(100))
# modelDNN.add(Activation("relu"))
# modelDNN.add(Dropout(0.5))


mergedOutput = layer.Concatenate()([modelCNN.output, modelDNN.output])

out = Dense(100, activation='relu')(mergedOutput)
out = Dropout(0.5)(out)
out = Dense(100, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(cnn_dnn.data.n_output_neurons, activation='softmax')(out)

mergedModel = models.Model([modelCNN.input, modelDNN.input], out)

cnn_dnn.build_model(mergedModel)

cnn_dnn.train_models()

cnn_dnn.eval_model()

cnn_dnn.plot_metrics()
cnn_dnn.plot_confusion_matrix(norm_matrix = True)


'''
cnn_dnn.plot_prenet_nodes()
cnn_dnn.plot_class_differences()
cnn_dnn.plot_discriminators()
cnn_dnn.plot_classification()

cnn_dnn.plot_output_output_correlation(plot=True)
cnn_dnn.plot_input_output_correlation(plot=False)
'''
