#global imports
import numpy as np
import os
import sys
import socket

import matplotlib.pyplot as plt

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

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

#import DRACO_Frameworks.CNN.CNN as CNN
import DRACO_Frameworks.CNN.data_frame
import DRACO_Frameworks.CNN.DNN_like_CNN as DNN
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
savepath = workpath + "/DNN_like_CNN_"+str(key)+""


dnn = DNN.DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    train_epochs        = 100,
    early_stopping      = 20,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"]
    )

'''
# Create a test image
data = cnn.data.get_train_data_cnn();
print(data.shape)
data = data[0]
print(np.squeeze(data, axis=2).shape)

plt.imshow( np.squeeze(data, axis=2).T, cmap = "Greens",
    extent = (-2.5,2.5,-np.pi,np.pi), aspect = 'equal')
plt.xlabel("eta")
plt.ylabel("phi")
plt.tight_layout()
plt.savefig(savepath + "example.pdf")
'''


# Build model

model = models.Sequential()
# add input layer
model.add(layer.Dense(
    100,
    input_dim = 775,
    kernel_regularizer = keras.regularizers.l2(1e-5)))

# loop over all dens layers

model.add(layer.Dense(
    100,
    kernel_regularizer = keras.regularizers.l2(1e-5)))
model.add(layer.Dropout(0.3))

model.add(layer.Dense(
    100,
    kernel_regularizer = keras.regularizers.l2(1e-5)))
model.add(layer.Dropout(0.3))

# create output layer
model.add(layer.Dense(
    dnn.data.n_output_neurons,
    activation = "softmax",
    kernel_regularizer = keras.regularizers.l2(1e-5)))


dnn.build_model(model)

dnn.train_models()

dnn.eval_model()

dnn.plot_metrics()
dnn.plot_confusion_matrix(norm_matrix = True)
'''
cnn.plot_prenet_nodes()
cnn.plot_class_differences()
cnn.plot_discriminators()
cnn.plot_classification()

cnn.plot_output_output_correlation(plot=True)
cnn.plot_input_output_correlation(plot=False)
'''
