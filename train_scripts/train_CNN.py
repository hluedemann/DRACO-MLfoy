#global imports
import numpy as np
import os
import sys
import socket

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
import DRACO_Frameworks.CNN.data_frame
import DRACO_Frameworks.CNN.CNN as CNN
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
savepath = workpath + "/CNN_Phi_"+str(key)+""


cnn = CNN.CNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    train_epochs        = 10,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 0
    )


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
plt.savefig(savepath + "/example1.pdf")
'''
cnn1 = CNN.CNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    train_epochs        = 10,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 0
    )


# Create a test image
data = cnn1.data.get_train_data_cnn();
print(data.shape)
data = data[0]
print(np.squeeze(data, axis=2).shape)

plt.imshow( np.squeeze(data, axis=2).T, cmap = "Greens",
    extent = (-2.5,2.5,-np.pi,np.pi), aspect = 'equal')
plt.xlabel("eta")
plt.ylabel("phi")
plt.tight_layout()
plt.savefig(savepath + "/example2.pdf")
'''



model = models.Sequential()

# CONV -> RELU -> POOL
model.add(Conv2D(32, (2, 2), padding="same", input_shape = cnn.data.size_input_image))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(2,2)))


# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, (4, 4), padding="same"))
#model.add(Activation("relu"))
#model.add(AveragePooling2D(pool_size=(2, 2)))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
#model.add(BatchNormalization())
# softmax classifier
model.add(Dense(cnn.data.n_output_neurons))
model.add(Activation("softmax"))


cnn.build_model()

cnn.train_models()

#cnn.eval_model()

#cnn.plot_metrics()
#cnn.plot_confusion_matrix(norm_matrix = True)
'''
cnn.plot_prenet_nodes()
cnn.plot_class_differences()
cnn.plot_discriminators()
cnn.plot_classification()

cnn.plot_output_output_correlation(plot=True)
cnn.plot_input_output_correlation(plot=False)
'''
