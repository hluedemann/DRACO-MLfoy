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
# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.CNN_like_DNN.CNN as CNN
import variable_sets.topVariables_T as variable_set

JTcategory      = sys.argv[1]
variables       = variable_set.variables[JTcategory]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"
inPath = workpath + "/train_samples"
savepath = workpath + "/CNN_2_"+str(JTcategory)+""

cmatrix_file = workpath+"/confusionMatrixData/CNN_"+str(JTcategory)+".h5"

cnn = CNN.CNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = JTcategory,
    train_variables     = variables,
    train_epochs        = 10,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 0
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
plt.savefig(savepath + "/example1.pdf")

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
'''

cnn.build_model()
cnn.train_models()
cnn.eval_model()

cnn.plot_metrics()
cnn.plot_discriminators()

# plotting
cnn.save_confusionMatrix(location = cmatrix_file, save_roc = True)
cnn.plot_confusionMatrix(norm_matrix = True)

cnn.plot_outputNodes()

#cnn.plot_output_output_correlation(plot=True)
#cnn.plot_input_output_correlation(plot=False)
