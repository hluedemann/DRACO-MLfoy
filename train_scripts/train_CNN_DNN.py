# global imports
import os
import sys

import matplotlib
matplotlib.use('Agg')

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.CNN_DNN.CNN_DNN as CNN_DNN
import variable_sets.topVariables_T as variable_set

JTcategory      = sys.argv[1]
variables       = variable_set.variables[JTcategory]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"
inPath = workpath + "/train_samples"
savepath = workpath + "/CNN_DNN_"+str(JTcategory)+""

cmatrix_file = workpath+"/confusionMatrixData/CNN_DNN_"+str(JTcategory)+".h5"


cnn_dnn = CNN_DNN.CNN_DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = JTcategory,
    train_variables     = variables,
    batch_size          = 5000,
    train_epochs        = 10,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.5,
    eval_metrics        = ["acc"],
    phi_padding         = 0
    )



'''
modelCNN = models.Sequential()

modelCNN.add(Conv2D(128, (4, 4), padding="same", input_shape = cnn_dnn.data.size_input_image))
modelCNN.add(Activation("relu"))
modelCNN.add(AveragePooling2D(pool_size=(2,2)))

modelCNN.add(Conv2D(64, (4, 4), padding="same"))
modelCNN.add(Activation("relu"))
modelCNN.add(AveragePooling2D(pool_size=(2, 2)))
modelCNN.add(Conv2D(32, (4, 4), padding="same"))
modelCNN.add(Activation("relu"))
modelCNN.add(AveragePooling2D(pool_size=(2, 2)))
modelCNN.add(Flatten())

modelDNN = models.Sequential()
modelDNN.add(Dense(100, input_shape = (cnn_dnn.data.n_input_neurons,)))
modelDNN.add(Dropout(0.3))
modelDNN.add(Dense(100))


# modelDNN.add(Activation("relu"))
# modelDNN.add(Dropout(0.5))
# modelDNN.add(Dense(100))
# modelDNN.add(Activation("relu"))
# modelDNN.add(Dropout(0.5))


mergedOutput = layer.Concatenate()([modelCNN.output, modelDNN.output])

out = Dense(100, activation='relu')(mergedOutput)
out = Dropout(0.3)(out)
out = Dense(100, activation='relu')(out)
out = Dropout(0.3)(out)
out = Dense(cnn_dnn.data.n_output_neurons, activation='softmax')(out)

mergedModel = models.Model([modelCNN.input, modelDNN.input], out)
'''


num_runs = 10

for i in range(num_runs):

    cnn_dnn.build_model()
    cnn_dnn.train_models()
    cnn_dnn.eval_model()
    cnn_dnn.plot_metrics()
    cnn_dnn.plot_discriminators()

    # plotting
    cnn_dnn.save_confusionMatrix(location = cmatrix_file, save_roc = True)
    cnn_dnn.plot_confusionMatrix(norm_matrix = True)
    cnn_dnn.plot_outputNodes()
