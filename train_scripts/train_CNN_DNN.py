# global imports
import os
import sys

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
savepath = workpath + "/DNN_"+str(JTcategory)+"/"

cmatrix_file = workpath+"/confusionMatrixData/topVariablesTight_"+str(JTcategory)+".h5"


cnn_dnn = CNN_DNN.CNN_DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = JTcategory,
    train_variables     = variables,
    batch_size          = 5000,
    train_epochs        = 5,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 10
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


cnn_dnn.build_model()

cnn_dnn.train_models()

cnn_dnn.eval_model()

cnn_dnn.plot_metrics()

cnn_dnn.plot_discriminators()

cnn_dnn.plot_confusion_matrix(norm_matrix = True)


# plotting 
cnn_dnn.save_confusionMatrix(location = cmatrix_file, save_roc = True)
cnn_dnn.plot_confusionMatrix(norm_matrix = True)

dnn.plot_outputNodes()


#dnn.plot_output_output_correlation(plot=True)
#dnn.plot_input_output_correlation(plot=False)

'''
cnn_dnn.plot_prenet_nodes()
cnn_dnn.plot_class_differences()
cnn_dnn.plot_discriminators()
cnn_dnn.plot_classification()

cnn_dnn.plot_output_output_correlation(plot=True)
cnn_dnn.plot_input_output_correlation(plot=False)
'''
