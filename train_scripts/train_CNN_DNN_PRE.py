# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.CNN_DNN.CNN_DNN_PRE as CNN_DNN_PRE
import variable_sets.topVariables_T as variable_set

JTcategory      = sys.argv[1]
variables       = variable_set.variables[JTcategory]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"
inPath = workpath + "/train_samples"
savepath = workpath + "/CNN_DNN_PRE_"+str(JTcategory)+""

cmatrix_file = workpath+"/confusionMatrixData/topVariablesTight_"+str(JTcategory)+".h5"


cnn_dnn_pre = CNN_DNN_PRE.CNN_DNN_PRE(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = JTcategory,
    train_variables     = variables,
    batch_size          = 5000,
    train_epochs        = 1,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.5,
    eval_metrics        = ["acc"],
    phi_padding         = 0
    )


cnn_dnn_pre.build_model()

cnn_dnn_pre.train_models()

cnn_dnn_pre.eval_model()
cnn_dnn_pre.plot_metrics()
cnn_dnn_pre.plot_discriminators()

# plotting 
cnn_dnn_pre.save_confusionMatrix(location = cmatrix_file, save_roc = True)
cnn_dnn_pre.plot_confusionMatrix(norm_matrix = True)


cnn_dnn_pre.plot_outputNodes()





'''
cnn_dnn.plot_prenet_nodes()
cnn_dnn.plot_class_differences()
cnn_dnn.plot_discriminators()
cnn_dnn.plot_classification()

cnn_dnn.plot_output_output_correlation(plot=True)
cnn_dnn.plot_input_output_correlation(plot=False)
'''
