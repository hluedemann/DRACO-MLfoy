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
# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import variable_sets.topVariables_T as variable_set

JTcategory      = sys.argv[1]
variables       = variable_set.variables[JTcategory]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

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


workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"
inPath = workpath + "/train_samples"
savepath = workpath + "/DNN_Aachen_"+str(JTcategory)+""

cmatrix_file = workpath+"/confusionMatrixData/DNN_Aachen_"+str(JTcategory)+".h5"

dnn_aachen = DNN_Aachen.DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = JTcategory,
    train_variables     = variables,
    prenet_targets      = prenet_targets,
    train_epochs        = 50,
    early_stopping      = 20,
    eval_metrics        = ["acc"],
    test_percentage     = 0.2)


num_runs = 15

for i in range(num_runs):



    dnn_aachen.build_model()
    dnn_aachen.train_models()
    dnn_aachen.eval_model()

    dnn_aachen.plot_metrics()
    #dnn_aachen.plot_prenet_nodes()
    #dnn_aachen.plot_class_differences()
    dnn_aachen.plot_discriminators()

    dnn_aachen.save_confusionMatrix(location = cmatrix_file, save_roc = True)
    dnn_aachen.plot_confusionMatrix(norm_matrix = True)


    #dnn_aachen.plot_output_output_correlation(plot=True)
    #dnn_aachen.plot_input_output_correlation(plot=False)
