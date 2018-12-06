#global imports
import numpy as np
import os
import sys
import socket

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

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
savepath = workpath + "/CNN"+str(key)+""


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
    eval_metrics        = ["acc"]
    )


cnn.build_model()

cnn.train_models()

cnn.eval_model()

cnn.plot_metrics()
cnn.plot_confusion_matrix(norm_matrix = True)
'''
cnn.plot_prenet_nodes()
cnn.plot_class_differences()
cnn.plot_discriminators()
cnn.plot_classification()

cnn.plot_output_output_correlation(plot=True)
cnn.plot_input_output_correlation(plot=False)
'''
