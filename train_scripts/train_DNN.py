
# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import variable_sets.topVariables_T as variable_set

JTcategory      = sys.argv[1]
variables       = variable_set.variables[JTcategory]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"
inPath = workpath + "/train_samples"
savepath = workpath + "/DNN_"+str(JTcategory)+"/"

cmatrix_file = workpath+"/confusionMatrixData/topVariablesTight_"+str(JTcategory)+".h5"

if not os.path.exists(os.path.dirname(cmatrix_file)):
    os.makedirs(os.path.dirname(cmatrix_file))

dnn = DNN.DNN(
    in_path         = inPath,
    save_path       = savepath,
    event_classes   = event_classes,
    event_category  = JTcategory,
    train_variables = variables,
    train_epochs    = 500,
    early_stopping  = 20,
    eval_metrics    = ["acc"],
    test_percentage = 0.2)

dnn.build_model()
dnn.train_model()
dnn.eval_model()
dnn.plot_metrics()

dnn.plot_discriminators()


#dnn.get_input_weights()
#dnn.plot_metrics()

# plotting 
dnn.save_confusionMatrix(location = cmatrix_file, save_roc = True)
dnn.plot_confusionMatrix(norm_matrix = True)

dnn.plot_outputNodes()


#dnn.plot_output_output_correlation(plot=True)
#dnn.plot_input_output_correlation(plot=False)
