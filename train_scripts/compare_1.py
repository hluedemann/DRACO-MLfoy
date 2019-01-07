# global imports
import rootpy.plotting as rp
import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.CNN_DNN.CNN_DNN as CNN_DNN
import DRACO_Frameworks.DNN.variable_info as variable_info

category_vars = {
    "4j_ge3t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b}
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }


event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"

key = sys.argv[1]

inPath   = workpath + "/train_samples"
savepath = workpath + "/compare_2_"+str(key)

# Define the models
cnn_dnn = CNN_DNN.CNN_DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    batch_size          = 5000,
    train_epochs        = 10,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 0)


dnn = DNN.DNN(
    in_path         = inPath,
    save_path       = savepath,
    event_classes   = event_classes,
    event_category  = categories[key],
    train_variables = category_vars[key],
    train_epochs    = 500,
    early_stopping  = 20,
    eval_metrics    = ["acc"])

def plot_confusion_matrix(confusion_matrix,
                          error_confusion_matrix,
                          xticklabels,
                          yticklabels,
                          title,
                          roc,
                          roc_err,
                          save_path,
                          norm_matrix = True,
                          difference = False):
    ''' generate confusion matrix '''
    n_classes = confusion_matrix.shape[0]

    # norm confusion matrix if wanted
    if norm_matrix:
        cm = np.empty( (n_classes, n_classes), dtype = np.float64 )
        cm_err = np.empty( (n_classes, n_classes), dtype = np.float64 )
        for yit in range(n_classes):
            evt_sum = float(sum(confusion_matrix[yit,:]))
            for xit in range(n_classes):
                cm[yit,xit] = confusion_matrix[yit,xit]/evt_sum
                cm_err[yit,xit] = error_confusion_matrix[yit,xit]/evt_sum

        confusion_matrix = cm
        error_confusion_matrix = cm_err

    plt.clf()

    plt.figure( figsize = [10,10])

    plt.title(title, fontsize=15)

    minimum = np.min( confusion_matrix )/(np.pi**2.0 * np.exp(1.0)**2.0)
    maximum = np.max( confusion_matrix )*(np.pi**2.0 * np.exp(1.0)**2.0)

    x = np.arange(0, n_classes+1, 1)
    y = np.arange(0, n_classes+1, 1)

    xn, yn = np.meshgrid(x,y)

    if difference:
        plt.pcolormesh(xn, yn, confusion_matrix, cmap = "summer")
    else:
        plt.pcolormesh(xn, yn, confusion_matrix,
            norm = LogNorm( vmin = max(minimum, 1e-6), vmax = min(maximum,1.)),
            cmap="jet")

    plt.colorbar()

    plt.xlim(0, n_classes)
    plt.ylim(0, n_classes)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    # add textlabel
    for yit in range(n_classes):
        for xit in range(n_classes):
            plt.text(
                xit+0.5, yit+0.5,
                "{:.3f} \n+- {:.3f}".format(confusion_matrix[yit, xit], error_confusion_matrix[yit, xit]),
                horizontalalignment = "center",
                verticalalignment = "center")

    plt_axis = plt.gca()
    plt_axis.set_xticks(np.arange( (x.shape[0] -1)) + 0.5, minor = False )
    plt_axis.set_yticks(np.arange( (y.shape[0] -1)) + 0.5, minor = False )

    plt_axis.set_xticklabels(xticklabels)
    plt_axis.set_yticklabels(yticklabels)

    plt_axis.set_aspect("equal")
    plt.annotate("ROC_Score: {:.3f} +- {:.3f}".format(roc, roc_err), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)
    plt.text(4.0, 9.5, "ROC_Score: " + str(roc), fontsize=14)

    out_path = save_path
    plt.savefig(out_path)
    print("saved confusion matrix at "+str(out_path))
    plt.clf()

    return confusion_matrix, error_confusion_matrix


# Run the dnn and dnn_aachen for num_runs train_samples and store the
# confusion matrix and the auc score for every run

num_runs = 5
cm_dnn = []
cm_cnn_dnn = []
auc_score_dnn = []
auc_score_cnn_dnn = []


for i in range(0, num_runs):

    print("##### Run number: {} ######".format(i))

    dnn.build_model()
    dnn.train_model()
    dnn.eval_model()

    dnn.plot_confusion_matrix(savePath = savepath + "/confusion_matrix_dnn_{}.pdf".format(i), norm_matrix = True)

    cm_dnn.append(dnn.confusion_matrix)
    auc_score_dnn.append(dnn.roc_auc_score)

    cnn_dnn.build_model()
    cnn_dnn.train_models()
    cnn_dnn.eval_model()

    cnn_dnn.plot_confusion_matrix(savePath = savepath + "/confusion_matrix_cnn_dnn_{}.pdf".format(i), norm_matrix = True)

    cm_cnn_dnn.append(cnn_dnn.confusion_matrix)
    auc_score_cnn_dnn.append(cnn_dnn.main_roc_score)


# Calculate the mean confusion_matrix and plot it

result_cm_dnn = cm_dnn[0]
result_cm_cnn_dnn = cm_cnn_dnn[0]
result_auc_dnn = auc_score_dnn[0]
result_auc_cnn_dnn = auc_score_cnn_dnn[0]


# Calculate the confusion matrices
for i in range(1, num_runs):
    result_cm_dnn += cm_dnn[i]
    result_cm_cnn_dnn += cm_cnn_dnn[i]
    result_auc_dnn += auc_score_dnn[i]
    result_auc_cnn_dnn += auc_score_cnn_dnn[i]


result_cm_dnn /= num_runs
result_cm_cnn_dnn /= num_runs
result_auc_dnn /= num_runs
result_auc_cnn_dnn /= num_runs

# Calculate the standard error of the mean
dnn_error = result_cm_dnn * 0.0
cnn_dnn_error = result_cm_cnn_dnn * 0.0
auc_err_dnn = 0.0
auc_err_cnn_dnn = 0.0

for i in range(0, num_runs):
    dnn_error += (cm_dnn[i] - result_cm_dnn)**2 / (num_runs * (num_runs - 1))
    cnn_dnn_error += (cm_cnn_dnn[i] - result_cm_cnn_dnn)**2 / (num_runs * (num_runs - 1))
    auc_err_dnn += (auc_score_dnn[i] - result_auc_dnn)**2 / (num_runs * (num_runs - 1))
    auc_err_cnn_dnn += (auc_score_cnn_dnn[i] - result_auc_cnn_dnn)**2 / (num_runs  * (num_runs - 1))

dnn_error = np.sqrt(dnn_error)
cnn_dnn_error = np.sqrt(cnn_dnn_error)
auc_err_dnn = np.sqrt(auc_err_dnn)
auc_err_cnn_dnn = np.sqrt(auc_err_cnn_dnn)


print("Result cm_dnn: {}".format(result_cm_dnn))
print("Result cm_cnn_dnn: {}".format(result_cm_cnn_dnn))


result_cm_dnn_norm, err_cm_dnn_nomr = plot_confusion_matrix(
                      confusion_matrix = result_cm_dnn,
                      error_confusion_matrix = dnn_error,
                      xticklabels = dnn.data.classes,
                      yticklabels = dnn.data.classes,
                      title = "Confusion Matrix DNN: Category: {}".format(str(key)),
                      roc = result_auc_dnn,
                      roc_err = auc_err_dnn,
                      save_path = savepath + "/confusion_matrix_dnn.pdf",
                      norm_matrix = True,
                      difference = False)
result_cm_cnn_dnn_norm, err_cm_cnn_dnn = plot_confusion_matrix(
                      confusion_matrix = result_cm_cnn_dnn,
                      error_confusion_matrix = cnn_dnn_error,
                      xticklabels = dnn.data.classes,
                      yticklabels = dnn.data.classes,
                      title = "Confusion Matrix Merged CNN And DNN: Category: {}".format(str(key)),
                      roc = result_auc_cnn_dnn,
                      roc_err = auc_err_cnn_dnn,
                      save_path = savepath + "/confusion_matrix_cnn_dnn.pdf",
                      norm_matrix = True,
                      difference = False)

# Calculate the confusion matrix of the differences of the two confusion matrices
result_cm_difference_nomr = result_cm_dnn_norm - result_cm_cnn_dnn_norm
roc_difference = result_auc_dnn - result_auc_cnn_dnn

# Calculate the errorporpagation
error_difference = np.sqrt(err_cm_dnn_nomr**2 + err_cm_cnn_dnn**2)
roc_error_difference = np.sqrt(auc_err_dnn**2 + auc_err_cnn_dnn**2)

plot_confusion_matrix(confusion_matrix = result_cm_difference_nomr,
                      error_confusion_matrix = error_difference,
                      xticklabels = dnn.data.classes,
                      yticklabels = dnn.data.classes,
                      title = "Confusion Matrix Difference: Category: {}".format(str(key)),
                      roc = roc_difference,
                      roc_err = roc_error_difference,
                      save_path = savepath + "/confusion_matrix_differences.pdf",
                      norm_matrix = False,
                      difference = True)



# Print the auc-scores

print("Mean auc score DNN: {}".format(result_auc_dnn))
print("Mean auc score DNN_Aachen: {}".format(result_auc_cnn_dnn))
print("Mean auc err DNN: {}".format(auc_err_dnn))
print("Mean auc err DNN_Aachen: {}".format(auc_err_cnn_dnn))
