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
savepath = workpath + "/hist2D_1_"+str(key)+"/"

# Define the models
cnn_dnn = CNN_DNN.CNN_DNN(
    in_path             = inPath,
    save_path           = savepath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    batch_size          = 5000,
    train_epochs        = 2,
    early_stopping      = 5,
    optimizer           = "adam",
    test_percentage     = 0.2,
    eval_metrics        = ["acc"],
    phi_padding         = 0
    )


dnn = DNN.DNN(
    in_path         = inPath,
    save_path       = savepath,
    event_classes   = event_classes,
    event_category  = categories[key],
    train_variables = category_vars[key],
    train_epochs    = 2,
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


dnn.build_model()
dnn.train_model()
dnn.eval_model()

predict_vector_dnn = dnn.model_prediction_vector
predict_classes_dnn = dnn.predicted_classes

cnn_dnn.build_model()
cnn_dnn.train_models()
cnn_dnn.eval_model()

predict_vector_cnn_dnn = cnn_dnn.mainnet_predicted_vector
predict_classes_cnn_dnn = cnn_dnn.predicted_classes



# Plot 2d hist of dnn and dnn_aachen predicted classes

plt.clf()
plt.figure( figsize = [10,10])
plt.title("2D-Hist of predictions", fontsize=15)

plt.hist2d(predict_classes_cnn_dnn, predict_classes_dnn, bins=6, range=[[0, 6], [0, 6]], cmap="Greens")

plt.xticks(np.arange(0.5, 6.5, 1), event_classes)
plt.yticks(np.arange(0.5, 6.5, 1), event_classes)

plt.xlabel("Prediction Merged CNN and DNN", fontsize=14)
plt.ylabel("Prediction DNN", fontsize=14)

# correlation
correlation_matrix = np.corrcoef(predict_classes_cnn_dnn, predict_classes_dnn)
plt.annotate("Correlation: {:.3f}".format(correlation_matrix[0][1]), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)
plt.colorbar()
plt.savefig(savepath + "hist2d.pdf")

# Plot the 2d hists of the one hot output for every class

for i in range(len(dnn.event_classes)):

    dnn_data = predict_vector_dnn[:, i]
    cnn_dnn_data = predict_vector_cnn_dnn[:, i]

    plt.clf()
    plt.figure( figsize = [10,10])
    plt.title("Predictions of calss {}".format(dnn.event_classes[i]), fontsize=15)

    plt.hist2d(cnn_dnn_data, dnn_data, bins=50, cmap="Greens")
    plt.xlabel("Prediction Merged CNN and DNN", fontsize=14)
    plt.ylabel("Prediction DNN", fontsize=14)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    correlation_matrix = np.corrcoef(cnn_dnn_data, dnn_data)
    plt.annotate("Correlation: {:.3f}".format(correlation_matrix[0][1]), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)
    plt.colorbar()

    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='grey')
    plt.savefig(savepath + "hist2d_prediction_class_{}.pdf".format(i+1))
