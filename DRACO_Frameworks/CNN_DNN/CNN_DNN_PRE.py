import os
import sys
import numpy as np
import json

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir  = os.path.dirname(DRACOdir)
sys.path.append(basedir)

# import with ROOT
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts

# imports with keras
import utils.generateJTcut as JTcut
#import architecture as arch
import data_frame

import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

import matplotlib.pyplot as plt
import pandas as pd

# Limit gpu usage
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score



class EarlyStoppingByLossDiff(keras.callbacks.Callback):
	def __init__(self, monitor = "loss", value = 0.01, min_epochs = 20, patience = 10, verbose = 0):
		super(keras.callbacks.Callback, self).__init__()
		self.val_monitor = "val_"+monitor
		self.train_monitor = monitor
		self.patience = patience
		self.n_failed = 0

		self.min_epochs = min_epochs
		self.value = value
		self.verbose = verbose

	def on_epoch_end(self, epoch, logs = {}):
		current_val = logs.get(self.val_monitor)
		current_train = logs.get(self.train_monitor)

		if current_val is None or current_train is None:
			warnings.warn("Early stopping requires {} and {} available".format(
				self.val_monitor, self.train_monitor), RuntimeWarning)

		if abs(current_val-current_train)/(current_train) > self.value and epoch > self.min_epochs:
			if self.verbose > 0:
				print("Epoch {}: early stopping threshold reached".format(epoch))
			self.n_failed += 1
			if self.n_failed > self.patience:
				self.model.stop_training = True



class CNN_DNN_PRE():
	def __init__(self, in_path, save_path,
				event_classes,
				event_category,
				train_variables,
				batch_size = 5000,
				train_epochs = 500,
				early_stopping = 10,
				optimizer = None,
				loss_function = "categorical_crossentropy",
				test_percentage = 0.2,
				eval_metrics = None,
				additional_cut = None,
				phi_padding = 0):

		# save some information

		# path to input files
		self.in_path = in_path
		# output directory for results
		self.save_path = save_path
		if not os.path.exists(self.save_path):
			os.makedirs( self.save_path )
		# list of classes
		self.event_classes = event_classes
		# name of event category (usually nJet/nTag category)
		self.event_category = event_category
		
		self.JTstring       = event_category
		self.event_category = JTcut.getJTstring(event_category)
		self.categoryLabel  = JTcut.getJTlabel(event_category)

		# list of input features
		self.train_variables = train_variables

		# batch size for training
		self.batch_size = batch_size
		# number of maximum training epochs
		self.train_epochs = train_epochs
		# number of early stopping epochs
		self.early_stopping = early_stopping
		# percentage of events saved for testing
		self.test_percentage = test_percentage

		# loss function for training
		self.loss_function = loss_function
		# additional metrics for evaluation of training process
		self.eval_metrics = eval_metrics

		# additional cut to be applied after variable norm
		self.additional_cut = additional_cut

		self.optimizer = optimizer
		self.phi_padding = phi_padding

		# load dataset
		self.data = self._load_datasets()
		self.data.get_train_data_cnn
		#print(self.data.get_train_data_cnn.values)
		out_path = self.save_path+"/checkpoints/"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		out_file = out_path+"/variable_norm.csv"
		#self.data.norm_csv.to_csv(out_file)
		print("saved variable norms at "+str(out_file))
		
		# make plotdir
		self.plot_path = self.save_path+"/plots/"
		if not os.path.exists(self.plot_path):
			os.makedirs(self.plot_path)

		# dict with architectures for analysis
		#self.architecture = arch.getArchitecture(self.JTstring)
		self.inputName = "inputLayer"
		self.outputName = "outputLayer"

		# optimizer for training
		if not(optimizer):
			self.optimizer = "adam"
		else:
			self.optimizer = optimizer


	def _load_datasets(self):
		''' load dataset '''
		return data_frame.DataFrame(
			path_to_input_files = self.in_path,
			classes             = self.event_classes,
			event_category      = self.event_category,
			train_variables     = self.train_variables,
			test_percentage     = self.test_percentage,
			norm_variables      = True,
			additional_cut      = self.additional_cut,
			phi_padding         = self.phi_padding)

	def load_trained_model(self):
		''' load an already trained model '''
		checkpoint_path = self.save_path + "/checkpoints/trained_main_net.h5py"

		self.main_net = keras.models.load_model(checkpoint_path)

		self.mainnet_eval = self.main_net.evaluate(
			self.data.get_test_data(as_matrix = True),
			self.data.get_test_labels())

		self.mainnet_predicted_vector = self.main_net.predict(
			self.data.get_test_data(as_matrix = True))

		self.predicted_classes = np.argmax( self.mainnet_predicted_vector, axis = 1)

		# save confusion matrix
		self.confusion_matrix = confusion_matrix(
			self.data.get_test_labels(as_categorical = False), self.predicted_classes)

		# print evaluations
		self.main_roc_score = roc_auc_score(self.data.get_test_labels(), self.mainnet_predicted_vector)
		print("mainnet test roc: {}".format(self.main_roc_score))
		if self.eval_metrics:
			print("mainnet test loss: {}".format(self.mainnet_eval[0]))
			for im, metric in enumerate(self.eval_metrics):
				print("mainnet test {}: {}".format(metric, self.mainnet_eval[im+1]))

	def predict_event_query(self, query ):
		events = self.data.get_full_df().query( query )
		print(str(events.shape[0]) + " events matched the query '"+str(query)+"'.")

		for index, row in events.iterrows():
			print("========== DNN output ==========")
			print("Event: "+str(index))
			for var in row.values:
				print(var)
			print("-------------------->")
			output = self.main_net.predict( np.array([list(row.values)]) )[0]
			for i, node in enumerate(self.event_classes):
				print(str(node)+" node: "+str(output[i]))
			print("-------------------->")



	def build_default_model(self):

		K.set_learning_phase(True)

		inputs = keras.layers.Input(
			shape = self.data.size_input_image,
			name = "input"
		)

		x = layer.Conv2D(32, kernel_size = (6,6),
				activation = "relu",
				padding = "same")(inputs)
		x = layer.AveragePooling2D( pool_size = (2,2), padding = "same" )(x)

		x = layer.Conv2D(64, kernel_size = (6,6),
				activation = "relu",
				padding = "same")(x)
		x = layer.AveragePooling2D( pool_size = (2,2), padding = "same" )(x)

		x = layer.Conv2D(128, kernel_size = (6,6),
				activation = "relu",
				padding = "same")(x)
		x = layer.AveragePooling2D( pool_size = (2,2), padding = "same" )(x)

		x = layer.Conv2D(256, kernel_size = (6,6),
				activation = "relu",
				padding = "same")(x)
		x = layer.AveragePooling2D( pool_size = (2,2), padding = "same" )(x)

		x = layer.Flatten()(x)

		x = layer.Dense(128, activation = "relu")(x)
		x = layer.Dropout(0.5)(x)
		x = layer.Dense(128, activation = "relu")(x)
		x = layer.Dropout(0.5)(x)

		outputs = layer.Dense(self.data.number_jet_categories, activation = "softmax")(x)

		pre_net = models.Model(input = [inputs], outputs = [outputs])
		pre_net.summary()

		# main_net

		main_input = keras.layers.Input(
				shape = (self.data.n_input_neurons, ),
				name = "main_input")

		y = layer.Concatenate()([main_input, pre_net.output])
		y = layer.Dense(100, activation="relu")(y)
		y = layer.Dropout(0.5)(y)
		y = layer.Dense(100, activation = "relu")(y)
		y = layer.Dropout(0.5)(y)
		y = layer.Dense(self.data.n_output_neurons, activation = "softmax")(y)

		main_net = models.Model(input = [inputs, main_input], outputs = [y])
		main_net.summary()

		return pre_net, main_net


	def build_model(self, pre_net = None, main_net = None):
		''' build a DNN model
			if none is specified use default model '''


		pre_net, main_net = self.build_default_model()

		for layer in pre_net.layers:
			layer.trainable = True

		pre_net.compile(
			loss = self.loss_function,
			optimizer = self.optimizer,
			metrics = self.eval_metrics)

		# ste pre net untrainable
		for layer in pre_net.layers:
			layer.trainable = False

		# compile main net
		main_net.compile(
			loss = self.loss_function,
			optimizer = self.optimizer,
			metrics = self.eval_metrics)

		# save compiled nets
		self.pre_net = pre_net
		self.main_net = main_net


		# save net information
		# save net information
		out_file = self.save_path+"/pre_net_summmary.yml"
		yml_pre_net = self.pre_net.to_yaml()
		with open(out_file, "w") as f:
			f.write(yml_pre_net)

		out_file = self.save_path+"/main_net_summmary.yml"
		yml_main_net = self.main_net.to_yaml()
		with open(out_file, "w") as f:
			f.write(yml_main_net)


	def train_models(self):
		''' train prenet first then the main net '''

		# checkpoint files
		cp_path = self.save_path + "/checkpoints"
		if not os.path.exists(cp_path):
			os.makedirs(cp_path)


		# add early stopping if activated
		callbacks = None
		if self.early_stopping:
			callbacks = [EarlyStoppingByLossDiff(
				monitor = "loss",
				value = 0.01,
				min_epochs = 100,
				patience = 20,
				verbose = 1)]

		for layer in self.pre_net.layers:
			layer.trainable = True

		# train pre net
		print("Training pre net ........")
		self.trained_pre_net = self.pre_net.fit(
			x = self.data.get_train_data_cnn(as_matrix = True),
			y = self.data.get_train_number_jets(),
			batch_size = self.batch_size,
			epochs = self.train_epochs,
			shuffle = True,
			validation_split = 0.25,
			sample_weight = self.data.get_train_weights()
			)

		# set pre net as untrainable
		for layer in self.pre_net.layers:
			layer.trainable = False

		out_file = cp_path + "trained_pre_net.h5py"
		self.pre_net.save(out_file)
		print("saved trained prenet model at "+str(out_file))

		prenet_config = self.pre_net.get_config()
		out_file = cp_path +"trained_pre_net_config"
		with open(out_file, "w") as f:
			f.write( str(prenet_config))
		print("saved prenet model config at "+str(out_file))

		out_file = cp_path +"trained_pre_net_weights.h5"
		self.pre_net.save_weights(out_file)
		print("wrote trained prenet weights to "+str(out_file))


		# add early stopping if activated
		callbacks = None
		if self.early_stopping:
			callbacks = [EarlyStoppingByLossDiff(
				monitor = "loss",
				value = 0.01,
				min_epochs = 100,
				patience = 20,
				verbose = 1)]

		# train main net
		print("Training main net .....")
		self.trained_main_net = self.main_net.fit(
			x = [self.data.get_train_data_cnn(), self.data.get_train_data(as_matrix = True)],
			y = self.data.get_train_labels(),
			batch_size = self.batch_size,
			epochs = self.train_epochs + 0,
			shuffle = True,
			validation_split = 0.25,
			sample_weight = self.data.get_train_weights())


		# save trained model
		out_file = cp_path + "trained_main_net.h5py"
		self.main_net.save(out_file)
		print("saved trained model at "+str(out_file))

		mainnet_config = self.main_net.get_config()
		out_file = cp_path + "trained_main_net_config"
		with open(out_file, "w") as f:
			f.write( str(mainnet_config))
		print("saved model config at "+str(out_file))

		out_file = cp_path +"trained_main_net_weights.h5"
		self.main_net.save_weights(out_file)
		print("wrote trained weights to "+str(out_file))



	def eval_model(self):
		''' evaluate trained model '''

		# prenet evaluation
		self.prenet_eval = self.pre_net.evaluate(
			self.data.get_test_data_cnn(as_matrix = True),
			self.data.get_test_number_jets())

		# save history of eval metrics
		self.prenet_history = self.trained_pre_net.history

		# save predicitons
		self.prenet_predicted_vector = self.pre_net.predict(
			self.data.get_test_data_cnn(as_matrix = True) )

		# print evaluations
		if self.eval_metrics:
			print("prenet test loss: {}".format(self.prenet_eval[0]))
			for im, metric in enumerate(self.eval_metrics):
				print("prenet test {}: {}".format(metric, self.prenet_eval[im+1]))


		# main net evaluation
		self.mainnet_eval = self.main_net.evaluate(
			[self.data.get_test_data_cnn(), self.data.get_test_data(as_matrix = True)],
			self.data.get_test_labels())

		# save history of eval metrics
		self.mainnet_history = self.trained_main_net.history

		# save predictions
		self.mainnet_predicted_vector = self.main_net.predict(
			[self.data.get_test_data_cnn(), self.data.get_test_data(as_matrix = True)] )

		# save predicted classes with argmax
		self.predicted_classes = np.argmax( self.mainnet_predicted_vector, axis = 1)

		# save confusion matrix
		self.confusion_matrix = confusion_matrix(
			self.data.get_test_labels(as_categorical = False), self.predicted_classes)

		# print evaluations
		self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.mainnet_predicted_vector)
		print("mainnet test roc: {}".format(self.roc_auc_score))
		if self.eval_metrics:
			print("mainnet test loss: {}".format(self.mainnet_eval[0]))
			for im, metric in enumerate(self.eval_metrics):
				print("mainnet test {}: {}".format(metric, self.mainnet_eval[im+1]))


	def save_confusionMatrix(self, location, save_roc):
		''' save confusion matrix as a line in output file '''
		flattened_matrix = self.confusion_matrix.flatten()
		labels = ["{}_in_{}_node".format(pred, true) for true in self.event_classes for pred in self.event_classes]
		data = {label: [float(flattened_matrix[i])] for i, label in enumerate(labels)}
		data["ROC"] = [float(self.roc_auc_score)]
		df = pd.DataFrame.from_dict(data)
		with pd.HDFStore(location, "a") as store:
			store.append("data", df, index = False)
		print("saved confusion matrix at "+str(location))


	# --------------------------------------------------------------------
	# result plotting functions
	# --------------------------------------------------------------------

	def plot_metrics(self):
		''' plot history of loss function and evaluation metrics '''

		metrics = ["loss"]
		if self.eval_metrics: metrics += self.eval_metrics

		for metric in metrics:
			# main net
			plt.clf()
			train_history = self.mainnet_history[metric]
			val_history = self.mainnet_history["val_"+metric]

			n_epochs = len(train_history)
			epochs = np.arange(1,n_epochs+1,1)

			plt.plot(epochs, train_history, "b-", label = "train", lw = 2)
			plt.plot(epochs, val_history, "r-", label = "validation", lw = 2)
			plt.title("train and validation "+str(metric)+" of mainnet")

			plt.grid()
			plt.xlabel("epoch")
			plt.ylabel(metric)

			plt.legend()

			out_path = self.save_path + "/mainnet_history_"+str(metric)+".pdf"
			plt.savefig(out_path)
			print("saved plot of "+str(metric)+" at "+str(out_path))

	def plot_outputNodes(self, log = False, cut_on_variable = None):
		''' plot distribution in outputNodes '''
		nbins = 21
		bin_range = [0., 0.7]

		plotNodes = plottingScripts.plotOutputNodes(
			data                = self.data,
			prediction_vector   = self.mainnet_predicted_vector,
			event_classes       = self.event_classes,
			nbins               = nbins,
			bin_range           = bin_range,
			signal_class        = "ttHbb",
			event_category      = self.categoryLabel,
			plotdir             = self.plot_path,
			logscale            = log)

		if cut_on_variable:
			plotNodes.set_cutVariable(
				cutClass = cut_on_variable["class"],
				cutValue = cut_on_variable["value"])

		plotNodes.set_printROCScore(True)
		plotNodes.plot(ratio = False)
		
		
	def plot_discriminators(self, log = False):
		''' plot all events classified as one category '''
		nbins = 15
		bin_range = [0.2, 0.7]

		plotDiscrs = plottingScripts.plotDiscriminators(
			data                = self.data,
			prediction_vector   = self.mainnet_predicted_vector,
			event_classes       = self.event_classes,
			nbins               = nbins,
			bin_range           = bin_range,
			signal_class        = "ttHbb",
			event_category      = self.categoryLabel,
			plotdir             = self.plot_path,
			logscale            = log)

		plotDiscrs.set_printROCScore(True)
		plotDiscrs.plot(ratio = False)


	def plot_confusionMatrix(self, norm_matrix = True):
		''' plot confusion matrix '''
		plotCM = plottingScripts.plotConfusionMatrix(
			data                = self.data,
			prediction_vector   = self.mainnet_predicted_vector,
			event_classes       = self.event_classes,
			event_category      = self.categoryLabel,
			plotdir             = self.save_path)

		plotCM.set_printROCScore(True)

		plotCM.plot(norm_matrix = norm_matrix)


	'''
	def plot_confusion_matrix(self, norm_matrix = True):
		#generate confusion matrix
		n_classes = self.confusion_matrix.shape[0]

		# norm confusion matrix if wanted
		if norm_matrix:
			cm = np.empty( (n_classes, n_classes), dtype = np.float64 )
			for yit in range(n_classes):
				evt_sum = float(sum(self.confusion_matrix[yit,:]))
				for xit in range(n_classes):
					cm[yit,xit] = self.confusion_matrix[yit,xit]/evt_sum

			self.confusion_matrix = cm

		plt.clf()

		plt.figure( figsize = [10,10])

		minimum = np.min( self.confusion_matrix )/(np.pi**2.0 * np.exp(1.0)**2.0)
		maximum = np.max( self.confusion_matrix )*(np.pi**2.0 * np.exp(1.0)**2.0)

		x = np.arange(0, n_classes+1, 1)
		y = np.arange(0, n_classes+1, 1)

		xn, yn = np.meshgrid(x,y)

		plt.pcolormesh(xn, yn, self.confusion_matrix,
			norm = LogNorm( vmin = max(minimum, 1e-6), vmax = min(maximum,1.)), cmap="jet")
		plt.colorbar()

		plt.xlim(0, n_classes)
		plt.ylim(0, n_classes)

		plt.xlabel("Predicted")
		plt.ylabel("True")
		plt.title("ROC-AUC value: {:.4f}".format(self.main_roc_score), loc = "left")

		# add textlabel
		for yit in range(n_classes):
			for xit in range(n_classes):
				plt.text(
					xit+0.5, yit+0.5,
					"{:.3f}".format(self.confusion_matrix[yit, xit]),
					horizontalalignment = "center",
					verticalalignment = "center")

		plt_axis = plt.gca()
		plt_axis.set_xticks(np.arange( (x.shape[0] -1)) + 0.5, minor = False )
		plt_axis.set_yticks(np.arange( (y.shape[0] -1)) + 0.5, minor = False )

		plt_axis.set_xticklabels(self.data.classes)
		plt_axis.set_yticklabels(self.data.classes)

		plt_axis.set_aspect("equal")

		out_path = self.save_path + "/confusion_matrix.pdf"
		plt.savefig(out_path)
		print("saved confusion matrix at "+str(out_path))
		plt.clf()
	'''
