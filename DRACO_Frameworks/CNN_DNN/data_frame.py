import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical

import matplotlib.pyplot as plt

class DataFrame(object):
	def __init__(self, path_to_input_files,
				classes, event_category,
				train_variables,
				test_percentage = 0.1,
				norm_variables = False,
				additional_cut = None,
				lumi = 41.5,
				phi_padding = 0):

		''' takes a path to a folder where one h5 per class is located
			the events are cut according to the event_category
			variables in train_variables are used as input variables
			variables in prenet_targets are used as classes for the pre net
			the dataset is shuffled and split into a test and train sample
				according to test_percentage
			for better training, the variables can be normed to std(1) and mu(0) '''

		# Specify the dimensions of the input image
		self.etabins = 25
		self.phibins = 31
		self.indices = ["eta{}phi{}".format(eta,phi) for eta in range(self.etabins) for phi in range(self.phibins)]

		# loop over all classes and extract data as well as event weights
		class_dataframes = list()

		for cls in classes:
			class_file = path_to_input_files + "/" + cls + "_cnn.h5"
			print("-"*50)
			print("loading class file "+str(class_file))

			with pd.HDFStore( class_file, mode = "r" ) as store:
				cls_df = store.select("data")
				print("number of events before selections: "+str(cls_df.shape[0]))


			# apply event category cut
			cls_df.query(event_category, inplace = True)
			self.event_category = event_category
			print("number of events after selections:  "+str(cls_df.shape[0]))



			# add event weight
			cls_df = cls_df.assign(total_weight = lambda x: x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom)

			weight_sum = sum(cls_df["total_weight"].values)
			class_weight_scale = 1.
			if "ttH" in cls: class_weight_scale *= 1.0
			cls_df = cls_df.assign(train_weight = lambda x: class_weight_scale*x.total_weight/weight_sum)
			print("weight sum of train_weight: "+str( sum(cls_df["train_weight"].values) ))

			# add lumi weight
			cls_df = cls_df.assign(lumi_weight = lambda x: x.Weight_XS * x.Weight_GEN_nom * lumi)

			# add data to list of dataframes
			class_dataframes.append( cls_df )
			print("-"*50)

		# concatenating all dataframes
		df = pd.concat( class_dataframes )
		del class_dataframes


		# add class_label translation
		index = 0
		self.class_translation = {}
		for cls in classes:
			self.class_translation[cls] = index
			index += 1
		self.classes = classes
		self.index_classes = [self.class_translation[c] for c in classes]

		df["index_label"] = pd.Series( [self.class_translation[c] for c in df["class_label"].values], index = df.index )
		df["is_ttH"] = pd.Series( [1 if c=="ttHbb" else 0 for c in df["class_label"].values], index = df.index )


		# norm weights to mean(1)
		df["train_weight"] = df["train_weight"]*df.shape[0]/len(classes)
		#df["train_weight"] = df["train_weight"]*df.shape[0]/len(classes)

		# save some meta data about net
		self.n_input_neurons = len(train_variables)
		self.n_output_neurons = len(classes)
		self.size_input_image = [self.etabins, self.phibins, 1]

		# shuffle dataframe
		df = shuffle(df, random_state = 333)
		# norm variables if wanted

		unnormed_df = df.copy()

		if norm_variables:
			norm_csv = pd.DataFrame(index=train_variables, columns=["mu", "std"])
			for v in train_variables:
				norm_csv["mu"][v] = unnormed_df[v].mean()
				norm_csv["std"][v] = unnormed_df[v].std()
			df[train_variables] = (df[train_variables] - df[train_variables].mean())/df[train_variables].std()
			self.norm_csv = norm_csv

		if additional_cut:
			df.query( additional_cut, inplace = True )

		self.unsplit_df = df.copy()
		# split test sample
		n_test_samples = int( df.shape[0]*test_percentage )
		self.df_test = df.head(n_test_samples)
		self.df_train = df.tail(df.shape[0] - n_test_samples )
		#self.df_test_unnormed = unnormed_df.head(n_test_samples)

		# print some counts
		print("total events after cuts:  "+str(df.shape[0]))
		print("events used for training: "+str(self.df_train.shape[0]))
		print("events used for testing:  "+str(self.df_test.shape[0]))
		del df

		# save variable lists
		self.train_variables = train_variables
		#self.prenet_targets = prenet_targets
		self.output_classes = classes

		## CNN data
		self.df_cnn_train = self.df_train[self.indices]
		self.df_cnn_train /= 255.0
		self.X_cnn = self.df_cnn_train.values.reshape(-1, *self.size_input_image)

		self.df_cnn_test = self.df_test[self.indices]
		self.df_cnn_test /= 255.0
		self.X_cnn_test = self.df_cnn_test.values.reshape(-1, *self.size_input_image)


		# Number of jets - labels

		train_number_jets = self.df_train["nJets_mAOD"].values
		test_number_jets =self.df_test["nJets_mAOD"].values

		self.min_jets = 3
		self.max_jets = 12
		self.train_number_jets = [self.max_jets if j>=self.max_jets \
				else self.min_jets if j<=self.min_jets \
				else j for j in train_number_jets]
		self.test_number_jets = [self.max_jets if j>=self.max_jets \
				else self.min_jets if j<=self.min_jets \
				else j for j in test_number_jets]

		self.train_number_jets = to_categorical(self.train_number_jets)
		self.test_number_jets = to_categorical(self.test_number_jets)
		self.number_jet_categories = self.train_number_jets.shape[1]



		if phi_padding != 0:
			 # padding in phi plane
			# add rows to top and bottom of image in the phi coordinate
			# representing the rotational dimension of phi
			self.X_cnn = np.concatenate(
				(self.X_cnn[:,:,-phi_padding:], self.X_cnn, self.X_cnn[:,:,:phi_padding]),
				axis = 2)
			print("data shape after padding: {}".format(self.X_cnn.shape))

			self.X_cnn_test = np.concatenate(
				(self.X_cnn_test[:,:,-phi_padding:], self.X_cnn_test, self.X_cnn_test[:,:,:phi_padding]),
				axis = 2)

			# edit input shape adjustment
			self.size_input_image[1] += 2*phi_padding
			print("input shape after padding: {}".format(self.size_input_image))
			self.image_size = self.size_input_image[0]*self.size_input_image[1]*self.size_input_image[2]
			print("image size after padding: {}".format(self.image_size))

	# train data -----------------------------------
	def get_train_data(self, as_matrix = True):
		if as_matrix: return self.df_train[ self.train_variables ].values
		else:         return self.df_train[ self.train_variables ]

	def get_train_data_cnn(self, as_matrix = True, normed = True):
		if as_matrix:
			return self.X_cnn

	def get_train_weights(self):
		return self.df_train["train_weight"].values

	def get_train_labels(self, as_categorical = True):
		if as_categorical: return to_categorical( self.df_train["index_label"].values )
		else:              return self.df_train["index_label"].values

	def get_train_number_jets(self):
		return self.train_number_jets

	#def get_prenet_train_labels(self):
	#    return self.df_train[ self.prenet_targets ].values

	# test data ------------------------------------
	def get_test_data(self, as_matrix = True, normed = True):
		if not normed: return self.df_test_unnormed[ self.train_variables ]
		if as_matrix:  return self.df_test[ self.train_variables ].values
		else:          return self.df_test[ self.train_variables ]

	def get_test_data_cnn(self, as_matrix = True, normed = True):
		#if not normed: return self.df_test_unnormed[ self.train_variables ]
		if as_matrix:
			return self.X_cnn_test

	def get_test_number_jets(self):
		return self.test_number_jets


	def get_test_weights(self):
		return self.df_test["total_weight"].values
	def get_lumi_weights(self):
		return self.df_test["lumi_weight"].values

	def get_test_labels(self, as_categorical = True):
		if as_categorical: return to_categorical( self.df_test["index_label"].values )
		else:              return self.df_test["index_label"].values

	#def get_prenet_test_labels(self, as_matrix = True):
	#    return self.df_test[ self.prenet_targets ].values

	def get_class_flag(self, class_label):
		return pd.Series( [1 if c==class_label else 0 for c in self.df_test["class_label"].values], index = self.df_test.index ).values

	def get_ttH_flag(self):
		return self.df_test["is_ttH"].values

	# full sample ----------------------------------
	def get_full_df(self):
		return self.unsplit_df[self.train_variables]
