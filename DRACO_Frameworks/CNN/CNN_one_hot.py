# global imports
import keras
import keras.models as models
import keras.layers as layer
from keras.callbacks import EarlyStopping
import keras.backend as K

import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
# local imports
import data_frame




class CNN():
    def __init__(self, in_path, save_path,      
                class_label = "class_label", 
                phi_padding = None,
                batch_size = 128, 
                train_epochs = 20,
                optimizer = "adam", 
                loss_function = "mean_squared_error", 
                eval_metrics = None):

        # saving some information

        # path to input files
        self.in_path = in_path
        # output directory for result files
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )
        # name of classification variable
        self.class_label = class_label
        # phi padding for rotational symmetries
        self.phi_padding = phi_padding

        # batch size for training
        self.batch_size  = batch_size
        # number of training epochs
        self.train_epochs = train_epochs
        # optimizer
        self.optimizer = optimizer
        # loss_function
        self.loss_function = loss_function
        # eval_metrics
        self.eval_metrics = eval_metrics
        
        

    def load_datasets(self):
        ''' load train and validation dataset '''
        self.train_data = data_frame.DataFrame( 
            self.in_path+"_train.h5", output_label = self.class_label,
            phi_padding = self.phi_padding )
        self.val_data = data_frame.DataFrame( 
            self.in_path+"_val.h5", output_label = self.class_label,
            phi_padding = self.phi_padding )

        self.num_classes = self.train_data.num_classes

    def build_default_model(self):
        ''' default CNN model for testing purposes
            has three conv layers with max pooling and one densly connected layer '''
        model = models.Sequential()
        
        # input layer
        model.add(
            layer.Conv2D( 32, kernel_size = (4,4), activation = "linear", padding = "same", 
            input_shape = self.train_data.input_shape ))
        model.add(
            layer.AveragePooling2D( pool_size = (4,4), padding = "same" ))
        model.add(
            layer.Dropout(0.5))

        # second layer
        model.add(
            layer.Conv2D( 64, kernel_size = (4,4), activation = "linear", padding = "same"))
        model.add(
            layer.AveragePooling2D( pool_size = (4,4), padding = "same" ))
        model.add(
            layer.Dropout(0.5))
    
        # third layer
        model.add(
            layer.Conv2D( 256, kernel_size = (3,3), activation = "linear", padding = "same"))
        model.add(
            layer.MaxPooling2D( pool_size = (2,2), padding = "same" ))
        model.add(
            layer.Dropout(0.5))

        # dense layer
        model.add(
            layer.Flatten())
        model.add(
            layer.Dense( 128, activation = "sigmoid" ))
        model.add(
            layer.Dropout(0.5))

        # output
        model.add( 
            layer.Dense( self.num_classes, activation = "softmax" ))
        return model


    def build_model(self, model = None):
        ''' build a CNN model
            if none is specified use default constructor '''
        if model == None:
            print("loading default model")
            model = self.build_default_model()
        
        # compile the model
        model.compile(
            loss        = self.loss_function,
            optimizer   = self.optimizer,
            metrics     = self.eval_metrics)
        
        self.model = model

        # model summary
        self.model.summary()
        out_file = self.save_path+"/model_summary.yml"

        yaml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yaml_model)
        print("saved model summary at "+str(out_file))

    def train_model(self, earlyStopping = False):
        
        if earlyStopping:
            
            early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            
            self.trained_model = self.model.fit(
                x = self.train_data.X,
                y = self.train_data.one_hot,
                batch_size = self.batch_size,
                epochs = self.train_epochs,
                shuffle = True,
                validation_data = (self.val_data.X, self.val_data.one_hot),
                callbacks=[early_stopping_monitor])
            
        else:    
            self.trained_model = self.model.fit(
                x = self.train_data.X,
                y = self.train_data.one_hot,
                batch_size = self.batch_size,
                epochs = self.train_epochs,
                shuffle = True,
                validation_data = (self.val_data.X, self.val_data.one_hot ))
        
        # get the number of epochs trained (in case of early stopping)
        self.early_stopping_epochs = len(self.trained_model.history['loss'])
        
        
        # save trained model
        out_file = self.save_path +"/trained_model.h5py"
        self.model.save(out_file)
        print("saved trained model at "+str(out_file))

        model_config = self.model.get_config()
        out_file = self.save_path +"/trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config) )
        print("saved model config at "+str(out_file))

        out_file = self.save_path +"/trained_model_weights.h5"
        self.model.save_weights(out_file)
        print("wrote trained net weights to "+str(out_file))

    def eval_model(self):
        # loading test examples
        self.test_data = data_frame.DataFrame( 
            self.in_path+"_test.h5", output_label = self.class_label,
            phi_padding = self.phi_padding )

        self.target_names = [self.test_data.inverted_label_dict[i] for i in range(
            self.test_data.min_jets, self.test_data.max_jets+1)]

        self.test_eval = self.model.evaluate(
            self.test_data.X, self.test_data.one_hot)
        print("test loss:     {}".format( self.test_eval[0] ))
        for im, metric in enumerate(self.eval_metrics):
            print("test {}: {}".format( metric, self.test_eval[im+1] ))

        self.history = self.trained_model.history
        
        self.predicted_vector = self.model.predict( self.test_data.X )
        self.predicted_classes = np.argmax( self.predicted_vector, axis = 1)
        
        self.predicted_classes = np.array([self.test_data.max_jets if j >= self.test_data.max_jets \
                             else self.test_data.min_jets if j <= self.test_data.min_jets \
                             else j for j in self.predicted_classes])

        self.confusion_matrix = confusion_matrix(
            self.test_data.Y, self.predicted_classes )

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def print_classification_examples(self):
        ''' print some examples of classifications '''

        correct = np.where( self.predicted_classes == self.test_data.Y )[0]
        print("found {} correct classifications".format(len(correct)))
        incorrect = np.where( self.predicted_classes != self.test_data.Y)[0]
        print("found {} incorrect classifications".format(len(incorrect)))

        # plot correct examples
        plt.clf()
        plt.figure(figsize = [5,5])
        for i, sample in enumerate(correct[:4]):
            plt.subplot(2,2,i+1)
            plt.imshow( 
                self.test_data.X[sample].reshape(*self.test_data.input_shape[:2]).T,
                cmap = "Greens")
        
            plt.title( "Predicted {}\nTrue {}".format(
                self.test_data.inverted_label_dict[self.predicted_classes[sample]],
                self.test_data.inverted_label_dict[self.test_data.Y[sample]] ))
            plt_axis = plt.gca()
            plt_axis.get_xaxis().set_visible(False)
            plt_axis.get_yaxis().set_visible(False)

        plt.tight_layout()
        out_path = self.save_path + "/correct_prediction_examples.pdf"
        plt.savefig( out_path )
        print("saved examples of correct predictions to "+str(out_path))
        plt.clf()
        
        # plot incorrect examples
        plt.figure(figsize = [5,5])
        for i, sample in enumerate(incorrect[:4]):
            plt.subplot(2,2,i+1)
            plt.imshow( 
                self.test_data.X[sample].reshape(*self.test_data.input_shape[:2]).T,
                cmap = "Greens")

            plt.title( "Predicted {}\nClass {}".format(
                self.test_data.inverted_label_dict[self.predicted_classes[sample]],
                self.test_data.inverted_label_dict[self.test_data.Y[sample]] ))
            plt_axis = plt.gca()
            plt_axis.get_xaxis().set_visible(False)
            plt_axis.get_yaxis().set_visible(False)

        plt.tight_layout()
        out_path = self.save_path + "/incorrect_prediction_examples.pdf"
        plt.savefig( out_path )
        print("saved examples of incorrect predictions to "+str(out_path))
        plt.clf()


    def print_classification_report(self):
        ''' print a classification report '''

        report = classification_report( 
            self.test_data.Y, self.predicted_classes,
            target_names = self.target_names )    

        print("classification report:")
        print(report)
        out_path = self.save_path + "/classification_report"
        with open(out_path, "w") as f:
            f.write(report)
        print("saved classification report to "+str(out_path))

    
    def plot_metrics(self):
        ''' plot history of loss function and metrics '''

        epochs = range(self.early_stopping_epochs)
        metrics = ["loss"] + self.eval_metrics

        for metric in metrics:
            plt.clf()
            train_history = self.history[metric]
            val_history = self.history["val_"+metric]
            
            plt.plot(epochs, train_history, "b-", label = "train", lw = 2.5)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2.5)
            plt.title("train and validation "+str(metric))

            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(metric)

            plt.legend()
            
            out_path = self.save_path + "/history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))

    def plot_discriminators(self, log = False):
        ''' plot discriminator for output classes '''

        for i in range(self.num_classes):
            values = self.predicted_vector[:,i]

            bkg_values = []
            bkg_labels = []
            n_bkg_evts = 0
            n_sig_evts = 0

            for j in range(self.num_classes):
                filtered_values = [values[k] for k in range(len(values)) if self.test_data.Y[k] == j]
                if i == j:
                    sig_values = filtered_values
                    sig_label = self.test_data.inverted_label_dict[j]
                    n_sig_evts += len(filtered_values)
                else:
                    bkg_values.append(filtered_values)
                    bkg_labels.append(self.test_data.inverted_label_dict[j])
                    n_bkg_evts += len(filtered_values)
        
            if n_sig_evts == 0: continue
            # plot the discriminator output
            plt.clf()
            plt.figure( figsize = [5,5] )
            # stack backgrounds
            plt.hist( bkg_values, stacked = True, histtype = "stepfilled", 
                        bins = 20, range = [0,1], label = bkg_labels, log = log)

            # get signal weights
            bkg_sig_ratio = 1.* n_bkg_evts / n_sig_evts     
            sig_weights = [bkg_sig_ratio]*len(sig_values)
            sig_label += "*{:.3f}".format(bkg_sig_ratio)

            # plot signal shape
            plt.hist( sig_values, histtype = "step", weights = sig_weights,
                        bins = 20, range = [0,1], label = sig_label, log = log)

            plt.legend(loc = "upper center")
            plt.xlabel("discriminator output")
            plt.title("discriminator for {}".format(
                self.test_data.inverted_label_dict[i]))
            
            out_path = self.save_path +"/discriminator_{}.pdf".format(
                self.test_data.inverted_label_dict[i].replace(" ","_"))
        
            plt.savefig(out_path)
            print("plot for discriminator of {} saved at {}".format(
                self.test_data.inverted_label_dict[i], out_path))

            plt.clf()



    def plot_confusion_matrix(self):
        ''' generate confusion matrix for classification '''
  
        plt.clf()
        plt.figure( figsize = (1.5*self.num_classes, 1.5*self.num_classes) )
        
        matplotlib.rcParams.update({'font.size': 22}) 
        
        minimum = np.min( self.confusion_matrix ) /(np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max( self.confusion_matrix ) *(np.pi**2.0 * np.exp(1.0)**2.0)

        n_classes = self.confusion_matrix.shape[0]       
        #x = np.arange(self.train_data.min_jets, self.train_data.max_jets+1, 1)
        #y = np.arange(self.train_data.min_jets, self.train_data.max_jets+1, 1)
        x = np.arange(0,n_classes+1,1)
        y = np.arange(0,n_classes+1,1)

        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, self.confusion_matrix, 
            norm = LogNorm( vmin = max(minimum, 1e-6), vmax = maximum ))
        plt.colorbar()

        plt.xlim(0, n_classes)
        plt.ylim(0, n_classes)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for yit in range(n_classes):
            for xit in range(n_classes):
                plt.text( xit+0.5, yit+0.5,
                    "{:.1f}".format(self.confusion_matrix[yit, xit]),
                    horizontalalignment = "center",
                    verticalalignment = "center")

        plt_axis = plt.gca()
        plt_axis.set_xticks(np.arange( (x.shape[0] -1)) + 0.5, minor = False )
        plt_axis.set_yticks(np.arange( (y.shape[0] -1)) + 0.5, minor = False )
        plt_axis.set_xticklabels(self.target_names)
        plt_axis.set_yticklabels(self.target_names)

        plt_axis.set_aspect("equal")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        out_path = self.save_path+"/confusion_matrix.pdf"
        plt.savefig(out_path)
        print("saved confusion matrix at "+str(out_path))
        plt.clf()



    def plot_filters(self, layer_number, x, y):
        ''' Plot the filters of the layer number layer_number in a x by y figure '''
        ''' layer_number has to ba a conv2d layer                                '''
        
        filters = self.model.layers[layer_number].get_weights()
        filters = filters[0]
        #num_filters = self.model.layers[layer_number].output_shape[3]
        
        if filters.size == 0:
            print("No filters to visualize! Can only take Conv2D layers as input.")
            return
        
        plt.clf()
        fig = plt.figure()
        fig.suptitle('Visualization of the filters from layer {}'.format(layer_number), fontsize=16)
        
        for j in range(x * y):
            ax = fig.add_subplot(y, x, j+1)
            ax.matshow(filters[:,:,0,j], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            
        plt.tight_layout()
        
        out_path = self.save_path + "/filters_layer" + str(layer_number) + ".pdf"
        plt.savefig(out_path)
        print("saved filter visualization at " + str(out_path))
        plt.clf()
        
        
        
        
    def plot_layer_output(self, num_layers):
        ''' Plot the outputs of the intermediate layers. num_layers discribes the number of     '''
        ''' layers that should be visualized. This has to be less or equal to the number of     '''
        ''' layers which have an image like shape as output                                      '''
        
        ## Plot the original image
        img_org = self.test_data.X[0:1, :, :, :]
        label = self.test_data.inverted_label_dict[self.test_data.Y[0]]
        
        
        plt.clf()
        plt.figure()
        plt.title('Input image')
        plt.xlabel('eta')
        plt.ylabel('phi')
        plt.imshow(np.rollaxis(img_org.reshape(*self.test_data.input_shape[:2].T),1, 0), cmap="Greens")
        out_path = self.save_path + "/input_image.pdf"
        plt.savefig(out_path)
        plt.clf()
    
        ## Visualize 12 images for each layer
    
        for i in range(num_layers):
            
            plt.clf()
            plt.figure(figsize = [5,5])      
            plt.suptitle("Output of layer {} ({})".format(i, label))
            output_fn = K.function([self.model.layers[0].input], [self.model.layers[i].output])
            img = output_fn([img_org])
            img = np.array(img)
            img = img[0,0,:,:,:]
        
            
            for j in range(12):
                plt.subplot(3, 4, j+1)
                plt.xlabel('eta')
                plt.ylabel('phi')
                plt.imshow(np.rollaxis(img[:,:,j], 1, 0), cmap="Greens")
                plt.tight_layout()
            out_path = self.save_path + "/visualize_layer" + str(i) + ".pdf"
            print("saved layer visualization at " + str(out_path))
            plt.savefig(out_path)
            plt.clf()
            plt.close("all")
                
                
            
        
        

        
        
        
        
        
        
        
        
        
        

