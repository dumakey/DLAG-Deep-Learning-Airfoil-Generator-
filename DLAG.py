# -*- coding: utf-8 -*-
import os
from shutil import rmtree, copytree
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
from collections import OrderedDict
import pickle
import cv2 as cv
from random import randint

import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()

import reader
import dataset_processing
import models
import dataset_augmentation
import postprocessing
import airfoil_reader


class CGenTrainer:

    def __init__(self, launch_file):

        class parameter_container:
            pass
        class dataset_container:
            pass
        class design_container:
            pass
        class model_container:
            pass
        class predictions_container:
            pass


        self.parameters = parameter_container()
        self.datasets = dataset_container()
        self.design = design_container()
        self.model = model_container()
        self.predictions = predictions_container()

        # Setup general parameters
        casedata = reader.read_case_setup(launch_file)
        self.parameters.analysis = casedata.analysis
        self.parameters.design_parameters_train = casedata.design_parameters_train
        self.parameters.design_parameters_des = casedata.design_parameters_des
        self.parameters.training_parameters = casedata.training_parameters
        self.parameters.img_processing = casedata.img_processing
        self.parameters.img_size = casedata.img_resize
        self.parameters.samples_generation = casedata.samples_generation
        self.parameters.data_augmentation = casedata.data_augmentation
        self.parameters.activation_plotting = casedata.activation_plotting
        self.case_dir = casedata.case_dir

        # Sensitivity analysis variable identification
        sens_vars = [item for item in self.parameters.training_parameters.items() if
                     item[0] not in ('enc_hidden_layers', 'dec_hidden_layers') if type(item[1]) == list]
        self.parameters.sens_variable = sens_vars[0] if len(sens_vars) != 0 else None

        # Check for model reconstruction
        if self.parameters.analysis['import'] == True:
            self.model.imported = True
            model, history = self.reconstruct_model()
            self.model.Model = [model]
            self.model.History = [history]
        else:
            self.model.imported = False

    def __str__(self):
        class_name = type(self).__name__

        return '{}, a class to generate airfoil contours based on Bayesian Deep learning algorithms'.format(class_name)

    def launch_analysis(self):

        analysis_ID = self.parameters.analysis['type']
        analysis_list = {
                        'singletraining': self.singletraining,
                        'sensanalysis': self.sensitivity_analysis_on_training,
                        'traingenerate': self.traingenerate,
                        'generate': self.airfoil_generation,
                        'datagen': self.data_generation,
                        'plotdata': self.plot_data,
                        'contourairfoil':self.contour_airfoil,
                        }

        analysis_list[analysis_ID]()

    def sensitivity_analysis_on_training(self):

        # Retrieve sensitivity variable
        sens_variable = self.parameters.sens_variable

        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size
        design_parameters_train = self.parameters.design_parameters_train
        airfoil_analysis = self.parameters.analysis['airfoil_analysis']

        # Read dataset
        self.datasets.samples_train, self.datasets.data_train, self.datasets.b_train, \
        self.datasets.samples_cv, self.datasets.data_cv, self.datasets.b_cv, \
        self.datasets.samples_test, self.datasets.data_test, self.datasets.b_test = \
        dataset_processing.get_datasets(case_dir,design_parameters_train,training_size,img_size,airfoil_analysis)

        if self.model.imported == False:
            self.train_model(sens_variable)
        self.export_model_performance(sens_variable)
        self.export_model(sens_variable)
        self.export_nn_log()
        
    def singletraining(self):

        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size
        design_parameters_train = self.parameters.design_parameters_train
        airfoil_analysis = self.parameters.analysis['airfoil_analysis']

        # Read dataset
        self.datasets.samples_train, self.datasets.data_train, self.datasets.b_train, \
        self.datasets.samples_cv, self.datasets.data_cv, self.datasets.b_cv, \
        self.datasets.samples_test, self.datasets.data_test, self.datasets.b_test = \
        dataset_processing.get_datasets(case_dir,design_parameters_train,training_size,img_size,airfoil_analysis)

        # Convert datasets to tensorflow format
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets((self.datasets.data_train,self.datasets.b_train),
                                                   (self.datasets.data_cv,self.datasets.b_cv),
                                                   (self.datasets.data_test,self.datasets.b_test),batch_size)

        # Train
        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_nn_log()
    
    def traingenerate(self):
    
        # Training
        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size
        design_parameters = self.parameters.design_parameters
        airfoil_analysis = self.parameters.analysis['airfoil_analysis']

        # Read dataset
        self.datasets.samples_train, self.datasets.data_train, self.datasets.b_train, \
        self.datasets.samples_cv, self.datasets.data_cv, self.datasets.b_cv, \
        self.datasets.samples_test, self.datasets.data_test, self.datasets.b_test = \
        dataset_processing.get_datasets(case_dir,design_parameters,training_size,img_size,airfoil_analysis)

        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train, self.datasets.data_cv,
                                                   self.datasets.data_test, batch_size)
        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_nn_log()
        
        # Generation
        model_dir = os.path.join(case_dir,'Results',str(self.parameters.analysis['case_ID']),'Model')
        generation_dir = os.path.join(case_dir,'Results','pretrained_model')
        if os.path.exists(generation_dir):
            rmtree(generation_dir)
        copytree(model_dir,generation_dir)
        self.model.imported = True
        self.airfoil_generation()
    
    def airfoil_generation(self):

        if self.model.imported == True:
            storage_dir = os.path.join(self.case_dir,'Results','pretrained_model','Airfoil_generation')
        else:
            storage_dir = os.path.join(self.case_dir,'Results','Airfoil_generation')
        if os.path.exists(storage_dir):
            rmtree(storage_dir)
        os.makedirs(storage_dir)

        # Set design parameters dictionary
        # Read case parameters from "pretrained model" folder
        casedata = reader.read_case_logfile(os.path.join(self.case_dir,'Results','pretrained_model','DLAG.log'))
        design_parameters_on_logfile = [item for item in casedata.design_parameters_train.keys() if item != 'xdzdx']  # exclude (training) slope controlpoints x-locations
        design_parameters_on_launch = self.parameters.design_parameters_des
        bcheck = set([True if item in design_parameters_on_logfile else False
                      for item in self.parameters.design_parameters_des.keys() if not item.startswith('dzdx')])
        if bcheck != {True}: # if not all design parameters are included in the (design) parameters used for training
            self.parameters.design_parameters_des = OrderedDict(casedata.design_parameters_train)
            # delete the parameter corresponding to the specification of the slope controlpoints x-loc
            if 'dzdx' in self.parameters.design_parameters_des:
                del self.parameters.design_parameters_des['dzdx']
            # Assign the specified design parameters to the "available" training design parameters
            for parameter, value in design_parameters_on_launch.items():
                if parameter in self.parameters.design_parameters_des.keys():
                    self.parameters.design_parameters_des[parameter] = value
            # If not all the specified design parameters match with the training design parameters
            if None in self.parameters.design_parameters_des.values():
                raise Exception('There are design parameters specified which were not used for training.\n'
                                'The list of design parameters used for training are:\n'
                                '{}'.format(list(casedata.design_parameters_train.keys())))

        # Add the curvature control points design parameters, if they exist
        dzdx_cp = [design_parameters_on_launch[item] for item in design_parameters_on_launch.keys()
                   if item == 'dzdx_c' or item == 'dzdx_t']
        if dzdx_cp:
            if 'dzdx_c' in design_parameters_on_launch.keys():
                self.parameters.design_parameters_des['dzdx'] = ('camber',design_parameters_on_launch['dzdx_c'])
                del self.parameters.design_parameters_des['dzdx_c']
            elif 'dzdx_t' in design_parameters_on_launch.keys():
                self.parameters.design_parameters_des['dzdx'] = ('thickness',design_parameters_on_launch['dzdx_t'])
                del self.parameters.design_parameters_des['dzdx_t']
        casedata.design_parameters_des = self.parameters.design_parameters_des.copy()

        # Read parameters
        case_dir = self.case_dir
        n_samples = self.parameters.samples_generation['n_samples']
        training_size = casedata.training_parameters['train_size']
        img_size = casedata.img_size

        if self.model.imported == False:
            self.singletraining()

        if not hasattr(self, 'data_train'):
            samples_train, data_train, b_train, samples_cv, data_cv, b_cv, samples_test, data_test, b_test = \
                dataset_processing.get_datasets(case_dir,self.parameters.design_parameters_train,training_size,img_size)
            for model in self.model.Model:
                postprocessing.plot_dataset_samples(data_train,b_train,model.predict,n_samples,img_size,storage_dir,stage='Train')
                postprocessing.plot_dataset_samples(data_cv,b_cv,model.predict,n_samples,img_size,storage_dir,stage='Cross-validation')
                postprocessing.plot_dataset_samples(data_test,b_test,model.predict,n_samples,img_size,storage_dir,stage='Test')

        ## GENERATE NEW DATA - SAMPLING ##
        X_samples = self.generate_samples(casedata)
        postprocessing.plot_generated_samples(X_samples,img_size,storage_dir)

    def contour_airfoil(self):

        storage_dir = os.path.join(self.case_dir,'Results','Airfoil_contours')
        if os.path.exists(storage_dir):
            rmtree(storage_dir)
        os.makedirs(storage_dir)


    def airfoil_reconstruction(self, plot_full_airfoil=False):

        m = 10
        airfoil_container = airfoil_reader.get_aerodata({},self.case_dir,self.parameters.analysis['airfoil_analysis'],add_geometry=True)
        for name,airfoil in airfoil_container.items():
            x = airfoil['x']
            zc = airfoil['zc']
            ac = airfoil_reader.AirfoilScanner.get_internal_parameters(zc,x,m)
            zc_r = airfoil_reader.AirfoilScanner.reconstruct_z(zc,x,order=m,provide_a=True,a=ac)

            zt = airfoil['zt']
            at = airfoil_reader.AirfoilScanner.get_internal_parameters(zt,x,m)
            zt_r = airfoil_reader.AirfoilScanner.reconstruct_z(zt,x,order=m,provide_a=True,a=at)

            if plot_full_airfoil == True:
                # Plot the whole airfoil (US + LS)
                xu = airfoil['xu']
                zu = airfoil['zu']
                au = airfoil_reader.AirfoilScanner.get_internal_parameters(zu,xu,m)
                zu_r = airfoil_reader.AirfoilScanner.reconstruct_z(zu,xu,order=m,provide_a=True,a=au)

                xl = airfoil['xl']
                zl = airfoil['zl']
                al = airfoil_reader.AirfoilScanner.get_internal_parameters(zl,xl,m)
                zl_r = airfoil_reader.AirfoilScanner.reconstruct_z(zl,xl,order=m,provide_a=True,a=al)

                max_z = 1.2*max(zu)
                min_z = 1.2*min(zl)

                fig, ax = plt.subplots(2,2)
                ax[0,0].plot(x,zc,'r',label='real')
                ax[0,0].plot(x,zc_r,'b',label='reconstructed')
                ax[0,0].set_ylim(0,max_z)
                ax[0,0].set_ylabel('zc')
                ax[0,0].set_xticks([])

                ax[0,1].plot(x,zt,'r',label='real')
                ax[0,1].plot(x,zt_r,'b',label='reconstructed')
                ax[0,1].set_ylim(0,max_z)
                ax[0,1].set_xticks([])
                ax[0,1].set_yticks([])
                ax[0,1].set_ylabel('zt')

                ax[1,0].plot(xu,zu,'r',label='real')
                ax[1,0].plot(xu,zu_r,'b',label='reconstructed')
                ax[1,0].set_ylim(0,max_z)
                ax[1,0].set_xlabel('x')
                ax[1,0].set_ylabel('z')
                ax[1,0].set_ylabel('zu')

                ax[1,1].plot(xl,zl,'r',label='real')
                ax[1,1].plot(xl,zl_r,'b',label='reconstructed')
                ax[1,1].set_ylim(min_z,max_z)
                ax[1,1].set_yticks([])
                ax[1,1].set_xlabel('x')
                ax[1,1].set_ylabel('zl')
                ax[1,1].legend()

                figurename = '%s_airfoil_full_geometry.png' %name
            else:
                fig, ax = plt.subplots(1,1)
                ax[0,0].plot(x,zc,x,zc_r,label='Camber')
                ax[0,1].plot(x,zt,x,zt_r,label='Thickness')
                ax[0,:].set_ylim(0,max_z)
                ax[:,:].set_xlabel('x')
                ax[:,:].set_ylabel('z')
                ax.legend()

                figurename = '%s_airfoil_basic_geometry.png' %name

            # Save figure to folder
            folder_path = os.path.join(self.case_dir,'Results','airfoil_plots')
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            plt.savefig(os.path.join(folder_path,figurename),dpi=200,format=None,orientation="landscape")

    def data_generation(self):

        transformations = [{k:v[1:] for (k,v) in self.parameters.img_processing.items() if v[0] == 1}][0]
        augdata_size = self.parameters.data_augmentation[1]
        self.generate_augmented_data(transformations,augdata_size)
        
    def plot_data(self):
    
        dataset_folder = os.path.join(self.case_dir,'Datasets')
        airfoil_fpaths = []
        for (root, case_dirs, _) in os.walk(os.path.join(dataset_folder,'geometry')):
            for case_dir in case_dirs:
                files = [os.path.join(root,case_dir,file) for file in os.listdir(os.path.join(root,case_dir)) if file.endswith('.dat')]
                airfoil_fpaths += files
        
        dataset_processing.plot_dataset(dataset_folder,airfoil_fpaths,dataset_type='Training')

    def generate_augmented_data(self, transformations, augmented_dataset_size=1):

        # Set storage folder for augmented dataset
        augmented_dataset_dir = os.path.join(self.case_dir,'Datasets','plots','Augmented')

        # Unpack data
        _, X = dataset_processing.read_dataset(case_folder=self.case_dir,airfoil_analysis=None,dataset_folder='To_augment')

        # Generate new dataset
        data_augmenter = dataset_augmentation.datasetAugmentationClass(X,transformations,augmented_dataset_size,augmented_dataset_dir)
        data_augmenter.transform_images()
        data_augmenter.export_augmented_dataset()

    def train_model(self, sens_var=None):

        # Parameters
        input_dim = self.parameters.img_size
        if 'xdzdx' in self.parameters.design_parameters_train.keys():
            n_dest = len(self.parameters.design_parameters_train.keys()) + len(self.parameters.design_parameters_train['xdzdx'][1]) - 1
        else:
            n_dest = len(self.parameters.design_parameters_train.keys())
        latent_dim = self.parameters.training_parameters['latent_dim']
        enc_hidden_layers = self.parameters.training_parameters['enc_hidden_layers']
        dec_hidden_layers = self.parameters.training_parameters['dec_hidden_layers']
        alpha = self.parameters.training_parameters['learning_rate']
        nepoch = self.parameters.training_parameters['epochs']
        batch_size = self.parameters.training_parameters['batch_size']
        l2_reg = self.parameters.training_parameters['l2_reg']
        l1_reg = self.parameters.training_parameters['l1_reg']
        dropout = self.parameters.training_parameters['dropout']
        activation = self.parameters.training_parameters['activation']

        self.model.Model = []
        self.model.History = []
        Model = models.VAEC
        if sens_var == None:  # If it is a one-time training
            self.model.Model.append(Model(input_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                               l1_reg,dropout,activation,mode='train'))

            self.model.History.append(self.model.Model[-1].fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                               y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                               validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                               self.datasets.data_cv),steps_per_epoch=200,validation_steps=None,verbose=1))

        else: # If it is a sensitivity analysis
            if type(alpha) == list:
                for learning_rate in alpha:
                    if self.model.imported == False:
                        model = Model(input_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,learning_rate,
                                           l2_reg,l1_reg,dropout,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                        y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                        steps_per_epoch=200,validation_steps=None,verbose=1,
                                                        validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                        self.datasets.data_cv)))
            elif type(l2_reg) == list:
                for regularizer in l2_reg:
                    if self.model.imported == False:
                        model = Model(input_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,regularizer,
                                           l1_reg,dropout,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                        y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                        steps_per_epoch=200,validation_steps=None,verbose=1,
                                                        validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                        self.datasets.data_cv)))
            elif type(l1_reg) == list:
                for regularizer in l1_reg:
                    if self.model.imported == False:
                        model = Model(input_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                           regularizer,dropout,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                        y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                        steps_per_epoch=200,validation_steps=None,verbose=1,
                                                        validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                        self.datasets.data_cv)))
            elif type(dropout) == list:
                for rate in dropout:
                    if self.model.imported == False:
                        model = Model(input_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                           l1_reg,rate,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                        y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                        steps_per_epoch=200,validation_steps=None,verbose=1,
                                                        validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                        self.datasets.data_cv)))
            elif type(activation) == list:
                for act in activation:
                    if self.model.imported == False:
                        model = Model(input_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                           l1_reg,dropout,act,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                        y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                        steps_per_epoch=200,validation_steps=None,verbose=1,
                                                        validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                        self.datasets.data_cv)))
            elif type(latent_dim) == list:
                for dim in latent_dim:
                    if self.model.imported == False:
                        model = Model(input_dim,n_dest,dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,l1_reg,dropout,
                                           activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(x=[self.datasets.data_train,self.datasets.b_train],
                                                        y=self.datasets.data_train,batch_size=batch_size,epochs=nepoch,
                                                        verbose=1,steps_per_epoch=200,
                                                        validation_data=([self.datasets.data_cv,self.datasets.b_cv],
                                                        self.datasets.data_cv)))

    def generate_samples(self, parameters):

        def build_design_vector(parameters, case_folder):

            if 'dzdx' in parameters.design_parameters_des.keys():
                n_dpar = len(parameters.design_parameters_des.keys()) - 1 + len(parameters.design_parameters_des['dzdx'][-1])
            else:
                n_dpar = len(parameters.design_parameters_des.keys())
            b = np.zeros((n_dpar,))
            i = 0
            for parameter, value in parameters.design_parameters_des.items():
                if parameter != 'dzdx':
                    b[i] = value
                    i += 1
                else:
                    for dzdx in value[1]:
                        b[i] = dzdx
                        i += 1

            # Normalize vector
            if 'dzdx' in parameters.design_parameters_des.keys():
                airfoil_analysis = parameters.design_parameters_des['dzdx'][0]
            _, b_tr, _ = dataset_processing.get_design_data(parameters.design_parameters_train,airfoil_analysis,case_folder)
            scaler = QuantileTransformer().fit(b_tr)  # the data is fit to the whole amount of samples (this can affect training)
            b_norm = scaler.transform(np.expand_dims(b,axis=0))

            return tf.convert_to_tensor(b_norm)

        ## BUILD DECODER ##
        output_dim = parameters.img_size
        latent_dim = parameters.training_parameters['latent_dim']
        alpha = parameters.training_parameters['learning_rate']
        dec_hidden_layers = parameters.training_parameters['dec_hidden_layers']
        activation = parameters.training_parameters['activation']
        training_size = parameters.training_parameters['train_size']
        batch_size = parameters.training_parameters['batch_size']
        n_samples = self.parameters.samples_generation['n_samples']
        if 'dzdx' in parameters.design_parameters_des.keys():
            n_dpar = len(parameters.design_parameters_des.keys()) - 1 + len(parameters.design_parameters_des['dzdx'][-1])
        else:
            n_dpar = len(parameters.design_parameters_des.keys())

        decoder = models.VAEC(output_dim,n_dpar,latent_dim,[],dec_hidden_layers,alpha,0.0,0.0,0.0,activation,'sample')  # No regularization

        # Generate new samples
        X_samples = []
        for model in self.model.Model:
            # Retrieve decoder weights
            j = 0
            for layer in model.layers:
                if layer.name.startswith('decoder') == False:
                    j += len(layer.weights)
                else:
                    break
            decoder_input_layer_idx = j

            decoder_weights = model.get_weights()[decoder_input_layer_idx:]
            decoder.set_weights(decoder_weights)

            ## SAMPLE IMAGES ##
            samples = np.zeros([n_samples,np.prod(output_dim)])
            for i in range(n_samples):
                t = tf.random.normal(shape=(1,latent_dim))
                b_des = build_design_vector(parameters,self.case_dir)
                samples[i,:] = decoder.predict([t,b_des],steps=1)
            X_samples.append(samples)

        return X_samples

    def export_model_performance(self, sens_var=None):

        try:
            History = self.model.History
        except:
            raise Exception('There is no evolution data for this model. Train model first.')
        else:
            if type(History) == list:
                N = len(History)
            else:
                N = 1
                History = [History]

            # Loss evolution plots #
            Nepochs = self.parameters.training_parameters['epochs']
            epochs = np.arange(1,Nepochs+1,1)

            case_ID = self.parameters.analysis['case_ID']
            for i,h in enumerate(History):
                loss_train = h.history['loss']
                loss_cv = h.history['val_loss']

                fig, ax = plt.subplots(1)
                ax.plot(epochs,loss_train,label='Training',color='r')
                ax.plot(epochs,loss_cv,label='Cross-validation',color='b')
                ax.grid()
                ax.set_xlabel('Epochs',size=12)
                ax.set_ylabel('Loss',size=12)
                ax.tick_params('both',labelsize=10)
                ax.legend()
                plt.suptitle('Loss evolution case = {}'.format(str(case_ID)))

                if sens_var:
                    if type(sens_var[1][i]) == str:
                        storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance',
                                                   '{}={}'.format(sens_var[0],sens_var[1][i]))
                    else:
                        storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance',
                                                   '{}={:.3f}'.format(sens_var[0],sens_var[1][i]))
                    loss_plot_filename = 'Loss_evolution_{}_{}={}.png'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                    loss_filename = 'Model_loss_{}_{}={}.csv'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                    metrics_filename = 'Model_metrics_{}_{}={}.csv'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance')
                    loss_plot_filename = 'Loss_evolution_{}.png'.format(str(case_ID))
                    loss_filename = 'Model_loss_{}.csv'.format(str(case_ID))
                    metrics_filename = 'Model_metrics_{}.csv'.format(str(case_ID))
                    
                if os.path.exists(storage_dir):
                    rmtree(storage_dir)
                os.makedirs(storage_dir)
                fig.savefig(os.path.join(storage_dir,loss_plot_filename),dpi=200)
                plt.close()

                # Metrics #
                metrics_name = [item for item in h.history if item not in ('loss','val_loss')]
                metrics_val = [(metric,h.history[metric][0]) for metric in metrics_name if metric.startswith('val')]
                metrics_train = [(metric,h.history[metric][0]) for metric in metrics_name if not metric.startswith('val')]

                rows = [metric[0] for metric in metrics_train]
                metric_fun = lambda L: np.array([item[1] for item in L])
                metrics_data = np.vstack((metric_fun(metrics_train),metric_fun(metrics_val))).T
                metrics = pd.DataFrame(index=rows,columns=['Training','CV'],data=metrics_data)
                metrics.to_csv(os.path.join(storage_dir,metrics_filename),sep=';',decimal='.')

                # Loss
                loss_data = np.vstack((list(epochs), loss_train, loss_cv)).T
                loss = pd.DataFrame(columns=['Epoch', 'Training', 'CV'], data=loss_data)
                loss.to_csv(os.path.join(storage_dir,loss_filename), index=False, sep=';', decimal='.')

    def export_model(self, sens_var=None):

        N = len(self.model.Model)
        case_ID = self.parameters.analysis['case_ID']
        for i in range(N):
            if sens_var:
                if type(sens_var[1][i]) == str:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'
                                               .format(sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'
                                               .format(sens_var[0],sens_var[1][i]))
                model_json_name = 'DLAG_model_{}_{}={}_arquitecture.json'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_weights_name = 'DLAG_model_{}_{}={}_weights.h5'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_folder_name = 'DLAG_model_{}_{}={}'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
            else:
                storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model')
                model_json_name = 'DLAG_model_{}_arquitecture.json'.format(str(case_ID))
                model_weights_name = 'DLAG_model_{}_weights.h5'.format(str(case_ID))
                model_folder_name = 'DLAG_model_{}'.format(str(case_ID))

            if os.path.exists(storage_dir):
                rmtree(storage_dir)
            os.makedirs(storage_dir)

            # Export history training
            with open(os.path.join(storage_dir,'History'),'wb') as f:
                pickle.dump(self.model.History[i].history,f)

            # Save model
            # Export model arquitecture to JSON file
            model_json = self.model.Model[i].to_json()
            with open(os.path.join(storage_dir,model_json_name),'w') as json_file:
                json_file.write(model_json)
            self.model.Model[i].save(os.path.join(storage_dir,model_folder_name.format(str(case_ID))))

            # Export model weights to HDF5 file
            self.model.Model[i].save_weights(os.path.join(storage_dir,model_weights_name))

    def reconstruct_model(self, mode='train'):

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        try:
            casedata = reader.read_case_logfile(os.path.join(storage_dir,'DLAG.log'))
            img_dim = casedata.img_size
            n_dest = len(casedata.design_parameters_train.keys()) + len(casedata.design_parameters_train['xdzdx'][1]) - 1
            latent_dim = casedata.training_parameters['latent_dim']
            enc_hidden_layers = casedata.training_parameters['enc_hidden_layers']
            dec_hidden_layers = casedata.training_parameters['dec_hidden_layers']
            activation = casedata.training_parameters['activation']

            # Load weights into new model
            Model = models.VAEC(img_dim,n_dest,latent_dim,enc_hidden_layers,dec_hidden_layers,0.001,0.0,0.0,0.0,activation,
                               mode)
            weights_filename = [file for file in os.listdir(storage_dir) if file.endswith('.h5')][0]
            Model.load_weights(os.path.join(storage_dir,weights_filename))
            class history_container:
                pass
            History = history_container()
            with open(os.path.join(storage_dir,'History'),'rb') as f:
                History.history = pickle.load(f)
            History.epoch = None
            History.model = Model
        except:
            tf.config.run_functions_eagerly(True) # Enable eager execution
            try:
                model_folder = next(os.walk(storage_dir))[1][0]
            except:
                print('There is no model stored in the folder')

            alpha = self.parameters.training_parameters['learning_rate']
            loss = models.loss_function

            Model = tf.keras.models.load_model(os.path.join(storage_dir,model_folder),custom_objects={'loss':loss},compile=False)
            Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x, y: loss,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            tf.config.run_functions_eagerly(False) # Disable eager execution

            # Reconstruct history
            class history_container:
                pass
            History = history_container()
            try:
                with open(os.path.join(storage_dir,'History'),'rb') as f:
                    History.history = pickle.load(f)
                History.epoch = np.arange(1,len(History.history['loss'])+1)
                History.model = Model
            except:
                History.epoch = None
                History.model = None

        return Model, History

    def export_nn_log(self):
        def update_log(parameters, model):
            training = OrderedDict()
            training['TRAINING SIZE'] = parameters.training_parameters['train_size']
            training['LEARNING RATE'] = parameters.training_parameters['learning_rate']
            training['L2 REGULARIZER'] = parameters.training_parameters['l2_reg']
            training['L1 REGULARIZER'] = parameters.training_parameters['l1_reg']
            training['DROPOUT'] = parameters.training_parameters['dropout']
            training['ACTIVATION'] = parameters.training_parameters['activation']
            training['NUMBER OF EPOCHS'] = parameters.training_parameters['epochs']
            training['BATCH SIZE'] = parameters.training_parameters['batch_size']
            training['LATENT DIMENSION'] = parameters.training_parameters['latent_dim']
            training['ENCODER HIDDEN LAYERS'] = parameters.training_parameters['enc_hidden_layers']
            training['DECODER HIDDEN LAYERS'] = parameters.training_parameters['dec_hidden_layers']
            training['OPTIMIZER'] = [model.optimizer._name for model in model.Model]
            training['METRICS'] = [model.metrics_names[-1] if model.metrics_names != None else None for model in model.Model]

            design = OrderedDict()
            design['DESIGN PARAMETERS TRAIN'] = [item.upper() for item in parameters.design_parameters_train.keys() if item != 'xdzdx']
            design['DESIGN PARAMETERS TRAIN'].append('XDZDX_C' if parameters.analysis['airfoil_analysis'] == 'camber' else 'XDZDX_T')
            design['XDZDX CONTROLPOINTS TRAIN'] = parameters.design_parameters_train['xdzdx'][-1]

            analysis = OrderedDict()
            analysis['CASE ID'] = parameters.analysis['case_ID']
            analysis['ANALYSIS'] = parameters.analysis['type']
            analysis['AIRFOIL ANALYSIS'] = parameters.analysis['airfoil_analysis']
            analysis['DESIGN AIRFOIL ANALYSIS'] = parameters.analysis['airfoil_analysis_des']
            analysis['IMPORTED MODEL'] = parameters.analysis['import']
            analysis['LAST TRAINING LOSS'] = ['{:.3f}'.format(history.history['loss'][-1]) for history in model.History]
            analysis['LAST CV LOSS'] = ['{:.3f}'.format(history.history['val_loss'][-1]) for history in model.History]

            architecture = OrderedDict()
            architecture['INPUT SHAPE'] = parameters.img_size

            return training, design, analysis, architecture


        parameters = self.parameters
        if parameters.analysis['type'] == 'sensanalysis':
            varname, varvalues = parameters.sens_variable
            for value in varvalues:
                parameters.training_parameters[varname] = value
                training, design, analysis, architecture = update_log(parameters,self.model)

                case_ID = parameters.analysis['case_ID']
                if type(value) == str:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'.format(varname,value))
                else:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'.format(varname,value))
                with open(os.path.join(storage_folder,'DLAG.log'),'w') as f:
                    f.write('DLAG log file\n')
                    f.write('==================================================================================================\n')
                    f.write('->ANALYSIS\n')
                    for item in analysis.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->TRAINING\n')
                    for item in training.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->DESIGN\n')
                    for item in design.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->ARCHITECTURE\n')
                    for item in architecture.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->MODEL\n')
                    for model in self.model.Model:
                        model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.write('==================================================================================================\n')

        else:
            training, design, analysis, architecture = update_log(self.parameters,self.model)
            case_ID = parameters.analysis['case_ID']
            storage_folder = os.path.join(self.case_dir,'Results',str(case_ID))
            with open(os.path.join(storage_folder,'Model','DLAG.log'),'w') as f:
                f.write('DLAG log file\n')
                f.write(
                    '==================================================================================================\n')
                f.write('->ANALYSIS\n')
                for item in analysis.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->TRAINING\n')
                for item in training.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->DESIGN\n')
                for item in design.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->ARCHITECTURE\n')
                for item in architecture.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->MODEL\n')
                for model in self.model.Model:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write(
                    '==================================================================================================\n')
if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\DLAG\Scripts\launcher.dat'
    trainer = CGenTrainer(launcher)
    trainer.launch_analysis()