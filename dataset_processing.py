import os

import matplotlib
import numpy as np
import cv2 as cv
from shutil import rmtree
from collections import OrderedDict
import matplotlib.pyplot as plt
plt.ioff()
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from preprocessing import ImageTransformer
import airfoil_reader
 

def preprocess_tf_data(im_tilde, im):

    im_tilde_tf = tf.cast(im_tilde,tf.float32)
    im_tf = tf.cast(im,tf.float32)

    return im_tilde_tf, im_tf

def preprocess_data(im_tilde, im):

    im_tilde = im_tilde.astype(np.float32)
    im = im.astype(np.float32)

    return im_tilde, im

def create_dataset_pipeline(dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):

    dataset_tensor = tf.data.Dataset.from_tensor_slices(dataset)
    if is_train:
        dataset_tensor = dataset_tensor.shuffle(buffer_size=dataset[0].shape[0]).repeat()
    dataset_tensor = dataset_tensor.map(preprocess_tf_data,num_parallel_calls=num_threads)
    dataset_tensor = dataset_tensor.batch(batch_size)
    dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

    return dataset_tensor

def get_tensorflow_datasets(data_train,data_cv,data_test,batch_size=32):

    dataset_train = create_dataset_pipeline(data_train,is_train=True,batch_size=batch_size)
    dataset_cv = create_dataset_pipeline(data_cv,is_train=False,batch_size=1)
    dataset_test = preprocess_data(data_test[0],data_test[1])

    return dataset_train, dataset_cv, dataset_test

def preprocess_image(img, new_dims):

    m = len(img)
    imgs_processed = np.zeros((m,new_dims[1]*new_dims[0]),dtype=np.float32)
    for i in range(m):
        # Resize
        if img[i].shape[0:2] != (new_dims[1],new_dims[0]):
            img_processed = ImageTransformer.resize(img[i],new_dims)
        else:
            img_processed = img[i]
        # Invert
        img_processed = cv.bitwise_not(img_processed)
        # Normalize
        img_processed = img_processed/255.
        imgs_processed[i] = img_processed.reshape((np.prod(img_processed.shape[0:])))

    return imgs_processed

def read_dataset(case_folder, airfoil_analysis, format='png'):

    plots_folder = os.path.join(case_folder,'Datasets','plots','Training',airfoil_analysis)
    if airfoil_analysis == 'camber':
        airfoil_fpaths = [os.path.join(plots_folder,file) for file in os.listdir(plots_folder)
                          if not file.endswith('_s%s' %format)]
    elif airfoil_analysis == 'thickness':
        airfoil_fpaths = [os.path.join(plots_folder,file) for file in os.listdir(plots_folder)]

    airfoil_img = []
    airfoil_name = []
    for ipath in airfoil_fpaths:
        img = cv.imread(ipath)
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        airfoil_img.append(gray_img)
        airfoil_name.append((os.path.basename(ipath).split('.')[0]))
   
    return airfoil_name, airfoil_img
    
def plot_dataset(dataset_folder, fpaths, dataset_type='Training'):
    
    plots_folder = os.path.join(dataset_folder,'plots',dataset_type)
    if os.path.exists(plots_folder):
        rmtree(plots_folder)
    os.makedirs(os.path.join(plots_folder,'camber'))
    os.makedirs(os.path.join(plots_folder,'thickness'))
    os.makedirs(os.path.join(plots_folder,'upperside'))
    os.makedirs(os.path.join(plots_folder,'lowerside'))

    line_width = 10
    for fpath in fpaths:
        airfoil_scanner = airfoil_reader.AirfoilScanner(fpath,{})
        xu, zu, xl, zl, x, zc, zt, name = airfoil_scanner.get_geometry()

        if not name.endswith('_s'):
            fig, ax = plt.subplots(1)
            ax.plot(x,zc,color='k',linewidth=line_width)
            ax.axis('off')
            fig.savefig(os.path.join(plots_folder,'camber',name),dpi=200)
            plt.close()

        fig, ax = plt.subplots(1)
        ax.plot(x,zt,color='k',linewidth=line_width)
        ax.axis('off')
        fig.savefig(os.path.join(plots_folder,'thickness',name), dpi=200)
        plt.close()
        
        fig, ax = plt.subplots(1)
        ax.plot(xl,zl,color='k',linewidth=line_width)
        ax.axis('off')
        fig.savefig(os.path.join(plots_folder,'lowerside',name),dpi=200)
        plt.close()
        
        fig, ax = plt.subplots(1)
        ax.plot(xu,zu,color='k',linewidth=line_width)
        ax.axis('off')
        fig.savefig(os.path.join(plots_folder,'upperside',name),dpi=200)
        plt.close()

def add_design_parameters(samples, X, airfoil_data, case_folder):

    r, m = X.shape  # number of samples and dimension of samples in the dataset provided
    s = len(airfoil_data[samples[0]].values()) # get size of design vector
    X_augm = np.zeros([r,m+s],dtype='float')
    for i,sample in enumerate(samples):
        b = list(airfoil_data[sample].values())
        X_augm[i,:] = np.concatenate((b,X[i,:]),axis=0)
        
    return X_augm
    
def get_datasets(case_folder, design_parameters, training_size, img_dims, cond_analysis=True, airfoil_analysis='camber'):

    # Read original datasets
    samples, X = read_dataset(case_folder,airfoil_analysis,format='png')
    # Resize images, if necessary
    X = preprocess_image(X,img_dims)

    # Read dataset images 
    samples_train, samples_val = train_test_split(samples,train_size=training_size,shuffle=True)
    samples_train_mask = [True if sample in samples_train else False for sample in samples]
    X_train = X[samples_train_mask]
    samples_val_mask = [True if sample in samples_val else False for sample in samples]
    X_val = X[samples_val_mask]

    samples_cv, samples_test = train_test_split(samples_val,train_size=0.75,shuffle=True)
    samples_cv_mask = [True if sample in samples_cv else False for sample in samples_val]
    X_cv = X_val[samples_cv_mask]
    samples_test_mask = [True if sample in samples_test else False for sample in samples_val]
    X_test = X_val[samples_test_mask]

    # Get design parameters and normalize design matrix
    aerodata = airfoil_reader.get_aerodata(design_parameters,case_folder,airfoil_analysis,add_geometry=False)
    aerodata_norm = dict.fromkeys(aerodata)
    r = len(aerodata)  # number of total (training + validation) samples
    s = len(aerodata[samples[0]].keys()) # get size of design vector
    b = np.zeros([r,s],dtype='float64')
    for j,airfoil in enumerate(aerodata.values()):
        b[j,:] = list(airfoil.values())
    scaler = QuantileTransformer().fit(b)  # the data is fit to the whole amount of samples (this can affect training)
    b_norm = scaler.transform(b)
    # Insert normalized parameters to aerodata (normalized) dictionary
    for j,airfoil in enumerate(aerodata_norm.keys()):
        aerodata_norm[airfoil] = OrderedDict(zip(aerodata[samples[0]].keys(),b_norm[j,:]))

    if cond_analysis == True:
        # Augment datasets with design parameters
        X_train = add_design_parameters(samples_train,X_train,aerodata_norm,case_folder)
        X_cv = add_design_parameters(samples_cv,X_cv,aerodata_norm,case_folder)
        X_test = add_design_parameters(samples_test,X_test,aerodata_norm,case_folder)

    data_train = (X_train, X_train)
    data_cv = (X_cv, X_cv)
    data_test = (X_test, X_test)
    
    return samples_train, data_train, samples_cv, data_cv, samples_test, data_test