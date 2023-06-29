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

def preprocess_data(im_tilde, im):

    im_tilde = im_tilde.astype(np.float32)
    im = im.astype(np.float32)

    return im_tilde, im

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

def read_dataset(case_folder, airfoil_dzdx_analysis, dataset_folder, format='png'):

    plots_folder = os.path.join(case_folder,'Datasets','plots',dataset_folder)
    airfoil_fpaths = []
    for (root, case_dirs, _) in os.walk(plots_folder):
        if airfoil_dzdx_analysis == 'camber' or airfoil_dzdx_analysis == None:
            for case_dir in case_dirs:
                files = [os.path.join(root,case_dir,file) for file in os.listdir(os.path.join(root,case_dir))
                         if file.endswith(format) if not file.endswith('_s%s' %format)]
                airfoil_fpaths += files
        elif airfoil_dzdx_analysis == 'thickness':
            for case_dir in case_dirs:
                files = [os.path.join(root,case_dir,file) for file in os.listdir(os.path.join(root,case_dir))
                         if file.endswith(format)]
                airfoil_fpaths += files

    airfoil_img = []
    airfoil_name = []
    for ipath in airfoil_fpaths:
        img = cv.imread(ipath)
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        airfoil_img.append(gray_img)
        airfoil_name.append((os.path.basename(ipath).split('.')[0]))
   
    return airfoil_name, airfoil_img
    
def plot_dataset(dataset_folder, fpaths, dataset_type='Originals'):
    
    plots_folder = os.path.join(dataset_folder,'plots',dataset_type)
    if os.path.exists(plots_folder):
        rmtree(plots_folder)
    os.makedirs(os.path.join(plots_folder,'camber'))
    os.makedirs(os.path.join(plots_folder,'thickness'))
    os.makedirs(os.path.join(plots_folder,'upperside'))
    os.makedirs(os.path.join(plots_folder,'lowerside'))
    os.makedirs(os.path.join(plots_folder,'full_airfoil'))

    ylim_c = (-0.015,0.15)
    ylim_t = (-0.01,0.3)
    ylim_u = (-0.05,0.35)
    ylim_l = (-0.1,0.15)
    line_width = 6
    for fpath in fpaths:
        airfoil_scanner = airfoil_reader.AirfoilScanner(fpath,{},airfoil_analysis='full')
        xu, zu, xl, zl, x, zc, zt, name = airfoil_scanner.get_geometry()

        if not name.endswith('_s'):
            fig, ax = plt.subplots(1)
            ax.plot(x,zc,color='k',linewidth=line_width)
            ax.axis('off')
            ax.set_ylim(ylim_c);
            fig.savefig(os.path.join(plots_folder,'camber',name),dpi=200,bbox_inches='tight',pad_inches=0)
            plt.close()

        fig, ax = plt.subplots(1)
        ax.plot(x,zt,color='k',linewidth=line_width)
        ax.axis('off')
        ax.set_ylim(ylim_t);
        fig.savefig(os.path.join(plots_folder,'thickness',name),dpi=200,bbox_inches='tight',pad_inches=0)
        plt.close()
        
        fig, ax = plt.subplots(1)
        ax.plot(xl,zl,color='k',linewidth=line_width)
        ax.axis('off')
        ax.set_ylim(ylim_l);
        fig.savefig(os.path.join(plots_folder,'lowerside',name),dpi=200,bbox_inches='tight',pad_inches=0)
        plt.close()
        
        fig, ax = plt.subplots(1)
        ax.plot(xu,zu,color='k',linewidth=line_width)
        ax.axis('off')
        ax.set_ylim(ylim_u);
        fig.savefig(os.path.join(plots_folder,'upperside',name),dpi=200,bbox_inches='tight',pad_inches=0)
        plt.close()

        fig, ax = plt.subplots(1)
        ax.plot(xu,zu,color='r',linewidth=line_width-2)
        ax.plot(xl,zl,color='b',linewidth=line_width-2)
        ax.plot(x,zc,color='g',linestyle='--',linewidth=line_width-5)
        ax.axis('off')
        ax.set_ylim((ylim_l[0],ylim_u[1]));
        fig.savefig(os.path.join(plots_folder,'full_airfoil',name),dpi=200,bbox_inches='tight',pad_inches=0)
        plt.close()

def get_design_dataset(samples, X, airfoil_data, design_parameters):

    r = X.shape[0]  # number of samples and dimension of samples in the dataset provided
    s = len(airfoil_data[samples[0]].values()) # get size of design vector
    b = np.zeros([r,s],dtype='float')

    # Retrieve indexes of the design parameters to take
    idx = []
    for i,v in enumerate(design_parameters.values()):
        if v == 1:
            idx.append(i)
        if type(v) == tuple:
            idx = idx + [i+ii for ii in range(len(v[1]))]

    # Build design parameters array
    for i,sample in enumerate(samples):
        airfoil_parameters = list(airfoil_data[sample].values())
        for j in idx:
            b[i,j] = airfoil_parameters[j]

    return b

def get_design_data(design_parameters, airfoil_dzdx_analysis, geo_folder):

    # Get design parameters and normalize design matrix
    aerodata = airfoil_reader.get_aerodata(design_parameters,geo_folder,airfoil_dzdx_analysis,mode='train',add_geometry=False)
    airfoils = list(aerodata.keys())
    parameters = list(aerodata[airfoils[0]].keys())

    r = len(aerodata)  # number of total (training + validation) samples
    if 'xdzdx' in design_parameters.keys():
        s = 8 + len(design_parameters['xdzdx'][1]) # get size of design vector
    else:
        s = 8
    b = np.zeros([r,s],dtype='float64')
    for j,airfoil in enumerate(aerodata.values()):
        b[j,:] = list(airfoil.values())

    return airfoils, b, parameters

def get_datasets(case_folder, design_parameters, training_size, img_dims, airfoil_dzdx_analysis=None):

    # Read original datasets
    samples, X = read_dataset(case_folder,airfoil_dzdx_analysis,dataset_folder='Training',format='png')
    # Resize images, if necessary
    X = preprocess_image(X,img_dims)

    # Read dataset images 
    samples_train, samples_val = train_test_split(samples,train_size=training_size,shuffle=True)
    samples_train = [sample for sample in samples if not sample in samples_val]
    samples_val = [sample for sample in samples if not sample in samples_train]

    samples_train_mask = [True if not sample in samples_val else False for sample in samples]
    X_train = X[samples_train_mask]
    samples_val_mask = [True if not sample in samples_train else False for sample in samples]
    X_val = X[samples_val_mask]

    # Split validation set into cross-validation set and test set
    samples_cv, samples_test = train_test_split(samples_val,train_size=0.75,shuffle=True)
    samples_cv = [sample for sample in samples_val if not sample in samples_test]
    samples_test = [sample for sample in samples_val if not sample in samples_cv]

    samples_cv_mask = [True if sample in samples_cv else False for sample in samples_val]
    X_cv = X_val[samples_cv_mask]
    samples_test_mask = [True if sample in samples_test else False for sample in samples_val]
    X_test = X_val[samples_test_mask]

    # Get design data and normalize
    geo_folder = os.path.join(case_folder,'Datasets','geometry','originals')
    airfoils, b, parameters_name = get_design_data(design_parameters,airfoil_dzdx_analysis,geo_folder)
    scaler = QuantileTransformer().fit(b)  # the data is fit to the whole amount of samples (this can affect training)
    b_norm = scaler.transform(b)

    # Insert normalized parameters to aerodata (normalized) dictionary
    aerodata_norm = dict.fromkeys(airfoils)
    for j,airfoil in enumerate(airfoils):
        aerodata_norm[airfoil] = OrderedDict(zip(parameters_name,b_norm[j,:]))

    # Get design dataset
    b_train = get_design_dataset(samples_train,X_train,aerodata_norm,design_parameters)
    b_cv = get_design_dataset(samples_cv,X_cv,aerodata_norm,design_parameters)
    b_test = get_design_dataset(samples_test,X_test,aerodata_norm,design_parameters)

    return samples_train, X_train, b_train, samples_cv, X_cv, b_cv, samples_test, X_test, b_cv
