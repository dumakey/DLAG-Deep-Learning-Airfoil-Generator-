import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from random import randint
import dataset_processing
import tensorflow as tf


def plot_generated_samples(datasets, img_size, storage_dir):

    width, height = img_size
    n_datasets = len(datasets)
    n_rows = 4
    n_cols = 2
    for k,dataset in enumerate(datasets):
        n_samples = dataset.shape[0]
        n_figs = int(np.ceil(n_samples/(n_rows*n_cols)))
        s = 0
        for n in range(n_figs):
            fig, ax = plt.subplots(n_rows,n_cols,sharex=True,figsize=(10,10))
            # hide axis
            for ii in range(n_rows):
                for jj in range(n_cols):
                    ax[ii,jj].axis('off')
            for i in range(n_rows):
                for j in range(n_cols):
                    x = np.reshape(dataset[s],(height,width))*255  # Un-scale image
                    x = x.astype('uint8')
                    _, x = cv.threshold(x,50,255,cv.THRESH_BINARY)
                    # x_dec = cv.bitwise_not(x_dec)
                    ax[i,j].imshow(x,cmap='Greys_r')
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                    s += 1
                    if s == n_samples:
                        break
                if s == n_samples:
                    break

            if n_figs > 1:
                if n_datasets == 1:
                    plt.savefig(os.path.join(storage_dir,'Generated_samples_{}.png'.format(n+1)), dpi=100)
                else:
                    plt.savefig(os.path.join(storage_dir,'Dataset_{}_generated_samples_{}.png'.format(k+1,n+1)), dpi=100)
            else:
                if n_datasets == 1:
                    plt.savefig(os.path.join(storage_dir,'Generated_samples.png'), dpi=100)
                else:
                    plt.savefig(os.path.join(storage_dir,'Dataset_{}_generated_samples.png'.format(k+1)), dpi=100)
            plt.close()

def plot_dataset_samples(dataset, b, predictor, n_samples, img_size, storage_dir, stage='Train'):

    dataset_prep, _ = dataset_processing.preprocess_data(dataset,dataset)
    width, height = img_size
    m = dataset.shape[0]
    
    ## PLOT GENERATED TRAINING DATA ##
    n_rows = 5
    n_figs = int(np.ceil(n_samples/n_rows))
    s = 0
    for j in range(n_figs):
        fig, ax = plt.subplots(n_rows,2,sharex=True,figsize=(10,10))
        # hide axis
        for ii in range(n_rows):
            for jj in range(2):
                ax[ii,jj].axis('off')
                ax[ii,jj].set_xticks([])
                ax[ii,jj].set_yticks([])
        ax[0,0].title.set_text('Predicted')
        ax[0,1].title.set_text(stage)
        for ii in range(n_rows):
            i = randint(0,m-1)
            # Predict
            x_unrolled = dataset_prep[i].reshape((1,height*width))
            b_des = np.reshape(b[i],(1,b[i].size))
            x_pred = predictor([x_unrolled,b_des])
            x_pred = np.reshape(x_pred,(height,width))*255  # Un-scale image
            x_pred = x_pred.astype('uint8')
            _, x_pred = cv.threshold(x_pred,50,255,cv.THRESH_BINARY)
            # x_pred = cv.bitwise_not(x_pred)

            # Plot
            x = dataset[i].reshape((height,width))
            # x = cv.bitwise_not(x)
            ax[ii,0].imshow(x_pred,cmap='Greys_r')
            ax[ii,1].imshow(x,cmap='Greys_r')
            s += 1
            if s == n_samples:
                break

        if n_figs > 1:
            plt.savefig(os.path.join(storage_dir,stage+'_'+'training_samples_{}.png'.format(j+1)), dpi=100)
        else:
            plt.savefig(os.path.join(storage_dir,stage+'_'+'training_samples.png'), dpi=100)
        plt.close()



