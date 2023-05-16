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
    n_rows = 1
    n_cols = 1

    for k,dataset in enumerate(datasets):
        n_samples = dataset.shape[0]
        n_figs = int(np.ceil(n_samples/(n_rows*n_cols)))
        s = 0
        for n in range(n_figs):
            fig, ax = plt.subplots(n_rows,n_cols,sharex=True,figsize=(10,10))
            if n_rows == 1 or n_cols == 1:
                ax = np.reshape(ax,(1,1))
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
                    figname = 'Generated_samples_{}'.format(n+1)
                else:
                    figname = 'Dataset_{}_generated_samples_{}.png'.format(k+1,n+1)
            else:
                if n_datasets == 1:
                    figname = 'Generated_samples'
                else:
                    figname = 'Dataset_{}_generated_samples.png'.format(k+1)
            plt.savefig(os.path.join(storage_dir,figname+'.png'),dpi=100,bbox_inches='tight',pad_inches=0)
            plt.close()

def plot_dataset_samples(dataset, b, names, predictor, n_samples, img_size, storage_dir, stage='Train'):

    dataset_prep, _ = dataset_processing.preprocess_data(dataset,dataset)
    width, height = img_size
    m = dataset.shape[0]

    ## PLOT GENERATED TRAINING DATA ##
    n_rows = 5
    n_figs = int(np.ceil(n_samples/n_rows))

    s = 0
    for j in range(n_figs):
        fig, ax = plt.subplots(n_rows,2,sharex=True,figsize=(10,10))
        if n_rows == 1:
            ax = np.reshape(ax,(1,2))
        # hide axis
        for ii in range(n_rows):
            for jj in range(2):
                ax[ii,jj].axis('off')
                ax[ii,jj].set_xticks([])
                ax[ii,jj].set_yticks([])
        ax[0,0].title.set_text('Predicted\n')
        ax[0,1].title.set_text(stage+'\n')
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
            ax[ii,0].text(0.5,0.2,names[i],fontsize=10,color='r')
            ax[ii,1].text(0.5,0.2,names[i],fontsize=10,color='r')
            s += 1
            if s == n_samples:
                break

        if n_figs > 1:
            figname = stage+'_'+'training_samples_{}'.format(j+1)
        else:
            figname = stage + '_' + 'training_samples_{}'

        plt.savefig(os.path.join(storage_dir,figname+'.png'),dpi=200)
        plt.close()

def get_mean_contour(img, threshold=200, ylims=None):

    # Define scaling variables
    height, width = img.shape
    real_height = ylims[1] - ylims[0]
    scale_height = real_height/height
    scale_width = 1/width

    slices = [[]] * width
    for i in range(width): # loop in width
        # Compute mask per image slice
        mask = np.zeros(height,dtype=int)
        for j in range(height): # loop in height
            mask[j] = 1 if img[j,i] > threshold else 0
        # Get slices
        slices[i] = np.where(mask == 1)[0]

    # Scale
    slices_scaled = [scale_height*(height - slice) for slice in slices if slice != []]

    '''
    x = [scale_width*randint(0,len(slices_scaled)) for i in range(5)]
    for i in range(len(x)):
        plt.hist(slices_scaled[i],bins=10,density=True,label='{:.2f}'.format(x[i]))
        plt.legend()
    '''
    # Get mean coordinate
    y = np.zeros((len(slices_scaled),))
    for i in range(len(slices_scaled)):
        y[i] = np.mean(slices_scaled[i])

    return y




