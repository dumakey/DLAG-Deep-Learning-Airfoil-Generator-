import os
import numpy as np
import cv2 as cv
from math import sinh, cosh
from matplotlib import pyplot as plt
from random import randint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
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
                    figname = 'Dataset_{}_generated_samples_{}'.format(k+1,n+1)
            else:
                if n_datasets == 1:
                    figname = 'Generated_samples'
                else:
                    figname = 'Dataset_{}_generated_samples'.format(k+1)
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

def plot_generated_variational_sample(datasets, variational_space, img_size, storage_dir):

    width, height = img_size
    n_datasets = len(datasets)
    var_space = [variation for variation in variational_space]
    n_rows = 3
    n_cols = 2

    for k,dataset in enumerate(datasets):
        n_latent_dim = dataset.shape[0]
        n_samples = dataset.shape[1]
        n_figs = int(np.ceil(n_samples/(n_rows*n_cols)))
        for t in range(n_latent_dim):
            figs_folder = os.path.join(storage_dir,'Latent_dim_t{}'.format(t+1))
            if os.path.exists(figs_folder):
                rmtree(figs_folder)
            os.makedirs(figs_folder)
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
                        x = np.reshape(dataset[t,s],(height,width))*255  # Un-scale image
                        x = x.astype('uint8')
                        _, x = cv.threshold(x,50,255,cv.THRESH_BINARY)
                        ax[i,j].imshow(x,cmap='Greys_r')
                        ax[i,j].text(0.5,0.2,'Var={:.2f}%'.format(var_space[s]),fontsize=12,color='r')
                        ax[i,j].set_xticks([])
                        ax[i,j].set_yticks([])
                        s += 1
                        if s == n_samples:
                            break
                    if s == n_samples:
                        break

                if n_figs > 1:
                    if n_datasets == 1:
                        figname = 'Generated_samples_latent_dim_ti={}_{}'.format(t+1,n+1)
                    else:
                        figname = 'Dataset_{}_latent_dim_ti={}_generated_samples_{}'.format(k+1,t+1,n+1)
                else:
                    if n_datasets == 1:
                        figname = 'Generated_samples_latent_dim_ti={}'.format(t+1)
                    else:
                        figname = 'Dataset_{}_generated_samples_latent_dim_ti={}'.format(k+1,t+1)
                plt.savefig(os.path.join(figs_folder,figname+'.png'),dpi=100,bbox_inches='tight',pad_inches=0)
                plt.close()

def chop(x, y, y_ref):
    # Chop
    if y[0] < y_ref[0]: # lower chop
        m = y.size
        i_y0 = np.where(y[:m//2] <= y_ref[0])[0]
        x = x[i_y0[-1]:]
        y = y[i_y0[-1]:]
    if y[-1] < y_ref[-1]: # upper chop
        m = np.argmax(y)
        i_yf = np.where(y[m:] <= y_ref[-1])[0]
        x = x[:(m+i_yf[0])]
        y = y[:(m+i_yf[0])]

    return x, y
def get_mean_contour(img, threshold=200, ylims=None):

    # Define scaling variables
    height, width = img.shape
    real_height = ylims[1] - ylims[0]
    scale_height = real_height/height
    scale_width = 1/width

    # Filter along x-slice, according to bright intensity
    slices = [[]] * width
    for i in range(width): # loop in width
        # Compute mask per image slice
        mask = np.zeros(height,dtype=int)
        for j in range(height): # loop in height
            mask[j] = 1 if img[j,i] > threshold else 0
        # Get slices
        slices[i] = np.where(mask == 1)[0]

    # Scale
    K = 1.5 # std-dev distance to remove outliers
    slice_scaled = np.zeros((width,))
    for i in range(width):
        if len(slices[i]) != 0:
            Q75, Q25 = np.percentile(slices[i],[75,25])
            DQ = Q75 - Q25
            upper_bound = Q75 + K*DQ
            lower_bound = Q25 - K*DQ

            i_lower_bound = np.where(slices[i] < lower_bound)[0]
            if ~len(i_lower_bound) == 0:
                slices[i][i_lower_bound] = np.nan
            i_upper_bound = np.where(slices[i] > upper_bound)[0]
            if ~len(i_upper_bound) == 0:
                slices[i][i_upper_bound] = np.nan

            slice = np.array([item for item in slices[i] if item != np.nan])

            slice_scaled[i] = scale_height*(height - np.mean(slice))
        else:
            slice_scaled[i] = np.nan

    '''
    x = [scale_width*randint(0,len(slices_scaled)) for i in range(5)]
    for i in range(len(x)):
        plt.hist(slices_scaled[i],bins=10,density=True,label='{:.2f}'.format(x[i]))
        plt.legend()
    '''
    # Filter
    n = 0
    for i in range(width):
        if ~np.isnan(slice_scaled[i]):
            n += 1
    y_filt = np.zeros((n,))
    x_filt = np.zeros((n,))
    x = np.linspace(0,1,width)
    j = 0
    for i in range(width):
        if ~np.isnan(slice_scaled[i]):
            y_filt[j] = slice_scaled[i]
            x_filt[j] = x[i]
            j += 1

    # Fit to polynomial shape
    deg = 4
    coeffs = np.polyfit(x_filt,y_filt,deg=deg)
    pol = np.poly1d(coeffs)
    x_ref = x
    y_ref = np.array([pol(x) for x in x_ref])

    # Chop to adjust to y-limits
    x_ref, y_ref = chop(x_ref,y_ref,y_filt)

    return x_ref, y_ref

def close_contours(x_open, zu_open, zl_open):

    def f(z, a, b, c, d, e, f):

        return a*z**5 + b*z**4 + c*z**3 + d*z**2 + e*z + f

    def f_p(z, a, b, c, d, e):

        return 5*a*z**4 + 4*b*z**3 + 3*c*z**2 + 2*d*z + e

    def f_pp(z, a, b, c, d):

        return 20*z**3 + 12*b*z**2 + 6*c*z + 2*d

    def F(C, *args):

        # C = a, b, c, d, e, f

        x, z, dxdz, d2xdz2 = args

        F = np.zeros((6,))
        F[0] = f_p(z[0],C[0],C[1],C[2],C[3],C[4]) - dxdz[0]
        F[1] = f_p(z[1],C[0],C[1],C[2],C[3],C[4]) - dxdz[1]
        F[2] = f_pp(z[0],C[0],C[1],C[2],C[3]) - d2xdz2[0]
        F[3] = f_pp(z[1],C[0],C[1],C[2],C[3]) - d2xdz2[1]
        F[4] = f(z[0],C[0],C[1],C[2],C[3],C[4],C[5]) - x[0]
        F[5] = f(z[1],C[0],C[1],C[2],C[3],C[4],C[5]) - x[1]

        return F


    i_xmax = min(np.argmax(zu_open),np.argmax(zl_open))//2
    i_xmin = 15
    # Construct x and z coordinates range
    x = np.concatenate((np.flipud(x_open[i_xmin:i_xmax]),np.array([x_open[0]]),x_open[i_xmin:i_xmax]))
    z = np.concatenate((-np.flipud(zu_open[i_xmin:i_xmax]),-np.array([zu_open[0]]),-np.flipud(zl_open[i_xmin:i_xmax])))

    # Compute F-function parameters
    i_z1 = (i_xmax-i_xmin)//2
    i_z2 = (i_xmax-i_xmin) + 1 + (i_xmax-i_xmin)//2

    zeval = np.zeros((2,))
    zeval[0] = z[i_z1]
    zeval[1] = z[i_z2]

    xeval = np.zeros((2,))
    xeval[0] = x[i_z1]
    xeval[1] = x[i_z2]

    dxdz = np.gradient(x,z)
    dxdz_eval = np.zeros((2,))
    dxdz_eval[0] = dxdz[i_z1]
    dxdz_eval[1] = dxdz[i_z2]

    d2xdz2 = np.gradient(dxdz,z)
    d2xdz2_eval = np.zeros((2,))
    d2xdz2_eval[0] = d2xdz2[i_z1]
    d2xdz2_eval[1] = d2xdz2[i_z2]

    X0 = [1,1,1,1,1,1]
    coeffs = fsolve(F,X0,args=(xeval,zeval,dxdz_eval,d2xdz2_eval))

    zq = np.linspace(z[0],z[-1],100)
    xq = f(zq,*coeffs)

    fig, ax = plt.subplots(1,)
    ax.scatter(z,x)
    ax.scatter(zeval[0],x[i_z1],color='r',label='z2')
    ax.scatter(zeval[1],x[i_z2],color='g',label='z3')
    ax.plot(zq,xq,color='c',linestyle='--',label='Polynomial')
    #ax.set_ylim(0.02,0.17);
    #ax.set_xlim(-0.12,0.008);
    print()


