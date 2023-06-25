import os
import numpy as np
import cv2 as cv
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

def generate_le(x_open, zu_open, zl_open):

    tmax_ratio = (zu_open[0] - zl_open[0])/(max(zu_open) - min(zl_open))
    if tmax_ratio < 0:
        # Computation of chordwise positions where dzdx = 0, searching only in the first half of the airfoil
        N = x_open.size//2
        dzudx = np.gradient(zu_open[:N],x_open[:N])
        x_zu_open_max = interp1d(dzudx[:N],x_open[:N],kind='quadratic')(0)

        dzldx = np.gradient(zl_open[:N],x_open[:N])
        x_zl_open_max = interp1d(dzldx[:N],x_open[:N],kind='quadratic')(0)

        i_xmax = min(np.where(x_open >= x_zu_open_max)[0][0],np.where(x_open >= x_zl_open_max)[0][0])
        i_xumin = 1
        i_xlmin = 1
        x = x_open[0:i_xmax]
        zu = zu_open[0:i_xmax]
        zl = zl_open[0:i_xmax]

        x_le = interp1d(zu-zl,x)(0)
        z_le = interp1d(x,zu,fill_value='extrapolate')(x_le)
    elif tmax_ratio > 0 and tmax_ratio < 0.1:
        x_le = 0.85*x_open[0] + 1e-05
        z_le = 0.5*(zu_open[0] + zl_open[0])
        i_xumin = i_xlmin = 0
    else:
        # Computation of chordwise positions where dzdx = 0, searching only in the first half of the airfoil
        N = x_open.size//2
        dzudx = np.gradient(zu_open[:N],x_open[:N])
        x_zu_open_max = interp1d(dzudx[:N],x_open[:N],fill_value='extrapolate')(0)

        dzldx = np.gradient(zl_open[:N],x_open[:N])
        x_zl_open_max = interp1d(dzldx[:N],x_open[:N],fill_value='extrapolate')(0)

        i_xmax = min(np.where(x_open >= x_zu_open_max)[0][0],np.where(x_open >= x_zl_open_max)[0][0])
        if i_xmax == 0:
            i_xmax = N//2
        i_xumin = 2
        i_xlmin = 2
        # Construct x and z coordinates range
        x = np.concatenate((np.flipud(x_open[i_xumin:i_xmax]),x_open[i_xlmin:i_xmax]))
        z = np.concatenate((-np.flipud(zu_open[i_xumin:i_xmax]),-zl_open[i_xlmin:i_xmax]))

        # 1. Intersection of upper-lower curves
        i_z1 = i_xmax - i_xumin - 1
        i_z2 = i_z1 + 1
        if z[i_z2] < 0:
            zmax = 0
        else:
            zmax = 10*z[i_z2]
        if z[-1] < z[-2]:
            zmin = z[-1]
        else:
            zmin = z[0]
        z_int = np.linspace(zmin,zmax)
        xu_int = interp1d(z[0:i_z1],x[0:i_z1],fill_value='extrapolate')(z_int)
        xl_int = interp1d(z[i_z2:],x[i_z2:],fill_value='extrapolate')(z_int)

        # 2. Compute intersection
        z_inter = interp1d(xu_int-xl_int,z_int,fill_value='extrapolate')(0)
        x_inter = interp1d(z_int,xu_int,fill_value='extrapolate')(z_inter)

        # 3. Bisector vector
        dxudz_int = np.gradient(xu_int,z_int)
        dxldz_int = np.gradient(xl_int,z_int)
        dxudz_inter = interp1d(z_int,dxudz_int,fill_value='extrapolate')(z_inter)
        dxldz_inter = interp1d(z_int,dxldz_int,fill_value='extrapolate')(z_inter)

        z1 = z_inter
        x1 = x_inter
        dz = z[1] - z[0]
        z2 = z1 - dz
        if all(dxldz_int < 0):
            z3 = z2
            dxldz_inter = -dxldz_inter
        else:
            z2 = z1 - dz
            z3 = z1 + dz
        x2 = x1 - dxudz_inter*dz
        x3 = x1 + dxldz_inter*dz

        a = np.array([z2-z1,x2-x1])
        b = np.array([z3-z1,x3-x1])
        c = np.sqrt(np.dot(b,b)) * a + np.sqrt(np.dot(a,a)) * b
        k = 5e5  # norm factor to amplify bisector vector
        z4 = z1 + k*c[0]
        x4 = x1 + k*c[1]

        # 4. Compute intersection of bisector and chord-line between upper and lower sides
        N = 30
        chord_ul_z = np.linspace(z[i_z1],z[i_z2],N)
        chord_ul_x = np.linspace(x[i_z1],x[i_z2],N)
        bisector_x = interp1d([z1,z4],[x1,x4],fill_value='extrapolate')(chord_ul_z)
        bisector_z = interp1d([x1,x4],[z1,z4],fill_value='extrapolate')(bisector_x)
        z5 = interp1d(bisector_x-chord_ul_x,chord_ul_z,fill_value='extrapolate')(0)
        x5 = interp1d(bisector_z,bisector_x,fill_value='extrapolate')(z5)

        # 5. Compute leading edge point
        z_le = 0.5 * (z1 + z5)
        x_le = interp1d([z1,z5],[x1,x5],fill_value='extrapolate')(z_le)
        z_le = -z_le
        '''
        plt.figure()
        plt.scatter(z,x,color='b',label='base points')
        plt.plot(z_int,xu_int,color='r')
        plt.plot(z_int,xl_int,color='g')
        plt.scatter(z1,x1,color='m',label='inter')
        plt.scatter(z[i_z1],x[i_z1],color='r',label='z1')
        plt.scatter(z[i_z2],x[i_z2],color='r',label='z2')
        plt.scatter(z2,x2,color='y',label='P2')
        plt.scatter(z3,x3,color='g',label='P3')
        plt.scatter(-z_le,x_le,color='c',label='PLE')
        plt.xlabel('z')
        plt.ylabel('x')
        plt.legend()
        plt.show(block=True)
        '''
    # Generate new upper and lower sides
    xu_round = np.concatenate((np.array([x_le]),x_open[i_xumin:]))
    zu_round = np.concatenate((np.array([z_le]),zu_open[i_xumin:]))
    xl_round = np.concatenate((np.array([x_le]),x_open[i_xlmin:]))
    zl_round = np.concatenate((np.array([z_le]),zl_open[i_xlmin:]))

    return xu_round, zu_round, xl_round, zl_round

def generate_te(xu_open, zu_open, xl_open, zl_open):

    xmin = min(max(xu_open),max(xl_open))
    xmax = 1.2
    x_int = np.linspace(xmin,xmax,30)
    zu_int = interp1d(xu_open,zu_open,fill_value='extrapolate')(x_int)
    zl_int = interp1d(xl_open,zl_open,fill_value='extrapolate')(x_int)

    # Compute intersection
    x_te = interp1d(zu_int-zl_int,x_int,fill_value='extrapolate')(0)
    if x_te > xmin and x_te < xmax:
        z_te = interp1d(x_int,zu_int,fill_value='extrapolate')(x_te)
    else:
        x_te = xmin
        z_te = 0.5*(zu_open[-1] + zl_open[-1])

    # Concatenate
    xu_closed = np.concatenate((xu_open,np.array([x_te])))
    zu_closed = np.concatenate((zu_open,np.array([z_te])))
    xl_closed = np.concatenate((xl_open,np.array([x_te])))
    zl_closed = np.concatenate((zl_open,np.array([z_te])))

    '''
    plt.figure()
    plt.plot(xu_open,zu_open,label='Upperside open',color='g',alpha=0.7)
    plt.plot(xl_open,zl_open,label='Lowerside open',color='g',alpha=0.7)
    plt.plot(xu_closed,zu_closed,label='Upperside open',color='r')
    plt.plot(xl_closed,zl_closed,label='Lowerside open',color='r')
    plt.ylabel('z')
    plt.xlabel('x')
    plt.legend()
    '''
    return xu_closed, zu_closed, xl_closed, zl_closed




