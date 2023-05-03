import os
import numpy as np
import tensorflow as tf


def swish(x, beta=1):
    return x * tf.keras.backend.sigmoid(beta * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sampling(args):
    """Returns sample from a distribution N(args[0], diag(args[1]))
    Sampling from the distribution q(t|x) = N(t_mean, exp(t_log_var)) with reparametrization trick.

    The sample should be computed with reparametrization trick.

    The inputs are tf.Tensor
        args[0]: (batch_size x latent_dim) mean of the desired distribution
        args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution

    Returns:
        A tf.Tensor of size (batch_size x latent_dim), the samples.
    """
    t_mean, t_log_var = args

    epsilon = tf.random.normal(tf.shape(t_log_var),name="epsilon")

    return t_mean + epsilon * tf.exp(t_log_var/2)


#@tf.function
def loss_function(x, x_decoded, t_mean, t_log_var):
    """Returns the value of negative Variational Lower Bound

    The inputs are tf.Tensor
        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
        x_decoded: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)

    Returns:
        A tf.Tensor with one element (averaged across the batch), VLB
    """

    loss = tf.reduce_sum(x * tf.math.log(x_decoded + 1e-19) + (1 - x) * tf.math.log(1 - x_decoded + 1e-19), axis=1)
    regularisation = 0.5 * tf.reduce_sum(-t_log_var + tf.math.exp(t_log_var) + tf.math.square(t_mean) - 1, axis=1)

    return tf.reduce_mean(-loss + regularisation, axis=0)

def conv2D_block(X, num_channels, f, p, s, dropout, **kwargs):
    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    if p != 0:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
    else:
        net = X
    net = tf.keras.layers.Conv2D(num_channels, kernel_size=f, strides=s, padding='valid',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)

    if activation == 'leakyrelu':
        rate = 0.3
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net


def get_padding(f, s, nin, nout):
    padding = []
    for i in range(f.__len__()):
        p = int(np.floor(0.5 * ((nout - 1) * s[i] + f[i] - nin)))
        nchout = int(np.floor((nin + 2 * p - f[i]) / s[i] + 1))
        if nchout != nout:
            padding.append(p + 1)
        else:
            padding.append(p)

    return padding


def encoder_lenet_c(input_dim, latent_dim, hidden_layers, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
        
    in_shape = (input_dim[1],input_dim[0])
    input_shape = tuple(list(in_shape) + [1])
    X_input = tf.keras.layers.Input(shape=(np.prod(input_shape),))
    net = tf.keras.layers.Reshape(input_shape)(X_input)
    net = conv2D_block(net,num_channels=32,f=5,p=0,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                               'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    for layer in hidden_layers:
        net = tf.keras.layers.Dense(units=layer,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation(activation)(net)
        net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=2*latent_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)

    encoder_c = tf.keras.Model(inputs=X_input,outputs=net,name='encoder_lenet_c')
    
    return encoder_c

def decoder_lenet_c(output_dim, latent_dim, hidden_layers, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    out_shape = np.prod(output_dim)
    adap_dim = int(np.sqrt(hidden_layers))
    adap_layer_shape = tuple(list((adap_dim,adap_dim)) + [1])
        
    X_input = tf.keras.layers.Input(shape=latent_dim)
    for layer in hidden_layers:
        net = tf.keras.layers.Dense(units=layer,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X_input)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation(activation)(net)
        net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=adap_dim**2,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Reshape(adap_layer_shape)(net)
    net = conv2D_block(net,num_channels=32,f=5,p=0,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                                  'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=out_shape,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('sigmoid')(net)

    decoder_c = tf.keras.Model(inputs=X_input,outputs=net,name='decoder_lenet_c')
    
    return decoder_c
    
def encoder_c(input_dim, hidden_dim, latent_dim, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
    '''
    Encoder network.
    Returns the mean and the log variances of the latent distribution
    '''
        
    encoder_c = tf.keras.Sequential(name='encoder_c')
    encoder_c.add(tf.keras.Input(shape=(input_dim,)))
    for hidden_layer_dim in hidden_dim:
        encoder_c.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))
        encoder_c.add(tf.keras.layers.BatchNormalization())
        if activation == 'leakyrelu':
            rate = 0.3
            encoder_c.add(tf.keras.layers.LeakyReLU(rate))
        else:
            encoder_c.add(tf.keras.layers.Activation(activation))
        encoder_c.add(tf.keras.layers.Dropout(dropout))
    encoder_c.add(tf.keras.layers.Dense(2*latent_dim,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))

    return encoder_c
    
    
def decoder_c(latent_dim, hidden_dim, output_dim, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
    '''
    Decoder network
    It assumes that the image is a normalized black & white image so each pixel ranges between 0 and 1
    '''
    
    decoder_c = tf.keras.Sequential(name='decoder_c')
    decoder_c.add(tf.keras.Input(shape=(latent_dim,)))
    for hidden_layer_dim in hidden_dim:
        decoder_c.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))
        decoder_c.add(tf.keras.layers.BatchNormalization())
        if activation == 'leakyrelu':
            rate = 0.3
            decoder_c.add(tf.keras.layers.LeakyReLU(rate))
        else:
            decoder_c.add(tf.keras.layers.Activation(activation))
        decoder_c.add(tf.keras.layers.Dropout(dropout))
    decoder_c.add(tf.keras.layers.Dense(output_dim,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg),activation='sigmoid'))
    
    return decoder_c

def VAEC(input_dim, n_dpar, latent_dim, encoder_hidden_layers, decoder_hidden_layers, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0,
        activation='relu', mode='train'):

    in_shape_unrolled = np.prod(input_dim)

    ## DEFINE MODEL ##
    if mode == 'train':
        # Encoder
        x = tf.keras.Input(shape=(in_shape_unrolled,))
        des_vec = tf.keras.Input(shape=(n_dpar,))
        e = encoder_c(in_shape_unrolled+n_dpar,encoder_hidden_layers,latent_dim,l2_reg,l1_reg,dropout,activation)
        h = e(tf.keras.layers.concatenate([x,des_vec]))
        
        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        d = decoder_c(latent_dim+n_dpar,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
        x_decoded = d(tf.keras.layers.concatenate([t,des_vec]))

        # Declare inputs/outputs for the model
        input = x
        output = x_decoded
    elif mode == 'sample':
        # Decoder
        t = tf.keras.Input(shape=(latent_dim,))
        des_vec = tf.keras.Input(shape=(n_dpar,))
        t_mean = tf.zeros_like(t)
        t_log_var = tf.zeros_like(t)
        d = decoder_c(latent_dim+n_dpar,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
        x_decoded = d(tf.keras.layers.concatenate([t,des_vec]))

        # Declare inputs/outputs for the model
        input = [t,des_vec]
        output = x_decoded
        x = x_decoded

    loss = loss_function(x,x_decoded,t_mean,t_log_var)
    model = tf.keras.Model(input,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x,y: loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model
