import math
import re
from random import randint

def read_case_setup(launch_filepath):
    file = open(launch_filepath, 'r')
    data = file.read()
    data = re.sub('%.*\n','', data)

    class setup:
        pass

    casedata = setup()
    casedata.case_dir = None
    casedata.analysis = dict.fromkeys(['case_ID','type', 'import'], None)
    casedata.design_parameters_train = dict()
    casedata.design_parameters_des = dict()
    casedata.training_parameters = dict()
    casedata.img_resize = [None,None]
    casedata.img_processing = {'rotation': [None, None, None, None],
                               'translation': [None, None, None],
                               'zoom': [None, None],
                               'filter': [None, None, None, None, None],
                               'flip': [None, None]
                               }
    casedata.samples_generation = {'n_samples': None}
    casedata.activation_plotting = {'n_samples': None, 'n_cols': None, 'rows2cols_ratio': None}
    casedata.data_augmentation = [None, None]

    ############################################### Data directory #####################################################
    match = re.search('DATADIR\s*=\s*(.*).*', data)
    if match:
        casedata.case_dir = match.group(1)

    ################################################## Analysis ########################################################
    casedata.analysis['case_ID'] = randint(1,9999)
    # Type of analysis
    match = re.search('TYPEANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['type'] = str.lower(match.group(1))

    # Conditional analysis
    match = re.search('CONDITIONALANALYSIS\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['conditional'] = int(match.group(1))

    # Type of airfoil analysis (camber/thickness)
    match = re.search('AIRFOILANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['airfoil_analysis'] = str.lower(match.group(1))

    # Type of airfoil analysis (camber/thickness)
    match = re.search('AIRFOILANALYSIS_DES\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['airfoil_analysis_des'] = str.lower(match.group(1))

    # Import
    match = re.search('IMPORTMODEL\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['import'] = int(match.group(1))

    ## Dataset augmentation
    match = re.search('AUGDATA\s*=\s*(\d).*', data)
    if match:
        casedata.data_augmentation[0] = int(match.group(1))
        match_factor = re.search('AUGDATASIZE\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.data_augmentation[1] = float(match_factor.group(1))

    ############################################# Training parameters ##################################################
    # Latent dimension
    match = re.search('LATENTDIM\s*=\s*(.*)', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['latent_dim'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Encoder hidden dimension
    match = re.search('ENCHIDDENDIM\s*=\s*\((.*)\).*', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['enc_hidden_layers'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Decoder hidden dimension
    match = re.search('DECHIDDENDIM\s*=\s*\((.*)\).*', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['dec_hidden_layers'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Training dataset size
    match = re.search('TRAINSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['train_size'] = 0.75
        else:
            casedata.training_parameters['train_size'] = float(match.group(1))

    # Learning rate
    match = re.search('LEARNINGRATE\s*=\s*(.*)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        casedata.training_parameters['learning_rate'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]

    # L2 regularizer
    match = re.search('L2REG\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['l2_reg'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['l2_reg'] = 0.0

    # L1 regularizer
    match = re.search('L1REG\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['l1_reg'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['l1_reg'] = 0.0

    # Dropout
    match = re.search('DROPOUT\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['dropout'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['dropout'] = 0.0

    # Number of epochs
    match = re.search('EPOCHS\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['epochs'] = 1
        else:
            casedata.training_parameters['epochs'] = int(match.group(1))

    # Batch size
    match = re.search('BATCHSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['batch_size'] = None
        else:
            casedata.training_parameters['batch_size'] = int(match.group(1))

    # Activation function
    match = re.search('ACTIVATION\s*=\s*((\w+)\s*(,?\s*\w+,?)*)\s*\n+.*', data)
    if match:
        matches = re.findall('(\w+)',match.group(1))
        if matches:
            if len(matches) == 1:
                casedata.training_parameters['activation'] = str.lower(matches[0])
            else:
                casedata.training_parameters['activation'] = [str.lower(item) for item in matches]

    ############################################### Design parameters ##################################################
    # Number of samples
    match = re.search('NSAMPLESGEN\s*=\s*(\d+|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.samples_generation['n_samples'] = 1
        else:
            casedata.samples_generation['n_samples'] = int(match.group(1))

    # Boolean to indicate whether to supply or not a set of latent vectors
    match = re.search('SUPPLYLATENT\s*=\s*(\d|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.samples_generation['supply_latent'] = None
        else:
            casedata.samples_generation['supply_latent'] = int(match.group(1))

    # Design parameters (training)
    match = re.search('DPARAMETERS_TRAIN\s*=\s*\(\s*LERADIUS\s*\,\s*(\d)\s*\,\s*TEANGLE\s*\,\s*(\d)\s*\,'
                      '\s*TMAX\s*\,\s*(\d)\s*\,\s*ZMAX\s*\,\s*(\d)\s*\,\s*ZMIN\s*\,\s*(\d)\s*\,\s*ZLE\s*\,\s*(\d)\s*\,'
                      '\s*ZTE\s*\,\s*(\d)\s*\,\s*DZDX_C\s*\,\s*(\d)\s*\,\s*DZDX_T\s*\,\s*(\d)\s*\).*',data)
    if match:
        casedata.design_parameters_train['parameters'] = []
        if match.group(1) == '1':
            casedata.design_parameters_train['parameters'].append('leradius')
        if match.group(2) == '1':
            casedata.design_parameters_train['parameters'].append('teangle')
        if match.group(3) == '1':
            casedata.design_parameters_train['parameters'].append('tmax')
        if match.group(4) == '1':
            casedata.design_parameters_train['parameters'].append('zmax')
        if match.group(5) == '1':
            casedata.design_parameters_train['parameters'].append('zmin')
        if match.group(6) == '1':
            casedata.design_parameters_train['parameters'].append('zle')
        if match.group(7) == '1':
            casedata.design_parameters_train['parameters'].append('zte')
        if match.group(8) == '1':
            casedata.design_parameters_train['parameters'].append('dzdx_c')
        if match.group(9) == '1':
            casedata.design_parameters_train['parameters'].append('dzdx_t')

    # DZDX controlpoints (training)
    match = re.search('XDZDX_CP_TRAIN\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.design_parameters_train['xdzdx_cp'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.design_parameters_train['xdzdx_cp'] = None

    # Slope design parameters standardisation (training)
    casedata.design_parameters_train_std = dict.fromkeys(casedata.design_parameters_train['parameters'])
    if casedata.design_parameters_train['xdzdx_cp'] != None:
        if 'dzdx_c' in casedata.design_parameters_train_std:
            del casedata.design_parameters_train_std['dzdx_c']
            casedata.design_parameters_train_std['xdzdx'] = ('camber',casedata.design_parameters_train['xdzdx_cp'])
        elif 'dzdx_t' in casedata.design_parameters_train_std:
            del casedata.design_parameters_train_std['dzdx_t']
            casedata.design_parameters_train_std['xdzdx'] = ('thickness',casedata.design_parameters_train['xdzdx_cp'])
    casedata.design_parameters_train = casedata.design_parameters_train_std
    del casedata.design_parameters_train_std

    # Design parameters (generation)
    match = re.search('DPARAMETERS_DES\s*=\s*\(\s*LERADIUS\s*\,\s*(.*|NONE)\s*\,\s*TEANGLE\s*\,\s*(.*|NONE)\s*\,'
                      '\s*TMAX\s*\,\s*(.*|NONE)\s*\,\s*ZMAX\s*\,\s*(.*|NONE)\s*\,\s*ZMIN\s*\,\s*(.*|NONE)\s*\,'
                      '\s*ZLE\s*\,\s*(.*|NONE)\s*\,\s*ZTE\s*\,\s*(.*|NONE)\s*\,\s*DZDX_C\s*\,\s*\(*(.*|NONE)\s*\)*\s*\,'
                      '\s*DZDX_T\s*\,\s*\(*(.*|NONE)\)*\s*\).*', data)
    if match:
        if match.group(1) != 'NONE':
            casedata.design_parameters_des['leradius'] = float(match.group(1))
        if match.group(2) != 'NONE':
            casedata.design_parameters_des['teangle'] = math.radians(float(match.group(2)))
        if match.group(3) != 'NONE':
            casedata.design_parameters_des['tmax'] = float(match.group(3))
        if match.group(4) != 'NONE':
            casedata.design_parameters_des['zmax'] = float(match.group(4))
        if match.group(5) != 'NONE':
            casedata.design_parameters_des['zmin'] = float(match.group(5))
        if match.group(6) != 'NONE':
            casedata.design_parameters_des['zle'] = float(match.group(6))
        if match.group(7) != 'NONE':
            casedata.design_parameters_des['zte'] = float(match.group(7))
        if match.group(8) != 'NONE':
            dzdx_c_matches = re.findall('(\-*\d+\.*\d+)', match.group(8))
            if len(dzdx_c_matches) != 0:
                casedata.design_parameters_des['dzdx_c'] = [float(item) for item in dzdx_c_matches]
            else:
                casedata.design_parameters_des['dzdx_c'] = None
        if match.group(9) != 'NONE':
            dzdx_t_matches = re.findall('(\-*\d+\.*\d+)', match.group(9))
            if len(dzdx_t_matches) != 0:
                casedata.design_parameters_des['dzdx_t'] = [float(item) for item in dzdx_t_matches]
            else:
                casedata.design_parameters_des['dzdx_t'] = None

    ######################################## Image processing parameters ###############################################
    # Image resize
    match_dist = re.search('IMAGERESIZE\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
    if match_dist:
        casedata.img_resize[0] = int(match_dist.group(1))
        casedata.img_resize[1] = int(match_dist.group(2))
        casedata.img_resize = tuple(casedata.img_resize)

    # Rotation
    match = re.search('ROTATION\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['rotation'][0] = int(match.group(1))
        match_angle = re.search('ROTATIONANGLE\s*=\s*([\+|\-]?\d+\.?\d*).*', data)
        if match_angle:
            casedata.img_processing['rotation'][1] = float(match_angle.group(1))
        match = re.search('ROTATIONCENTER\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
        if match:
            if match.group(1) != 'NONE':
                casedata.img_processing['rotation'][2] = int(match.group(1))
            elif match.group(2) != 'NONE':
                casedata.img_processing['rotation'][3] = int(match.group(2))

    # Translation
    match = re.search('TRANSLATION\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['translation'][0] = int(match.group(1))
        match_dist = re.search('TRANSLATIONDIST\s*=\s*\(([\+|\-]?\d+|NONE)\,+([\+|\-]?\d+|NONE)\).*', data)
        if match_dist:
            casedata.img_processing['translation'][1] = float(match_dist.group(1))
            casedata.img_processing['translation'][2] = float(match_dist.group(2))

    # Zoom
    match = re.search('ZOOM\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['zoom'][0] = int(match.group(1))
        match_factor = re.search('ZOOMFACTOR\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.img_processing['zoom'][1] = float(match_factor.group(1))
    # Filter
    match = re.search('FILTER\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['filter'][0] = int(match.group(1))
        match_type = re.search('FILTERTYPE\s*=\s*(\w+).*', data)
        casedata.img_processing['filter'][1] = str.lower(match_type.group(1))
        if match_type:
            if str.lower(match_type.group(1)) == 'gaussian':
                filter_param = re.search(
                    'FILTERPARAM\s*=\s*\(\s*SIZE\s*\,\s*(\d+|NONE)\s*\,\s*SIGMA\s*\,\s*(\d+|NONE)\s*\).*', data)
                casedata.img_processing['filter'][2] = int(filter_param.group(1))
                casedata.img_processing['filter'][3] = int(filter_param.group(2))
        elif str.lower(match_type.group(1)) == 'bilateral':
            filter_param = re.search(
                'FILTERPARAM\s*=\s*\(\s*(D)\s*\,\s*(\d+|NONE)\s*\,\s*SIGMACOLOR\s*\,\s*(\d+|NONE)\s*SIGMASPACE\s*\,\s*(\d+|NONE)\s*\).*',
                data)
            casedata.img_processing['filter'][2] = int(filter_param.group(1))
            casedata.img_processing['filter'][3] = int(filter_param.group(2))
            casedata.img_processing['filter'][4] = int(filter_param.group(3))

        # Flip
        match = re.search('FLIP\s*=\s*(\d).*', data)
        if match:
            casedata.img_processing['flip'][0] = int(match.group(1))
            match_type = re.search('FLIPTYPE\s*=\s*(\w+).*', data)
            if match_type:
                casedata.img_processing['flip'][1] = str.lower(match_type.group(1))

    ######################################### Activation plotting parameters ###########################################
    # Number of samples
    match = re.search('NSAMPLESACT\s*=\s*(\d+).*', data)
    if match:
        casedata.activation_plotting['n_samples'] = int(match.group(1))

    # Number of columns
    match = re.search('NCOLS\s*=\s*(\d+).*', data)
    if match:
        casedata.activation_plotting['n_cols'] = int(match.group(1))

    # Rows-to-columns figure ratio
    match = re.search('ROWS2COLS\s*=\s*(\d+).*', data)
    if match:
        casedata.activation_plotting['rows2cols_ratio'] = int(match.group(1))

    return casedata

def read_case_logfile(log_filepath):

    file = open(log_filepath, 'r')
    data = file.read()
    data = re.sub('%.*\n','', data)

    class setup:
        pass

    casedata = setup()
    casedata.analysis = dict.fromkeys(['case_ID','type', 'import'], None)
    casedata.training_parameters = dict()
    casedata.design_parameters_train = dict()
    casedata.img_resize = [None,None]

    ################################################## Analysis ########################################################
    # Case ID
    match = re.search('CASE ID\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        casedata.analysis['case_ID'] = int(match.group(1))

    # Type of analysis
    match = re.search('ANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['type'] = str.lower(match.group(1))

    # Type of airfoil analysis (camber/thickness)
    match = re.search('AIRFOIL ANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['airfoil_analysis'] = str.lower(match.group(1))

    # Type of airfoil analysis (camber/thickness)
    match = re.search('DESIGN AIRFOIL ANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['airfoil_analysis_des'] = str.lower(match.group(1))

    # Image shape
    match = re.search('INPUT SHAPE\s*=\s*\((.*)\).*', data)
    if match:
        casedata.img_size = [int(item) for item in re.findall('\d+',match.group(1))]
        casedata.img_size = tuple(casedata.img_size)

    # Import
    match = re.search('IMPORTED MODEL\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['import'] = int(match.group(1))

    ############################################# Training parameters ##################################################
    # NN architecture
    match = re.search('ARCHITECTURE\s*=\s*(\w+).*', data)
    if match:
        casedata.training_parameters['architecture'] = str.lower(match.group(1))

    # Latent dimension
    match = re.search('LATENT DIMENSION\s*=\s*\[*(.*)\]*', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['latent_dim'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Encoder hidden dimension
    match = re.search('ENCODER HIDDEN LAYERS\s*=\s*\[(.*)\].*', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['enc_hidden_layers'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Decoder hidden dimension
    match = re.search('DECODER HIDDEN LAYERS\s*=\s*\[(.*)\].*', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['dec_hidden_layers'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Training dataset size
    match = re.search('TRAINING SIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['train_size'] = 0.75
        else:
            casedata.training_parameters['train_size'] = float(match.group(1))

    # Learning rate
    match = re.search('LEARNING RATE\s*=\s*\[*(.*)\]*', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        casedata.training_parameters['learning_rate'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]

    # L2 regularizer
    match = re.search('L2 REGULARIZER\s*=\s*\[*(.*|NONE)\]*', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['l2_reg'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['l2_reg'] = 0.0

    # L1 regularizer
    match = re.search('L1 REGULARIZER\s*=\s*\[*(.*|NONE)\]*', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['l1_reg'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['l1_reg'] = 0.0

    # Dropout
    match = re.search('DROPOUT\s*=\s*\[*(.*|NONE)\]*', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['dropout'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['dropout'] = 0.0

    # Number of epochs
    match = re.search('NUMBER OF EPOCHS\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['epochs'] = 1
        else:
            casedata.training_parameters['epochs'] = int(match.group(1))

    # Batch size
    match = re.search('BATCH SIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['batch_size'] = None
        else:
            casedata.training_parameters['batch_size'] = int(match.group(1))

    # Activation function
    match = re.search('ACTIVATION\s*=\s*\[*(.*)\]*\s*.*', data)
    if match:
        matches = re.findall('(\w+)',match.group(1))
        if matches:
            if len(matches) == 1:
                casedata.training_parameters['activation'] = str.lower(matches[0])
            else:
                casedata.training_parameters['activation'] = [str.lower(item) for item in matches]

    ############################################### Design parameters ##################################################
    # Design parameters (training)
    match = re.search('DESIGN PARAMETERS TRAIN\s*=\s*\[((\'\w+\')\s*(,?\s*\'\w+\',?)*)\]\s*.*',data)
    if match:
        matches = re.findall('(\w+)',match.group(1))
        if matches:
            if len(matches) == 1:
                casedata.design_parameters_train['parameters'] = str.lower(matches[0])
            else:
                casedata.design_parameters_train['parameters'] = [str.lower(item) for item in matches]

    # DZDX controlpoints (training)
    match = re.search('XDZDX CONTROLPOINTS TRAIN\s*=\s*\[(.*)\]', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.design_parameters_train['xdzdx_cp'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.design_parameters_train['xdzdx_cp'] = None

    # Design parameters standardisation (training)
    casedata.design_parameters_train_std = dict.fromkeys(casedata.design_parameters_train['parameters'])
    if casedata.design_parameters_train['xdzdx_cp'] != None:
        if 'xdzdx_c' in casedata.design_parameters_train_std:
            del casedata.design_parameters_train_std['xdzdx_c']
            casedata.design_parameters_train_std['xdzdx'] = ('camber',casedata.design_parameters_train['xdzdx_cp'])
        elif 'xdzdx_t' in casedata.design_parameters_train_std:
            del casedata.design_parameters_train_std['xdzdx_t']
            casedata.design_parameters_train_std['xdzdx'] = ('thickness',casedata.design_parameters_train['xdzdx_cp'])
    casedata.design_parameters_train = casedata.design_parameters_train_std
    del casedata.design_parameters_train_std


    return casedata