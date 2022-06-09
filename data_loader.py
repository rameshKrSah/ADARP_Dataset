
import numpy as np
import sys

import utils as utl

def create_dataset(stress, not_stress, path=True, reshape=True, train_test=True, balance_classes=False, oversampling_method=None):
    """Create dataset given stress and not-stress data or path. 
    
    stress -- path or data of stress class
    not_stress -- path or data of not-stress class
    path -- whether values of first and seconds arguments are path or not (default True)
    reshape -- whether to reshape the data as (n_samples, window_length, n_channels) (default True)
    train_test -- whether to split the data into train and test sets (default True)
    balance_classes -- whether to balance the data between the classes (default True)
    oversampling_method -- initialized methods from imblearn.over_sampling such as RandomOverSampler, SMOTE, ADASYN

    """
    # if path load the datasets
    if path:
        stress_segments = utl.read_data(stress)
        not_stress_segments = utl.read_data(not_stress)
    else:
        stress_segments = stress
        not_stress_segments = not_stress
 
    # majority class undersampling: randomly drop samples from not-stress class to have equal number samples
    if balance_classes == True:
        not_stress_segments = utl.select_random_samples(not_stress_segments, stress_segments.shape[0])

    # concatenate the stress and not-stress data
    x = np.concatenate([stress_segments, not_stress_segments], axis=0)
    y = np.concatenate([
        np.ones(len(stress_segments), dtype=int),
        np.zeros(len(not_stress_segments), dtype=int)
    ], axis=0)

    # remove segments from memory
    del stress_segments
    del not_stress_segments
    print(f"X: {x.shape}, Y: {y.shape}")

    x_tr = x
    y_tr = y
    x_ts, y_ts = [], []

    # split into train, test
    if train_test:
        x_tr, x_ts, y_tr, y_ts = utl.split_into_train_val_test(x, y, test_split=0.3)     

    # SMOTE oversampling to balance the classes. Do this only for the training set
    if (oversampling_method != None) & (balance_classes == False):
        if len(x_tr.shape) == 2:
            x_tr, y_tr = oversampling_method.fit_resample(x_tr, y_tr)

        elif len(x_tr.shape) == 3:
            org_shape = x_tr.shape
            x_tr, y_tr = oversampling_method.fit_resample(x_tr.reshape(-1, org_shape[1] * org_shape[2]), y_tr)
            x_tr = x_tr.reshape(-1, org_shape[1], org_shape[2])

    # reshape is instructed
    if reshape:
        if len(x_tr.shape) == 2:
            x_tr = x_tr.reshape(-1, x_tr.shape[1], 1)
            if x_ts is np.ndarray:
                x_ts = x_ts.reshape(-1, x_ts.shape[1], 1)

        elif len(x_tr.shape) == 3:
            # the case for acceleration data with 3 channels
            x_tr = x_tr.transpose([0, 2, 1])
            if x_ts is np.ndarray:
                x_ts = x_ts.transpose([0, 2, 1])

    return x_tr, y_tr, x_ts, y_ts
    
def load_wesad_data(baseline_path, amusement_path, stressed_path, combine_amusement=False, reshape=True, train_test=True):
    """Get training and testing set for the WESAD dataset. 
    
    baseline_path: path to the baseline data
    amusement_path: path to the amusement data
    stressed_path: path to the stressed data

    combine_amusement: whether to combine amusement class or not; default false
    reshape: whether to reshape the 2d data into 3d or not; default true
    train_test; whether to split the data into training and testing set or not; default true

    Return features and labels.
    """
    X, Y = combine_class_data(baseline_path, amusement_path, stressed_path, combine_amusement)

    if reshape:
        X = X.reshape(-1, X.shape[1], 1)

    if train_test:
        x_train, x_test, y_train, y_test = utl.split_into_train_test(X, Y, test_split=0.25)
        return x_train, x_test, y_train, y_test, utl.get_hot_labels(y_train), utl.get_hot_labels(y_test)
    else:
        return X, Y


def combine_class_data(baseline_path, amusement_path, stressed_path, include_amusement = False):
    """
        Load the data for different stress class for the WESAD dataset and return X, Y. 
        If amusement is included into the baseline class, the labels assigned to amusement is 0 
        same as that of baseline class. 
            
        baseline_path (string): path to baseline data, 
        amusement_path (string): path to the amusement data,
        stressed_path (string): path to stressed data
        include_amusement (Boolean): whether to include amusement data into baseline or not. 
        By default amusement data is not included into baseline.
            
        X, Y : NumPy arrays.
    """
    
    # load the segments
    baseline_segments = utl.read_data(baseline_path)
    stress_segments = utl.read_data(stressed_path)
    
    # combine the baseline and stress segments
    X = np.concatenate([baseline_segments, stress_segments], axis = 0)
    Y = np.concatenate([np.zeros(baseline_segments.shape[0], dtype=int), 
                       np.ones(stress_segments.shape[0], dtype=int)
                       ])
    # include the amusement data is indicated
    if include_amusement:
        amusement_segments = utl.read_data(amusement_path)
        X = np.concatenate([X, amusement_segments])
        Y = np.concatenate([Y, np.zeros(amusement_segments.shape[0], dtype=int)])
    
    return X, Y
    
