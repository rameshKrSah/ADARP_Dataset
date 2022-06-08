import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import pprint
import tensorflow as tf
from tensorflow import keras

def check_nan_finiteness(npArray):
  """Check whether there are NaN and infinity values in the Numpy array"""
  print(f"Is NaN {np.any(np.isnan(npArray))}")
  print(f"Is Finite {np.all(np.isfinite(npArray))}")

def remove_nan_infiniteness(npArray):
    """Replace NaN with zero and infinity with large value"""
    return np.nan_to_num(npArray)

def difference_of_list(list1: list, list2: list) -> list:
    """ Given two list, return the difference of the two lists. The difference constains the elements in list1 that are not in list2
    list1: A list
    list2: Another list

    return list
    """
        
    return list(set(list1).difference(set(list2)))

class PlotLosses(keras.callbacks.Callback):
    """
        Keras Callback to plot the training loss and accuracy of the training and validation sets.
    """
    def __init__(self, metrics):
        self.i = 0
        self.epoch = []
        self.metrics_names = metrics
        self.metrics = {}

        for name in self.metrics_names:
            self.metrics[name] = []
            self.metrics['val_'+name] = []

        self.fig = plt.figure()
        self.logs = []
        self.tf_version = float(tf.__version__[:3])

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.epoch.append(self.i)

        # extract the metrics from the logs
        for name in self.metrics_names:
            # get the training metric
            tr_value = logs.get(name)

            # get the validation metric
            try:
                val_value = logs.get('val_'+name)
            except:
                val_value = 0.0

            # store the metric: for f1-score we get two values one for each class. 
            # We only want the value for the positive class
            self.metrics[name].append(tr_value)
            self.metrics['val_'+name].append(val_value)

        self.i += 1
        f, axes = plt.subplots(len(self.metrics_names), 1, sharex=True, 
                               figsize=(12, 4 * len(self.metrics_names)))
        clear_output(wait=True)
        
        for name, ax in zip(self.metrics_names, axes):
            ax.plot(self.epoch, self.metrics.get(name), label=name)
            ax.plot(self.epoch, self.metrics.get('val_'+name), label="val "+name)
            ax.legend()

        axes[-1].set_xlabel("Epoch")
        plt.show()
		
		
def print_confusion_matrix(confusion_matrix, class_names, activities, 
  figsize = (12, 6), fontsize=10):
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the output figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    fig = fig = plt.gcf()
    heatmap.yaxis.set_ticklabels(activities, rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(activities, rotation=90, ha='right', fontsize=fontsize)
    plt.show()
	
	
def get_features_labels_from_df(data_df, shape_y, shape_z):
    """
	    Given a dataframe with class as column, separate the features and class label
	    and normalize the feature with min-max scaler and encode label as one-hot 
	    vector.

        Arguments:
        data_df (pandas DataFrame): dataframe
        shape_y (int) : Number of channels for the sensor data
        shape_z (int) : Length of the window segment

        Returns:
        Normalized features in the range (-1.0, 1.0), label, and one hot encoded label
    """
    labels = data_df['Class'].values.astype(int)
    features = data_df.drop(['Class'], axis = 1).values
    
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    features = scaler.fit_transform(features)
    
    features = features.reshape(-1, shape_y, shape_z)
    features = np.transpose(features, (0, 2, 1))
	
    labels_one_hot = keras.utils.to_categorical(labels, np.max(labels)+1)
    
    return features, labels, labels_one_hot	
	
def min_max_scale(data):
  """
    Min-Max scale the data in the range [-1.0, 1.0]
    The data is expected to have the shape (n_samples, segment_length, n_channels)
  
    Return the scaled data in the original shape.
  """
  _, segment_length, n_channels = data.shape

  # flatten the data
  features = data.reshape(-1, segment_length * n_channels)

  # scale the data
  scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
  features = scaler.fit_transform(features)
  
  # reshape the data
  features = features.reshape(-1, n_channels, segment_length)
  features = np.transpose(features, (0, 2, 1))

  return features
  
def standard_scaler(data):
  """ Normalize the data to have zero mean and unit standard devication
    The data is expected to have the shape (n_samples, segment_length, n_channels)
  
    Return the scaled data in the original shape.
  """
  _, segment_length, n_channels = data.shape

  # flatten the data
  features = data.reshape(-1, segment_length * n_channels)

  # scale the data
  scaler = StandardScaler(with_mean=False, with_std=False)
  features = scaler.fit_transform(features)
  
  # reshape the data
  features = features.reshape(-1, n_channels, segment_length)
  features = np.transpose(features, (0, 2, 1))

  return features
	
def get_cnn_model(input_shape, n_output_classes, learning_rate):
    """ 
        Returns a 1D CNN model with arch 100 - 50 - GlobalMaxPool1D - 64 - Dropout(0.3) - n_classes. 
        We have used this 1D CNN model extensively in Adversarial research projects.

        Arguments: 
        input_shape (tuple) : Shape of the input
        n_output_classes (int) : number of output classes 
        learning_rate (float) : learning rate for the Adam optimizer

        Returns: 
        A 1D CNN model ready for training, with categorical cross entropy loss and Adam optimizer.
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters = 100, kernel_size = (10), strides = 2, activation = tf.nn.relu, input_shape = input_shape),
        keras.layers.Conv1D(filters = 50, kernel_size = (5), strides = 1, activation = tf.nn.relu),
        keras.layers.GlobalMaxPool1D(),
        #keras.layers.Flatten(),
        keras.layers.Dense(units = 64, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        keras.layers.Dense(units = n_output_classes, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(learning_rate=learning_rate), 
                      metrics = ['accuracy'])
    
    return temp_model


def save_data(path, data):
    """
        Given a path and data, save the data to the path as a pickle file.

        Arguments:
        path (string) : file path with .pkl extension
        data : data values; can be a single container or multiple containers
    """
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def read_data(path, n_vaues=None):
    """
        Given a path, read the file and return the contents.

        Arguments:
        path (string) : File path with .pkl extension
        n_values (int) : Number of containers expected to be read. 
    """
    
    f = open(path, "rb")
    d = pickle.load(f)
    f.close()
    return d

def stylize_axis(ax, xticks=True, yticks=False, top_right_spines=True,
                    bottom_left_spines=False):
    """
        Given an axis, stylize it by removing ticks and spines. Default choice for
        ticks and spines are given. Modify as needed.

        Arguments:
        ax (matplotlib.axes.Ax): matplotlib axis
        xticks (Boolean): whether to make xticks visible or not (True by Default)
        yticks (Boolean): whether to make yticks visible or not (False by Default)
        top_right_spines (Boolean): whether to make top_right_spines visible or not (True by Default)
        bottom_left_spines (Boolean): whether to make bottom_left_spines visible or not (False by Default)

    """
    if xticks:
        ax.set_xticks([])

    if yticks:
        ax.set_yticks([])

    if top_right_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if bottom_left_spines:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


def print_metrics(met_dict):
    """
    	Given a metrics dictionary, print the values. 
    """
    print("Loss: {:.3f}".format(met_dict['Loss']))
    print("Accuracy: {:.3f} %".format(met_dict['Accuracy'] * 100))

    print("Precision score: {:.3f}".format(met_dict['Precision']))
    print("Recall score: {:.3f}".format(met_dict["Recall"]))
    print("F1 score: {:.3f}".format(met_dict['F1 Score']))
    print("ROC AUC: {:.3f}".format(met_dict['ROC AUC']))

def precision_recall_f1_score(y_true, y_pred):
  """ Compute precision, recall, and f1 score given y and y predicted.
    y_true and y_pred are labels (not hot encoded)
    Return a dictionary containing Precision, Recall, and F1 Score
  """
  # whether binary or multi-class classification
  if len(np.unique(y_true)) == 2:
    average_case = 'binary'
  else:
    average_case = 'macro'

  recall = recall_score(y_true, y_pred, average=average_case)
  precision = precision_score(y_true, y_pred, average=average_case)
  print(f"Precision {precision} \nRecall {recall}")

  f1_score_cal = f1_score(y_true, y_pred, average=average_case)
  print("F1 score {:.3f}, with formula {:.3f}".format(f1_score_cal,
        2 * ((precision * recall) / (precision + recall))))

  return {'Precision': precision, 'Recall': recall, 'F1 Score': f1_score_cal}


def compute_performance_metrics(model, x, y, metric_names):
    """
        Given a model (TensorFlow) and (x, y), we compute accuracy, loss, True Positive, False Negative,
        False Positive, True Negative, Recall, Precision, f1 score, Average Precision Recall, ROC AUC, 
        and classification report.

        Arguments:
            model: tensorflow model
            x: feature vector
            y: label vector (one hot encoded)

        Returns: A dictionary containint, Accuracy, Loss, True Positive, False Positive, False Negative, 
                True Negative, Recall, Precision, f1 score, roc_auc_score
    """
    y_true = np.argmax(y, axis=1)
    if len(np.unique(y_true)) > 2:
      print("This only works for binary classification")
      return {}

    # get the metrics  
    metrics = model.evaluate(x, y)
    rt = dict()
    for name, val in zip(metric_names, metrics):
      rt[name] = val
  
    # the loss is always at first position and accuracy the second
    loss, acc = metrics[0], metrics[1] * 100 
    print("Accuracy {:.3f}, Loss {:.3f}".format(acc, loss))

    y_probs = model.predict(x)
    y_pred = np.argmax(y_probs, axis=1)

    tp, fp, tn, fn = (0, 0, 0, 0)

    try:
      # we can only do this in binary case
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
      print("Not a binary classification problem")
    
    print("True Positive ", tp)
    print("False Positive ", fp)
    print("True Negative ", tn)
    print("False Negative ", fn)

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print("Recall {:.3f}, with formula {:.3f}".format(recall, (tp / (tp + fn))))
    print("Precision {:.3f}, with formula {:.3f}".format(precision, (tp / (tp + fp))))

    f1_score_cal = f1_score(y_true, y_pred)
    print("F1 score {:.3f}, with formula {:.3f}".format(f1_score_cal,
           2 * ((precision * recall) / (precision + recall))))

    print("Average precision score {:.3f}".format(average_precision_score(y_true, y_pred)))

    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC Score {:.3f}".format(roc_auc))
    
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    pprint.pprint(clf_report)
    # print(clf_report.keys())

    rt_dict = {'Accuracy': acc,
            'Loss': loss,
            'True Positive': tp, 
            'False Positive': fp, 
            'True Negative': tn, 
            'False Negative': fn,
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1_score_cal,
            'ROC AUC': roc_auc
            }

    return rt_dict

def split_into_train_test(X, Y, test_split = 0.25):
    """ 
        Given data (X, Y), split the data into training and testing sets.
        Validation is 10 percent of the training set.

        Arguments:
            X (numpy.ndarray): Data vector
            Y (numpy.ndarray): Label vector
            test_split (float): Test split (0.25 by default)

        Returns:
            x_train, y_train, x_test, and y_test
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length")
    
    # split the data
    random_state = 42
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=random_state, 
                                                        shuffle=True, stratify=Y)
    
    # x_val = np.array([])
    # y_val = np.array([])
    # if val_split > 0.0:
    #     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=random_state, 
    #                                                       shuffle=True, stratify=y_train)

    print("Training set {} \nTest set {}".format(x_train.shape, x_test.shape))
    return x_train, x_test, y_train, y_test

def select_random_samples(data, n_samples):
    """
        @brief: Select n_samples random samples from the data
        @param: data (array)
        @param: n_samples (int) Number of samples to randomly select from the data.

        @return: Randomly selected samples
    """
    length = data.shape[0]
    print(length, n_samples)
    if n_samples >= length:
        return data
    else:
        random_index = np.random.randint(low=0, high=length, size=n_samples)
        return data[random_index]


def get_hot_labels(Y):
    """
        Given label vector, return the one hot encoded label vector.

        Arguments:
            Y (numpy.ndarray): label vector
        
        Returns:
            One hot encoded label vector.
    """
    return keras.utils.to_categorical(Y, np.max(Y) + 1, dtype=int)


def find_min_max(X):
    """ Return the minimum and maximum value of X """
    return np.min(X), np.max(X)


def load_data_with_preprocessing(data_path):
    """
        Given a data path, load the data (must be in the format (X, Y)) and 
        scale the X in range [-1.0, 1.0] and return scaled x and y.

        Arguments:
            data_path (string): Pickle file path
    
        Returns:
            (X, Y)
    """
    # load the file
    f = open(data_path, "rb")
    try:
        x, y = pickle.load(f)
        f.close()
    except:
        f.close()
        return

    # check for same length
    if len(x) != len(y):
        raise ValueError("Unequal X and Y sizes")
    
#     print(x.shape, y.shape)
#     wherenane = np.argwhere(np.isnan(x))[:, 1]
#     print(np.unique(wherenane, return_counts=True))
    # do we need preprocessing 
    print("Before Scaling: Min - Max {}".format(find_min_max(x)))
    scaler = MinMaxScaler((-1.0, 1.0))
    x = scaler.fit_transform(x)
    print("After Scaling: Min - Max {}".format(find_min_max(x)))
    
    return x, y


def cross_validation(model_function, X, Y, n_CV, test_split, val_split, batch_size=32, epochs=50):
    """
        @brief: Do cross validation for n_CV times and returns the results.

        @param: model_function : A function that returns the model after calling it.
        @param: X (array): Total data
        @param: Y (array): Total label
        @param: test_split (float): The percentage of samples to be included in the test set
        @param: val_split (float): The percentage of samples to be included in the validation set.
        @param: batch_size (int): Default 32
        @param: epochs (int): Default 50

        @return: Results of the cross validation, a dictionary
    """
    x_tr, x_val, x_ts, y_tr, y_val, y_ts = split_into_train_test(X, Y, test_split, val_split=0.0)
    y_tr_hot = get_hot_labels(y_tr)
    y_ts_hot = get_hot_labels(y_ts)

    results_dict = {}
    metrics_arr = []
    for i in range(n_CV):
        model = model_function()
        results = evaluate_model(model, x_tr, y_tr_hot, x_ts, y_ts_hot, validation_split=val_split, 
                                 batch_size=batch_size, epochs=epochs)
        metrics_arr.append(results)
        train_report = compute_performance_metrics(model, x_tr, y_tr)
        test_report = compute_performance_metrics(model, x_ts, y_ts)
        results_dict[i] = {"Training Loss": results[0], "Training Accuracy": results[1], 
                            "Test Loss": results[2], "Test Accuracy": results[3],
                            "Training True Positive": train_report[0], "Training False Positive": train_report[1], 
                            "Training True Negative": train_report[2], "Training False Negative": train_report[3], 
                            "Training Recall": train_report[4], "Training Precision": train_report[5], 
                            "Training F1 Score": train_report[6], "Training ROC AUC": train_report[7],
                            "Training Report": train_report[8],
                            "Test True Positive": test_report[0], "Test False Positive": test_report[1], 
                            "Test True Negative": test_report[2], "Test False Negative": test_report[3], 
                            "Test Recall": test_report[4], "Test Precision": test_report[5], 
                            "Test F1 Score": test_report[6], "Test RO AUC": test_report[7],
                            "Test Report": test_report[8]}

    metrics_arr = np.array(metrics_arr).reshape(n_CV, 4)
    print("Average Training Set Accuracy {:.3f}".format(np.average(metrics_arr[:, 1].ravel())))
    print("Average Testing Set Accuracy {:.3f}".format(np.average(metrics_arr[:, 3].ravel())))

    return results_dict


def evaluate_model(model, x_tr, y_tr, x_ts, y_ts, val_split=0.0, 
                   batch_size=32, epochs=50, callbacks=[], 
                   metric_names=['accuracy', 'loss']):
    """
        @brief: Train the model and evaluate it on training and test set and return the results.

        @param: model: TF model
        @param: x_tr: training x
        @param: y_tr: training y
        @param: x_ts: test x
        @param: y_ts: test y
        @param: val_split: validation set split
        @param: BATCH_SIZE (int): default value 32
        @param: EPOCHS (int): default value 50
        @param: callbacks: TF callback functions
        @param: metric_names

        @return: Train and test metrics
    """
    # plot loss function
    plot_loss_cb = PlotLosses(metric_names)
    cbs = [plot_loss_cb]
    
    # append other callbacks
    for c in callbacks:
      cbs.append(c)

    # fit the model
    model_history = model.fit(x_tr, y_tr, batch_size = batch_size, epochs = epochs, 
                              validation_split = val_split, verbose = 0, callbacks = cbs)

    # get the performance values
    train_metrics = model.evaluate(x_tr, y_tr)
    test_metrics = model.evaluate(x_ts, y_ts)
    
    return train_metrics, test_metrics


def segment_sensor_reading(values, window_duration, overlap_percentage,
                           sampling_frequency):
    """
        Sliding window segmentation of the values array for the given window
        duration and overlap percentage.

    param values: 1D array of values to be segmented
    param window_duration: Window duration in seconds
    param overlap_percentage: Float value in the range (0 < overlap_percentage < 1)
    param sampling_frequency: Frequency in Hz
    """

    total_length = len(values)
    window_length = sampling_frequency * window_duration
    segments = []
    if(total_length < window_length):
        return segments
    
    start_index = 0
    end_index = start_index + window_length
    increment_size = int(window_length * (overlap_percentage))
    
    while(1):
        # print(start_index, end_index)
	
        # get the segment
        v = values[start_index:end_index]

        # save the segment
        segments.append(v)

        # change the start and end index values
        start_index += increment_size
        end_index += increment_size 

        if (start_index > total_length) | (end_index > total_length):
        #print("we are done, no more segments possible")
            break
        
    segments = np.array(segments).reshape(len(segments), window_length)
    return segments


def create_tf_dataset(X, Y, batch_size, test_size=0.3):
  """ Create train and test TF dataset from X and Y
    The prefetch overlays the preprocessing and model execution of a training step. 
    While the model is executing training step s, the input pipeline is reading the data for step s+1.
    AUTOTUNE automatically tune the number for sample which are prefeteched automatically. 
    
    Keyword arguments:
    X -- numpy array
    Y -- numpy array
    batch_size -- integer
  """
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  
  X = X.astype('float32')
  Y = Y.astype('float32')
  
  x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size = 0.3, random_state=42, stratify=Y, shuffle=True)
  
  print(f"Train size: {x_tr.shape[0]}")
  print(f"Test size: {x_ts.shape[0]}")

  train_dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
  train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)
  
  test_dataset = tf.data.Dataset.from_tensor_slices((x_ts, y_ts))
  test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)
  
  return train_dataset, test_dataset
    
def check_continuity(array):
    """
        Check whether the array contains continous values or not like 1, 2, 3, 4, ..
    """
    max_v = max(array)
    min_v = min(array)
    n = len(array)
#     print(n, min_v, max_v)
    if max_v - min_v + 1 == n:
#         print("Given array has continous values")
        return True
    else:
#         print("Given array is not continous")
        return False
        

if __name__ == "__main__":
    print("Script with utilities functions used throughout the research projects.")
    print("Availabel Functions are:")
    print(get_cnn_model.__doc__)
    print(get_features_labels_from_df.__doc__)
    print(print_confusion_matrix.__doc__)
    print(PlotLosses.__doc__)
    print(save_data.__doc__)
    print(read_data.__doc__)
    print(stylize_axis.__doc__)
    print(print_metrics.__doc__)
    print(compute_performance_metrics.__doc__)
    print(split_into_train_test.__doc__)
    print(get_hot_labels.__doc__)
    print(find_min_max.__doc__)
    print(load_data_with_preprocessing.__doc__)
    print(evaluate_model.__doc__)
    print(cross_validation.__doc__)
    print(segment_sensor_reading.__doc__)
    print(create_tf_dataset.__doc__)
    
