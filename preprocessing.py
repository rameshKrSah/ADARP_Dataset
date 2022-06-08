import numpy as np
import scipy.signal as sps
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def resample_data(gsrdata, prevSR, newSR):
  '''Calculates rolling mean
    Function to calculate moving average over the passed data
	
    Parameters
    ----------
    gsrdata : 1-d array
        array containing the gsr data
    prevSR : int or float 
        the previous sample rate of the data
    newSR : int or float
        the new sample rate of the data
		
    Returns
    -------
    data : 1-d array
        array containing the resampled data
  '''
  number_of_samples = int(round(len(gsrdata) * float(newSR) / prevSR))
  data = sps.resample(gsrdata, number_of_samples)
  
  return data

	
def normalization(gsrdata):
  '''Min Max normalization
    Function to calculate normalized gsr data
	
    Parameters
    ----------
    gsrdata : 1-d array
        array containing the gsr data
		
    Returns
    -------
    n_gsrdata : 1-d array
        normalized gsr data
  '''
  gsrdata = gsrdata - (np.min(gsrdata))
  gsrdata /= (np.max(gsrdata) - np.min(gsrdata))
  n_gsrdata = gsrdata
  return n_gsrdata

def rolling_mean(data, windowsize, sample_rate):
  '''calculates rolling mean
    Function to calculate moving average over the passed data
	
    Parameters
    ----------
    data : 1-d array
        array containing the gsr data
    windowsize : int or float 
        the moving average window size in seconds 
    sample_rate : int or float
        the sample rate of the data set
		
    Returns
    -------
    rol_mean : 1-d array
        array containing computed rolling mean
  '''
  avg_hr = (np.mean(data))
  data_arr = np.array(data)
	
  t_windowsize = int(windowsize*sample_rate)
  t_shape = data_arr.shape[:-1] + (data_arr.shape[-1] - t_windowsize + 1, t_windowsize)
  t_strides = data_arr.strides + (data_arr.strides[-1],)
  sep_win = np.lib.stride_tricks.as_strided(data_arr, shape=t_shape, strides=t_strides)
  rol_mean = np.mean(sep_win, axis=1)
	
  missing_vals = np.array([avg_hr for i in range(0, int(abs(len(data_arr) - len(rol_mean))/2))])
  rol_mean = np.insert(rol_mean, 0, missing_vals)
  rol_mean = np.append(rol_mean, missing_vals)

  #only to catch length errors that sometimes unexplicably occur. 
  ##Generally not executed, excluded from testing and coverage
  if len(rol_mean) != len(data): # pragma: no cover
    lendiff = len(rol_mean) - len(data)
    if lendiff < 0:
      rol_mean = np.append(rol_mean, 0)
    else:
      rol_mean = rol_mean[:-1]
	  
  return rol_mean

def min_max_scale(data):
  """
    Min-Max scale the data in the range [-1.0, 1.0]
    The data is expected to have the shape (samples, length, channels)
  
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
	