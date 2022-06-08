import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, freqz

def lpf(order = 4, fs = 4, cutoff = 1):
    """Get a low pass filter.

    order: Order for the Butterworth filter
    fs: Sampling frequency
    cutoff: Cut off frequency

    Retuns polynomial of IIR filter.

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def hpf(order = 1, fs = 4, cutoff = 0.05):
    """Get a high pass filter.

    order: Order for the Butterworth filter
    fs: Sampling frequency
    cutoff: Cut off frequency

    Retuns polynomial of IIR filter.

    """
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a


def butter_lowpassfilter(data, cutoff, sample_rate, order=2):
  '''Standard Butterworth lowpass filter for 1d-data
	
    Parameters
    ----------
	data : 1-d array
        array containing the data
    cutoff : int or float
        frequency in Hz that acts as cutoff for filter.
    sample_rate : int or float
        sample rate of the supplied signal
    order : int
        filter order, defines the strength of the roll-off
        around the cutoff frequency.
        default: 2
    
    Returns
    -------
    y : 1-d array
        filtered data
  '''
  b, a = lpf(order, sample_rate, cutoff)
  y = filtfilt(b, a, data)
  return y


def butter_highpassfilter(data, cutoff, sample_rate, order=2):
    '''Standard Butterworth highpass filter for 1d-data.

    Parameters
    ----------
    data : 1-d array
        array containing the data
    cutoff : int or float
        frequency in Hz that acts as cutoff for filter.
    sample_rate : int or float
        sample rate of the supplied signal
    order : int
        filter order, defines the strength of the roll-off
        around the cutoff frequency.
        default: 2

    Returns
    -------
    y : 1-d array
        filtered data
    '''
    b, a = hpf(order, sample_rate, cutoff)
    y = lfilter(b, a, data)
    return y

def plot_freq_response(sample_rate, cutoff, order, filter_type='lowpass'):
    """ Given sampling frequency and cutoff frequency, draw the frequency response. 
    """
    if filter_type == 'lowpass':
        b, a = lpf(order, sample_rate, cutoff)

    elif filter_type == 'highpass':
        b, a = hpf(order, sample_rate, cutoff)

    else:
        print("Undefined filter type. Either Lowpass or Highpass supported")
        return

    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*sample_rate*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*sample_rate)
    plt.title(f"{filter_type} filter frequency fesponse")
    plt.xlabel('Frequency [Hz]')
    plt.grid()