from cmath import exp
from math import floor, pi
import numpy as np
from scipy.signal import butter, filtfilt
from skimage.measure import block_reduce

def highpass_filter(data, cutoff=3000, fs=48000, order=5):
  """Runs a Butterworth high-pass filter on the given data.

  Keyword arguments:
  data -- the data as a numpy array
  cutoff -- the frequency below which to attenuate (default 1500 Hz)
  fs -- the sampling rate of the audio (default 48000 Hz)
  order -- the order of the filter (default 5)

  Returns:
  The filtered data as a numpy array.
  """
  
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='highpass')
  return filtfilt(b, a, data)


def lowpass_filter(data, cutoff=12600, fs=48000, order=5):
  """Runs a Butterworth low-pass filter on the given data.

  Keyword arguments:
  data -- the data as a numpy array
  cutoff -- the frequency above which to attenuate (default 1500 Hz)
  fs -- the sampling rate of the audio (default 48000 Hz)
  order -- the order of the filter (default 5)

  Returns:
  The filtered data as a numpy array.
  """
  
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='lowpass')
  return filtfilt(b, a, data)


def convert_to_baseband(data, center_freq, bandwidth, fs=48000, new_fs=None):
  """Converts the data to baseband and downsamples it.

  Keyword arguments:
  data -- the data as a numpy array
  center_freq -- the frequency around which the bandwidth is centered
  bandwidth -- the bandwidth of the desired frequency range, must divide fs
  fs -- the sampling rate

  Returns:
  A 2-tuple of the downsampled data as a numpy array and the new sampling rate. Raises a ValueError if bandwidth does not divide fs."""
  if fs % bandwidth != 0:
    raise ValueError('Bandwidth does not divide the sampling rate.')
  if new_fs is None:
    new_fs = bandwidth
  baseband_signal = np.array([datum * exp(-2j*pi*(center_freq/fs)*i) for i, datum in enumerate(data)])
  baseband_signal = lowpass_filter(baseband_signal, cutoff=new_fs, fs=fs)
  baseband_signal = block_reduce(baseband_signal, block_size=(floor(fs/new_fs),), func=np.mean)
  return (baseband_signal.real, new_fs)

