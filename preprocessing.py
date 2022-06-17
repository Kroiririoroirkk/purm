from cmath import exp
from math import floor, pi
import numpy as np
from scipy.signal import butter, filtfilt
from skimage.measure import block_reduce

from config import FREQ_CUTOFF

def highpass_filter(data, cutoff=FREQ_CUTOFF[0], fs=48000, order=5):
  """Runs a Butterworth high-pass filter on the given data.

  Keyword arguments:
  data -- the data as a numpy array
  cutoff -- the frequency below which to attenuate
  fs -- the sampling rate of the audio (default 48000 Hz)
  order -- the order of the filter (default 5)

  Returns:
  The filtered data as a numpy array.
  """
  
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='highpass')
  return filtfilt(b, a, data)


def lowpass_filter(data, cutoff=FREQ_CUTOFF[1], fs=48000, order=5):
  """Runs a Butterworth low-pass filter on the given data.

  Keyword arguments:
  data -- the data as a numpy array
  cutoff -- the frequency above which to attenuate
  fs -- the sampling rate of the audio (default 48000 Hz)
  order -- the order of the filter (default 5)

  Returns:
  The filtered data as a numpy array.
  """
  
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='lowpass')
  return filtfilt(b, a, data)


def bandpass_filter(data, cutoffs=FREQ_CUTOFF, fs=48000, order=5):
  """Runs a Butterworth low-pass filter on the given data.

  Keyword arguments:
  data -- the data as a numpy array
  cutoffs -- the frequencies below and above which to attenuate as a 2-tuple
  fs -- the sampling rate of the audio (default 48000 Hz)
  order -- the order of the filter (default 5)

  Returns:
  The filtered data as a numpy array.
  """
  
  nyq = 0.5 * fs
  b, a = butter(order, [cutoffs[0]/nyq, cutoffs[1]/nyq], btype='bandpass')
  return filtfilt(b, a, data)


def convert_to_baseband(data, center_freq, bandwidth, fs, new_fs=None):
  """Converts the data to baseband and downsamples it.

  Keyword arguments:
  data -- the data as a numpy array
  center_freq -- the frequency around which the bandwidth is centered
  bandwidth -- the bandwidth of the desired frequency range, must divide fs
  fs -- the sampling rate

  Returns:
  A 2-tuple of the downsampled data as a numpy array and the new sampling rate. Raises a ValueError if bandwidth does not divide fs.
  """
  if fs % bandwidth != 0:
    raise ValueError('Bandwidth does not divide the sampling rate.')
  if new_fs is None:
    new_fs = bandwidth
  baseband_signal = np.array([datum * exp(-2j*pi*(center_freq/fs)*i) for i, datum in enumerate(data)])
  baseband_signal = lowpass_filter(baseband_signal, cutoff=new_fs, fs=fs)
  baseband_signal = block_reduce(baseband_signal, block_size=(floor(fs/new_fs),), func=np.mean)
  return (baseband_signal.real, new_fs)


def preprocess(audio_vec, fs):
  """Preprocesses the audio vector.

  Keyword arguments:
  audio_vec -- the audio vector as a numpy array
  fs -- the sampling rate

  Returns:
  A 2-tuple of the new audio vector and the sampling rate.
  """
  audio_vec = bandpass_filter(audio_vec)
  # audio_vec, fs = convert_to_baseband(audio_vec, 7800, 9600, fs, new_fs=12000)
  return audio_vec, fs
