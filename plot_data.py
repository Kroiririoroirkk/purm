import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile

from config import CallType
from process_data import timestamp_to_frame
from validation import parse_output_line


def plot_training_data(sampling_rate=48000):
  """Plots the training data as spectograms.

  Keyword arguments:
  sampling_rate -- sampling rate in Hz (default 48000)

  Returns:
  Nothing. Displays spectograms.
  """

  for call_type in CallType:
    vecs = np.loadtxt(f'training_data/{call_type.filename}.csv', delimiter=',', dtype=int)
    for vec in vecs:
      plt.specgram(vec, Fs=sampling_rate)
      plt.title(call_type.filename)
      plt.show()


def plot_dictionary_data(sampling_rate=48000):
  """Plots the dictionary data as spectograms.

  Keyword arguments:
  sampling_rate -- sampling rate in Hz (default 48000)

  Returns:
  Nothing. Displays spectograms.
  """

  for call_type in CallType:
    vecs = np.loadtxt(f'dictionaries/{call_type.filename}.csv', delimiter=',').transpose()
    for vec in vecs:
      plt.specgram(vec, Fs=sampling_rate)
      plt.title(call_type.filename)
      plt.show()


def plot_output_data(audio_filename='audio/aviary_2019-05-01_1556722860.000-1556723760.000_audio.wav', channel_number=8, output_filename='output/plot.csv'):
  """Plots the output data as spectograms.

  Keyword arguments:
  audio_filename -- the filename of the audio file
  channel_number -- the channel number of the audio file
  output_filename -- the filename of the output file

  Returns:
  Nothing. Displays spectograms.
  """
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
  audio_vec = audio_vec[:, channel_number]

  with open(output_filename, 'r') as f:
    output_lines = [parse_output_line(s) for s in f.read().strip().split('\n')]

  for start_time, end_time in output_lines:
    start_index = timestamp_to_frame(start_time, sampling_rate)
    end_index = timestamp_to_frame(end_time, sampling_rate)
    sub_vec = audio_vec[start_index : end_index]
    plt.specgram(sub_vec, Fs=sampling_rate)
    plt.show()


#plot_training_data()
#plot_dictionary_data()
#plot_output_data()
