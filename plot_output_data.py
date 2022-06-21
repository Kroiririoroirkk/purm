import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import scipy.io.wavfile
import sounddevice as sd
import sys

from extract_training_data import timestamp_to_frame
from validation import parse_output_line


def plot_output_data(audio_filename, channel_number, output_filename='output/output.csv'):
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

  for start_time, end_time, threshold in output_lines:
    start_index = timestamp_to_frame(start_time, sampling_rate)
    end_index = timestamp_to_frame(end_time, sampling_rate)
    sub_vec = audio_vec[start_index : end_index]
    plt.specgram(sub_vec, Fs=sampling_rate)
    plt.title(threshold)
    button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
    button = Button(button_axes, 'Play')
    button.on_clicked(lambda _: sd.play(sub_vec, sampling_rate))
    plt.show()


if __name__ == '__main__':
  audio_filename = sys.argv[1]
  channel = 0
  if audio_filename == 'audio/aviary_2019-05-01_1556722860.000-1556723760.000_audio.wav':
    channel = 8
  elif audio_filename == 'audio/aviary_2019-06-01_1559399640.000-1559400540.000_audio.wav':
    channel = 15
  plot_output_data(audio_filename, channel)
