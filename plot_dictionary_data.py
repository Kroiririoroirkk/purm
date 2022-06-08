import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import sounddevice as sd

from config import CallType


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
      button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
      button = Button(button_axes, 'Play')
      button.on_clicked(lambda _: sd.play(vec, sampling_rate))
      plt.show()


if __name__ == '__main__':
  plot_dictionary_data()
