import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import sounddevice as sd
import sys

from config import CallType


def plot_training_data(filenames, sampling_rate=48000):
  """Plots the training data as spectograms.

  Keyword arguments:
  filenames -- a list of filenames to display
  sampling_rate -- sampling rate in Hz (default 48000)

  Returns:
  Nothing. Displays spectograms.
  """

  for fn in filenames:
    vecs = np.loadtxt(fn, delimiter=',', dtype=np.int16)
    for i, vec in enumerate(vecs):
      plt.specgram(vec, Fs=sampling_rate)
      plt.title(i+1)
      button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
      button = Button(button_axes, 'Play')
      button.on_clicked(lambda _: sd.play(vec, sampling_rate))
      plt.show()


if __name__ == '__main__':
  plot_training_data(sys.argv[1:])
