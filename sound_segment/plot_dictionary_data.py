import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import sounddevice as sd
import sys

from config import CallType


def plot_dictionary_data(sampling_rate=48000, call_types=CallType):
  """Plots the dictionary data as spectograms.

  Keyword arguments:
  sampling_rate -- sampling rate in Hz (default 48000)
  call_types -- a list of CallTypes to display (default all)

  Returns:
  Nothing. Displays spectograms.
  """

  for call_type in call_types:
    vecs = np.loadtxt(f'dictionaries/{call_type.filename}.csv', delimiter=',').transpose()
    for vec in vecs:
      plt.specgram(cum_vec, Fs=sampling_rate)
      plt.title(call_type.filename)
      button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
      button = Button(button_axes, 'Play')
      button.on_clicked(lambda _: sd.play(vec.astype(np.int16), sampling_rate))
      plt.show()


if __name__ == '__main__':
  call_types = []
  for arg in sys.argv[1:]:
    call_types.append(CallType.from_str(arg))
  if call_types:
    plot_dictionary_data(call_types=call_types)
  else:
    plot_dictionary_data()
