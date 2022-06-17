from cmath import exp
from math import pi
import numpy as np
from scipy.fft import fft, ifft
import sys

from config import CallType


for arg in sys.argv[1:]:
  call_type = CallType.from_str(arg)
  print(f'Generating Fourier dictionary for {call_type.filename}...')
  vecs = np.loadtxt(f'training_data/{call_type.filename}.csv', delimiter=',')
  avg_vec = np.mean(vecs, axis=0)
  ##
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Button
  import sounddevice as sd
  plt.specgram(avg_vec, Fs=48000)
  plt.title(call_type.filename)
  button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
  button = Button(button_axes, 'Play')
  button.on_clicked(lambda _: sd.play(avg_vec, 48000))
  plt.show()
  ##
  transformed_vec = fft(avg_vec, overwrite_x=True)
  fourier_dictionary = []
  s = transformed_vec.shape
  l = s[0]
  for n in range(0, l, 1000):
    displacement_vec = np.fromfunction(lambda i: exp(-2j*pi*n/l), s)
    new_vec = avg_vec * displacement_vec
    fourier_dictionary.append(new_vec)
  fourier_dictionary = np.column_stack(fourier_dictionary)
  np.savetxt(f'dictionaries/{call_type.filename}.csv', fourier_dictionary, delimiter=',')
