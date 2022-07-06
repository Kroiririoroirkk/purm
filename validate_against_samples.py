import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
import numpy as np
import sounddevice as sd
import sys

from config import CallType
from omp import orthogonal_mp


def validate_against_samples2(d, call_samples, non_call_samples, call_type, sampling_rate=48000, plot=True):
  """Validate the algorithm's output against labeled numpy vectors.

  Keyword arguments:
  d -- a dictionary
  call_samples -- a list of numpy vectors which include the call
  non_call_samples -- a list of numpy vectors which do not include the call
  call_type -- what call type to look for
  sampling_rate -- sampling rate in Hz (default 48000)
  plot -- whether or not to generate plots (default True)

  Returns:
  A 2-tuple of the sensitivity and specificity of the test
  """

  true_positive_count = 0
  false_positive_count = 0
  real_threshold = 10**(call_type.threshold/10)

  call_samples_matr = np.column_stack(call_samples)
  sparse_reps = orthogonal_mp(d, call_samples_matr, call_type.sparsity)

  pos_sinrs = []
  for i, vec in enumerate(call_samples):
    approximate_signal = d @ sparse_reps[:,i]
    signal_strength = np.linalg.norm(approximate_signal)**2
    background = vec - approximate_signal
    background_strength = np.linalg.norm(background)**2
    sinr = signal_strength/background_strength
    sinr_log = 10*math.log(sinr,10)
    pos_sinrs.append(sinr_log)
    if sinr > real_threshold:
      true_positive_count += 1
    elif plot:
      plt.specgram(vec, Fs=sampling_rate)
      plt.title(f'False negative, SINR (dB): {sinr_log:.2f}')
      button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
      button = Button(button_axes, 'Play')
      button.on_clicked(lambda _: sd.play(vec.astype(np.int16), sampling_rate))
      plt.show()

  non_call_samples_matr = np.column_stack(non_call_samples)
  neg_sparse_reps = orthogonal_mp(d, non_call_samples_matr, call_type.sparsity)

  neg_sinrs = []
  for i, vec in enumerate(non_call_samples):
    approximate_signal = d @ neg_sparse_reps[:,i]
    signal_strength = np.linalg.norm(approximate_signal)**2
    background = vec - approximate_signal
    background_strength = np.linalg.norm(background)**2
    sinr = signal_strength/background_strength
    sinr_log = 10*math.log(sinr,10)
    neg_sinrs.append(sinr_log)
    if sinr > real_threshold:
      false_positive_count += 1
      if plot:
        plt.specgram(vec, Fs=sampling_rate)
        plt.title(f'False positive, SINR (dB): {sinr_log:.2f}')
        button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
        button = Button(button_axes, 'Play')
        button.on_clicked(lambda _: sd.play(vec.astype(np.int16), sampling_rate))
        plt.show()

  plt.hist([pos_sinrs, neg_sinrs], bins=range(-20,350,10), label=['Whistle', 'Noise'])
  plt.legend()
  plt.show()

  sensitivity = true_positive_count / len(call_samples)
  specificity = 1 - false_positive_count / len(non_call_samples)

  return sensitivity, specificity


if __name__ == '__main__':
  call_type = CallType.from_str(sys.argv[1])
  call_samples = np.loadtxt(sys.argv[2], delimiter=',')
  non_call_samples = np.loadtxt(sys.argv[3], delimiter=',')
  plot = True
  try:
    if sys.argv[4] == 'noplot':
      plot = False
  except IndexError:
    pass
  d = np.loadtxt(f'dictionaries/{sys.argv[1]}.csv', delimiter=',')
  sensitivity, specificity = validate_against_samples2(d, call_samples, non_call_samples, call_type, plot=plot)
  print(f'The sensitivity is {sensitivity:.2%}.')
  print(f'The specificity is {specificity:.2%}.')

