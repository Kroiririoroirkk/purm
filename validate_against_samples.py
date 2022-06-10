import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
import numpy as np
import sounddevice as sd
import sys

from ksvd import omp
from config import CallType
from extract_training_data import get_training_data


def convert_to_length(vec, length):
  """Ensure that a numpy vector has a certain amount of entries by looping or cutting.

  Keyword arguments:
  vec -- the numpy vector
  length -- an integer

  Returns:
  A vector that has the number of entries given by length
  """
  if len(vec) < length:
    vec = np.tile(vec, math.ceil(length/len(vec)))
  if len(vec) > length:
    vec = vec[:length]
  return vec


def validate_against_samples(vec_dict, call_type, sampling_rate=48000, plot=True):
  """Validate the algorithm's output against a dictionary of labeled numpy vectors.

  Keyword arguments:
  vec_dict -- a dictionary that maps CallTypes to lists of numpy vectors
  call_type -- what call type to look for
  sampling_rate -- sampling rate in Hz (default 48000)
  plot -- whether or not to generate plots (default True)

  Returns:
  A 2-tuple of the sensitivity and specificity of the test
  """
  d = np.loadtxt(f'dictionaries/{call_type.filename}.csv', delimiter=',')

  true_positive_count = 0
  false_positive_count = 0
  real_threshold = 10**(call_type.threshold/10)
  call_duration_frames = math.floor(call_type.duration * sampling_rate)
  for call_type2, vecs in vec_dict.items():
    for vec in vecs:
      vec = convert_to_length(vec, call_duration_frames)
      sparse_rep = omp(d, vec, call_type.sparsity)
      approximate_signal = d @ sparse_rep
      signal_strength = np.linalg.norm(approximate_signal)**2
      background = vec - approximate_signal
      background_strength = np.linalg.norm(background)**2
      sinr = signal_strength/background_strength
      if sinr > real_threshold:
        if call_type == call_type2:
          true_positive_count += 1
        else:
          false_positive_count += 1
          if plot:
            plt.specgram(vec, Fs=sampling_rate)
            plt.title(f'False positive, SINR (dB): {10*math.log(sinr,10):.2f}')
            button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
            button = Button(button_axes, 'Play')
            button.on_clicked(lambda _: sd.play(vec, sampling_rate))
            plt.show()
      else:
        if (call_type == call_type2) and plot:
          plt.specgram(vec, Fs=sampling_rate)
          plt.title(f'False negative, SINR (dB): {10*math.log(sinr,10):.2f}')
          button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
          button = Button(button_axes, 'Play')
          button.on_clicked(lambda _: sd.play(vec, sampling_rate))
          plt.show()
        
  sensitivity = true_positive_count / len(vec_dict[call_type])
  specificity = 1 - false_positive_count / sum(len(vec_dict[ct]) for ct in vec_dict if ct != call_type)

  return sensitivity, specificity


if __name__ == '__main__':
  audio_filename = f'{sys.argv[1]}'
  annotations_filename = f'{sys.argv[2]}'
  call_type = CallType.from_str(sys.argv[3])
  plot = True
  try:
    if sys.argv[4] == 'noplot':
      plot = False
  except IndexError:
    pass
  vec_dict = get_training_data(audio_filename, annotations_filename, 15)
  sensitivity, specificity = validate_against_samples(vec_dict, call_type, plot=plot)
  print(f'The sensitivity is {sensitivity:.2%}.')
  print(f'The specificity is {specificity:.2%}.')

