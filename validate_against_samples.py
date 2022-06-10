import numpy as np
import sys

from ksvd import omp
from config import CallType
from extract_training_data import get_training_data


def validate_against_samples(vec_dict, call_type):
  """Validate the algorithm's output against a dictionary of labeled numpy vectors.

  Keyword arguments:
  vec_dict -- a dictionary that maps CallTypes to lists of numpy vectors
  call_type -- what call type to look for

  Returns:
  A 2-tuple of the sensitivity and specificity of the test
  """
  d = np.loadtxt(f'dictionaries/{call_type.filename}.csv', delimiter=',')

  true_positive_count = 0
  false_positive_count = 0
  real_threshold = 10**(call_type.threshold/10)
  for call_type2, vecs in vec_dict.items():
    if call_type != call_type2: # Figure out how to deal with different call lengths later
      continue
    for vec in vecs:
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
          false_positive_count += 1 # Add informative plots later
  sensitivity = true_positive_count / len(vec_dict[call_type])
  specificity = 1 - false_positive_count / sum(len(vec_dict[ct]) for ct in vec_dict if ct != call_type)

  return sensitivity, specificity


if __name__ == '__main__':
  audio_filename = f'{sys.argv[1]}'
  annotations_filename = f'{sys.argv[2]}'
  call_type = CallType.from_str(sys.argv[3])
  vec_dict = get_training_data(audio_filename, annotations_filename, 8)
  sensitivity, specificity = validate_against_samples(vec_dict, call_type)
  print(f'The sensitivity is {sensitivity:.2%}.')
  print(f'The specificity is {specificity:.2%}.')

