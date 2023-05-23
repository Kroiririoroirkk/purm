import math
import numpy as np
import os
from scipy.fft import fft
import scipy.io.wavfile
import sys

from .config import CallType, get_channel
from .omp import orthogonal_mp
from .preprocessing import preprocess


def detect_sounds(audio_vec, call_type, fourier=False, sampling_rate=48000, overlap=0.25):
  """Detects where specific sounds are present in an audio file given a dictionary for the sounds. Assumes the sampling rate of the audio file is the same as that used for the dictionary.

  Keyword arguments:
  audio_vec -- the audio file as a numpy array
  call_type -- the CallType to detect for
  fourier -- whether to apply a Fourier transform to the audio (default False)
  sampling_rate -- the sampling rate (default 48000)
  overlap -- how much to move the window each time, relative to the length of the call type

  Returns:
  A list of 3-tuples, where the first entry in each tuple represents the start time, the second entry represents the end time, and the third entry represents the signal-to-interference-and-noise ratio (in dB). Raises an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  audio_vec, sampling_rate = preprocess(audio_vec, sampling_rate, call_type)

  d = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),f'dictionaries/{call_type.filename}.csv'), delimiter=',')
  sample_length = int(call_type.duration * sampling_rate)

  pointer = 0
  timestamps = []
  real_threshold = 10**(call_type.threshold/10)

  to_omp = []

  while True:
    segment = audio_vec[pointer : pointer+sample_length]
    if len(segment) != sample_length:
      break
    if fourier:
      segment = fft(segment, overwrite_x=True)
    to_omp.append(segment)
    pointer += math.floor(sample_length * overlap)

  sparse_reps = orthogonal_mp(d, np.column_stack(to_omp), call_type.sparsity)

  pointer = 0
  for segment, sparse_rep in zip(to_omp, sparse_reps.T):
    approximate_signal = d @ sparse_rep
    signal_strength = np.linalg.norm(approximate_signal)**2
    background = segment - approximate_signal
    background_strength = np.linalg.norm(background)**2
    sinr = signal_strength/background_strength
    if sinr > real_threshold:
      start_time = pointer/sampling_rate
      end_time = start_time + sample_length/sampling_rate
      timestamps.append((start_time, end_time, 10*math.log(sinr, 10)))
    pointer += math.floor(sample_length * overlap)
  return timestamps


if __name__ == '__main__':
  audio_filename = sys.argv[1]
  call_type = CallType.from_str(sys.argv[2])
  channel = get_channel(audio_filename)
  fourier = False
  try:
    if sys.argv[3] == 'fourier':
      fourier = True
  except IndexError:
    pass
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
  if sampling_rate != 48000:
    raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
  times = detect_sounds(audio_vec[:,channel], call_type, fourier, sampling_rate)
  np.savetxt(f'./output/output.csv', times, fmt='%.2f', delimiter=',')

