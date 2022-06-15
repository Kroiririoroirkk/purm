import math
import numpy as np
import scipy.io.wavfile
import sys

from config import CallType
from omp import orthogonal_mp
from preprocessing import preprocess


def detect_sounds(audio_filename, dictionary_filename, k, threshold, channel_number):
  """Detects where specific sounds are present in an audio file given a dictionary for the sounds. Assumes the sampling rate of the audio file is the same as that used for the dictionary.

  Keyword arguments:
  audio_filename -- the filename of the audio file
  dictionary_filename -- the filename of the dictionary file
  k -- the sparsity constraint
  threshold -- determines how sensitive the algorithm is (in dB)
  channel_number -- which channel of the audio to use (zero-indexed)

  Returns:
  A list of 3-tuples, where the first entry in each tuple represents the start time, the second entry represents the end time, and the third entry represents the signal-to-interference-and-noise ratio (in dB). Raises an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
  if sampling_rate != 48000:
    raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
  audio_vec = audio_vec[:, channel_number]
  audio_vec, sampling_rate = preprocess(audio_vec, sampling_rate)

  d = np.loadtxt(dictionary_filename, delimiter=',')
  sample_length = len(d[:,0])

  pointer = 0
  timestamps = []
  real_threshold = 10**(threshold/10)
  while True:
    segment = audio_vec[pointer : pointer+sample_length]
    if len(segment) != sample_length:
      break
    sparse_rep = orthogonal_mp(d, segment, k)
    approximate_signal = d @ sparse_rep
    signal_strength = np.linalg.norm(approximate_signal)**2
    background = segment - approximate_signal
    background_strength = np.linalg.norm(background)**2
    sinr = signal_strength/background_strength
    if sinr > real_threshold:
      start_time = pointer/sampling_rate
      end_time = start_time + sample_length/sampling_rate
      timestamps.append((start_time, end_time, 10*math.log(sinr, 10)))
    pointer += math.floor(sample_length/4)
  return timestamps


if __name__ == '__main__':
  audio_filename = f'{sys.argv[1]}'
  call_type = CallType.from_str(sys.argv[2])
  dictionary_filename = f'dictionaries/{call_type.filename}.csv'
  channel = 0
  if audio_filename == 'audio/aviary_2019-05-01_1556722860.000-1556723760.000_audio.wav':
    channel = 8
  elif audio_filename == 'audio/aviary_2019-06-01_1559399640.000-1559400540.000_audio.wav':
    channel = 15
  times = detect_sounds(audio_filename, dictionary_filename, call_type.sparsity, call_type.threshold, channel)
  np.savetxt(f'./output/output.csv', times, fmt='%.2f', delimiter=',')

