import math
import numpy as np
import random
import scipy.io.wavfile
import sys

from config import CallType
from extract_training_data import parse_line, timestamp_to_frame


def get_noise(audio_filenames, annotations_filenames, channel_numbers):
  """Reads audio files and annotations files to obtain noise data (for filtering purposes).

  Keyword arguments:
  audio_filenames -- a list of the filenames of the audio files
  annotations_filenames -- a list of the filenames of the annotations files
  channel_numbers -- which channels of the audio to use (zero-indexed)

  Returns:
  A list of numpy 1D vectors. Raises an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  l = []

  for audio_filename, annotations_filename, channel_number in zip(audio_filenames, annotations_filenames, channel_numbers):
    sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
    if sampling_rate != 48000:
      raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
    audio_vec = list(audio_vec[:, channel_number]) # ignore all channels except one

    with open(annotations_filename, 'r') as f:
      lines = [parse_line(s) for s in f.read().strip().split('\n')]
      lines = [(s,e) for s,e,_ in lines]

    for start_time, end_time in lines:
      start_index = timestamp_to_frame(start_time, sampling_rate)
      end_index = timestamp_to_frame(end_time, sampling_rate)
      audio_vec[start_index:end_index] = [np.nan]*(end_index-start_index)

    chunk_size = timestamp_to_frame(CallType.BURBLE.duration, sampling_rate)
    for i in range(0, len(audio_vec), chunk_size*20): # This multiplier parameter is pretty arbitrary, change it to get more or fewer non-samples
      subvec = audio_vec[i:i+chunk_size]
      subvec = [i for i in subvec if not np.isnan(i)]
      if len(subvec) != chunk_size:
        continue
      l.append(subvec)

  random.shuffle(l)
  return l


if __name__ == '__main__':
  if len(sys.argv)%2 != 1:
    print("Error: Number of audio files and annotation files do not match")
  else:
    divisor = math.floor(len(sys.argv)/2)
    audio_files = sys.argv[1:1+divisor]
    annotations_files = sys.argv[1+divisor:]
    channel_numbers = [get_channel(fn) for fn in audio_files]
    vec_list = get_noise(audio_files, annotations_files, channel_numbers)
    np.savetxt(f'training_data/noise.csv', vec_list, delimiter=',', fmt='%.2f')

