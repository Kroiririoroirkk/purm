import math
import numpy as np
import random
import scipy.io.wavfile
import sys

from config import CallType, get_channel
from extract_training_data import parse_line, timestamp_to_frame
from preprocessing import preprocess


def get_non_sample_data(audio_filenames, annotations_filenames, channel_numbers, call_type):
  """Reads audio files and annotations files to obtain sample data which is not a certain call type (for validation purposes).

  Keyword arguments:
  audio_filenames -- a list of the filenames of the audio files
  annotations_filenames -- a list of the filenames of the annotations files
  channel_number -- which channel of the audio to use (zero-indexed)
  call_type -- the CallType to exclude

  Returns:
  A list of numpy 1D vectors. Raises an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  l = []

  for audio_filename, annotations_filename, channel_number in zip(audio_filenames, annotations_filenames, channel_numbers):
    sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
    if sampling_rate != 48000:
      raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
    audio_vec = audio_vec[:, channel_number] # ignore all channels except one
    audio_vec, sampling_rate = preprocess(audio_vec, sampling_rate, call_type)

    with open(annotations_filename, 'r') as f:
      lines = [parse_line(s) for s in f.read().strip().split('\n')]
      lines = [(s,e) for s,e,c in lines if c == call_type]

    for start_time, end_time in lines:
      start_index = timestamp_to_frame(start_time, sampling_rate)
      end_index = timestamp_to_frame(end_time, sampling_rate)
      audio_vec[start_index:end_index] = [np.nan]*(end_index-start_index)

    chunk_size = timestamp_to_frame(call_type.duration, sampling_rate)
    for i in range(0, len(audio_vec), chunk_size*20): # This multiplier parameter is pretty arbitrary, change it to get more or fewer non-samples
      subvec = audio_vec[i:i+chunk_size]
      subvec = [i for i in subvec if not np.isnan(i)]
      if len(subvec) != chunk_size:
        continue
      l.append(subvec)

  random.shuffle(l)
  return l


if __name__ == '__main__':
  if len(sys.argv)%2 != 0:
    print("Error: Number of audio files and annotation files do not match")
  else:
    divisor = math.floor((len(sys.argv)-2)/2)
    call_type = CallType.from_str(sys.argv[1])
    audio_files = sys.argv[2:2+divisor]
    annotations_files = sys.argv[2+divisor:]
    channel_numbers = [get_channel(fn) for fn in audio_files]
    vec_list = get_non_sample_data(audio_files, annotations_files, channel_numbers, call_type)
    np.savetxt(f'training_data/non_sample_{call_type.filename}.csv', vec_list, delimiter=',', fmt='%.2f')

