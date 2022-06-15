import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import sys

from config import CallType, ROLLS
from preprocessing import preprocess


def parse_line(s):
  """Parses a line of the input text file.

  Keyword arguments:
  s -- the line

  Returns:
  A 3-tuple of the start timestamp (in seconds), the end timestamp (in seconds), and the audio type (as a CallType).

  Raises:
  A ValueError if the input line is ill-formed.
  """
  try:
    start_time, end_time, call_type = s.strip().split('\t')
  except ValueError:
    raise ValueError("Ill-formed input line: " + s)
  return (float(start_time.strip()),
          float(end_time.strip()),
          CallType.from_str(call_type.strip()))


def timestamp_to_frame(s, f):
  """Converts a timestamp in seconds to a audio sample number, rounded down.

  Keyword arguments:
  s -- the second number (can be a decimal)
  f -- the sampling rate in Hz

  Returns:
  A sample number (zero indexed)
  """
  return math.floor(s*f)


def get_training_data(audio_filenames, annotations_filenames, channel_numbers):
  """Reads audio files and annotations files to obtain training data.

  Keyword arguments:
  audio_filename -- a list of the filenames of the audio files
  annotations_filename -- a list of the filenames of the annotations files
  channel_number -- which channel of the audio to use (zero-indexed)

  Returns:
  A dictionary with CallTypes as keys and lists of numpy 1D vectors as values. Raises an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  d = dict()
  for call_type in CallType:
    d[call_type] = []

  for audio_filename, annotations_filename, channel_number in zip(audio_filenames, annotations_filenames, channel_numbers):
    sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
    if sampling_rate != 48000:
      raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
    audio_vec = audio_vec[:, channel_number] # ignore all channels except one
    audio_vec, sampling_rate = preprocess(audio_vec, sampling_rate)

    with open(annotations_filename, 'r') as f:
      lines = [parse_line(s) for s in f.read().strip().split('\n')]

    for start_time, end_time, call_type in lines:
      duration_frames = timestamp_to_frame(call_type.duration, sampling_rate)
      avg_time = (start_time + end_time)/2
      avg_index = timestamp_to_frame(avg_time, sampling_rate)
      start_index = avg_index - math.floor(duration_frames/2)
      end_index = start_index + duration_frames
      sub_vec = audio_vec[start_index : end_index]
      if len(sub_vec) != duration_frames:
        continue
      for roll in ROLLS:
        roll_frames = math.floor(duration_frames * roll)
        d[call_type].append(np.roll(sub_vec, roll_frames))

  return d


if __name__ == '__main__':
  if len(sys.argv)%3 != 1:
    print("Error: Number of audio files, annotation files, and channel numbers do not match")
  else:
    divisor = math.floor(len(sys.argv)/3)
    audio_files = sys.argv[1:1+divisor]
    annotations_files = sys.argv[1+divisor:1+2*divisor]
    channel_numbers = [int(s) for s in sys.argv[1+2*divisor:]]
    vec_d = get_training_data(audio_files, annotations_files, channel_numbers)
    for call_type, vecs in vec_d.items():
      np.savetxt(f'training_data/{call_type.filename}.csv', vecs, delimiter=',', fmt='%d')

