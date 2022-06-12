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


def get_training_data(audio_filename, annotations_filename, channel_number):
  """Reads an audio file and annotation file to obtain training data.

  Keyword arguments:
  audio_filename -- the filename of the audio file
  annotations_filename -- the filename of the annotations file
  channel_number -- which channel of the audio to use (zero-indexed)

  Returns:
  A dictionary with CallTypes as keys and lists of numpy 1D vectors as values. Raises an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
  if sampling_rate != 48000:
    raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
  audio_vec = audio_vec[:, channel_number] # ignore all channels except one
  audio_vec, sampling_rate = preprocess(audio_vec, sampling_rate)

  with open(annotations_filename, 'r') as f:
    lines = [parse_line(s) for s in f.read().strip().split('\n')]

  d = dict()
  for call_type in CallType:
    d[call_type] = []
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
  audio_file = sys.argv[1]
  annotations_file = sys.argv[2]
  vec_d = get_training_data(audio_file, annotations_file, 8)
  for call_type, vecs in vec_d.items():
    np.savetxt(f'training_data/{call_type.filename}.csv', vecs, delimiter=',', fmt='%d')
