import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io.wavfile
import sys

from config import CallType, DatasetType, ROLLS, get_channel
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
  audio_filenames -- a list of the filenames of the audio files
  annotations_filenames -- a list of the filenames of the annotations files
  channel_number -- which channels of the audio to use (zero-indexed)

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
    audio_vecs = dict()
    for call_type in CallType:
      audio_vecs[call_type] = preprocess(audio_vec, sampling_rate, call_type)

    with open(annotations_filename, 'r') as f:
      lines = [parse_line(s) for s in f.read().strip().split('\n')]

    for start_time, end_time, call_type in lines:
      duration_frames = timestamp_to_frame(call_type.duration, audio_vecs[call_type][1])
      avg_time = (start_time + end_time)/2
      avg_index = timestamp_to_frame(avg_time, audio_vecs[call_type][1])
      start_index = avg_index - math.floor(duration_frames/2)
      end_index = start_index + duration_frames
      sub_vec = audio_vecs[call_type][0][start_index : end_index]
      if len(sub_vec) != duration_frames:
        continue
      for roll in ROLLS:
        roll_frames = math.floor(duration_frames * roll)
        d[call_type].append(np.roll(sub_vec, roll_frames))

  d_with_datasettypes = dict()
  for call_type, vecs in d.items():
    random.shuffle(vecs)
    partition1 = math.floor(len(vecs)*DatasetType.INIT.proportion)
    partition2 = partition1 + math.floor(len(vecs)*DatasetType.TRAIN.proportion)
    init_vecs = vecs[:partition1]
    train_vecs = vecs[partition1:partition2]
    validate_vecs = vecs[partition2:]
    d_with_datasettypes[(call_type, DatasetType.INIT)] = init_vecs
    d_with_datasettypes[(call_type, DatasetType.TRAIN)] = train_vecs
    d_with_datasettypes[(call_type, DatasetType.VALIDATE)] = validate_vecs

  return d_with_datasettypes


if __name__ == '__main__':
  if len(sys.argv)%3 != 1:
    print("Error: Number of audio files, annotation files, and channel numbers do not match")
  else:
    divisor = math.floor(len(sys.argv)/2)
    audio_files = sys.argv[1:1+divisor]
    annotations_files = sys.argv[1+divisor:]
    channel_numbers = [get_channel(fn) for fn in audio_files]
    vec_d = get_training_data(audio_files, annotations_files, channel_numbers)
    for (call_type, dataset_type), vecs in vec_d.items():
      np.savetxt(f'training_data/{dataset_type.filename}_{call_type.filename}.csv', vecs, delimiter=',', fmt='%.2f')

