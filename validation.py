import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.io.wavfile
import sounddevice as sd
import sys

from config import CallType
from extract_training_data import parse_line, timestamp_to_frame


def parse_output_line(s):
  """Parses a line of the algorithm's output text file.

  Keyword arguments:
  s -- the line

  Returns:
  A 3-tuple of the start timestamp (in seconds), the end timestamp (in seconds), and the signal-to-interference-and-noise ratio (in dB).

  Raises:
  A ValueError if the input line is ill-formed.
  """
  try:
    start_time, end_time, sinr = s.strip().split(',')
  except ValueError:
    raise ValueError("Ill-formed input line: " + s)
  return (float(start_time.strip()),
          float(end_time.strip()),
          float(sinr.strip()))


def within_any(n, intervals):
  """Checks if a number n is in any of the intervals.

  Keyword arguments:
  n -- a number
  intervals -- a list of two-tuples

  Returns:
  True if there is a tuple (a,b) in intervals such that a < n < b and false otherwise.
  """
  return any(a < n < b for a,b in intervals)


def validate(audio_filename, channel_number, annotations_filename, call_type, manual=False, output_filename='output/output.csv'):
  """Validate the algorithm's output against manual annotations.

  Keyword arguments:
  audio_filename -- the filename of the audio
  annotations_filename -- the filename of the manual annotations
  call_type -- what call type to look for
  manual -- whether or not to manually detect false positives (default False)
  output_filename -- the filename of the output (default 'output/output.csv')

  Returns:
  A 2-tuple of the sensitivity and specificity of the test
  """
  with open(annotations_filename, 'r') as f:
    annotation_lines = [parse_line(s) for s in f.read().strip().split('\n')]
  annotation_lines = [(s,e) for s,e,t in annotation_lines if t == call_type]

  with open(output_filename, 'r') as f:
    output_lines = [parse_output_line(s) for s in f.read().strip().split('\n')]
  output_lines = [(s,e) for s,e,_ in output_lines]

  true_positive_count = 0
  for start_time, end_time in annotation_lines:
    if within_any((start_time + end_time)/2, output_lines):
      true_positive_count += 1
  sensitivity = true_positive_count / len(annotation_lines)

  false_positive_count = 0
  if manual:
    sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
    if sampling_rate != 48000:
      raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
    audio_vec = audio_vec[:, channel_number] # ignore all channels except one
    for start_time, end_time in output_lines:
      start_index = timestamp_to_frame(start_time, sampling_rate)
      end_index = timestamp_to_frame(end_time, sampling_rate)
      sub_vec = audio_vec[start_index : end_index]
      plt.specgram(sub_vec, Fs=sampling_rate)
      play_button_axes = plt.axes([0.81, 0.05, 0.1, 0.075])
      play_button = Button(play_button_axes, 'Play')
      play_button.on_clicked(lambda _: sd.play(sub_vec, sampling_rate))
      correct_button_axes = plt.axes([0.81-0.15*2, 0.05, 0.1, 0.075])
      correct_button = Button(correct_button_axes, 'Correct')
      correct_button.on_clicked(lambda _: plt.close())
      incorrect_button_axes = plt.axes([0.81-0.15, 0.05, 0.1, 0.075])
      incorrect_button = Button(incorrect_button_axes, 'Incorrect')
      def on_incorrect_click(_):
        nonlocal false_positive_count
        false_positive_count = false_positive_count + 1
        plt.close()
      incorrect_button.on_clicked(on_incorrect_click)
      plt.show()
  else:
    for start_time, end_time in output_lines:
      if not within_any((start_time + end_time)/2, annotation_lines):
        false_positive_count += 1
  specificity = 1 - (false_positive_count / len(output_lines))

  return sensitivity, specificity


if __name__ == '__main__':
  audio_filename = sys.argv[1]
  annotations_filename = sys.argv[2]
  call_type = CallType.from_str(sys.argv[3])
  manual = False
  try:
    if sys.argv[4] == 'manual':
      manual = True
  except IndexError:
    pass
  if audio_filename == 'audio/aviary_2019-05-01_1556722860.000-1556723760.000_audio.wav':
    channel = 8
  elif audio_filename == 'audio/aviary_2019-06-01_1559399640.000-1559400540.000_audio.wav':
    channel = 15
  sensitivity, specificity = validate(audio_filename, channel, annotations_filename, call_type, manual)
  print(f'The sensitivity is {sensitivity:.2%}.')
  print(f'The specificity is {specificity:.2%}.')

