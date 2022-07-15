import matplotlib.pyplot as plt
import sys

from config import CallType
from extract_training_data import parse_line


def plot_call_durations(annotations_filename):
  """Reads an annotation file and generates a histogram of the call durations by call type.

  Keyword arguments:
  annotations_filename -- the filename of the annotations file

  Returns:
  Nothing. Generates histograms using matplotlib.
  """
  durations = dict()
  with open(annotations_filename, 'r') as f:
    lines = [parse_line(s) for s in f.read().strip().split('\n')]
  for call_type in CallType:
    durations[call_type] = []
  for start_time, end_time, call_type in lines:
    durations[call_type].append(end_time - start_time)
  for call_type, durs in durations.items():
    plt.hist(durs)
    plt.title(call_type.filename)
    plt.show()


if __name__ == '__main__':
  plot_call_durations(sys.argv[1])
