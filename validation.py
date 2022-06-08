import sys

from config import CallType
from process_data import parse_line


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


def validate(annotations_filename, output_filename, call_type):
  """Validate the algorithm's output against manual annotations.

  Keyword arguments:
  annotations_filename -- the filename of the annotations
  output_filename -- the filename of the output
  call_type -- what call type to look for

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
  for start_time, end_time in output_lines:
    if not within_any((start_time + end_time)/2, annotation_lines):
      false_positive_count += 1
  specificity = 1 - (false_positive_count / len(output_lines))

  return sensitivity, specificity


if __name__ == '__main__':
  annotations_filename = f'{sys.argv[1]}'
  output_filename = f'{sys.argv[2]}'
  call_type = CallType.from_str(sys.argv[3])
  sensitivity, specificity = validate(annotations_filename, output_filename, call_type)
  print(f'The sensitivity is {sensitivity:.2%}.')
  print(f'The specificity is {specificity:.2%}.')

