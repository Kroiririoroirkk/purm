import matplotlib.pyplot as plt
import scipy

from process_data import timestamp_to_frame
from validation import parse_output_line

def view_output(audio_filename, output_filename, channel_number):
  """Reads an audio file and an output file and generates spectograms.

  Keyword arguments:
  audio_filename -- the filename of the audio file
  output_filename -- the filename of the output file
  channel_number -- which channel of the audio to use (zero-indexed)

  Returns:
  Nothing. Displays spectograms. Throws an error if the audio is not at the expected sampling rate of 48 kHz.
  """
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)
  if sampling_rate != 48000:
    raise ValueError('The audio is not at the expected sampling rate of 48 kHz.')
  audio_vec = audio_vec[:, channel_number] # ignore all channels except one

  with open(output_filename, 'r') as f:
    lines = [parse_output_line(s) for s in f.read().strip().split('\n')]

  for start_time, end_time in lines:
    start_index = timestamp_to_frame(start_time, sampling_rate)
    end_index = timestamp_to_frame(end_time, sampling_rate)
    sub_vec = audio_vec[start_index : end_index]
    plt.specgram(sub_vec, Fs=sampling_rate)
    plt.show()

