import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys


if __name__ == '__main__':
  # Usage: python plot_spectrogram.py <filename> <start_time (in s)> <end_time (in s)> <width (in inches)> <height (in inches)>
  fs, vec = scipy.io.wavfile.read(sys.argv[1])
  vec = vec[:, 0]
  start = int(float(sys.argv[2])*fs)
  end = int(float(sys.argv[3])*fs)
  fig = plt.figure(dpi=400)
  fig.set_size_inches(float(sys.argv[4]), float(sys.argv[5]))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.specgram(vec[start:end], Fs=fs)
  plt.savefig('plot.png')
