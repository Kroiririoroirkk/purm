import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram
import sys

from config import CallType


def plot_spectral_densities(filenames, sampling_rate=48000):
  """Plots the spectral densities of the files averaged together.

  Keyword arguments:
  filenames -- a list of filenames to display
  sampling_rate -- sampling rate in Hz (default 48000)

  Returns:
  Nothing. Displays an average spectral density plot.
  """

  freq_vals = []
  spectral_densities = []
  for fn in filenames:
    vecs = np.loadtxt(fn, delimiter=',')
    for vec in vecs:
      freq_vals, spectral_density = periodogram(vec, fs=sampling_rate)
      spectral_densities.append(spectral_density)
      #plt.semilogy(freq_vals, spectral_density)
      #plt.ylim([1e-4, 1e4])
      #plt.show()
  avg_spectral_density = np.mean(spectral_densities, axis=0)
  plt.semilogy(freq_vals, avg_spectral_density)
  plt.ylim([1e-4, 1e4])
  plt.minorticks_on()
  plt.show()


if __name__ == '__main__':
  plot_spectral_densities(sys.argv[1:])
