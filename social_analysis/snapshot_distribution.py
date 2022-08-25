from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
  with np.load(sys.argv[1]) as f:
    ts, positions = f['ts'], f['positions']
  
  dt = ts[1] - ts[0]
  time_bin_size = 5.0 # In seconds
  di = int(time_bin_size/dt)
  
  N = 15 # Number of birds
  pairs = list(combinations(range(0,N), 2))
  
  dists = dict()
  for pair in pairs:
    dists[pair] = []
  for i in range(0, len(ts), di):
    for pair in pairs:
      pos_1 = positions[i][pair[0]]
      pos_2 = positions[i][pair[1]]
      if pos_1[0] and pos_2[0]:
        dists[pair].append(np.linalg.norm(pos_1-pos_2))
      else:
        dists[pair].append(np.nan)

  dists_list = [val for vals in dists.values() for val in vals if not np.isnan(val)]
  plt.hist(dists_list, bins=50, range=(0,7))
  plt.title(f'Pairwise distances for {sys.argv[1][10:-4]}\nper every 5 seconds')
  plt.xlabel('Pairwise distance')
  plt.ylabel('Count')
  plt.savefig('plot.png')
