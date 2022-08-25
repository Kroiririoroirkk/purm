import csv
from itertools import combinations
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
  # Usage: python avg_distance_heatmap.py <positions_file> <id_file>
  with np.load(sys.argv[1]) as f:
    ts, positions = f['ts'], f['positions']
  with open(sys.argv[2], newline='') as csvfile:
    id_reader = csv.reader(csvfile)
    ids = {int(k)-1: v.upper() for k,v in id_reader}
  
  N = 15 # Number of birds
  pairs = list(combinations(range(0,N), 2))
  
  dists = dict()
  for pair in pairs:
    dists[ids[pair[0]], ids[pair[1]]] = []
  for i in range(0, len(ts)):
    for pair in pairs:
      k = ids[pair[0]], ids[pair[1]]
      pos_1 = positions[i][pair[0]]
      pos_2 = positions[i][pair[1]]
      if pos_1[0] and pos_2[0]:
        dists[k].append(np.linalg.norm(pos_1-pos_2))
      else:
        dists[k].append(np.nan)

  for k, v in dists.items():
    v = [dist for dist in v if not np.isnan(dist)]
    dists[k] = sum(v)/len(v)

  image = np.empty((N,N))
  birds = sorted(ids.values())
  for i in range(0,N):
    for j in range(0,N):
      if i == j:
        image[i][j] = 0
      else:
        try:
          image[i][j] = dists[birds[i],birds[j]]
        except KeyError:
          image[i][j] = dists[birds[j],birds[i]]

  plt.title(f'Average pairwise distances for\n{sys.argv[1][10:-4]}')
  plt.imshow(image, norm=Normalize(0, 4))
  plt.colorbar()
  plt.xticks(ticks=range(0,N), labels=birds, rotation='vertical')
  plt.xlabel('Bird')
  plt.yticks(ticks=range(0,N), labels=birds)
  plt.ylabel('Bird')
  plt.savefig('heatmap.png', bbox_inches='tight')
