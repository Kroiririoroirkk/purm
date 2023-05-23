import matplotlib.pyplot as plt
import numpy as np
import sys

from overlay_video.readin import get_annotations, get_3D_annotations, AnnoEntry3D, CAMERAS, View, Quarter, FRAME_SIZE
from sound_localize.cameras.cameras import CameraSystem

CENTER = np.array([(6.003-0.002)/2, (2.376-0.077)/2, (2.476-0.022)/2])

if __name__ == '__main__':
  # Usage: python plot_distance_from_center.py <position_time_series> <bird_id_filename> <plot_title>
  means = []
  anno_position = sys.argv[1]
  id_filename = sys.argv[2]
  title = sys.argv[3]

  with np.load(anno_position) as f:
    positions = f['positions']
  for i in range(15):
    bird_pos = positions[:, i]
    dists = []
    for pos in bird_pos:
      if not np.isnan(pos[0]):
        dists.append(np.linalg.norm(pos - CENTER))
    means.append(np.mean(dists))

  BIRDS = []
  with open(id_filename, 'r') as f:
    for line in f.readlines():
      _, name = line.strip().split(',')
      name = name.upper()
      BIRDS.append(name)
  print(BIRDS, means)
  data_to_plot = list(zip(BIRDS, means))
  data_to_plot.sort(key=lambda t: t[0])
  sorted_birds, sorted_means = zip(*data_to_plot)
  plt.bar(sorted_birds, sorted_means, color=['g','g','g','g','g','g','g','g','g','b','b','b','b','b','b'])
  plt.title(title)
  plt.xlabel('Bird')
  plt.ylabel('Mean distance from center (m)')
  plt.savefig('mean_distance_graph.png')

