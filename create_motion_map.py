import matplotlib.pyplot as plt
import sys

from overlay_video.readin import get_annotations, get_3D_annotations, AnnoEntry3D, CAMERAS, View, Quarter, FRAME_SIZE
from sound_localize.cameras.cameras import CameraSystem


BIRDS_TO_DISPLAY = list(range(1,15+1))


if __name__ == '__main__':
  anno_filename_prefix = sys.argv[1]
  id_filename = sys.argv[2]
  title = sys.argv[3]
  BIRDS = dict()
  MALE_BIRDS = []
  FEMALE_BIRDS = []
  with open(id_filename, 'r') as f:
    for line in f.readlines():
      num, name = line.strip().split(',')
      num = int(num)
      name = name.upper()
      BIRDS[num] = name
      if name.startswith('M'):
        MALE_BIRDS.append(num)
      elif name.startswith('F'):
        FEMALE_BIRDS.append(num)
  MALE_BIRDS.sort(key=lambda i: BIRDS[i])
  FEMALE_BIRDS.sort(key=lambda i: BIRDS[i])
  
  annos, _, _ = get_annotations(anno_filename_prefix)
  _, paths3d = get_3D_annotations(annos, 'sound_localize/cameras/aviary_2019-06-01_calibration.yaml')

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  for bird_no in MALE_BIRDS:
    _, xs, ys, zs, _ = zip(*paths3d[bird_no])
    ax.scatter(xs, ys, zs, label=f'{BIRDS[bird_no]}', marker='s', c='#ff0000')
    #ax.plot(xs, ys, zs)
  for bird_no in FEMALE_BIRDS:
    _, xs, ys, zs, _ = zip(*paths3d[bird_no])
    ax.scatter(xs, ys, zs, label=f'{BIRDS[bird_no]}', marker='^', c='#00ff00')
    #ax.plot(xs, ys, zs)
  ax.legend(loc=(1.2,0))
  plt.title(title)
  plt.show()
  fig.savefig('motion_map.png', dpi=600)
