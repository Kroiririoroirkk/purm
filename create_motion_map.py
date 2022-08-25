import matplotlib.pyplot as plt

from overlay_video.readin import get_annotations, get_3D_annotations, AnnoEntry3D, CAMERAS, View, Quarter, FRAME_SIZE
from sound_localize.cameras.cameras import CameraSystem


BIRDS_TO_DISPLAY = list(range(1,15+1))

"""
#aviary_2019-05-15_1557931500.000-1557932400.000_bird
MALE_BIRDS = [2,4,6,12,13,14]
FEMALE_BIRDS = [1,3,5,7,8,9,10,11,15]
BIRDS = {
  1: 'Female Yellow Blue',
  2: 'Male Blue Green',
  3: 'Female Pink Green',
  4: 'Male Green Teal',
  5: 'Female Pink Red',
  6: 'Male Teal Red',
  7: 'Female Yellow Green',
  8: 'Female Blue Pink',
  9: 'Female Teal Pink',
  10: 'Female Blue Teal',
  11: 'Female Yellow Teal',
  12: 'Male Pink Yellow',
  13: 'Male Blue Red',
  14: 'Male Red Green',
  15: 'Female Red Yellow'
}
"""

#"""
#aviary_2019-06-01_1559412240.000-1559413140.000_bird
MALE_BIRDS = [1, 5, 6, 8, 9, 15]
FEMALE_BIRDS = [2, 3, 4, 7, 10, 11, 12, 13, 14]
BIRDS = {
  1: 'Male Teal Red',
  2: 'Female Blue Teal',
  3: 'Female Pink Red',
  4: 'Female Yellow Blue',
  5: 'Male Red Green',
  6: 'Male Green Teal',
  7: 'Female Teal Pink',
  8: 'Male Pink Yellow',
  9: 'Male Blue Green',
  10: 'Female Red Yellow',
  11: 'Female Blue Pink',
  12: 'Female Yellow Teal',
  13: 'Female Yellow Green',
  14: 'Female Pink Green',
  15: 'Male Blue Red'
}
#"""
MALE_BIRDS.sort(key=lambda i: BIRDS[i])
FEMALE_BIRDS.sort(key=lambda i: BIRDS[i])


if __name__ == '__main__':
  annos, _ = get_annotations('jsons/aviary_2019-06-01_1559412240.000-1559413140.000/aviary_2019-06-01_1559412240.000-1559413140.000_bird')
  _, paths3d = get_3D_annotations(annos, 'sound_localize/cameras/aviary_2019-06-01_calibration.yaml')

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  for bird_no in MALE_BIRDS:
    _, xs, ys, zs, _ = zip(*paths3d[bird_no])
    ax.scatter(xs, ys, zs, label=f'{BIRDS[bird_no]}', marker='s')
    #ax.plot(xs, ys, zs)
  for bird_no in FEMALE_BIRDS:
    _, xs, ys, zs, _ = zip(*paths3d[bird_no])
    ax.scatter(xs, ys, zs, label=f'{BIRDS[bird_no]}', marker='^')
    #ax.plot(xs, ys, zs)
  ax.legend(loc=(1.2,0))
  plt.title('aviary_2019-06-01_1559412240.000-1559413140.000')
  plt.show()
  fig.savefig('motion_map.png', dpi=600)
