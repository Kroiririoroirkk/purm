import numpy as np
import sys
from torch import FloatTensor

from edit_video import determine_3D_pos
from overlay_video.readin import get_3D_annotations, get_annotations


FREQ = 10 # In Hertz
N = 15 # Number of birds


if __name__ == '__main__':
  # Usage: python create_location_time_series <annotations_json_file_prefix>
  anno_prefix = sys.argv[1]
  cam_sys, annotations = get_3D_annotations(get_annotations(anno_prefix)[0])
  ts = np.linspace(0, 15*60, 15*60*FREQ)
  positions = []
  for t in ts:
    l = []
    for bird_no in range(1,N+1):
      coord = determine_3D_pos(annotations, t, bird_no)
      if isinstance(coord, FloatTensor):
        x,y,z = coord.flatten().tolist()
        l.append([x,y,z])
      else:
        l.append([np.nan,np.nan,np.nan])
    positions.append(l)
  positions = np.array(positions)
  np.savez_compressed('positions.npz', ts=ts, positions=positions)
