import numpy as np
import pickle

from find_calls import CallType2, OutputEntry, CameraEntry
from sound_localize.cameras.cameras import CameraSystem

cam_sys = CameraSystem('sound_localize/cameras/aviary_2019-06-01_calibration.yaml')
cam_observes = np.swapaxes(cam_sys.perspective_projection(cam_sys.location),0,1)

cam_out = []
for cam_observe in cam_observes:
  cam_entries = []
  for i,(x,y,visible) in enumerate(cam_observe):
    if visible:
      cam_entries.append(CameraEntry(camera=i,x=x,y=y))
  cam_out.append(OutputEntry(start_t=0,end_t=10,call_type=CallType2.CHATTER,camera_positions=cam_entries,bird_id=None))

with open('output.pickle','wb') as f:
  pickle.dump(cam_out, f)

