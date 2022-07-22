from collections import namedtuple
import numpy as np
import pickle
import scipy.io.wavfile
import sys

from sound_localize.cameras.cameras import CameraSystem
from sound_localize.SoundLoc.SoundLoc import localize_sound
from segment_audio import CallType2, Timestamp


OutputEntry = namedtuple('OutputEntry', ['start_t', 'end_t', 'call_type', 'camera_positions', 'sinr'])
CameraEntry = namedtuple('CameraEntry', ['camera', 'x', 'y'])


if __name__ == '__main__':
  # Format: python localize_calls <audio_file.wav>
  print('Loading sound segmentation file...')
  with open('segment_output.pickle', 'rb') as f:
    timestamps = pickle.load(f)
  print('Sound segmentation file loaded.')

  audio_filename = sys.argv[1]
  print('Reading audio file...')
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)

  mic_pos = np.loadtxt('sound_localize/SoundLoc/data/aviary_mic_positions.txt', delimiter=',')
  cage_dims = np.loadtxt('sound_localize/SoundLoc/data/cage_coords.txt', delimiter=',')
  cam_sys = CameraSystem('sound_localize/cameras/aviary_2019-06-01_calibration.yaml')

  locs = []
  i = 0
  for ct, ts in timestamps.items():
    print(f'Localizing {ct.value}s...')
    timestamps[ct] = []
    for start_t, end_t, sinr in ts:
      audio_subvec = audio_vec[int(start_t*sampling_rate):int(end_t*sampling_rate)].T
      loc, _ = localize_sound(mic_pos, cage_dims, audio_subvec, sampling_rate)
      locs.append(loc)
      timestamps[ct].append((start_t, end_t, sinr, i))
      i += 1
  print('Projecting onto camera perspectives...')
  locs = np.row_stack(locs)
  locs = cam_sys.perspective_projection(locs)
  locs = np.swapaxes(locs, 0, 1)

  print('Creating output array...')
  output_arr = []
  for ct, ts in timestamps.items():
    for start_t, end_t, sinr, i in ts:
      cam_entries = [CameraEntry(camera=j, x=x, y=y)
        for j,(x,y,visible) in enumerate(locs[i]) if visible]
      output_arr.append(OutputEntry(start_t=start_t,
                        end_t=end_t,
                        call_type=ct,
                        camera_positions=cam_entries,
                        sinr=sinr))
  output_arr.sort(key=lambda o: o.start_t)

  print('Dumping to file...')
  with open('output.pickle', 'wb') as f:
    pickle.dump(output_arr, f)

