from collections import namedtuple
import enum
import numpy as np
import pickle
import scipy.io.wavfile
import sys
from torch import FloatTensor

from sound_segment.config import CallType, get_channel
from sound_segment.detect_sound import detect_sounds
from sound_localize.cameras.cameras import CameraSystem
from sound_localize.SoundLoc.SoundLoc import localize_sound


class CallType2(enum.Enum):
  SONG = "song"
  WHISTLE = "whistle"
  BURBLE = "burble"
  CHATTER = "chatter"


OutputEntry = namedtuple('OutputEntry', ['start_t', 'end_t', 'call_type', 'camera_positions', 'bird_id'])
CameraEntry = namedtuple('CameraEntry', ['camera', 'x', 'y'])


def cluster_overlapping(timestamps):
  """Given a list of potentially overlapping timestamps, concatenate together the ones which overlap.

  Keyword arguments:
  timestamps - a list of 2-tuples of start times and end times

  Returns:
  A list of 2-tuples of start times and end times.
  """
  new_timestamps = []
  for start_time, end_time in timestamps:
    if (not new_timestamps) or (start_time > new_timestamps[-1][1]):
      new_timestamps.append((start_time, end_time))
    else:
      old_start_time, _ = new_timestamps.pop()
      new_timestamps.append((old_start_time, end_time))
  return new_timestamps


def cluster_songs(burble_timestamps, whistle_timestamps):
  """Given a list of burble and whistle timestamps, identify song timestamps and whistle timestamps.

  Keyword arguments:
  burble_timestamps - a list of 2-tuples of start times and end times
  whistle_timestamps - a list of 2-tuples of start times and end times

  Returns:
  A 3-tuple, the first element being a list of 2-tuples for start times and end times of songs, the second element being a list of 2-tuples for start times and end times of whistles, the third element being a list of 2-tuples for start times and end times of burbles.
  """
  song_timestamps = []
  new_whistle_timestamps = []
  new_burble_timestamps = []
  for b_start, b_end in burble_timestamps:
    if song_timestamps and (song_timestamps[-1][0] < b_start < song_timestamps[-1][1]):
      continue # We already added this song to the song_timestamps list
    if whistle_timestamps:
      while whistle_timestamps[0][0] < b_start: # Add any whistles not associated with a song to the new_whistles_timestamps list
        w = whistle_timestamps.pop(0)
        new_whistle_timestamps.append(w)
      if whistle_timestamps[0][0] - b_start < 0.7: # We are going to say that a song is when the start of the whistle occurs within 0.7 seconds of the start of the burble (arbitrary number)
        _, w_end = whistle_timestamps.pop(0)
        song_timestamps.append((b_start, w_end))
      else:
        new_burble_timestamps.append((b_start, b_end))
    else:
      new_burble_timestamps.append((b_start, b_end))
  new_whistle_timestamps += whistle_timestamps # Make sure to include remaining whistles
  return song_timestamps, new_whistle_timestamps, new_burble_timestamps


if __name__ == '__main__':
  # Format: python find_calls <audio_file.wav>
  audio_filename = sys.argv[1]
  print('Reading audio file...')
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)

  timestamps = dict()
  for ct in [CallType.WHISTLE, CallType.BURBLE, CallType.CHATTER]:
    print(f'Detecting {ct.filename}s...')
    detected = detect_sounds(audio_vec, ct, get_channel(audio_filename), sampling_rate=sampling_rate)
    for s,e,sinr in detected:
      print(f'Found a {ct.filename} from {s:.3f} to {e:.3f} seconds (SINR = {sinr:.3f} dB)')
    detected = [(s,e) for s,e,_ in detected]
    timestamps[ct] = cluster_overlapping(detected)

  print('Processing sound detections...')
  s_timestamps, w_timestamps, b_timestamps = cluster_songs(timestamps[CallType.BURBLE], timestamps[CallType.WHISTLE])
  new_timestamps = {
    CallType2.SONG: s_timestamps,
    CallType2.WHISTLE: w_timestamps,
    CallType2.BURBLE: b_timestamps,
    CallType2.CHATTER: timestamps[CallType.CHATTER]
  }

  mic_pos = np.loadtxt('sound_localize/SoundLoc/data/aviary_mic_positions.txt', delimiter=',')
  cage_dims = np.loadtxt('sound_localize/SoundLoc/data/cage_coords.txt', delimiter=',')
  cam_sys = CameraSystem('sound_localize/cameras/aviary_2019-06-01_calibration.yaml')

  locs = []
  i = 0
  for ct, ts in new_timestamps.items():
    print(f'Localizing {ct.value}s...')
    new_timestamps[ct] = []
    for start_t, end_t in ts:
      audio_subvec = audio_vec[int(start_t*sampling_rate):int(end_t*sampling_rate)].T
      loc, bird_id = localize_sound(mic_pos, cage_dims, audio_subvec, sampling_rate)
      locs.append(loc)
      new_timestamps[ct].append((start_t, end_t, bird_id, i))
      i += 1
  print('Projecting onto camera perspectives...')
  locs = np.row_stack(locs)
  locs = cam_sys.perspective_projection(locs)
  locs = np.swapaxes(locs, 0, 1)

  print('Creating output array...')
  output_arr = []
  for ct, ts in new_timestamps.items():
    for start_t, end_t, bird_id, i in ts:
      cam_entries = [CameraEntry(camera=j, x=x, y=y)
        for j,(x,y,visible) in enumerate(locs[i]) if visible]
      output_arr.append(OutputEntry(start_t=start_t,
                        end_t=end_t,
                        call_type=ct,
                        camera_positions=cam_entries,
                        bird_id=bird_id))
  output_arr.sort(key=lambda o: o.start_t)

  print('Dumping to file...')
  with open('output.pickle', 'wb') as f:
    pickle.dump(output_arr, f)

