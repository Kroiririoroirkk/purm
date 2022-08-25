from collections import namedtuple
import enum
import pickle
import scipy.io.wavfile
import sys

from sound_segment.config import CallType, get_channel
from sound_segment.detect_sound import detect_sounds


class CallType2(enum.Enum):
  SONG = "song"
  WHISTLE = "whistle"
  BURBLE = "burble"
  CHATTER = "chatter"


Timestamp = namedtuple('Timestamp', ['start_t', 'end_t', 'sinr'])


def cluster_overlapping(timestamps):
  """Given a list of potentially overlapping timestamps, concatenate together the ones which overlap.

  Keyword arguments:
  timestamps - a list of Timestamp

  Returns:
  A list of Timestamp
  """
  new_timestamps = []
  for start_time, end_time, sinr in timestamps:
    if (not new_timestamps) or (start_time > new_timestamps[-1].end_t):
      new_timestamps.append(Timestamp(start_time, end_time, sinr))
    else:
      old_start_time, _, old_sinr = new_timestamps.pop()
      new_timestamps.append(Timestamp(old_start_time, end_time, max(old_sinr, sinr)))
  return new_timestamps


def cluster_songs(burble_timestamps, whistle_timestamps):
  """Given a list of burble and whistle timestamps, identify song timestamps and whistle timestamps.

  Keyword arguments:
  burble_timestamps - a list of Timestamp
  whistle_timestamps - a list of Timestamp

  Returns:
  A 3-tuple, the first element being a list of Timestamp of songs, the second element being a list of Timestamp of whistles, the third element being a list of Timestamp of burbles.
  """
  song_timestamps = []
  new_whistle_timestamps = []
  new_burble_timestamps = []
  for b_start, b_end, b_sinr in burble_timestamps:
    if song_timestamps and (song_timestamps[-1][0] < b_start < song_timestamps[-1][1]):
      continue # We already added this song to the song_timestamps list
    while whistle_timestamps and (whistle_timestamps[0][0] < b_start): # Add any whistles not associated with a song to the new_whistles_timestamps list
      w = whistle_timestamps.pop(0)
      new_whistle_timestamps.append(w)
    if whistle_timestamps and (whistle_timestamps[0][0] - b_end < 1): # We are going to say that a song is when the start of the whistle occurs within 1 second of the start of the burble (arbitrary number)
      _, w_end, w_sinr = whistle_timestamps.pop(0)
      song_timestamps.append(Timestamp(b_start, w_end, w_sinr))
    else:
      new_burble_timestamps.append(Timestamp(b_start, b_end, b_sinr))
  new_whistle_timestamps += whistle_timestamps # Make sure to include remaining whistles
  return song_timestamps, new_whistle_timestamps, new_burble_timestamps

if __name__ == '__main__':
  # Format: python segment_audio.py <audio_file.wav>
  audio_filename = sys.argv[1]
  print('Reading audio file...')
  sampling_rate, audio_vec = scipy.io.wavfile.read(audio_filename)

  timestamps = dict()
  for ct in [CallType.WHISTLE, CallType.BURBLE, CallType.CHATTER]:
    print(f'Detecting {ct.filename}s...')
    detected = detect_sounds(audio_vec, ct, get_channel(audio_filename), sampling_rate=sampling_rate)
    for s,e,sinr in detected:
      print(f'Found a {ct.filename} from {s:.3f} to {e:.3f} seconds (SINR = {sinr:.3f} dB)')
    timestamps[ct] = cluster_overlapping(Timestamp(s,e,sinr) for s,e,sinr in detected)

  print('Processing sound detections...')
  s_timestamps, w_timestamps, b_timestamps = cluster_songs(timestamps[CallType.BURBLE], timestamps[CallType.WHISTLE])
  new_timestamps = {
    CallType2.SONG: [t for t in s_timestamps if t.sinr > -8],
    CallType2.WHISTLE: [t for t in w_timestamps if t.sinr > -7], # Be more selective for whistles than songs (more likely to produce false positives)
    CallType2.BURBLE: [], # The algorithm is not well-attuned enough to detect these with any reliability
    CallType2.CHATTER: [t for t in timestamps[CallType.CHATTER] if t.sinr > -17]
  }
  print('Printing processed sound detections...')
  for ct, ts in new_timestamps.items():
    for s,e,sinr in ts:
      print(f'Found a {ct.value} from {s:.3f} to {e:.3f} seconds (SINR = {sinr:.3f} dB)')

  print('Dumping to file...')
  with open('segment_output.pickle', 'wb') as f:
    pickle.dump(new_timestamps, f)
