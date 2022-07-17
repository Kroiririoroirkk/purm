from collections import namedtuple
import cv2
import enum
from itertools import groupby
import json

# --------------------- #

N = 15
FPS = 25
FRAME_SIZE = (3840, 2400)
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.8
MALE_COLOR = (0, 255, 0)
FEMALE_COLOR = (255, 0, 255)
THICKNESS = 7
ANNOTATIONS_PREFIX = 'jsons/aviary_2019-06-01_1559412240.000-1559413140.000_bird'

# --------------------- #

class AnnoType(enum.Enum):
  START = '0'
  MIDDLE = '1'
  END = '2'

class View(enum.Enum):
  TOP = 'TOP'
  BOTTOM = 'BOTTOM'

class Quarter(enum.Enum):
  TL = 'TL'
  TR = 'TR'
  BL = 'BL'
  BR = 'BR'

AnnoEntry = namedtuple('AnnoEntry', ['uid', 'time', 'anno_type', 'x', 'y', 'quarter'])

class SongType(enum.Enum):
  CHATTER = 'Chatter'
  CHATTER_UNSURE = 'Chatter? Unsure'
  MALE_DIR = 'Male Directed Song'
  FEMALE_DIR = 'Female Directed Song'
  MALE_COUNTER = 'Male Countersong'
  INDIRECT = 'Indirect Song'
  UNKNOWN_SONG = 'Song, Unsure'
  HEADS_UP = 'Heads-up Display'

  @staticmethod
  def from_str(s):
    d = {
      'Chatter': SongType.CHATTER,
      'Chatter? I have no idea': SongType.CHATTER_UNSURE,
      'Male Directed Song': SongType.MALE_DIR,
      'Female Directed Song': SongType.FEMALE_DIR,
      'Male Countersong': SongType.MALE_COUNTER,
      'Indirect Song': SongType.INDIRECT,
      'Song, unsure': SongType.UNKNOWN_SONG,
      'Song, not sure': SongType.UNKNOWN_SONG,
      'Heads Up Display': SongType.HEADS_UP
    }
    return d[s]

SongEntry = namedtuple('SongEntry', ['uid', 'start_time', 'end_time', 'song_type'])

class Sex(enum.Enum):
  MALE = MALE_COLOR
  FEMALE = FEMALE_COLOR

  @staticmethod
  def from_str(s):
    d = {'m': Sex.MALE, 'f': Sex.FEMALE}
    return d[s]

# --------------------- #

def get_annotations():
  annotations = dict()
  song_annotations = dict()

  for i in range(N):
    annotations[i+1] = {
      View.TOP: [],
      View.BOTTOM: []
    }
    song_annotations[i+1] = {
      View.TOP: [],
      View.BOTTOM: []
    }
    with open(f'{ANNOTATIONS_PREFIX}{i+1:02d}.json') as f:
      j = json.load(f)
      for number, descr in j['file'].items():
        if descr['fname'].endswith('top.mp4'):
          top = number
        elif descr['fname'].endswith('bot.mp4'):
          bottom = number
      for uid, sub_j in j['metadata'].items():
        if len(sub_j['z']) == 1:
          time = sub_j['z'][0]
        elif len(sub_j['z']) == 2:
          start_time, end_time = sub_j['z']          
          song_type = SongType.from_str(sub_j['av']['1'])
          entry = SongEntry(
            uid=uid,
            start_time=start_time,
            end_time=end_time,
            song_type=song_type
          )
          song_annotations[i+1][view].append(entry)
          continue
        else:
          continue
        if sub_j['vid'] == top:
          view = View.TOP
        else:
          view = View.BOTTOM
        if len(sub_j['xy']) != 3:
          raise ValueError('Unexpected behavior')
        _, x, y = sub_j['xy']
        x, y = int(x), int(y)
        if x > FRAME_SIZE[0]/2:
          if y > FRAME_SIZE[1]/2:
            quarter = Quarter.BR
          else:
            quarter = Quarter.TR
        else:
          if y > FRAME_SIZE[1]/2:
            quarter = Quarter.BL
          else:
            quarter = Quarter.TL
        anno_type = AnnoType(sub_j['av']['2'])
        entry = AnnoEntry(
          uid=uid,
          time=time,
          anno_type=anno_type,
          x=x,
          y=y,
          quarter=quarter
        )
        annotations[i+1][view].append(entry)
    for view in View:
      annotations[i+1][view].sort(key=lambda e: e.time)
      groups = []
      for _, g in groupby(annotations[i+1][view], key=lambda e: e.time):
        groups.append(list(g))
      annotations[i+1][view] = groups
  return annotations, song_annotations


if __name__ == '__main__':
  print(get_annotations())
