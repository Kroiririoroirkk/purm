from collections import namedtuple
import cv2
import enum
from itertools import groupby
import json
import numpy as np
from torch import FloatTensor

from sound_localize.cameras.cameras import CameraSystem

# --------------------- #

N = 15
FRAME_SIZE = (3840, 2400)
HALF_FRAME_SIZE = (1920, 1200)
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.8
MALE_COLOR = (0, 255, 0)
FEMALE_COLOR = (255, 0, 255)
THICKNESS = 7

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
AnnoEntry3D = namedtuple('AnnoEntry3D', ['t', 'x', 'y', 'z', 'anno_type'])

CAMERAS = {
  (View.TOP, Quarter.TL): 3,
  (View.TOP, Quarter.TR): 2,
  (View.TOP, Quarter.BL): 7,
  (View.TOP, Quarter.BR): 6,
  (View.BOTTOM, Quarter.TL): 0,
  (View.BOTTOM, Quarter.TR): 1,
  (View.BOTTOM, Quarter.BL): 4,
  (View.BOTTOM, Quarter.BR): 5
}
CAMERAS_LOOKUP = {v:k for k,v in CAMERAS.items()}

class SongType(enum.Enum):
  CHATTER = 'Chatter'
  CHATTER_UNSURE = 'Chatter, Unsure'
  MALE_DIR = 'Male Directed Song'
  FEMALE_DIR = 'Female Directed Song'
  MALE_COUNTER = 'Male Countersong'
  INDIRECT = 'Indirect Song'
  UNKNOWN_SONG = 'Song, Unsure'
  HEADS_UP = 'Heads-up Display'
  UNKNOWN = 'Unknown'

  @staticmethod
  def from_str(s):
    d = {
      'Chatter': SongType.CHATTER,
      'chatter': SongType.CHATTER,
      'Chatter? I have no idea': SongType.CHATTER_UNSURE,
      'Chatter?': SongType.CHATTER_UNSURE,
      'Male Directed Song': SongType.MALE_DIR,
      'male directed song': SongType.MALE_DIR,
      'Female Directed Song': SongType.FEMALE_DIR,
      'female directed song': SongType.FEMALE_DIR,
      'Male Countersong': SongType.MALE_COUNTER,
      'Male Countersong?': SongType.MALE_COUNTER,
      'Countersong': SongType.MALE_COUNTER,
      'Indirect Song': SongType.INDIRECT,
      'Undirected Song': SongType.INDIRECT,
      'Song, unsure': SongType.UNKNOWN_SONG,
      'Song, Unsure': SongType.UNKNOWN_SONG,
      'Song, not sure': SongType.UNKNOWN_SONG,
      'Song': SongType.UNKNOWN_SONG,
      'song': SongType.UNKNOWN_SONG,
      'Heads Up Display': SongType.HEADS_UP,
      'heads up display': SongType.HEADS_UP,
      'HeadUp': SongType.HEADS_UP,
      'Heads Down Display': SongType.UNKNOWN,
      '_DEFAULT': SongType.UNKNOWN,
      'Unsure': SongType.UNKNOWN
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

WingspanRawEntry = namedtuple('WingspanRawEntry', ['uid', 'time', 'x1', 'y1', 'x2', 'y2', 'quarter'])
WingspanEntry = namedtuple('WingspanEntry', ['t', 'length'])

# --------------------- #

def get_annotation(fname):
  annos = {
    View.TOP: [],
    View.BOTTOM: []
  }
  song_annos = {
    View.TOP: [],
    View.BOTTOM: []
  }
  wingspan_annos = {
    View.TOP: [],
    View.BOTTOM: []
  }
  with open(fname) as f:
      j = json.load(f)
      for number, descr in j['file'].items():
        if descr['fname'].endswith('top.mp4'):
          top = number
        elif descr['fname'].endswith('bot.mp4'):
          bottom = number
      for uid, sub_j in j['metadata'].items():
        if len(sub_j['z']) == 1 and sub_j['xy'][0] == 1:
          time = sub_j['z'][0]
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
          annos[view].append(entry)
        elif len(sub_j['z']) == 1 and sub_j['xy'][0] == 5:
          time = sub_j['z'][0]
          if sub_j['vid'] == top:
            view = View.TOP
          else:
            view = View.BOTTOM
          if len(sub_j['xy']) != 5:
            raise ValueError('Unexpected behavior')
          _, x1, y1, x2, y2 = sub_j['xy']
          if x1 > FRAME_SIZE[0]/2:
            if y1 > FRAME_SIZE[1]/2:
              quarter = Quarter.BR
            else:
              quarter = Quarter.TR
          else:
            if y1 > FRAME_SIZE[1]/2:
              quarter = Quarter.BL
            else:
              quarter = Quarter.TL
          entry = WingspanRawEntry(
            uid=uid,
            time=time,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            quarter=quarter
          )
          wingspan_annos[view].append(entry)
        elif len(sub_j['z']) == 2:
          start_time, end_time = sub_j['z']
          song_type = SongType.from_str(sub_j['av']['1'])
          entry = SongEntry(
            uid=uid,
            start_time=start_time,
            end_time=end_time,
            song_type=song_type
          )
          song_annos[view].append(entry)
          continue
        else:
          continue
  for view in View:
    annos[view].sort(key=lambda e: e.time)
    groups = []
    for _, g in groupby(annos[view], key=lambda e: e.time):
      groups.append(list(g))
    annos[view] = groups

    wingspan_annos[view].sort(key=lambda e: e.time)
    groups = []
    for _, g in groupby(wingspan_annos[view], key=lambda e: e.time):
      groups.append(list(g))
    wingspan_annos[view] = groups
  return annos, song_annos, wingspan_annos


def get_annotations(anno_prefix):
  annotations = dict()
  song_annotations = dict()
  wingspan_annotations = dict()

  for i in range(N):
    annos, song_annos, wingspan_annos = get_annotation(f'{anno_prefix}{i+1:02d}.json')
    annotations[i+1] = annos
    song_annotations[i+1] = song_annos
    wingspan_annotations[i+1] = wingspan_annos
  return annotations, song_annotations, wingspan_annotations


def get_sexes(filename):
  with open(filename) as f:
    sexes = dict()
    lines = f.read().strip().split('\n')
    for line in lines:
      b_id, sex = line.strip().split(',')
      b_id, sex = int(b_id), Sex.from_str(sex)
      sexes[b_id] = sex
  return sexes


def get_3D_annotations(pos_annos, camera_file='sound_localize/cameras/aviary_2019-06-01_calibration.yaml'):
  cam_sys = CameraSystem(camera_file)
  paths3d = dict()
  for bird_no in range(1,N+1):
    anno_arr = []
    for view, view_annos in pos_annos[bird_no].items():
      for group in view_annos:
        for anno_entry in group:
          x = anno_entry.x if anno_entry.x < FRAME_SIZE[0]/2 else anno_entry.x - FRAME_SIZE[0]/2
          y = anno_entry.y if anno_entry.y < FRAME_SIZE[1]/2 else anno_entry.y - FRAME_SIZE[1]/2
          cam = CAMERAS[(view, anno_entry.quarter)]
          anno_type = anno_entry.anno_type
          anno_arr.append((anno_entry.time, x, y, cam, anno_type))
    anno_arr.sort(key=lambda e: e[0])
    groups = []
    for _, g in groupby(anno_arr, lambda e: e[0]):
      groups.append(list(g))
    anno_arr = groups
    points = []
    for es in anno_arr:
      point = []
      for i in range(cam_sys.num_cams):
        entry = None
        for e in es:
          if e[3] == i:
            entry = e
            break
        if entry:
          point_row = [e[1],e[2],1]
        else:
          point_row = [0,0,0]
        point.append(point_row)
      points.append(point)
    points = FloatTensor(np.swapaxes(points, 0, 1))
    points3d = cam_sys.initialize_points(points)
    points_arr = [AnnoEntry3D(t=anno_arr[i][0][0], x=float(p[0]), y=float(p[1]), z=float(p[2]), anno_type=anno_arr[i][0][4]) for i, p in enumerate(points3d)]
    paths3d[bird_no] = points_arr
  return cam_sys, paths3d


def get_wingspan_annotation(raw_wingspan_anno, camera_file='sound_localize/cameras/aviary_2019-06-01_calibration.yaml'):
  cam_sys = CameraSystem(camera_file)
  anno_arr = []
  if (not raw_wingspan_anno[View.TOP]) and (not raw_wingspan_anno[View.BOTTOM]):
    return []
  for view, view_annos in raw_wingspan_anno.items():
    for group in view_annos:
      for anno_entry in group:
        x1 = anno_entry.x1 if anno_entry.x1 < FRAME_SIZE[0]/2 else anno_entry.x1 - FRAME_SIZE[0]/2
        y1 = anno_entry.y1 if anno_entry.y1 < FRAME_SIZE[1]/2 else anno_entry.y1 - FRAME_SIZE[1]/2
        x2 = anno_entry.x2 if anno_entry.x2 < FRAME_SIZE[0]/2 else anno_entry.x2 - FRAME_SIZE[0]/2
        y2 = anno_entry.y2 if anno_entry.y2 < FRAME_SIZE[1]/2 else anno_entry.y2 - FRAME_SIZE[1]/2
        cam = CAMERAS[(view, anno_entry.quarter)]
        anno_arr.append((anno_entry.time, x1, y1, x2, y2, cam))
  anno_arr.sort(key=lambda e: e[0])
  groups = []
  for _, g in groupby(anno_arr, lambda e: e[0]):
    groups.append(list(g))
  anno_arr = groups
  points1 = []
  points2 = []
  for es in anno_arr:
    point1 = []
    point2 = []
    for i in range(cam_sys.num_cams):
      entry = None
      for e in es:
        if e[5] == i:
          entry = e
          break
      if entry:
        point_row1 = [e[1],e[2],1]
        point_row2 = [e[3],e[4],1]
      else:
        point_row1 = [0,0,0]
        point_row2 = [0,0,0]
      point1.append(point_row1)
      point2.append(point_row2)
    points1.append(point1)
    points2.append(point2)
  points1 = FloatTensor(np.swapaxes(points1, 0, 1))
  points1_3d = cam_sys.initialize_points(points1)
  points2 = FloatTensor(np.swapaxes(points2, 0, 1))
  points2_3d = cam_sys.initialize_points(points2)
  lengths_arr = [WingspanEntry(t=anno_arr[i][0][0], length=np.linalg.norm(p2-p1)) for i, (p1, p2) in enumerate(zip(points1_3d, points2_3d))]
  return cam_sys, lengths_arr


def to_composite_pos(quarter, pos):
  if quarter == Quarter.TL:
    return pos
  elif quarter == Quarter.TR:
    return (pos[0]+HALF_FRAME_SIZE[0], pos[1])
  elif quarter == Quarter.BL:
    return (pos[0], pos[1]+HALF_FRAME_SIZE[1])
  else:
    return (pos[0]+HALF_FRAME_SIZE[0], pos[1]+HALF_FRAME_SIZE[1])

