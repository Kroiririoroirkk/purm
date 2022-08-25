import cv2
import pickle
import sys

from segment_audio import CallType2
from localize_calls import OutputEntry, CameraEntry
from overlay_video.readin import FRAME_SIZE, FOURCC, FONT

FONT_SCALE = 5
THICKNESS = 12
SOUND_COLOR = (255, 0, 0)
LETTERS = {
  CallType2.WHISTLE: 's', # Too hard to distinguish between whistles and songs
  CallType2.SONG: 's',
  CallType2.CHATTER: 'c',
  CallType2.BURBLE: 'b' # Won't use
}

def overlay_video_sound_sources(fname, out_fname, entries, tl_cam, tr_cam, bl_cam, br_cam):
  """Overlay a video with sound sources.

  Keyword arguments:
  fname -- the filename of the video
  out_fname -- the filename to write to
  entries -- an array of OutputEntry
  tl_cam -- the camera number of the top left view
  tr_cam -- the camera number of the top right view
  bl_cam -- the camera number of the bottom left view
  br_cam -- the camera number of the bottom right view

  Returns:
  Nothing. Overlays the video.
  """
  cap = cv2.VideoCapture(fname)
  FPS = int(cap.get(cv2.CAP_PROP_FPS))
  out = cv2.VideoWriter(out_fname, FOURCC, FPS, FRAME_SIZE)
  vid_x, vid_y = FRAME_SIZE[0]//2, FRAME_SIZE[1]//2
  frame_num = 0
  while cap.isOpened():
    frame_num += 1
    print(f'Processing frame {frame_num} of {fname}')
    ret, frame = cap.read()
    if not ret:
      break
    time = frame_num / FPS
    while entries and (time > entries[0].end_t):
      entries.pop(0)
    for output_entry in entries:
      if output_entry.start_t < time:
        for cam_entry in output_entry.camera_positions:
          x, y = int(cam_entry.x), int(cam_entry.y)
          if cam_entry.camera == tl_cam:
            pos = (x, y)
          elif cam_entry.camera == tr_cam:
            pos = (x + vid_x, y)
          elif cam_entry.camera == bl_cam:
            pos = (x, y + vid_y)
          elif cam_entry.camera == br_cam:
            pos = (x + vid_x, y + vid_y)
          else:
            continue
          frame = cv2.putText(frame, f'{LETTERS[output_entry.call_type]}', pos, FONT, FONT_SCALE, SOUND_COLOR, THICKNESS)
      else:
        break
    out.write(frame)
  cap.release()
  out.release()

if __name__ == '__main__':
  # Format: python overlay_video_sound_sources <top_video.mp4> <bottom_video.mp4>
  top_fname = sys.argv[1]
  bottom_fname = sys.argv[2]

  print('Loading sound source file...')
  with open('output.pickle', 'rb') as f:
    output_arr = pickle.load(f)
  print('Sound source file loaded.')

  print('Overlaying top video...')
  overlay_video_sound_sources(top_fname, 'sound_localized_top_output.mp4', output_arr.copy(), 3, 2, 7, 6)
  print('Overlaying bottom video...')
  overlay_video_sound_sources(bottom_fname, 'sound_localized_bot_output.mp4', output_arr, 0, 1, 4, 5)

