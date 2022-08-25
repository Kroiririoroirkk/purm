import cv2
import sys
from torch import FloatTensor

from overlay_video.readin import FONT, FONT_SCALE, FOURCC, FRAME_SIZE, N, THICKNESS, CAMERAS_LOOKUP, AnnoType, View, get_3D_annotations, get_annotations, get_sexes, to_composite_pos

# --------------------- #

def determine_3D_pos(annotations, time, bird_number):
  """Helper function for determine_text_pos. Returns the 3D coordinates of a specific bird at a specific time as a FloatTensor. Edits the annotations list."""
  anns = annotations[bird_number]
  if not anns:
    return []
  prev_entry = None
  cur_entry = None
  is_final_entry = False
  for i, entry in enumerate(anns):
    cur_entry = entry
    if entry.t > time:
      if prev_entry:
        del anns[:i-1]
        break
      else:
        return []
    prev_entry = entry
  else:
    is_final_entry = True
  if prev_entry.anno_type == AnnoType.END:
    return []
  if is_final_entry:
    coord = FloatTensor([[cur_entry.x, cur_entry.y, cur_entry.z]])
  else:
    big_dt = cur_entry.t - prev_entry.t
    little_dt = time - prev_entry.t
    prop = little_dt / big_dt
    coord = FloatTensor([[prop*cur_entry.x + (1-prop)*prev_entry.x,
                          prop*cur_entry.y + (1-prop)*prev_entry.y,
                          prop*cur_entry.z + (1-prop)*prev_entry.z]])
  return coord

def determine_text_pos(annotations, time, bird_number, cam_sys):
  """Returns a text position for the given frame and bird (1-indexed). Edits the annotations list."""
  coord = determine_3D_pos(annotations, time, bird_number)
  if isinstance(coord, list):
    return []
  projected_points = cam_sys.perspective_projection(coord)
  cam_coords = []
  for cam, pos in enumerate(projected_points):
    if pos[0][2]:
      tup = (CAMERAS_LOOKUP[cam], (int(pos[0][0]), int(pos[0][1])))
      cam_coords.append(tup)
  return cam_coords


def edit_video(annotations, top_filename, bot_filename, sexes, cam_sys):
  top_cap = cv2.VideoCapture(top_filename)
  bot_cap = cv2.VideoCapture(bot_filename)
  FPS = int(top_cap.get(cv2.CAP_PROP_FPS))
  top_out = cv2.VideoWriter('top_output.mp4', FOURCC, FPS, FRAME_SIZE)
  bot_out = cv2.VideoWriter('bot_output.mp4', FOURCC, FPS, FRAME_SIZE)
  total_frame_count = int(min(top_cap.get(cv2.CAP_PROP_FRAME_COUNT),
                              bot_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  frame_num = 0
  while top_cap.isOpened():
    frame_num += 1
    print(f'Now labeling frame {frame_num} out of {total_frame_count}...')
    top_ret, top_frame = top_cap.read()
    bot_ret, bot_frame = bot_cap.read()
    if not top_ret or not bot_ret:
      break
    for i in range(N):
      for (view, quarter), pos in determine_text_pos(annotations, frame_num/FPS, i+1, cam_sys):
        composite_pos = to_composite_pos(quarter, pos)
        color = sexes[i+1].value
        if view == View.TOP:
          top_frame = cv2.putText(top_frame, f'{i+1}', composite_pos, FONT, FONT_SCALE, color, THICKNESS)
        elif view == View.BOTTOM:
          bot_frame = cv2.putText(bot_frame, f'{i+1}', composite_pos, FONT, FONT_SCALE, color, THICKNESS)
    top_out.write(top_frame)
    bot_out.write(bot_frame)
  top_cap.release()
  bot_cap.release()
  top_out.release()
  bot_out.release()


if __name__ == '__main__':
  # Usage: python edit_video <top_video_filename> <bottom_video_filename> <annotations_json_file_prefix> <sexes_filename>
  top_video = sys.argv[1]
  bottom_video = sys.argv[2]
  anno_prefix = sys.argv[3]
  cam_sys, annotations = get_3D_annotations(get_annotations(anno_prefix)[0])
  sexes = get_sexes(sys.argv[4])
  edit_video(annotations, top_video, bottom_video, sexes, cam_sys)

