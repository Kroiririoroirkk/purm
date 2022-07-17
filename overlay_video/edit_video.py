import cv2

from readin import FONT, FONT_SCALE, FOURCC, FPS, FRAME_SIZE, N, THICKNESS, AnnoType, View, get_annotations

# --------------------- #

annotations = get_annotations()[0]

def determine_text_pos(frame, bird_number, view):
  """Returns a text position for the given frame and bird (1-indexed)."""
  anns = annotations[bird_number][view]
  if not anns:
    return []
  time = frame / FPS
  last_group = None
  cur_group = None
  for i, entry_group in enumerate(anns):
    cur_group = entry_group
    if entry_group[0].time > time:
      if last_group:
        del anns[:i-1]
        break
      else:
        return []
    last_group = entry_group
  if last_group[0].anno_type == AnnoType.END:
    return []
  coords = []
  for e in last_group:
    found = False
    for e2 in cur_group:
      if (e.anno_type == AnnoType.MIDDLE or e2.anno_type == AnnoType.MIDDLE or (e.anno_type == AnnoType.START and e2.anno_type == AnnoType.END)) and (e.quarter == e2.quarter):
        found = True
        try:
          big_dt = e2.time - e.time
          little_dt = time - e.time
          prop = little_dt / big_dt
          coords.append((int(prop*e2.x + (1-prop)*e.x),
                         int(prop*e2.y + (1-prop)*e.y)))
          break
        except ZeroDivisionError:
          coords.append((e.x, e.y))
          break
    if not found:
      coords.append((e.x, e.y))
  return coords

def edit_video(filename, view, sexes):
  cap = cv2.VideoCapture(filename)
  out = cv2.VideoWriter('edited_' + filename, FOURCC, FPS, FRAME_SIZE)
  frame_num = 0
  while cap.isOpened():
    frame_num = frame_num + 1
    print(frame_num)
    ret, frame = cap.read()
    if not ret:
      break
    for i in range(N):
      for pos in determine_text_pos(frame_num, i+1, view):
        color = sexes[i+1].value
        frame = cv2.putText(frame, f'{i+1}', pos, FONT, FONT_SCALE, color, THICKNESS)
    out.write(frame)
  cap.release()
  out.release()

