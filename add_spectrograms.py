import cv2 as cv
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io.wavfile
import sys

from overlay_video.readin import FOURCC

DPI = 100 # Change depending on setup
WIDTH_PX = 7680 # Width of spectrogram in pixels (needs to match with video width)
HEIGHT_PX = 500 # Height of spectrogram in pixels
WIDTH_IN = WIDTH_PX/DPI
HEIGHT_IN = HEIGHT_PX/DPI
HEIGHT_SCALE = 0.45 # How much of the spectrogram to show (frequency range)

CACHE_WIDTH = 5*48000 # In audio frames
cache = dict()
def get_spectrogram(wav_file, t, fs):
  def load_cache(i):
    try:
      return cache[i]
    except KeyError:
      if i < 0:
        img = np.zeros((HEIGHT_PX, WIDTH_PX//2, 3), dtype=np.uint8)
        cache[i] = img
        return img
      else:
        fig = plt.figure()
        fig.set_size_inches(WIDTH_IN/2, HEIGHT_IN/HEIGHT_SCALE)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        start_i = i * CACHE_WIDTH
        end_i = start_i + CACHE_WIDTH
        _, _, _, ax_im = ax.specgram(wav_file[start_i:end_i], Fs=fs)
        img, _, _, _ = ax_im.make_image(None)
        img = img[:HEIGHT_PX]
        img = cv.cvtColor(img[::-1], cv.COLOR_RGBA2BGR)
        plt.close(fig)
        cache[i] = img
        return img

  frame = int(t * fs)
  cache_position, cache_index = math.modf(frame / CACHE_WIDTH)
  cache_index = int(cache_index)
  cache_position = int(cache_position * WIDTH_PX / 2)
  try:
    del cache[cache_index-2]
  except KeyError:
    pass
  start = load_cache(cache_index-1)
  middle = load_cache(cache_index)
  end = load_cache(cache_index+1)
  img = np.hstack((start[:, cache_position:], middle, end[:, :cache_position]))
  midpoint = img.shape[1]//2
  img[:, midpoint-3 : midpoint+3] = [0, 0, 255]
  return img
  

if __name__ == '__main__':
  video_filename = sys.argv[1]
  audio_fs, wav_file = scipy.io.wavfile.read(sys.argv[2])
  wav_file = wav_file[:,0]
  cap = cv.VideoCapture(video_filename)
  video_fs = int(cap.get(cv.CAP_PROP_FPS))
  total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
  frame_size = (original_width, original_height+HEIGHT_PX)
  out = cv.VideoWriter('spec.mp4', FOURCC, video_fs, frame_size)
  video_frame_num = 0
  while cap.isOpened():
    video_frame_num += 1
    print(f'Now labeling frame {video_frame_num} out of {total_frame_count}...')
    #if video_frame_num > 100:
    #  break
    ret, video_frame = cap.read()
    if not ret:
      break
    spec = get_spectrogram(wav_file, video_frame_num/video_fs, audio_fs)
    new_frame = np.vstack((video_frame, spec))
    out.write(new_frame)
  cap.release()
  out.release()
