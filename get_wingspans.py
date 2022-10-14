#!bin/python
import sys

import overlay_video.readin as r

if __name__ == '__main__':
  # Usage: ./get_wingspans.py <filename>
  fname = sys.argv[1]
  _,_,raw_annos = r.get_annotation(fname)
  _,annos = r.get_wingspan_annotation(raw_annos)
  for e in annos:
    print(f'{e.t}, {e.length}, {e.error}')
