import numpy as np
from sklearn.preprocessing import normalize
import sys

from config import CallType


for arg in sys.argv[1:]:
  call_type = CallType.from_str(arg)
  init_vecs = np.loadtxt(f'training_data/init_{call_type.filename}.csv', delimiter=',')
  training_vecs = np.loadtxt(f'training_data/train_{call_type.filename}.csv', delimiter=',')
  dict_vecs = np.concatenate((init_vecs, training_vecs)).T
  dict_vecs = normalize(dict_vecs, axis=0)
  np.savetxt(f'dictionaries/{call_type.filename}.csv', dict_vecs, delimiter=',')

