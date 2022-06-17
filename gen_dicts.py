import numpy as np
import sys

import ksvd
from config import CallType, KSVD_ITERS


for arg in sys.argv[1:]:
  call_type = CallType.from_str(arg)
  print(f'Performing k-SVD for {call_type.filename}...')
  init_vecs = np.loadtxt(f'training_data/init_{call_type.filename}.csv', delimiter=',')
  training_vecs = np.loadtxt(f'training_data/train_{call_type.filename}.csv', delimiter=',')
  svd_dictionary, _ = ksvd.ksvd(call_type.sparsity, None, init_vecs, training_vecs, KSVD_ITERS)
  np.savetxt(f'dictionaries/{call_type.filename}.csv', svd_dictionary, delimiter=',')

