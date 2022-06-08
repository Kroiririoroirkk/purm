import numpy as np
import sys

import ksvd
import process_data
from config import CallType


for arg in sys.argv[1:]:
  call_type = CallType.from_str(arg)
  print(f'Performing k-SVD for {call_type.filename}...')
  vecs = np.loadtxt(f'training_data/{call_type.filename}.csv', delimiter=',', dtype=int)
  svd_dictionary, _ = ksvd.ksvd(call_type.sparsity, call_type.dict_size, vecs)
  np.savetxt(f'dictionaries/{call_type.filename}.csv', svd_dictionary, delimiter=',')

