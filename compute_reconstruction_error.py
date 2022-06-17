import numpy as np
import sys

from config import CallType
from ksvd import frobdist
from omp import orthogonal_mp


if __name__ == '__main__':
  call_type = CallType.from_str(sys.argv[1])
  calls = np.loadtxt(sys.argv[2], delimiter=',').T
  d = np.loadtxt(f'dictionaries/{sys.argv[1]}.csv', delimiter=',')
  sparse_rep = orthogonal_mp(d, calls, call_type.sparsity)
  dist = frobdist(d @ sparse_rep, calls)
  print(dist / np.linalg.norm(calls))
