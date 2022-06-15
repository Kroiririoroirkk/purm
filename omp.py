from math import sqrt
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
import warnings


premature = (
  "Orthogonal matching pursuit ended prematurely due to linear"
  " dependence in the dictionary. The requested precision might"
  " not have been met."
)


def orthogonal_mp(X, y, n_nonzero_coefs):
  """Code copied from scikit-learn and extended to accomodate complex-valued entries."""
  if y.ndim == 1:
    y = y.reshape(-1, 1)
  if n_nonzero_coefs <= 0:
    raise ValueError("The number of atoms must be positive")
  if n_nonzero_coefs > X.shape[1]:
    raise ValueError(
      "The number of atoms cannot be more than the number of features"
    )
  G = X.conj().T @ X
  G = np.asfortranarray(G)
  Xy = X.conj().T @ y
  return orthogonal_mp_gram(G, Xy, n_nonzero_coefs, copy_Gram=(y.shape[1]>1))


def orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs, copy_Gram):
  """Code copied from scikit-learn and extended to accomodate complex-valued entries."""
  Xy = np.asarray(Xy)
  if Xy.ndim > 1 and Xy.shape[1] > 1:
    # or subsequent target will be affected
    copy_Gram = True
  if Xy.ndim == 1:
    Xy = Xy[:, np.newaxis]

  coef = np.zeros((len(Gram), Xy.shape[1]), dtype=Gram.dtype)

  n_iters = []
  for k in range(Xy.shape[1]):
    out = _gram_omp(
      Gram,
      Xy[:, k],
      n_nonzero_coefs
    )
    x, idx, n_iter = out
    coef[idx, k] = x
    n_iters.append(n_iter)

  if Xy.shape[1] == 1:
    n_iters = n_iters[0]

  return np.squeeze(coef)


def _gram_omp(Gram, Xy, n_nonzero_coefs):
  Gram = Gram.copy("F")

  min_float = np.finfo(Gram.dtype).eps
  nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (Gram,))
  (potrs,) = get_lapack_funcs(("potrs",), (Gram,))

  indices = np.arange(len(Gram))  # keeping track of swapping
  alpha = Xy
  delta = 0
  gamma = np.empty(0, dtype=Gram.dtype)
  n_active = 0

  max_features = n_nonzero_coefs

  L = np.empty((max_features, max_features), dtype=Gram.dtype)

  L[0, 0] = 1.0

  while True:
    lam = np.argmax(np.abs(alpha))
    if lam < n_active or alpha[lam] * np.conj(alpha[lam]) < min_float:
      # selected same atom twice, or inner product too small
      warnings.warn(premature, RuntimeWarning, stacklevel=3)
      break
    if n_active > 0:
      L[n_active, :n_active] = Gram[lam, :n_active]
      linalg.solve_triangular(
        L[:n_active, :n_active],
        L[n_active, :n_active],
        trans=0,
        lower=1,
        overwrite_b=True,
        check_finite=False,
      )
      v = nrm2(L[n_active, :n_active]) ** 2
      Lkk = Gram[lam, lam] - v
      if Lkk <= min_float:  # selected atoms are dependent
        warnings.warn(premature, RuntimeWarning, stacklevel=3)
        break
      if abs(Lkk.imag) < 0.00001: # I'm pretty sure this should be real-valued, but who knows
        L[n_active, n_active] = sqrt(Lkk.real)
      else:
        raise ValueError('Something weird happened.')
    else:
      if abs(Gram[lam, lam].imag) < 0.00001: # I'm pretty sure this should be real-valued, but who knows
        L[0, 0] = sqrt(Gram[lam, lam].real)
      else:
        raise ValueError('Something weird happened.')

    Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
    Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
    indices[n_active], indices[lam] = indices[lam], indices[n_active]
    Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
    n_active += 1
    # solves LL'x = X'y as a composition of two triangular systems
    gamma, _ = potrs(
      L[:n_active, :n_active], Xy[:n_active], lower=True, overwrite_b=False
    )
    beta = np.dot(Gram[:, :n_active], gamma)
    alpha = Xy - beta
    if n_active == max_features:
      break

  return gamma, indices[:n_active], n_active


def normalize(matr):
  """Return a matrix with unit columns (works for complex-valued matrices).

  Keyword arguments:
  matr -- a 2D numpy array

  Returns:
  A normalized copy of matr.
  """
  return matr / np.linalg.norm(matr, axis=0)

