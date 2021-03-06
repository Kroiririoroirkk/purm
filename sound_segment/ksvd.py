import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import warnings

from omp import orthogonal_mp


def l2dist(v1, v2):
  """Calculate the l2 norm distance between two numpy vectors.
  
  Keyword arguments:
  v1 -- a numpy vector
  v2 -- a numpy vector
  
  Returns:
  Euclidean distance between v1 and v2
  """
  return np.linalg.norm(v1-v2)

def frobdist(m1, m2):
  """Calculate the Frobenius norm distance between two numpy matrices.
  
  Keyword arguments:
  m1 -- a numpy matrix
  m2 -- a numpy matrix
  
  Returns:
  Frobenius distance between m1 and m2
  """
  return np.linalg.norm(m1-m2)


def naive_kmeans(k, dataset):
  """Perform k-means clustering on a dataset.
  
  Keyword arguments:
  k -- the number of clusters
  dataset -- a list of numpy vectors (1D)
  
  Returns:
  A list of k numpy vectors (1D)
  """
  # Initialize step
  centroids = random.sample(dataset, k)
  clusters = []

  while True:
    # Assignment step
    clusters_new = [[] for _ in range(k)]
    for j, d in enumerate(dataset):
      min_i = 0
      min_dist = l2dist(d, centroids[0])
      for i, centroid in enumerate(centroids):
        if i == 0:
          continue
        dist = l2dist(d, centroid)
        if dist < min_dist:
          min_i = i
          min_dist = dist
      clusters_new[min_i].append(j)
    
    # Update step
    centroids_new = []
    for cluster in clusters_new:
      if cluster:
        s = sum(dataset[c] for c in cluster)
        avg = s / len(cluster)
        centroids_new.append(avg)
      else:
        centroids_new.append(0)

    # Check for convergence
    if clusters == clusters_new:
      break

    clusters = clusters_new
    centroids = centroids_new

  return centroids


def ksvd(k, n, init_vecs, train_vecs, iters):
  """Generate a dictionary through the k-SVD algorithm for a dataset.
  
  Keyword arguments:
  k -- the maximum support size of the sparse representations
  n -- the size of the dictionary (only has an effect if n is less than the number of initialization vectors, can pass None)
  init_vecs -- a list of numpy vectors (with each vector being a row)
  train_vecs -- a list of numpy vectors (with each vector being a row)
  iters -- a number that determines how many loops to go through
  
  Returns:
  A 2-tuple of an dictionary matrix and a sparse representation matrix
  """
  # Initialize step
  if n and (n < len(init_vecs)):
    d = KMeans(n_clusters=n).fit(np.vstack(init_vecs)).cluster_centers_.transpose()
  else:
    n = len(init_vecs)
    d = np.column_stack(init_vecs)
  d = normalize(d, axis=0) # Normalizes the columns (l2 norm)
  dataset_matr = np.column_stack(train_vecs)

  trials = []
  errs = []
  labels = []
  
  for iter_n in range(iters):
    print('Iteration ' + str(iter_n+1))
    # Assignment step
    x = orthogonal_mp(d, dataset_matr, k)

    # Update step
    for i in range(n):
      d[:, i] = np.zeros(d.shape[0])
      indices = [j for j in range(len(train_vecs)) if x[i][j]]
      if not indices:
        continue
      x_col = x[i, indices]
      selected_signals = dataset_matr[:, indices]
      d_col = (selected_signals - (d @ x[:, indices])) @ x_col
      d_col = d_col / np.linalg.norm(d_col)
      x_col = (selected_signals - (d @ x[:, indices])).transpose() @ d_col
      d[:, i] = d_col
      x[i, indices] = x_col

    # Calculate reconstruction errors
    dataset_calc = d @ x
    for i in range(len(train_vecs)):
      trials.append(iter_n)
      errs.append(l2dist(dataset_calc[:, i], dataset_matr[:, i]))
      labels.append(i+1)

  plt.scatter(trials, errs)
  for trial, err, label in zip(trials, errs, labels):
    plt.annotate(label, (trial, err))
  plt.show()

  x = orthogonal_mp(d, dataset_matr, k)
  print('Final error: ' + str(frobdist(dataset_matr, d @ x) / np.linalg.norm(dataset_matr)))

  return d, x

