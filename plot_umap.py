import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import sys
import umap

from scipy.cluster.vq import kmeans, whiten

def plot_umap(filenames):
  """Does a UMAP on the training data.

  Keyword arguments:
  filenames -- a list of filenames to do a UMAP on

  Returns:
  Nothing. Displays a plot.
  """
  def detect_color(codebook, v):
    closest_i = None
    for i, centroid in enumerate(codebook):
      d = np.linalg.norm(v-centroid)
      print(i,d)
      if not closest_i or d < closest:
        closest_i = i
        closest = d
    return closest_i
  vecss = []
  for fn in filenames:
    vecss.append(np.loadtxt(fn, delimiter=","))
  vecs = np.concatenate(vecss)
  vecs = normalize(vecs)
  vecs = whiten(vecs)
  codebook, _ = kmeans(vecs, 4, thresh=1e-8)
  embedding = umap.UMAP(
    n_neighbors=50,
    min_dist=0.0,
    n_components=2
  ).fit_transform(vecs)
  plt.scatter(embedding[:,0], embedding[:,1], c=[detect_color(codebook, v) for v in vecs])
  plt.show()

if __name__ == '__main__':
  plot_umap(sys.argv[1:])
