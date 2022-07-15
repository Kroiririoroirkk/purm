from random import random
import numpy as np

import ksvd

dataset = []
v_1 = np.array([3,3,3,100,3,5,2,1,8]) / np.linalg.norm([3,3,3,100,3,5,2,1,8])
v_2 = np.array([1,3,50,20,23,90,-10,3,0]) / np.linalg.norm([1,3,50,20,23,90,-10,3,0])
v_3 = np.array([1,300,0,0,0,0,-1,3,0]) / np.linalg.norm([1,300,0,0,0,0,-1,3,0])
rng = np.random.default_rng()
for _ in range(1000):
    #a = rng.multivariate_normal(v_1, 0.1*np.identity(9))
    #b = rng.multivariate_normal(v_2, 0.1*np.identity(9))
    #c = rng.multivariate_normal(v_3, 0.1*np.identity(9))
    l = random()*100 - 50
    m = random()*100 - 50
    n = random()*100 - 50
    dataset.append(l*v_1+m*v_2+n*v_3)
d, sr = ksvd.ksvd(3, 3, dataset)

def dist(v):
  return np.linalg.norm(v - np.dot(v,v_1)*v_1 - np.dot(v,v_2)*v_2 - np.dot(v,v_3)*v_3)

print('Scores for each column', dist(d[:,0]), dist(d[:,1]), dist(d[:,2]))

random_v = np.array([random(),random(),random(),random(),random(),random(),random(),random(),random()])
random_v = random_v / np.linalg.norm(random_v)
print('Scores for a random vector', dist(random_v))
