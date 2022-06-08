import matplotlib.pyplot as plt

### With highpass filter
# sensitivities = [1.0000, 1.0000, 0.8802, 0.8099, 0.7686, 0.5661, 0.3967, 0.1653]
# specificities = [0.1326, 0.1731, 0.2687, 0.3038, 0.3146, 0.3038, 0.3154, 0.3410]
# thresholds = [-40, -35, -30, -28, -27, -25, -23, -20]

### With bandpass filter
# sensitivities = [1.0000, 0.8678, 0.6488, 0.2149]
# specificities = [0.1329, 0.1980, 0.2611, 0.3966]
# thresholds = [-35, -28, -25, -20]

### With bandpass filter (again)
sensitivities = [0.8678, 0.3430, 0.0207]
specificities = [0.2787, 0.3942, 0.2857]
thresholds = [-25, -20, -15]
tpr = sensitivities
fpr = [1-s for s in specificities]

plt.scatter(fpr, tpr)
for i, t in enumerate(thresholds):
  plt.annotate(t, (fpr[i], tpr[i]))

plt.show()
