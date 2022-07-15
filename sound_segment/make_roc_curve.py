import matplotlib.pyplot as plt

thresholds = [-20, -18, -16, -14, -12]
sensitivities = [0.9130, 0.8261, 0.6957, 0.5652, 0.3478]
specificities = [0.2267, 0.4222, 0.7022, 0.9200, 0.9822]

tpr = sensitivities
fpr = [1-s for s in specificities]

plt.scatter(fpr, tpr)
plt.xlim(left=0, right=1)
plt.ylim(bottom=0, top=1)
for i, t in enumerate(thresholds):
  plt.annotate(t, (fpr[i], tpr[i]))

plt.show()
