import matplotlib.pyplot as plt


# Plotting sensitivity vs K at 95±1% specificity
k_vals = [7, 10, 15, 20, 30, 50]
y_vals = [0.8929, 0.8214, 0.7857, 0.8929, 0.8929, 0.9286]

plt.plot(k_vals, y_vals, 'o')
plt.ylim(bottom=0, top=1)
plt.show()
