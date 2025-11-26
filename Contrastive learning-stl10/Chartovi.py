import matplotlib.pyplot as plt

methods = ["Random", "SwAV"]
linear_acc = [28.70, 70.50]
knn_acc = [23.94, 64.62]

plt.figure(figsize=(8,5))
plt.plot(methods, linear_acc, marker='o', label="Linear Eval")
plt.plot(methods, knn_acc, marker='o', label="k-NN Eval")

plt.ylabel("Accuracy (%)")
plt.title("STL10 – Comparison of Linear and k-NN evaluation")
plt.grid(alpha=0.3)
plt.legend()

plt.show()
