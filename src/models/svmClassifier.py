import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# -------------------------
# Load processed data
# -------------------------

X_train_path = "data/processedData/X_train_scaled.csv"
Y_train_path = "data/processedData/Y_train.csv"
X_test_path  = "data/processedData/X_test_scaled.csv"
Y_test_path  = "data/processedData/Y_test.csv"

X_train = pd.read_csv(X_train_path, index_col=0).values
Y_train = pd.read_csv(Y_train_path, index_col=0).values.ravel()

X_test = pd.read_csv(X_test_path, index_col=0).values
Y_test = pd.read_csv(Y_test_path, index_col=0).values.ravel()


# -------------------------
# Train SVM Model
# -------------------------

model = SVC(kernel="rbf")

model.fit(X_train, Y_train)


# -------------------------
# Predictions
# -------------------------

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, predictions))


# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(Y_test, predictions)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign (0)", "Malignant (1)"],
    yticklabels=["Benign (0)", "Malignant (1)"],
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()


# -------------------------
# PCA Visualization (2D)
# -------------------------

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

model_2d = SVC(kernel="rbf")
model_2d.fit(X_test_pca, Y_test)

# Create grid for decision boundary
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=Y_test,
    edgecolors="k",
    cmap=plt.cm.coolwarm
)

plt.title("SVM Decision Boundary (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()


# -------------------------
# 3D PCA Visualization
# -------------------------

pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_test)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=Y_test,
    cmap="coolwarm",
    s=60
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.title("3D PCA Projection")

plt.show()