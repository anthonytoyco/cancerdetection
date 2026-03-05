import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

"""
NEEDS TO BE MODIFIED WITH PROCESSED DATA AND FIX COMMENTS
"""

# load dataset
dataPath = "data/rawData/cancerDetectionDataRaw.csv"
df = pd.read_csv(dataPath)

# split into features and labels
# Drop the columns we don't want (ID and diagnosis)
# We keep everything from index 2 up to index 32
X = df.iloc[:, 2:].values

# Get the target (Diagnosis)
y = df.iloc[:, 1].values

# Transforms M/B into 1/0
le = LabelEncoder()
y = le.fit_transform(y)


# train test split (test_size=0.2 means 20% of data is saved for testing and rest is for training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create SVM model
model = SVC(kernel="rbf")

# train
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# evaluate (94% is still low for svm on unprocessed data according to gemini)
print("Accuracy:", accuracy_score(y_test, predictions))


"""
move into results once data is good and change var names
"""

# create confusion matrix
cm = confusion_matrix(y_test, predictions)

# plot using Seaborn
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

# PCA to reduce to 2D
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# train a model on the 2D data for the plot
model_2d = SVC(kernel="rbf")
model_2d.fit(X_test_pca, y_test)

# Create a grid (using more points instead of a fixed small step)
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1

# instead of h=0.02, we tell it to just make 500 steps.
# this works regardless of how big or small your numbers are
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

# predict and Plot
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(
    X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors="k", cmap=plt.cm.coolwarm
)
plt.title("SVM Decision Boundary (Unscaled Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# 3D PCA visual
pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_test)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# scatter plot for 3D
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_test, cmap="coolwarm", s=60)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D PCA")
plt.show()
