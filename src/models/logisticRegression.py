import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Load preprocessed data
X_train = pd.read_csv(
    "data/processedData/X_train_scaled.csv", index_col=0).values
Y_train = pd.read_csv("data/processedData/Y_train.csv",
                      index_col=0).values.ravel()

X_test = pd.read_csv(
    "data/processedData/X_test_scaled.csv", index_col=0).values
Y_test = pd.read_csv("data/processedData/Y_test.csv",
                     index_col=0).values.ravel()

# Train
# max_iter raised from default 100 to ensure convergence on this dataset
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, Y_train)

# Evaluate
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, predictions))
print()
print(classification_report(Y_test, predictions,
      target_names=["Benign (0)", "Malignant (1)"]))

# Confusion matrix
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
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# 2D PCA decision boundary
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

model_2d = LogisticRegression(max_iter=1000, random_state=42)
model_2d.fit(pca.transform(X_train), Y_train)

x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
            c=Y_test, edgecolors="k", cmap=plt.cm.coolwarm)
plt.title("Logistic Regression Decision Boundary (PCA, Scaled Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# 3D PCA scatter
pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_test)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=Y_test, cmap="coolwarm", s=60)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("Logistic Regression — 3D PCA")
plt.tight_layout()
plt.show()
