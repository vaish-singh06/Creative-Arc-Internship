import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("../data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ---- SCALE FEATURES ----
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

k_values = range(1, 31)
train_scores = []
val_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    val_scores.append(accuracy_score(y_val, knn.predict(X_val)))

plt.figure(figsize=(7,4))
plt.plot(k_values, train_scores, label='Train Acc')
plt.plot(k_values, val_scores, label='Validation Acc')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs K")
plt.legend()
plt.tight_layout()
plt.savefig("../plots/k_vs_accuracy.png")
plt.show()
