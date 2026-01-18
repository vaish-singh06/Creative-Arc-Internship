import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("../data/heart.csv")
X = df.drop("target", axis=1)  # REMOVE LABEL

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---- Elbow Method ----
inertia_vals = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_vals.append(kmeans.inertia_)

plt.figure(figsize=(7,4))
plt.plot(K_range, inertia_vals, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig("../plots/elbow.png")
plt.show()

# ---- Silhouette ----
sil_scores = {}
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil_scores[k] = silhouette_score(X, labels)

print("Silhouette Scores:", sil_scores)
best_k = max(sil_scores, key=sil_scores.get)
print("Selected K:", best_k)
