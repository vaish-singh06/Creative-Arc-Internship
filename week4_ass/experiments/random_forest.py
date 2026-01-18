import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("../data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train))
val_acc = accuracy_score(y_val, rf.predict(X_val))

print("Random Forest (default)")
print("Train Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)
print("Gap:", train_acc - val_acc)
print()

# ---- CONTROLLED HYPERPARAMETER: n_estimators ----
trees = [10, 50, 100, 150, 200, 300]
train_scores = []
val_scores = []

for n in trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
    val_scores.append(accuracy_score(y_val, rf.predict(X_val)))

plt.figure(figsize=(7,4))
plt.plot(trees, train_scores, label="Train Acc")
plt.plot(trees, val_scores, label="Validation Acc")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("RF: Accuracy vs n_estimators")
plt.legend()
plt.tight_layout()
plt.savefig("../plots/k_vs_accuracy.png")  # required
plt.show()
