import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---- LOAD DATA ----
df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ---- TRAIN/VAL SPLIT ----
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# ---- BASELINE DECISION TREE (NO DEPTH LIMIT) ----
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
val_acc = accuracy_score(y_val, clf.predict(X_val))

print("Decision Tree (no limit)")
print("Train Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)
print("Gap:", train_acc - val_acc)
print()

# ---- CONTROLLED HYPERPARAMETER EXPERIMENT (max_depth) ----
depths = range(1, 21)
train_scores = []
val_scores = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    val_scores.append(accuracy_score(y_val, clf.predict(X_val)))

# ---- PLOT ----
plt.figure(figsize=(7,4))
plt.plot(depths, train_scores, label="Train Acc")
plt.plot(depths, val_scores, label="Validation Acc")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Accuracy vs Depth")
plt.legend()
plt.tight_layout()
plt.savefig("../plots/bias_variance.png")  # required plot
plt.show()
