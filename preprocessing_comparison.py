import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Min-Max Normalization
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
print("Min-Max Normalized Data:\n", X_normalized)

# 3. Z-score Standardization
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
print("\nZ-score Standardized Data:\n", X_standardized)

# 3.1 Plot histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(X_normalized.flatten(), bins=20, color='skyblue')
plt.title('Min-Max Normalized Data')

plt.subplot(1, 2, 2)
plt.hist(X_standardized.flatten(), bins=20, color='salmon')
plt.title('Z-score Standardized Data')

plt.tight_layout()
plt.savefig("histogram_comparison.png")
plt.show()

# 4. Train and compare Logistic Regression models
def train_and_evaluate(X_data, label):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{label} Accuracy: {acc:.4f}")

print("\nLogistic Regression Model Performance:")
train_and_evaluate(X, "Original")
train_and_evaluate(X_normalized, "Min-Max Normalized")
train_and_evaluate(X_standardized, "Z-score Standardized")

# 5. Explanation:
print("\nExplanation:")
print("Normalization (Min-Max) is useful when features are on different scales and bounded.")
print("Standardization (Z-score) is preferred when data is normally distributed or for ML models like logistic regression, SVMs, and neural networks.")
