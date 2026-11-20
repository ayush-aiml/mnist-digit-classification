import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Built-in Handwritten Digits Dataset (0-9)
digits = load_digits()
X = digits.data      # Flattened pixel values
y = digits.target    # Actual digit labels (0 to 9)

print(f"Total images loaded: {X.shape[0]}")
print(f"Features per image: {X.shape[1]} pixels\n")

# 2. Split Data into Training (80%) and Testing (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale Pixel Values for Optimal Neural Network Performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Multi-Layer Perceptron (Neural Network) Classifier
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate Model Accuracy
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Validation Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, predictions))

# 6. Test Prediction on a Single Sample Image
sample_index = 0
sample_image = X_test_scaled[sample_index].reshape(1, -1)
predicted_digit = model.predict(sample_image)[0]
actual_digit = y_test[sample_index]

print(f"Sample Test -> Actual Digit: {actual_digit} | Predicted Digit: {predicted_digit}")
