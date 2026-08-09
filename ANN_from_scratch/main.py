import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from components.forward_propagation.forward_prop import forward_prop
from components.gradient_descent.gradient_descent import get_accuracy, get_predictions, gradient_descent

# Optional: makes the random initialization reproducible
np.random.seed(0)

# Load data
iris = load_iris(as_frame=True)
df = iris.frame

X_raw = df.drop("target", axis=1).values
y = df["target"].values

# Create training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.2,
    stratify=y,
    random_state=0
)

# Standardize features
# The network expects X shape: (number_of_features, number_of_samples)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).T
X_test = scaler.transform(X_test_raw).T




def predict(X, W1, b1, W2, b2):
    """Makes predictions for a given input matrix."""
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, y_train, alpha=0.2, iterations=500)

# Evaluate the model
train_predictions = predict(X_train, W1, b1, W2, b2)
test_predictions = predict(X_test, W1, b1, W2, b2)

print("Final Training Accuracy:", get_accuracy(train_predictions, y_train))
print("Final Test Accuracy:", get_accuracy(test_predictions, y_test))

# Take user inputs for custom prediction
print("\n--- Custom Input Prediction ---")
try:
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    user_input_raw = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_user = scaler.transform(user_input_raw).T

    prediction_idx = predict(X_user, W1, b1, W2, b2)[0]
    predicted_label = iris.target_names[prediction_idx]

    print(f"\nPredicted Class: {prediction_idx} ({predicted_label})")
except ValueError:
    print("Error: Invalid numeric input provided.")