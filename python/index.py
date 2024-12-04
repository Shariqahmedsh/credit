import math


# Helper Functions

# Calculate Mean
def calculate_mean(data):
    if not data:
        return 0
    return sum(data) / len(data)


# Calculate Standard Deviation
def calculate_std(data, mean):
    if len(data) <= 1:
        return 0
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))


# Logistic Regression Helper Functions
def sigmoid(x):
    # To prevent overflow errors
    if x > 20:
        return 1.0
    elif x < -20:
        return 0.0
    return 1 / (1 + math.exp(-x))


def logistic_regression_train(X, y, learning_rate=0.1, iterations=1000):
    weights = [0] * len(X[0])  # Initialize weights to 0
    bias = 0  # Initialize bias to 0
    for _ in range(iterations):
        for i in range(len(X)):
            linear_model = sum(weights[j] * X[i][j] for j in range(len(weights))) + bias
            prediction = sigmoid(linear_model)
            error = y[i] - prediction
            for j in range(len(weights)):
                weights[j] += learning_rate * error * X[i][j]
            bias += learning_rate * error
    return weights, bias


def logistic_regression_predict(X, weights, bias):
    return [sigmoid(sum(weights[j] * x[j] for j in range(len(weights))) + bias) for x in X]


# Decision Tree Helper Functions
class DecisionTree:
    def init(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(set(y)) == 1:  # If only one class, return that
            return y[0]

        if len(X[0]) == 0:  # If no features left, return the most common class
            return max(set(y), key=y.count)

        best_split = self._best_split(X, y)
        left_X, right_X, left_y, right_y = self._split(X, y, best_split)
        left_tree = self._build_tree(left_X, left_y)
        right_tree = self._build_tree(right_X, right_y)

        return {'split': best_split, 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_gain = 0
        best_split = None
        for feature_index in range(len(X[0])):
            feature_values = set(x[feature_index] for x in X)
            for value in feature_values:
                left_X, right_X, left_y, right_y = self._split(X, y, (feature_index, value))
                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, value)
        return best_split

    def _split(self, X, y, split):
        feature_index, value = split
        left_X = [x for i, x in enumerate(X) if x[feature_index] <= value]
        right_X = [x for i, x in enumerate(X) if x[feature_index] > value]
        left_y = [y[i] for i in range(len(y)) if X[i][feature_index] <= value]
        right_y = [y[i] for i in range(len(y)) if X[i][feature_index] > value]
        return left_X, right_X, left_y, right_y

    def _information_gain(self, parent_y, left_y, right_y):
        parent_entropy = self._entropy(parent_y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        total = len(y)
        value_counts = {x: y.count(x) for x in set(y)}
        return -sum((count / total) * math.log2(count / total) for count in value_counts.values())

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            feature_index, value = tree['split']
            if x[feature_index] <= value:
                return self._predict_single(x, tree['left'])
            else:
                return self._predict_single(x, tree['right'])
        return tree


# Fraud Detection Function Using Anomaly Detection (STD)
def detect_fraud(historical_data, new_transaction, threshold):
    if len(historical_data) <= 1:
        print("Insufficient data for reliable anomaly detection.")
        return 0

    mean = calculate_mean(historical_data)
    std_dev = calculate_std(historical_data, mean)

    if std_dev == 0:  # Handle cases where standard deviation is zero
        if new_transaction != mean:
            return 1  # Any deviation from the single transaction is considered fraud
        else:
            return 0

    if abs(new_transaction - mean) > threshold * std_dev:
        return 1
    else:
        return 0


# Main Program for User Input and Model Prediction
try:
    history_size = int(input("Enter the number of historical transactions: "))
    if history_size <= 0:
        raise ValueError("Number of historical transactions must be greater than zero.")

    historical_data = []
    print("Enter the historical transaction amounts:")
    for i in range(history_size):
        transaction = float(input(f"Transaction {i + 1}: "))
        historical_data.append(transaction)

    new_transaction = float(input("Enter the new transaction amount: "))
    threshold = float(input("Enter the anomaly threshold (e.g., 2 for 2 standard deviations): "))
    if threshold <= 0:
        raise ValueError("Threshold must be a positive number.")

    # Fraud Detection using Standard Deviation Method
    is_fraud_std_dev = detect_fraud(historical_data, new_transaction, threshold)

    # Prepare Data for Logistic Regression and Decision Tree
    X = [[x] for x in historical_data]  # Feature: transaction amount
    mean = calculate_mean(historical_data)
    y = [1 if abs(x - mean) > threshold else 0 for x in historical_data]  # Labels based on anomaly detection

    # Train Logistic Regression Model
    weights, bias = logistic_regression_train(X, y)
    logistic_predictions = logistic_regression_predict([[new_transaction]], weights, bias)

    # Train Decision Tree Model
    tree = DecisionTree()
    tree.fit(X, y)
    tree_predictions = tree.predict([[new_transaction]])

    # Output Results
    print(f"Fraud Detection (Standard Deviation Method): {'Fraud' if is_fraud_std_dev else 'No Fraud'}")
    print(f"Logistic Regression Prediction: {'Fraud' if logistic_predictions[0] > 0.5 else 'No Fraud'}")
    print(f"Decision Tree Prediction: {'Fraud' if tree_predictions[0] == 1 else 'No Fraud'}")

except ValueError as e:
    print(f"Invalid input: {e}")