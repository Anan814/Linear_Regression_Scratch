import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Step 2: Preprocess the data
def preprocess_data(df, target_column):
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert categorical variables to dummy variables (if any)
    X = pd.get_dummies(X, drop_first=True)
    
    # Normalize features (optional but recommended for gradient descent)
    X = (X - X.mean()) / X.std()
    
    # Add a bias term (intercept) to X
    X = np.c_[np.ones(X.shape[0]), X]
    
    return X, y

# Step 3: Implement Logistic Regression from Scratch
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.cost_history = []

    def sigmoid(self, z):
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.weights)
        # Avoid log(0) by clipping h
        h = np.clip(h, 1e-15, 1 - 1e-15)
        cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        
        for i in range(self.max_iter):
            h = self.sigmoid(X @ self.weights)
            gradient = (X.T @ (h - y)) / m
            self.weights -= self.learning_rate * gradient
            
            # Store cost for visualization
            self.cost_history.append(self.cost_function(X, y))

    def predict(self, X):
        return (self.sigmoid(X @ self.weights) >= 0.5).astype(int)

# Step 4: Implement Linear Regression from Scratch
class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.cost_history = []

    def cost_function(self, X, y):
        m = len(y)
        h = X @ self.weights
        cost = (1/(2*m)) * np.sum((h - y) ** 2)
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        
        for i in range(self.max_iter):
            h = X @ self.weights
            gradient = (X.T @ (h - y)) / m
            self.weights -= self.learning_rate * gradient
            
            # Store cost for visualization
            self.cost_history.append(self.cost_function(X, y))

    def predict(self, X):
        return X @ self.weights

# Step 5: Train and Evaluate the Model
def train_and_evaluate(X, y):
    # Check if the target variable is binary (Logistic Regression) or continuous (Linear Regression)
    if y.nunique() == 2:  # Binary classification
        model = LogisticRegression(learning_rate=0.01, max_iter=1000)
        model_type = "Logistic Regression"
    else:  # Continuous target (Linear Regression)
        model = LinearRegression(learning_rate=0.01, max_iter=1000)
        model_type = "Linear Regression"
    
    # Split the data into training and testing sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    if model_type == "Logistic Regression":
        accuracy = np.mean(y_pred == y_test)
        print("Logistic Regression Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
    else:
        mse = np.mean((y_pred - y_test) ** 2)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        print("Linear Regression Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
    
    # Plot the cost function
    plt.plot(model.cost_history)
    plt.title(f"{model_type} Cost Function")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

# Step 6: Main Function
def main(file_path, target_column):
    # Step 1: Load the dataset
    df = load_dataset(file_path)
    
    # Step 2: Preprocess the data
    X, y = preprocess_data(df, target_column)
    
    # Step 3: Train and evaluate the model
    train_and_evaluate(X, y)

# Step 7: Run the Program
# For ModifiedHeartDisease.csv
print("Results for ModifiedHeartDisease.csv:")
main("ModifiedHeartDisease.csv", "target")

# For CustomerService.csv
print("Results for CustomerService.csv:")
main("CustomerService.csv", "Time_to_Resolve")