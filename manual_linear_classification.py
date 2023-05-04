"""
**Linear Classification On UCI Breast Cancer Dataset**

This notebook aims to write a Linear classificatin model manually with hyperparameters. Then we test the proposed model on UCI breast cancer dataset. The procedure is as follows:

1. Developing Model

2. Imporing UCI Breast Cancer dataset

3. Testing Model on Dataset

4. Enhancing results using Hyperparameter Tunning

5. Cross-validating to find best model

**1. Developing Model**

1.1. Importing required libraries
"""

import numpy as np
from sklearn.metrics import accuracy_score

"""1.2. Coding Model"""

# Define a Python class called LinearClassifier
class LinearClassifier:
    # Define the constructor method with default hyperparameters
    def __init__(self, learning_rate=0.1, num_iterations=100, penalty='l1', C=1.0, regularization_strength=1.0):
        # Set the hyperparameters as instance variables
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.penalty = penalty
        self.C = C
        self.regularization_strength = regularization_strength
        self.weights = None  # Placeholder for the model weights
        self.bias = None  # Placeholder for the bias term

    # Method to set the hyperparameters of the model
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # Method to train the model on a dataset X and labels y
    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get the number of samples and features in the input dataset
        self.weights = np.zeros(n_features)  # Initialize the model weights to zero
        self.bias = 0  # Initialize the bias term to zero

        # Loop through the specified number of iterations
        for i in range(self.num_iterations):
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)  # Compute the predicted labels using the current weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Compute the derivative of the loss function with respect to the weights

            # Apply L2 regularization if specified
            if self.penalty == 'l2':
                dw += (1 / n_samples) * self.C * self.regularization_strength * self.weights
            # Apply L1 regularization if specified
            elif self.penalty == 'l1':
                dw += (1 / n_samples) * self.C * self.regularization_strength * np.sign(self.weights)

            db = (1 / n_samples) * np.sum(y_pred - y)  # Compute the derivative of the loss function with respect to the bias

            # Update the weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self  # Return the trained model

    # Method to predict the labels of a new dataset X using the trained weights and bias
    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)  # Compute the predicted labels using the trained weights and bias
        return np.round(y_pred)  # Round the predicted labels to the nearest integer (0 or 1)

    # Method to compute the sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Compute the sigmoid function

    # Method to compute the accuracy of the model on a dataset X and labels y
    def score(self, X, y):
        y_pred = self.predict(X)  # Predict the labels of the input dataset
        return accuracy_score(y, y_pred)  # Compute the accuracy of the predicted labels

    # Method to return the current hyperparameters of the model as a dictionary
    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
            "penalty": self.penalty,
            "C": self.C,
            "regularization_strength": self.regularization_strength
 }  # return the values of the model's parameters

"""**2. Imporing UCI Breast Cancer dataset**

This could be done in two ways. First we could download the data from its origin and import it in csv format.

Or we could use the scikit-learn library dataset which has our dataset.
"""

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

"""2.1. We split the data to Test and Train by 1 to 4 ratio.

Also use the 42 for random state so for upcoming analyzes we the exactly same Train and Test.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Turning Warnings to show it once
import warnings
warnings.filterwarnings('ignore')

"""**3.Testing Model on Dataset**

We test the model on Dataset without any tuning.
"""

# Instantiate the model
model = LinearClassifier()
# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

"""**4. Enhancing results using Hyperparameter Tunning**

In this code, we try to optimize the result by hyperparameter tuning.

the cv for this example is equal to "3"
"""

from sklearn.model_selection import train_test_split, GridSearchCV

# Create a LinearClassifier instance
classifier = LinearClassifier()

# Define the hyperparameter grid to search over
param_grid = {
    'learning_rate': np.arange(0, 0.1, 0.01),
    'num_iterations': [200, 500, 1000,1500],
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10.0],
    'regularization_strength': [0.01, 0.1, 1.0]
}

# Create a GridSearchCV instance to search over the hyperparameter grid
grid_search = GridSearchCV(classifier, param_grid, cv=3)

# Fit the GridSearchCV instance to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding classification accuracy
print("Best hyperparameters:", grid_search.best_params_)
print("Best classification accuracy:", grid_search.best_score_)

"""**5. Cross-validating to find best model**

Here we use the Hyperparameter Tuning and cross-validating to get the best result.

As previous, the params have the same range. but we change the cv value from 1 to 7, to get the best accuracy. For every cv, we have the best params and best accuracy associated with these cv and params.

"""

param_grid = {
    'learning_rate': np.arange(0, 0.1, 0.01),
    'num_iterations': [200, 500, 1000,1500],
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10.0],
    'regularization_strength': [0.01, 0.1, 1.0]
}

model = LinearClassifier()

# Define a range of cv values to try
cv_range = range(2, 8)

# Create an empty list to store the mean and std of the scores for each cv value
cv_scores_mean = []
cv_scores_std = []

# Loop over the cv_range and compute the mean and std of the scores for each cv value
for cv in cv_range:
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    print("CV:", cv)
    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)
    cv_scores = grid_search.cv_results_['mean_test_score']
    cv_scores_mean.append(np.mean(cv_scores))
    cv_scores_std.append(np.std(cv_scores))

# Print the cv values and corresponding mean and std of the scores
for i in range(len(cv_range)):
    print("CV = %d, mean = %f, std = %f" % (cv_range[i], cv_scores_mean[i], cv_scores_std[i]))

"""**The best model has accuracy of 92.31%**"""