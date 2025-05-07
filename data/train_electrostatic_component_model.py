import os

import numpy
import pandas

from matplotlib import pyplot

# Linear Models
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor,
                                  PassiveAggressiveRegressor)

# Neighbors
from sklearn.neighbors import KNeighborsRegressor

# Tree-based models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Support Vector Models
from sklearn.svm import SVR, NuSVR, LinearSVR

# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor

# Load additional classes and functions.
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


# Load the CSV files as a Pandas dataframes.
data = pandas.read_csv(os.path.join(os.getcwd(), 'anion-data.csv'))

# Compile the arrays for descriptors and values.
DESCRIPTORS, VALUES = [], []

for index, row in data.iterrows():

    DESCRIPTORS.append(numpy.array([float(descriptor) for descriptor in row.iloc[2:6]]))
    VALUES.append(float(row.iloc[1]))

DESCRIPTORS = numpy.array(DESCRIPTORS)
VALUES = numpy.array(VALUES)

# Put together the five-fold cross-validation sets.
validation_sets = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results for r2 and mae
r2_scores = {
    'Linear Regression': [],
    'Ridge': [],
    'Lasso': [],
    'Elastic Net': [],
    'Bayesian Ridge': [],
    'Huber Regressor': [],
    'Passive Aggressive Regressor': [],
    'K-nearest Neighbors Regressor': [],
    'Decision Tree Regressor': [],
    'Random Forest Regressor': [],
    'Gradient Boosting Regressor': [],
    'Ada Boost Regressor': [],
    'SVR': [],
    'NuSVR': [],
    'Linear SVR': [],
    'Gaussian Process Regressor': []
}

mae_scores = {
    'Linear Regression': [],
    'Ridge': [],
    'Lasso': [],
    'Elastic Net': [],
    'Bayesian Ridge': [],
    'Huber Regressor': [],
    'Passive Aggressive Regressor': [],
    'K-nearest Neighbors Regressor': [],
    'Decision Tree Regressor': [],
    'Random Forest Regressor': [],
    'Gradient Boosting Regressor': [],
    'Ada Boost Regressor': [],
    'SVR': [],
    'NuSVR': [],
    'Linear SVR': [],
    'Gaussian Process Regressor': []
}

# Test different modelling methods using default parameters.
for train_index, test_index in validation_sets.split(DESCRIPTORS, VALUES):

    # Split data into train and test sets
    X_train, X_test = DESCRIPTORS[train_index], DESCRIPTORS[test_index]
    y_train, y_test = VALUES[train_index], VALUES[test_index]

    # Define models and corresponding names
    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge()),
        ('Lasso', Lasso(random_state=42)),
        ('Elastic Net', ElasticNet(random_state=42)),
        ('Bayesian Ridge', BayesianRidge()),
        ('Huber Regressor', HuberRegressor()),
        ('Passive Aggressive Regressor', PassiveAggressiveRegressor(random_state=42)),
        ('K-nearest Neighbors Regressor', KNeighborsRegressor()),
        ('Decision Tree Regressor', DecisionTreeRegressor(random_state=42)),
        ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
        ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42)),
        ('Ada Boost Regressor', AdaBoostRegressor(random_state=42)),
        ('SVR', SVR()),
        ('NuSVR', NuSVR()),
        ('Linear SVR', LinearSVR(random_state=42)),
        ('Gaussian Process Regressor', GaussianProcessRegressor())
    ]

    # Loop through models
    for name, model in models:
        model.fit(X_train, y_train)
        results = model.predict(X_test)

        # Calculate r2 and mae
        r2 = r2_score(y_test, results)
        mae = mean_absolute_error(y_test, results)

        # Append results to the corresponding lists
        r2_scores[name].append(round(r2, 2))
        mae_scores[name].append(round(mae, 2))

# Print r2 scores table
print("R2:")
for name in r2_scores:
    print(f'{name}, {', '.join(map(str, r2_scores[name]))}')

# Print MAE scores table
print("\nMean Absolute Error (MAE):")
for name in mae_scores:
    print(f'{name}, {', '.join(map(str, mae_scores[name]))}')

# As expected, the ensemble methods had the best performance with Random forest regressor having the highest average R2.
for train_index, test_index in validation_sets.split(DESCRIPTORS, VALUES):

    # Split data into train and test sets
    X_train, X_test = DESCRIPTORS[train_index], DESCRIPTORS[test_index]
    y_train, y_test = VALUES[train_index], VALUES[test_index]
    
    # Train a random forest model.
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    results = model.predict(X_test)

    # Plot the descriptor importance.
    descriptor_importance = model.feature_importances_
    print(round(descriptor_importance[0], 2), round(descriptor_importance[1], 2), round(descriptor_importance[2], 2),
          round(descriptor_importance[3], 2))

    pyplot.figure(figsize=(10, 6))
    pyplot.barh(range(len(descriptor_importance)), descriptor_importance, align='center')
    pyplot.yticks(range(len(descriptor_importance)), data.columns[2:6])
    pyplot.xlabel('Feature Importance')
    pyplot.ylabel('Feature')
    pyplot.title("Random Forest Feature Importance's")
    pyplot.show()

    # Plot the correlation.
    margin = 10
    pyplot.figure(figsize=(8, 6))
    pyplot.scatter(y_test, results, color='blue', label='Correlated Data')
    pyplot.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Identity Line')
    pyplot.plot([min(y_test), max(y_test)], [min(y_test) + margin, max(y_test) + margin], 'r--',
                label=f'+{margin} kJ per mol')
    pyplot.plot([min(y_test), max(y_test)], [min(y_test) - margin, max(y_test) - margin], 'r--',
                label=f'-{margin} kJ per mol')
    pyplot.xlabel('Predicted binding affinity')
    pyplot.ylabel('Measured binding affinity')
    pyplot.title('Correlated Data Plot')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
