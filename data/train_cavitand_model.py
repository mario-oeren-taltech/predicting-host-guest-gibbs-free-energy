import os

import numpy
import pandas
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

ANALYSIS = True  # All sets are run if True.

# Load the CSV files as a Pandas dataframes.
data = pandas.read_csv(os.path.join(os.getcwd(), 'anion-data.csv'))
external_data = pandas.read_csv(os.path.join(os.getcwd(), 'anion-external-data.csv'))

TRN = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38,
       39, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71,
       75, 78, 80, 81, 83, 84, 86, 87, 89, 94, 97, 98, 100, 101, 104]
VAL = [5, 15, 18, 29, 43, 54, 67, 72, 74, 76, 82, 88, 91, 93, 95, 103]
TST = [1, 7, 17, 25, 34, 47, 60, 73, 77, 79, 85, 90, 92, 96, 99, 102]

TRN_DESCRIPTORS, TRN_VALUES, VAL_DESCRIPTORS, VAL_VALUES, TST_VALUES, TST_DESCRIPTORS = [], [], [], [], [], []

for index, row in data.iterrows():

    if index in TRN:
        TRN_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        TRN_VALUES.append(row.iloc[1])

    elif index in VAL:
        VAL_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        VAL_VALUES.append(row.iloc[1])

    else:  # TST
        TST_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        TST_VALUES.append(row.iloc[1])

# Register the descriptors and the respective values for the external data.
EXT_DESCRIPTORS, EXT_VALUES = [], []

for index, row in external_data.iterrows():
    EXT_DESCRIPTORS.append(row.iloc[2:])
    EXT_VALUES.append(row.iloc[1])

# # Grid search.
# rfr = RandomForestRegressor(random_state=42)
#
# # Define the hyperparameter grid (discrete values for each parameter)
# param_grid = {
#     'n_estimators': [100, 500, 1000, 5000],
#     'max_depth': [None, 10, 20, 30, 100],
#     'min_samples_split': [2, 5, 10, 40],
#     'min_samples_leaf': [1, 2, 4, 20],
#     'bootstrap': [True, False],
#     'oob_score': [True, False]
# }
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)
#
# # Fit the model with GridSearchCV
# grid_search.fit(TRN_DESCRIPTORS, TRN_VALUES)
#
# # Print the best hyperparameters found
# print("Best Hyperparameters:", grid_search.best_params_)
#
# # Predict on the test set using the best model
# y_pred = grid_search.predict(VAL_DESCRIPTORS)
#
# # Evaluate the model performance (mean absolute error)
# mae = mean_absolute_error(VAL_VALUES, y_pred)
# print(f"Mean Absolute Error: {mae:.2f}")


def train_model(trn_descriptors, trn_values, val_descriptors, val_values, set_name):
    """
    :param trn_descriptors: The descriptors for the training set.
    :param trn_values: The values for the training set.
    :param val_descriptors: The descriptors for the validation set.
    :param val_values: The values for the validation set.
    :param set_name: The name of the set.
    """

    # Initialise the random forest regressor.
    rfr = RandomForestRegressor(n_estimators=1000, random_state=42, oob_score=True)
    rfr.fit(trn_descriptors, trn_values)

    results = rfr.predict(trn_descriptors)
    print(f'The R2 score for the training set ({set_name}): {r2_score(trn_values, results)}.')
    print(f'The RMSE score for the training set: {numpy.sqrt(numpy.mean((numpy.array(trn_values) - results) ** 2))}')
    print(f'The MAE score for the training set: {mean_absolute_error(trn_values, results)}')

    results = rfr.predict(val_descriptors)
    print(f'The R2 score for the validation set ({set_name}): {r2_score(val_values, results)}.')
    print(f'The RMSE score for the validation set ({set_name}): {numpy.sqrt(numpy.mean((numpy.array(val_values) -
                                                                                        results) ** 2))}')
    print(f'The MAE score for the validation set ({set_name}): {mean_absolute_error(val_values, results)}')

    # Get the feature importance's.
    descriptor_importance = rfr.feature_importances_

    # Create a bar plot to visualize feature importance's.
    pyplot.figure(figsize=(10, 6))
    pyplot.barh(range(len(descriptor_importance)), descriptor_importance, align='center')
    pyplot.yticks(range(len(descriptor_importance)), data.columns[2:])
    pyplot.xlabel('Feature Importance')
    pyplot.ylabel('Feature')
    pyplot.title("Random Forest Feature Importance's")
    pyplot.show()

    # Plot the correlation.
    margin = 10
    pyplot.figure(figsize=(8, 6))
    pyplot.scatter(val_values, results, color='blue', label='Correlated Data')
    pyplot.plot([min(val_values), max(val_values)], [min(val_values), max(val_values)], 'k--', label='Identity Line')
    pyplot.plot([min(val_values), max(val_values)], [min(val_values) + margin, max(val_values) + margin], 'r--',
                label=f'+{margin} kJ per mol')
    pyplot.plot([min(val_values), max(val_values)], [min(val_values) - margin, max(val_values) - margin], 'r--',
                label=f'-{margin} kJ per mol')
    pyplot.xlabel('Predicted binding affinity')
    pyplot.ylabel('Measured binding affinity')
    pyplot.title('Correlated Data Plot')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

    return rfr


# Train the model.
model = train_model(TRN_DESCRIPTORS, TRN_VALUES, VAL_DESCRIPTORS, VAL_VALUES, 'Validation')

if ANALYSIS is True:

    # Test on the test set.
    tst_results = numpy.array(model.predict(TST_DESCRIPTORS))

    print(f'The R2 score for the test set: {r2_score(TST_VALUES, tst_results)}.')
    print(f'The RMSE score for the test set: {numpy.sqrt(numpy.mean((numpy.array(TST_VALUES) - tst_results) ** 2))}')
    print(f'The MAE score for the test set: {mean_absolute_error(TST_VALUES, tst_results)}')

    # Plot the correlation.
    margin = 10
    pyplot.figure(figsize=(8, 6))
    pyplot.scatter(TST_VALUES, tst_results, color='blue', label='Correlated Data')
    pyplot.plot([min(TST_VALUES), max(TST_VALUES)], [min(TST_VALUES), max(TST_VALUES)], 'k--', label='Identity Line')
    pyplot.plot([min(TST_VALUES), max(TST_VALUES)], [min(TST_VALUES) + margin, max(TST_VALUES) + margin], 'r--',
                label=f'+{margin} kJ per mol')
    pyplot.plot([min(TST_VALUES), max(TST_VALUES)], [min(TST_VALUES) - margin, max(TST_VALUES) - margin], 'r--',
                label=f'-{margin} kJ per mol')
    pyplot.xlabel('Predicted binding affinity')
    pyplot.ylabel('Measured binding affinity')
    pyplot.title('Correlated Data Plot')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

    # Test the external test set.
    ext_results = numpy.array(model.predict(EXT_DESCRIPTORS))

    print(
        f'SBF6-: {round(float(ext_results[0]), 1)} kJ per mol\n'
        f'FF6-:  {round(float(ext_results[1]), 1)} kJ per mol\n'
        f'ReO4-: {round(float(ext_results[2]), 1)} kJ per mol\n'
        f'ClO4-: {round(float(ext_results[3]), 1)} kJ per mol\n'
        f'I-:    {round(float(ext_results[5]), 1)} kJ per mol\n'
        f'BF4-:  {round(float(ext_results[4]), 1)} kJ per mol'
    )
