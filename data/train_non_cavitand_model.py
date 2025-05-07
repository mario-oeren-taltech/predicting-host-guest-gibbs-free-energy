import os

import numpy
import pandas
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

ANALYSIS = True  # All sets are run if True.

# Load the CSV files as a Pandas dataframes.
data = pandas.read_csv(os.path.join(os.getcwd(), 'anion-not-caviton-data.csv'))
external_data = pandas.read_csv(os.path.join(os.getcwd(), 'anion-external-data.csv'))

TRN = [

    # The training set from the caviton model.
    0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38,
    39, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71,
    75, 78, 80, 81, 83, 84, 86, 87, 89, 94, 97, 98, 100, 101, 104,

    # The validation set from the caviton model.
    5, 15, 18, 29, 43, 54, 67, 72, 74, 76, 82, 88, 91, 93, 95, 103,

    # Random complexes from the non-cavitons.
    106, 107, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 126, 127, 128, 129, 130, 135, 136,
    137, 138, 139, 140, 141, 142, 143, 145, 146, 148, 149, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164,
    166, 167, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193,
    194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 220,
    221, 223, 224, 225, 227, 228, 229, 230, 232, 233, 235, 236, 237, 238, 239, 241, 242, 246
]

TST = [

    # The test set from the caviton model.
    1, 7, 17, 25, 34, 47, 60, 73, 77, 79, 85, 90, 92, 96, 99, 102,

    # Random complexes from the non-cavitons.
    105, 109, 112, 123, 125, 131, 132, 133, 134, 144, 147, 150, 151, 158, 165, 168, 170, 180, 188, 189, 197, 205, 218,
    219, 222, 226, 231, 234, 240, 243, 244, 245

]

TRN_DESCRIPTORS, TRN_VALUES, TST_VALUES, TST_DESCRIPTORS = [], [], [], []

for index, row in data.iterrows():

    if index in TRN:
        TRN_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        TRN_VALUES.append(row.iloc[1])

    elif index in TST:
        TST_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        TST_VALUES.append(row.iloc[1])

    else:
        raise RuntimeError(f'index {index} is not used.')

# Register the descriptors and the respective values for the external data.
EXT_DESCRIPTORS, EXT_VALUES = [], []

for index, row in external_data.iterrows():
    EXT_DESCRIPTORS.append(row.iloc[2:])
    EXT_VALUES.append(row.iloc[1])


def train_model(trn_descriptors, trn_values, val_descriptors, val_values):
    """
    :param trn_descriptors: The descriptors for the training set.
    :param trn_values: The values for the training set.
    :param val_descriptors: The descriptors for the validation set.
    :param val_values: The values for the validation set.
    """

    # Initialise the random forest regressor.
    rfr = RandomForestRegressor(n_estimators=1000, random_state=42, oob_score=True)
    rfr.fit(trn_descriptors, trn_values)

    results = rfr.predict(trn_descriptors)
    print(f'The R2 score for the training set: {r2_score(trn_values, results)}.')
    print(f'The RMSE score for the training set: {numpy.sqrt(numpy.mean((numpy.array(trn_values) - results) ** 2))}')
    print(f'The MAE score for the training set: {mean_absolute_error(trn_values, results)}')

    results = rfr.predict(val_descriptors)
    print(f'The R2 score for the test set: {r2_score(val_values, results)}.')
    print(f'The RMSE score for the test set: {numpy.sqrt(numpy.mean((numpy.array(val_values) - results) ** 2))}')
    print(f'The MAE score for the test set: {mean_absolute_error(val_values, results)}')

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
model = train_model(TRN_DESCRIPTORS, TRN_VALUES, TST_DESCRIPTORS, TST_VALUES)

# Test the external test set.
ext_results = numpy.array(model.predict(EXT_DESCRIPTORS))

print(
    f'SbF6-: {round(float(ext_results[0]), 1)} kJ per mol\n'
    f'PF6-:  {round(float(ext_results[1]), 1)} kJ per mol\n'
    f'ReO4-: {round(float(ext_results[2]), 1)} kJ per mol\n'
    f'ClO4-: {round(float(ext_results[3]), 1)} kJ per mol\n'
    f'I-:    {round(float(ext_results[5]), 1)} kJ per mol\n'
    f'BF4-:  {round(float(ext_results[4]), 1)} kJ per mol'
)
