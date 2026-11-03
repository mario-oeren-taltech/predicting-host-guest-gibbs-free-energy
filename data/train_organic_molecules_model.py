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
data = pandas.read_csv(os.path.join(os.getcwd(), 'organic-molecules-data.csv'))
external_data = pandas.read_csv(os.path.join(os.getcwd(), 'anion-external-data.csv'))



TRN = [0, 1, 5, 8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 34, 35, 37, 38, 41, 42, 44, 45, 46, 47, 48, 49, 50, 53, 54, 55, 57, 59, 61, 62, 63, 64, 66, 67, 70, 73, 74, 76, 77, 78, 79, 80, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 104, 105, 106, 108, 109, 112, 113, 114, 115, 116, 118, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 138, 140, 141, 142, 144, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 164, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 177, 179, 180, 183, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 200, 201, 203, 204, 205, 206, 208, 209, 210, 211, 213, 215, 217, 218, 219, 220, 221, 222, 223, 225, 226, 228, 230, 232, 233, 237, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 254, 256, 257, 259, 262, 263, 264, 265, 266, 267, 268, 270, 271, 272, 274, 275, 276, 279, 281, 282, 283, 286, 287, 288, 289, 290, 292, 294, 296, 297, 298, 299, 300, 301, 302, 303, 306, 307, 308, 310, 311, 312, 313, 315, 317, 318, 319, 320, 321, 322, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 342, 343, 345, 346, 347, 348, 349, 350, 354, 355, 357, 360, 361]
VAL = [3, 4, 7, 14, 30, 33, 36, 39, 43, 56, 58, 65, 69, 75, 83, 98, 107, 120, 143, 145, 147, 160, 161, 170, 181, 182, 184, 185, 186, 195, 199, 202, 214, 216, 224, 236, 238, 242, 251, 255, 258, 260, 261, 269, 273, 280, 284, 291, 295, 305, 325, 326, 341, 353, 2, 6, 17, 19, 24, 40, 51, 52, 60, 68, 71, 72, 81, 102, 103, 110, 111, 117, 119, 131, 136, 137, 139, 146, 151, 162, 163, 176, 178, 207, 212, 227, 229, 231, 234, 235, 252, 253, 277, 278, 285, 293, 304, 309, 314, 316, 323, 324, 344, 351, 352, 356, 358, 359]


print(len(TRN))
print(len(VAL))


TRN_DESCRIPTORS, TRN_VALUES, VAL_DESCRIPTORS, VAL_VALUES, TST_VALUES, TST_DESCRIPTORS = [], [], [], [], [], []

for index, row in data.iterrows():

    if index in TRN:
        TRN_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        TRN_VALUES.append(row.iloc[1])

    elif index in VAL:
        VAL_DESCRIPTORS.append(row.iloc[2:])  # Replace with 2:7 for only energy.
        VAL_VALUES.append(row.iloc[1])



# Register the descriptors and the respective values for the external data.
EXT_DESCRIPTORS, EXT_VALUES = [], []

for index, row in external_data.iterrows():
    EXT_DESCRIPTORS.append(row.iloc[2:])
    EXT_VALUES.append(row.iloc[1])


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
    print(f'The RMSE score for the validation set ({set_name}): {numpy.sqrt(numpy.mean((numpy.array(val_values) - results) ** 2))}')
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



