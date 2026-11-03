import os
import joblib
import numpy
import pandas

# Load the CSV files as a Pandas dataframes.
data = pandas.read_csv('/app/input_dataset.csv')

model = joblib.load("/app/models/model.joblib")
# Register the descriptors and the respective values for the external data.
EXT_DESCRIPTORS, EXT_VALUES = [], []

for index, row in data.iterrows():
    EXT_DESCRIPTORS.append(row.iloc[2:])
    EXT_VALUES.append(row.iloc[1])



# Test the external test set.
ext_results = numpy.array(model.predict(EXT_DESCRIPTORS))


print(round(float(ext_results[0]), 1))
