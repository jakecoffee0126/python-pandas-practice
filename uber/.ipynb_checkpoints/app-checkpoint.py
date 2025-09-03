

'''
using print(df.isna().sum()) a lot, because we want to make sure
 we clean up all the NA value
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import os


'''
header=0 means that the headers for the variable names are to be found in the first row (note that 0 means the first row in Python)
sep="," means that "," is used as the separator between the values. This is because we are using the file type .csv (comma separated values)

When we load a data set using Pandas, all blank cells are automatically converted into "NaN" values.
'''

df = pd.read_csv('uber/ncr_ride_bookings.csv', header=0, sep=",")
# print(df.info())
# print(df["Booking Status"].value_counts())
# print(df.isna().sum())


# check all the data type for each column
# for col in df.columns:
#    print(df[col].dtype)

numeric_col = [col for col in df.columns if df[col].dtype != 'object']
# print(numeric_col)

# print out the header
# print(df.head(0))
# print(df.columns.tolist())


# After calling sns.heatmap(), it is necessary to call plt.show()
# to display the plot, especially in scripts or environments where plots are not automatically rendered.

sns.heatmap(df[numeric_col].corr())
# df[numeric_col].corr()
# plt.show()


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
target_columns = [
    'Avg VTAT',
    'Avg CTAT',
    'Booking Value',
    'Ride Distance',
    'Driver Ratings',
    'Customer Rating'
]

df[target_columns] = imp_mean.fit_transform(df[target_columns])
# print(df.isna().sum())


df["Cancelled Rides by Driver"] = df["Cancelled Rides by Driver"].replace(
    np.nan, 0)
# value_count, By default, rows that contain any NA values are omitted from the result.
total = df["Cancelled Rides by Driver"].value_counts()
# print(total)


imp_cat = SimpleImputer(strategy='most_frequent')
# fit_transform returns 2D array, so flattening it using [:, 0]
df["Payment Method"] = imp_cat.fit_transform(df[["Payment Method"]])[:, 0]

# print(df.isna().sum())


# df['Reason for cancelling by Customer'].value_counts()
# df['Reason for cancelling by Customer'].fillna('Unknown', inplace=True)

# df['Driver Cancellation Reason'].value_counts()
# df['Driver Cancellation Reason'].fillna('Unknown', inplace=True)

# df['Incomplete Rides Reason'].value_counts()
# df['Incomplete Rides Reason'].fillna('Unknown', inplace=True)

# df['Cancelled Rides by Customer'].value_counts()
# df['Cancelled Rides by Customer'].fillna('Unknown', inplace=True)

# print(df.isna().sum())
# print(df.describe())
# print(df.info())

# If Date and Time are separate columns
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')

# Combine Date and Time into a single datetime column (optional)
df['DateTime'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Time'].astype(str))


print(df['Booking Status'].unique())
# check the original and new csv file, and will know why use the map function
df["Booking Status"] = df["Booking Status"].map({
    'No Driver Found': 0,
    'Incomplete': 1,
    'Completed': 2,
    'Cancelled by Driver': 3,
    'Cancelled by Customer': 4
})


# with open("uber/ncr_ride_bookings_new.csv", "w"):
#    df.to_csv('uber/ncr_ride_bookings_new.csv', index=False)
