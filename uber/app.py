# https://www.kaggle.com/code/namannimble/uber-data-pre-processing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


df = pd.read_csv('uber/ncr_ride_bookings.csv')
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
df[numeric_col].corr()
plt.show()
