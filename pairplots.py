import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import pairplot
from sklearn.preprocessing import LabelEncoder
from unicodedata import numeric

df = pd.read_csv("ResearchInformation3.csv")

# Make Success Column in dataframe and append it to existing data.
successVal = []
for index, row in df.iterrows():
    overall = row["Overall"]
    if overall < 3.25:
        successVal.append("Not Successful")
    elif 3.25 <= overall < 3.50:
        successVal.append("Middling")
    else:
        successVal.append("Success")
df['Outcome'] = successVal

# Define relevant columns for out pairplot.
numeric_features = ['SSC', 'HSC', 'Computer', 'English', 'Last', 'Overall']
pairplot_features = numeric_features + ['Outcome']

# Ensure data in our columns is numeric to prevent errors, and drop columns that do not fit.
df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')
df_clean = df.dropna(subset=pairplot_features)

# Create the pairplot
sns.pairplot(df_clean[pairplot_features], hue='Outcome', palette='Set2', diag_kind='hist')
plt.suptitle('Pairplot of Student Features by Success Label', y=1.02)
plt.show()

# Secondary Pair plot for additional columns.
numeric_subplot_features = ['Overall', 'Last', 'Gaming', 'Income', 'Job']
overall_subplot_features = numeric_subplot_features + ['Outcome']
df_subset = df[overall_subplot_features].copy()

le = LabelEncoder()
for col in ['Gaming', 'Income', 'Job']:
    df_subset[col] = le.fit_transform(df_subset[col])

df_subset[numeric_subplot_features] = df_subset[numeric_subplot_features].apply(pd.to_numeric, errors='coerce').dropna()

sns.pairplot(df_subset, hue='Outcome', palette='Set2', diag_kind='hist')
plt.suptitle('Focused Pairplot: GPA, Background & Success', y=1.02)
plt.show()