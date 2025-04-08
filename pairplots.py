import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("ResearchInformation3.csv")

# Define what qualifies as a success(3.5 GPA).
df['Successful'] = (df['Overall'] >= 3.5).astype(int)

# Define relevant columns for out pairplot.
pairplot_features = ['SSC', 'HSC', 'Computer', 'English', 'Last', 'Overall', 'Successful']

# Ensure data in our columns is numeric to prevent errors. and drop columns that do not fit.
df[pairplot_features] = df[pairplot_features].apply(pd.to_numeric, errors='coerce')
df_clean = df.dropna(subset=pairplot_features)

# Create the pairplot
sns.pairplot(df_clean[pairplot_features], hue='Successful', palette='Set2', diag_kind='hist')
plt.suptitle('Pairplot of Student Features by Success Label', y=1.02)
plt.show()


# Secondary Pair plot for additional columns.
subset_features = ['Overall', 'Last', 'Gaming', 'Income', 'Job', 'Successful']
df_subset = df[subset_features].copy()

le = LabelEncoder()
for col in ['Gaming', 'Income', 'Job']:
    df_subset[col] = le.fit_transform(df_subset[col])

df_subset = df_subset.apply(pd.to_numeric, errors='coerce').dropna()

sns.pairplot(df_subset, hue='Successful', palette='Set2', diag_kind='hist')
plt.suptitle('Focused Pairplot: GPA, Background & Success', y=1.02)
plt.show()
