import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("ResearchInformation3.csv")
df['Successful'] = (df['Overall'] >= 3.5).astype(int)

# Encoding
df_encoded = df.copy()
le = LabelEncoder()

for col in ['Gaming', 'Hometown', 'Income']:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Setup for plotting
sns.set(style="whitegrid")

# Plot 1: Stacked bar chart - Gaming vs Success
plt.figure(figsize=(8, 6))
gaming_counts = df.groupby(['Gaming', 'Successful']).size().unstack(fill_value=0)
gaming_counts.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8, 6))
plt.title('Success vs Failure by Gaming Time')
plt.xlabel('Gaming Category')
plt.ylabel('Number of Students')
plt.legend(title='Successful', labels=['No', 'Yes'])
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Plot 2: Stacked bar chart - Hometown vs Success
plt.figure(figsize=(6, 5))
hometown_counts = df.groupby(['Hometown', 'Successful']).size().unstack(fill_value=0)
hometown_counts.plot(kind='bar', stacked=True, colormap='Set2', figsize=(6, 5))
plt.title('Success vs Failure by Hometown')
plt.xlabel('Hometown')
plt.ylabel('Number of Students')
plt.legend(title='Successful', labels=['No', 'Yes'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot 3: Stacked bar chart - Income vs Success
plt.figure(figsize=(10, 6))
income_counts = df.groupby(['Income', 'Successful']).size().unstack(fill_value=0)
income_counts.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 6))
plt.title('Success vs Failure by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Number of Students')
plt.legend(title='Successful', labels=['No', 'Yes'])
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()