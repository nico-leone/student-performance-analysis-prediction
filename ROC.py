import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc


file_path = "ResearchInformation3.csv"
df = pd.read_csv(file_path)

# Here is where we create target variable and encode our categories.
df['Successful'] = (df['Overall'] >= 3.5).astype(int)
categorical_cols = [
    'Department', 'Gender', 'Income', 'Hometown', 'Preparation',
    'Gaming', 'Attendance', 'Job', 'Extra', 'Semester'
]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Here is where we define our features and target of "Successful".
X = df.drop(columns=['Overall', 'Successful'])
y = df['Successful']
numeric_cols = ['HSC', 'SSC', 'Computer', 'English', 'Last']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Here we train our test slit and models.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Find out predicted probabilities.
log_probs = log_reg.predict_proba(X_test)[:, 1]
tree_probs = tree.predict_proba(X_test)[:, 1]

# Here we calculate ROC curves and AUC.
log_fpr, log_tpr, _ = roc_curve(y_test, log_probs)
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_probs)
log_auc = auc(log_fpr, log_tpr)
tree_auc = auc(tree_fpr, tree_tpr)

# Plot our data onto a graph.
plt.figure(figsize=(8, 6))
plt.plot(log_fpr, log_tpr, label=f'Logistic Regression (AUC = {log_auc:.2f})')
plt.plot(tree_fpr, tree_tpr, label=f'Decision Tree (AUC = {tree_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Logistic Regression vs Decision Tree')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()