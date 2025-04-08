from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "ResearchInformation3.csv"
df = pd.read_csv(file_path)

df.info(), df.head()
data = df.copy()

# Create our successful column.
data['Successful'] = (data['Overall'] >= 3.5).astype(int)
data = data.drop(columns=['Overall'])

# Encoding
categorical_cols = [
    'Department', 'Gender', 'Income', 'Hometown', 'Preparation',
    'Gaming', 'Attendance', 'Job', 'Extra', 'Semester'
]

le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop(columns=['Successful'])
y = data['Successful']

numeric_cols = ['HSC', 'SSC', 'Computer', 'English', 'Last']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Model training-----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set the models we intend to use
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train or models and evaluate their results.
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    }


# Create out plot
for name in results:
    cm = results[name]['Confusion Matrix']
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Show summarize results
summary = {
    model: {
        'Accuracy': round(results[model]['Accuracy'], 3),
        'ROC AUC': round(results[model]['ROC AUC'], 3) if results[model]['ROC AUC'] != "N/A" else "N/A"
    }
    for model in results
}

summary_df = pd.DataFrame(summary).T

print("Model Performance Summary:\n")
print(summary_df.to_string())

