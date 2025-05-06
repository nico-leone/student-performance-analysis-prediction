from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ResearchInformation3.csv")

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

data = df.copy()
data = data.drop(columns=['Overall'])

categorical_cols = [
    'Department', 'Gender', 'Income', 'Hometown', 'Preparation',
    'Gaming', 'Attendance', 'Job', 'Extra', 'Semester'
]

le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

le = LabelEncoder()
data['Outcome'] = le.fit_transform(data['Outcome'])

X = data.drop(columns=['Outcome'])
y = data['Outcome']

numeric_cols = ['HSC', 'SSC', 'Computer', 'English', 'Last']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

#Model training
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        y_test_binarized = label_binarize(y_test, classes=list(range(y_prob.shape[1])))
        roc_auc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
    else:
        roc_auc = "N/A"
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'ROC AUC': roc_auc
    }

class_labels = le.classes_

#Confusion Matrix Plot
for name in results:
    cm = results[name]['Confusion Matrix']
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

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