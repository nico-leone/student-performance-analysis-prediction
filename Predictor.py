import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
#import shap

df = pandas.read_csv("ResearchInformation3.csv")

features = [
    'SSC', 'HSC', 'Computer', 'English',
    'Gaming', 'Preparation', 'Attendance', 'Job',
    'Income', 'Extra', 'Hometown'
]
categorical_columnss = ['Gaming', 'Preparation', 'Attendance', 'Job', 'Income', 'Extra', 'Hometown']
numeric_columnss = ['SSC', 'HSC', 'Computer', 'English']

label_encoders = {}
for columns in categorical_columnss:
    encode = LabelEncoder()
    df[columns] = encode.fit_transform(df[columns])
    label_encoders[columns] = encode

scaler = StandardScaler()
df[numeric_columnss] = scaler.fit_transform(df[numeric_columnss])

#Model Training
X = df[features]
y = df['Overall']
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(eval_metric='rmse', random_state=42)
rf_model.fit(X, y)
xgb_model.fit(X, y)

print("\nStudent GPA Predictor for Random Forest & XGBoost")

while True:
    filename = input("\nEnter the name of a profile or type exit to quit: ")
    if filename.lower() == 'exit':
        break

    profile = {}

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key in categorical_columnss:
                    valid_classes = list(label_encoders[key].classes_)
                    if value not in valid_classes:
                        raise ValueError(f"Invalid value '{value}' for {key}. Expected one of: {valid_classes}")
                    profile[key] = label_encoders[key].transform([value])[0]
                elif key in numeric_columnss:
                    profile[key] = float(value)

        data = pandas.DataFrame([profile])
        data[numeric_columnss] = scaler.transform(data[numeric_columnss])

        predicted_rf_gpa = rf_model.predict(data)[0]
        predicted_xgb_gpa = xgb_model.predict(data)[0]

        print(f"\n🎓 Random Forest Predicted GPA: {predicted_rf_gpa:.2f}")
        print(f"🎓 XGBoost Predicted GPA: {predicted_xgb_gpa:.2f}")

        '''
        explainer_rf = shap.TreeExplainer(rf_model)
        explainer_xgb = shap.TreeExplainer(xgb_model)
        shap_values_rf = explainer_rf.shap_values(data)
        shap_values_xgb = explainer_xgb.shap_values(data)
        
        print("\n🔍 Feature Contributions (Random Forest):")
        for feature, value in zip(data.columns, shap_values_rf[0]):
            print(f"{feature}: {value:+.3f}")

        print("\n🔍 Feature Contributions (XGBoost):")
        for feature, value in zip(data.columns, shap_values_xgb[0]):
            print(f"{feature}: {value:+.3f}")
        '''

    except FileNotFoundError:
        print(f"File '{filename}' not found. Please try again.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



#print(df.groupby('Preparation')['Overall'].agg(['mean', 'count']))






