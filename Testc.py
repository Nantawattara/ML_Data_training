import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # ใช้ joblib แทน pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ดาวน์โหลดข้อมูลจาก Kaggle (ผู้ใช้ต้องดาวน์โหลดเองและวางไฟล์ไว้ในเครื่อง)
file_path = "Bank-Customer-Attrition-Insights-Data.csv"
df = pd.read_csv(file_path)

# เลือก 6 Feature ที่สำคัญที่สุด ตาม Feature Importance
selected_features = ["Complain", "Age", "IsActiveMember", "NumOfProducts", "Geography", "Balance"]

# Encode Categorical Feature ('Geography')
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])

# เลือกเฉพาะคอลัมน์ที่จำเป็น
X = df[selected_features]
y = df['Exited']

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ SMOTE เพื่อแก้ปัญหา Class Imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ทำ Standardization กับข้อมูลเชิงตัวเลข
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล Ensemble Learning
models = [
    ('Logistic Regression', LogisticRegression()),
    ('kNN', KNeighborsClassifier(n_neighbors=5)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100))
]

# ใช้ Voting Classifier รวมพลังโมเดลทั้งหมด
voting_clf = VotingClassifier(estimators=models, voting='soft')

# Train โมเดล
voting_clf.fit(X_train, y_train)

# บันทึกโมเดลเป็นไฟล์โดยใช้ joblib
joblib.dump(voting_clf, 'customer_churn_model.joblib')

# บันทึก StandardScaler โดยใช้ joblib
joblib.dump(scaler, 'scaler.joblib')

print("โมเดลถูกบันทึกเรียบร้อยแล้วเป็นไฟล์ customer_churn_model.joblib และ scaler.joblib")
