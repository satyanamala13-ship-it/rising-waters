import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
print("Libraries imported successfully!")
dataset = pd.read_excel("flood dataset.xlsx")
print("Dataset loaded successfully!")
print(dataset.head())
print("Shape:", dataset.shape)
dataset.info()
print(dataset.describe())
print(dataset.isnull().sum())
plt.figure(figsize=(8,5))
sns.histplot(dataset.iloc[:,0], kde=True)
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(y=dataset.iloc[:,0])
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("Dataset loaded successfully!")
print(dataset.head())
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print(cm)