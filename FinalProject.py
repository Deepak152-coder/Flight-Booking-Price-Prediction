print("\n" + "-"*60 + " Project: Flight Booking Price Prediction " + "-"*60 + "\n")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Clean_Dataset.csv")
df = df.drop(columns=["Unnamed: @"], errors='ignore')

price_bins = [0, 5000, 15000, float('inf')]
price_labels = ['Low', 'Medium', 'High']
df['price_category'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'price_category':
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['price', 'price_category'])
y = df['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("\n" + "-"*45 + " Confusion Matrix " + "-"*45 + "\n")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=price_labels, yticklabels=price_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n" + "-"*42 + " Classification Report " + "-"*42 + "\n")
print(classification_report(y_test, y_pred, target_names=price_labels))

print("\n" + "-"*40 + " Predicted vs Actual Output " + "-"*40 + "\n")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sample_results = results.sample(50, random_state=42).sort_index()

plt.figure(figsize=(12, 6))
plt.plot(sample_results['Actual'], 'o-', label='Actual', markersize=8)
plt.plot(sample_results['Predicted'], 's--', label='Predicted', markersize=6)
plt.title('Actual vs Predicted Flight Price Categories (Sample of 50)')
plt.ylabel('Price Category')
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "-"*73 + " End of Project " + "-"*73 + "\n")
