import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib

# Load and clean the data
df = pd.read_csv("ckd.csv")
df = df[['age', 'bp', 'sg', 'al', 'classification']]
df.dropna(subset=['age', 'bp', 'sg', 'al', 'classification'], inplace=True)

# Normalize and map classification to binary
df['classification'] = df['classification'].str.strip().str.lower()
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
df.dropna(subset=['classification'], inplace=True)

# Train-test split
X = df[['age', 'bp', 'sg', 'al']]
y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'ckd_model.pkl')


samples = [
    [18, 70, 1.020, 0],
    [25, 75, 1.015, 0],
    [30, 80, 1.025, 0],
    [40, 85, 1.020, 0],
    [50, 90, 1.010, 0],
    [60, 120, 1.005, 3],
    [70, 130, 1.010, 4],
    [55, 110, 1.008, 2],
    [65, 115, 1.006, 3],
    [75, 140, 1.004, 5],
]

df_samples = pd.DataFrame(samples, columns=['age', 'bp', 'sg', 'al'])
predictions = model.predict(df_samples)
print(df_samples)
print(predictions)
