import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# 1️ SIMULATE IoT BIN DATA


np.random.seed(42)

num_bins = 20
days = 15
records = []

areas = ['Residential', 'Commercial', 'Industrial']

for bin_id in range(1, num_bins + 1):
    area = np.random.choice(areas)
    base_fill = np.random.randint(5, 20)

    for day in range(days):
        for hour in range(0, 24, 3):
            timestamp = datetime.now() - timedelta(days=days-day, hours=hour)

            if area == "Residential":
                growth = np.random.uniform(2, 5)
            elif area == "Commercial":
                growth = np.random.uniform(3, 6)
            else:
                growth = np.random.uniform(1, 4)

            fill_level = min(100, base_fill + growth * day + np.random.uniform(0, 5))

            # Introduce rare anomaly
            if np.random.rand() < 0.02:
                fill_level = np.random.uniform(90, 100)

            records.append([bin_id, area, timestamp, hour, fill_level])

df = pd.DataFrame(records, columns=['bin_id', 'area', 'timestamp', 'hour', 'fill_level'])

print("\nSample Data:")
print(df.head())


# 2️ EDA


plt.figure()
sns.boxplot(x='area', y='fill_level', data=df)
plt.title("Area-wise Fill Level Distribution")
plt.show()


# 3️ PREPROCESSING



df_encoded = pd.get_dummies(df, columns=['area'], drop_first=True)

X = df_encoded.drop(['fill_level', 'timestamp'], axis=1)
y = df_encoded['fill_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4️ XGBOOST MODEL


model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nXGBoost Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# 5️ ANOMALY DETECTION


iso_forest = IsolationForest(contamination=0.02, random_state=42)
df_encoded['anomaly'] = iso_forest.fit_predict(X)

anomalies = df_encoded[df_encoded['anomaly'] == -1]

print("\nDetected Anomalies:")
print(anomalies[['bin_id', 'hour', 'fill_level']].head())


# 6️ URGENCY CLASSIFICATION


sample_bin = X_test.iloc[0:1]
predicted_fill = model.predict(sample_bin)[0]

print("\nPredicted Fill Level:", round(predicted_fill, 2), "%")

if predicted_fill > 85:
    print("⚠️ URGENT: Collection Required")
else:
    print("✅ Not Urgent Yet")

print("\nProject Completed: XGBoost + Anomaly Detection Integrated")
input("\nPress Enter to exit...")
