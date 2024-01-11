import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import numpy as np

# Wczytaj dane
df = pd.read_csv(r'C:\Users\adamk\Desktop\Output.csv', sep=';')
df.drop(columns=['Sets URL', 'Part URL'], axis=1, inplace=True)
df.dropna(axis=0, inplace=True)
df['Star rating'] = df['Star rating'].apply(lambda x: x.replace(',', '.'))
numeric_features = ['Set Price', 'Number of reviews', 'Star rating', 'year']

for feature in numeric_features: 
    df[feature] = pd.to_numeric(df[feature].apply(lambda x: x.replace(",","") if type(x) not in (int, float) else x))


encoders = {}
text_features = [feature for feature in df.columns if feature not in numeric_features]

for feature in text_features: 
    encoders[feature] = LabelEncoder()
    df[feature] = encoders[feature].fit_transform(df[feature])

plt.figure(figsize=(20,7))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
corr = df.corr()
corr = corr["Set Price"]
threshold = 0.005
low_corr_features = corr[abs(corr) < threshold].index
print(f"Removing these features: {low_corr_features}")
df_filtered = df.drop(low_corr_features, axis=1)
df_filtered.drop_duplicates(inplace=True)
plt.show()


features = df_filtered.drop('Set Price', axis=1)
target = df_filtered['Set Price']



scaler = StandardScaler()


features_scaled = scaler.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


pipe1 = Pipeline([
    ("regressor", RandomForestRegressor(n_jobs=-1))
])


pipe1.fit(X_train, y_train)


y_pred = pipe1.predict(X_test)


# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MeanAbsoluteError with RandomForestRegressor is {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
