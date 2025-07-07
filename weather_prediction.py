import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import timedelta

df = pd.read_csv("weather.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofyear
df = df.sort_values("Day")

X = df[['Day']]
y = df['Temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

df['Predicted_Temp'] = model.predict(X)

last_date = df['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
future_days = [d.timetuple().tm_yday for d in future_dates]

future_df = pd.DataFrame({
    'Date': future_dates,
    'Day': future_days
})
future_df['Predicted_Temp'] = model.predict(future_df[['Day']])

combined_df = pd.concat([df[['Date', 'Temperature', 'Predicted_Temp']], future_df[['Date', 'Predicted_Temp']]], ignore_index=True)

plt.figure(figsize=(10, 5))
sns.lineplot(data=combined_df, x='Date', y='Predicted_Temp', label='Predicted Temperature', color='red')
sns.scatterplot(data=df, x='Date', y='Temperature', label='Actual Temperature', color='blue')
plt.title("Historical & Future Temperature Prediction")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("Mean Squared Error:", mean_squared_error(y_test, model.predict(X_test)))
print("R² Score:", r2_score(y_test, model.predict(X_test)))
