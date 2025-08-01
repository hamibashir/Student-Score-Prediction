import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Veclar\Desktop\Internship\Elevvo\student-score-prediction-main\student-score-prediction-main\StudentPerformance.csv")

# Step 2: Explore the dataset
print("🔍 First 5 rows of the dataset:")
print(df.head())

print("\n📊 Summary statistics:")
print(df.describe())

# Step 3: Data Cleaning (check for nulls)
print("\n❓ Are there any missing values?")
print(df.isnull().sum())

# Step 4: Data Visualization
plt.figure(figsize=(8,5))
plt.scatter(df['Hours_Studied'], df['Exam_Score'], color='blue')
plt.title("Study Hours vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Step 5: Prepare data for training
X = df[['Hours_Studied']]
y = df['Exam_Score']

# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
from sklearn.preprocessing import PolynomialFeatures

# Step: Polynomial Regression (Bonus)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Make predictions
y_poly_pred = poly_model.predict(X_poly_test)

# Evaluate polynomial model
print("\n🌟 Polynomial Regression Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_poly_pred))
print("R² Score:", r2_score(y_test, y_poly_pred))

# Plotting the curve
import numpy as np
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_poly = poly_model.predict(X_range_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_range, y_range_poly, color='orange', label='Polynomial Regression')
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()

y_pred_lr = lr_model.predict(X_test)

# Step 7: Evaluate Linear Regression
print("\n📈 Linear Regression Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Plot predictions
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='green', label='Actual')
plt.plot(X_test, y_pred_lr, color='red', label='Linear Prediction')
plt.title("Linear Regression Prediction")
plt.xlabel("Hours Studied")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.show()

# Step 8: BONUS - Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate Polynomial Regression
print("\n🌟 Polynomial Regression (Degree=2) Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_poly))
print("R2 Score:", r2_score(y_test, y_pred_poly))

# Plot Polynomial Prediction (optional smooth curve)
X_plot = np.linspace(min(X['Hours_Studied']), max(X['Hours_Studied']), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot_poly = poly_model.predict(X_plot_poly)

plt.figure(figsize=(8,5))
plt.scatter(df['Hours_Studied'], df['Exam_Score'], color='blue', label='Actual Data')
plt.plot(X_plot, y_plot_poly, color='orange', label='Polynomial Regression')
plt.title("Polynomial Regression (Degree=2)")
plt.xlabel("Hours Studied")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.show()