import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# ✅ Load the dataset
data_path = "data/jobs.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Dataset not found at 'data/jobs.csv'. Please add the file.")

df = pd.read_csv(data_path)

# ✅ Check required columns
required_columns = ['experience', 'python', 'excel', 'sql', 'salary']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    raise ValueError(f"❌ Missing columns in CSV: {missing_cols}")

# ✅ Define features (X) and target (y)
X = df[['experience', 'python', 'excel', 'sql']]
y = df['salary']

# ✅ Create and train model
model = LinearRegression()
model.fit(X, y)

# ✅ Save the model
output_model_path = "model/salary_model.pkl"
joblib.dump(model, output_model_path)

print("✅ Model trained successfully!")
print(f"💾 Model saved to: {output_model_path}")
