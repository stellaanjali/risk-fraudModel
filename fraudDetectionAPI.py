import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("C://Users//ANJALI//FraudDetectionGeneration//fraud_dataset.csv")

# Convert timestamp to datetime (useful for future enhancements)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Handle missing or invalid data
df.dropna(inplace=True)  # Remove rows with missing values
df = df[df["amount"] > 0]  # Ensure transactions have valid amounts

# Feature Engineering: Create additional features for better prediction
df["transaction_ratio"] = df["amount"] / (df["debitor_avg_txn"] + 1)  # Avoid division by zero
df["log_debitor_balance"] = np.log1p(df["debitor_balance"])  # Log transformation to normalize large values
df["log_creditor_balance"] = np.log1p(df["creditor_balance"])
df["log_amount"] = np.log1p(df["amount"])

# Select Features and Target
features = [
    "debitor_balance", "creditor_balance", "amount", "debitor_txn_history", 
    "creditor_txn_history", "debitor_avg_txn", "transaction_ratio",
    "log_debitor_balance", "log_creditor_balance", "log_amount"
]
X = df[features]
y = df["fraudulent"]

# Standardize numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model with better hyperparameters
model = LogisticRegression(solver="liblinear", C=0.7, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler efficiently
joblib.dump(model, "fraud_model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)

# FastAPI app
app = FastAPI()

# Input schema for API with validation
class TransactionInput(BaseModel):
    debitor_balance: float = Field(..., gt=0, description="Debitor account balance, must be > 0")
    creditor_balance: float = Field(..., gt=0, description="Creditor account balance, must be > 0")
    amount: float = Field(..., gt=0, description="Amount being transferred, must be > 0")
    debitor_txn_history: int = Field(..., ge=0, description="Number of past transactions by debitor")
    creditor_txn_history: int = Field(..., ge=0, description="Number of past transactions by creditor")
    debitor_avg_txn: float = Field(..., ge=0, description="Average transaction amount of debitor")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    try:
        # Load the saved model and scaler
        model = joblib.load("fraud_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Convert input data to DataFrame
        input_data = pd.DataFrame([transaction.dict()])

        # Generate new features
        input_data["transaction_ratio"] = input_data["amount"] / (input_data["debitor_avg_txn"] + 1)
        input_data["log_debitor_balance"] = np.log1p(input_data["debitor_balance"])
        input_data["log_creditor_balance"] = np.log1p(input_data["creditor_balance"])
        input_data["log_amount"] = np.log1p(input_data["amount"])

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Predict fraud
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        return {
            "fraud_prediction": bool(prediction[0]),
            "fraud_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
