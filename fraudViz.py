import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc

# Load dataset
df = pd.read_csv("C://Users//ANJALI//FraudDetectionGeneration//fraud_dataset.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Handle missing values and create log transformations
df.dropna(inplace=True)
df["transaction_ratio"] = df["amount"] / (df["debitor_avg_txn"] + 1)
df["log_debitor_balance"] = np.log1p(df["debitor_balance"])
df["log_creditor_balance"] = np.log1p(df["creditor_balance"])
df["log_amount"] = np.log1p(df["amount"])

# Features & target variable
features = [
    "debitor_balance", "creditor_balance", "amount", "debitor_txn_history",
    "creditor_txn_history", "debitor_avg_txn", "transaction_ratio",
    "log_debitor_balance", "log_creditor_balance", "log_amount"
]
X = df[features]
y = df["fraudulent"]

# Load trained scaler
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Convert scaled features back to DataFrame for visualization
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled["fraudulent"] = y




plt.figure(figsize=(12, 6))
sns.histplot(df[df["fraudulent"] == 0]["amount"], bins=50, label="Non-Fraud", kde=True, color="blue")
sns.histplot(df[df["fraudulent"] == 1]["amount"], bins=50, label="Fraud", kde=True, color="red")
plt.xscale("log")  # Log scale for better visualization
plt.xlabel("Transaction Amount")
plt.ylabel("Density")
plt.title("Distribution of Transaction Amounts (Fraud vs. Non-Fraud)")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="fraudulent", y="amount", data=df, palette="coolwarm")
plt.yscale("log")  # Log scale for better visualization
plt.xlabel("Fraudulent Transaction")
plt.ylabel("Transaction Amount")
plt.title("Box Plot of Transaction Amounts by Fraud Status")
plt.xticks(ticks=[0, 1], labels=["Non-Fraud", "Fraud"])
plt.show()




# Load trained model
model = joblib.load("fraud_model.pkl")

# Get predicted probabilities
y_pred_prob = model.predict_proba(X_scaled)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Fraud Detection Model")
plt.legend()
plt.show()

# **Plot Fraud vs. Non-Fraud Transactions**
plt.figure(figsize=(12, 6))
sns.countplot(x=y, palette="coolwarm")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraudulent")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Non-Fraud", "Fraud"])
plt.show()
