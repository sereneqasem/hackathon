import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
df = pd.read_csv("/Users/chloegray/Documents/GitHub/hackathon/models/aggregated_invoice_data.csv")
print("data loaded:")
print(df.head())

#define features 
features = [
    "invoice_count",
    "avg_quantity",
    "avg_unit_price",
    "avg_invoice_amount",
    "max_invoice_amount"
]

X = df[features]
y = df["risk_flag"]

print("feature matrix shape:", X.shape)
print("target shape:", y.shape)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model trained. Accuracy on test set: {accuracy:.2f}")
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#predict risk for all customers and filter out risky ones
df["predicted_risk"] = model.predict(X)
risky_customers = df[df["predicted_risk"] == 1][
    ["customer_id", "avg_invoice_amount", "invoice_count", "max_invoice_amount"]
]
risky_json = risky_customers.to_dict(orient="records")
output_path = "/Users/chloegray/Documents/GitHub/hackathon/models/risky_customers.json"
with open(output_path, "w") as f:
    json.dump(risky_json, f, indent=2)

print(f"\n JSON export complete. {len(risky_json)} risky customers written to:\n{output_path}")

# pie chart comparing risky vs non risky 
risk_counts = df["predicted_risk"].value_counts()
labels = ["Not Risky", "Risky"]
colors = ["#66b3ff", "#ff6666"]

plt.figure(figsize=(6, 6))
plt.pie(risk_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Customer Risk Distribution")
plt.tight_layout()
plt.savefig("/Users/chloegray/Documents/GitHub/hackathon/models/risk_pie_chart.png")
plt.close()

# bar chart of top 10 risky customers based on invoice amount 
top_risky = df[df["predicted_risk"] == 1].nlargest(10, "avg_invoice_amount")

plt.figure(figsize=(10, 6))
sns.barplot(data=top_risky, x="avg_invoice_amount", y="customer_id", palette="rocket")
plt.title("Top 10 Risky Customers by Avg Invoice")
plt.xlabel("Avg Invoice Amount")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig("/Users/chloegray/Documents/GitHub/hackathon/models/top_risky_customers.png")
plt.close()