import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



df=pd.read_csv("transactions.csv")

print(df.shape)
print(df.info())
print(df.describe())

print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

print(df.groupby("label")["amount"].mean())

print(df.groupby("label")["transactions_last_24h"].mean())
print(df.groupby("label")["distance_from_usual_location_km"].mean())

print(df.groupby("label")[["amount","transactions_last_24h","distance_from_usual_location_km"]].median())

df.boxplot(column="amount", by="label")
plt.title("Transaction Amount by Risk Label")
plt.xlabel("Label (0 = Legitimate, 1 = Suspicious)")
plt.ylabel("Amount (₹)")
plt.show()

df.groupby("label")[["is_international", "is_new_device", "failed_transactions_last_24h"]].mean()

print(df.columns)
print(df.groupby("label")[['is_international', 'is_new_device', 'transactions_last_24h']].mean())

print(df.groupby("label")[["amount","avg_amount_last_30d"]].mean())

df["amount_ratio"]=(df["amount"]/df["avg_amount_last_30d"])
print(df.groupby("label")[["amount_ratio"]].mean())

print(df.groupby("label").mean(numeric_only=True).T)

df.boxplot(column="distance_from_usual_location_km", by="label")


plt.title("Distance From Usual Location by Risk Label")
plt.suptitle("")
plt.xlabel("Label (0 = Legitimate, 1 = Suspicious)")
plt.ylabel("Distance (km)")
plt.show()


df.boxplot(column="failed_transactions_last_24h", by="label")

import matplotlib.pyplot as plt

plt.title("Failed Transactions in Last 24 Hours by Risk Label")
plt.suptitle("")
plt.xlabel("Label (0 = Legitimate, 1 = Suspicious)")
plt.ylabel("Failed Transactions")
plt.show()



corr = df.select_dtypes(include="number").corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


print(pd.crosstab(
    df["transaction_type"],
    df["label"],
    normalize="index"
))

print(pd.crosstab(
    df["merchant_category"],
    df["label"],
    normalize="index"
).sort_values(1, ascending=False))

X=df.drop("label",axis=1)
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print(X_train.shape)
print(X_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())

categorical_cols = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_type",
    "payment_method"
]

numeric_cols = [
    "amount",
    "hour",
    "is_international",
    "is_new_device",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "account_age_days",
    "distance_from_usual_location_km",
    "failed_transactions_last_24h",
    "amount_ratio"
]

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:, 1]

print(y_prob[:10])

threshold = 0.20

y_pred_20 = (y_prob >= threshold).astype(int)

print(classification_report(y_test, y_pred_20))

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(classification_report(y_test, rf_pred))

rf_prob = rf_model.predict_proba(X_test)[:, 1]

print(rf_prob[:20])
print(rf_prob.max())

model.named_steps["classifier"].coef_

coefficients = model.named_steps["classifier"].coef_[0]

print(coefficients)

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

print(len(feature_names))
print(len(coefficients))
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

coef_df.sort_values(
    "abs_coefficient",
    ascending=False
).head(15)

print("Top features pushing toward SUSPICIOUS:")
print(
    coef_df[coef_df["coefficient"] > 0]
    .sort_values("coefficient", ascending=False)
    .head(10)
)

print("\nTop features pushing toward LEGITIMATE:")
print(
    coef_df[coef_df["coefficient"] < 0]
    .sort_values("coefficient")
    .head(10)
)
print(xgb.__version__)
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ))
])
xgb_model.fit(X_train, y_train)
# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))
# Logistic Regression metrics
lr_accuracy = accuracy_score(y_test, y_pred)
lr_precision = precision_score(y_test, y_pred)
lr_recall = recall_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)

# XGBoost metrics
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "Accuracy": [lr_accuracy, xgb_accuracy],
    "Precision": [lr_precision, xgb_precision],
    "Recall": [lr_recall, xgb_recall],
    "F1 Score": [lr_f1, xgb_f1]
})

print(comparison)
# Get the trained XGBoost classifier
xgb_classifier = xgb_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = xgb_model.named_steps["preprocessor"].get_feature_names_out()

# Get feature importance
importances = xgb_classifier.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Sort from most important to least important
importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(importance_df.head(15))

from sklearn.metrics import classification_report

# Get probability of class 1 (suspicious)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# Try different thresholds
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:

    y_pred_threshold = (xgb_prob >= threshold).astype(int)

    print(f"\n========== Threshold: {threshold} ==========")
    print(classification_report(y_test, y_pred_threshold))

threshold_results = []

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    y_pred_t = (xgb_prob >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred_t, zero_division=0),
        "recall": recall_score(y_test, y_pred_t, zero_division=0),
        "f1": f1_score(y_test, y_pred_t, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df)

best_threshold = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print("\nBest threshold based on F1:")
print(best_threshold)

final_threshold = 0.10

y_pred_final = (xgb_prob >= final_threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

cm = confusion_matrix(y_test, y_pred_final)

print("True Negatives :", cm[0, 0])
print("False Positives:", cm[0, 1])
print("False Negatives:", cm[1, 0])
print("True Positives  :", cm[1, 1])
# Take one transaction from the test set
sample_transaction = X_test.iloc[[0]]

# Get probability of being suspicious
risk_probability = xgb_model.predict_proba(sample_transaction)[0][1]



test_transaction = X_test.iloc[0].to_dict()





negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("Negative samples:", negative)
print("Positive samples:", positive)
print("scale_pos_weight:", scale_pos_weight)

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

xgb_balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])

# Train
xgb_balanced.fit(X_train, y_train)

# Predict
y_pred_balanced = xgb_balanced.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred_balanced))
# Confusion matrix
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print("Confusion Matrix:")
print(cm_balanced)

# Extract values
tn, fp, fn, tp = cm_balanced.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

# Main metrics
print("\nPrecision:", precision_score(y_test, y_pred_balanced))
print("Recall   :", recall_score(y_test, y_pred_balanced))
print("F1 Score :", f1_score(y_test, y_pred_balanced))

# Get suspicious probability from balanced XGBoost
xgb_balanced_prob = xgb_balanced.predict_proba(X_test)[:, 1]

# Test different thresholds
balanced_results = []

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:

    y_pred_t = (xgb_balanced_prob >= threshold).astype(int)

    balanced_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_test, y_pred_t, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred_t, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred_t, zero_division=0
        )
    })

balanced_threshold_df = pd.DataFrame(balanced_results)

print(balanced_threshold_df)

best_balanced = balanced_threshold_df.loc[
    balanced_threshold_df["f1"].idxmax()
]

print("\nBest threshold:")
print(best_balanced)

final_threshold = best_balanced["threshold"]

print("Final threshold:", final_threshold)
print("Precision:", best_balanced["precision"])
print("Recall:", best_balanced["recall"])
print("F1 Score:", best_balanced["f1"])

y_pred_final = (
    xgb_balanced_prob >= final_threshold
).astype(int)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final))

print("\nFinal Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))

final_model = xgb_model
final_threshold = 0.10

def predict_transaction(transaction):
    transaction_df = pd.DataFrame([transaction])

    probability = final_model.predict_proba(transaction_df)[0][1]

    prediction = int(probability >= final_threshold)

    if probability >= 0.30:
        risk_level = "HIGH"
    elif probability >= final_threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    decision = "SUSPICIOUS" if prediction == 1 else "LEGITIMATE"

    return {
        "risk_score": round(float(probability * 100), 2),
        "risk_level": risk_level,
        "decision": decision
    }