# 🛡️ Transaction Risk Detector

An ML-powered transaction risk detection system that analyzes transaction behavior and estimates the probability of a transaction being suspicious.

The project combines **machine learning, feature engineering, model evaluation, threshold tuning, and an interactive risk dashboard**.

---

## 🚀 Project Overview

Financial transaction datasets are highly imbalanced: legitimate transactions greatly outnumber suspicious ones.

Instead of relying only on accuracy, this project focuses on identifying suspicious transactions using:

- Transaction behavior
- Amount patterns
- Device and payment information
- Geographic distance
- Failed transaction activity
- International transactions
- New devices
- Recent transaction frequency

The trained model produces a **risk score**, **risk level**, and **decision** for each transaction.

---

## 🎯 Objective

Build a machine learning system that can:

1. Analyze transaction behavior
2. Identify suspicious patterns
3. Handle categorical and numerical features
4. Compare multiple ML models
5. Tune the classification threshold
6. Generate a transaction risk score
7. Provide an interactive interface for analysis

---

## 📊 Dataset

The dataset contains **20,000 transactions** with 17 original features.

### Target Distribution

| Class | Meaning | Count |
|------|---------|------:|
| `0` | Legitimate | 19,385 |
| `1` | Suspicious | 615 |

Suspicious transactions represent approximately **3.1%** of the dataset.

This class imbalance makes metrics such as **precision, recall and F1-score** important in addition to accuracy.

---

## 🧩 Features

The dataset contains information such as:

- Transaction amount
- Transaction type
- Merchant category
- Location
- Device type
- Payment method
- Transaction hour
- International transaction indicator
- New device indicator
- Transactions in the last 24 hours
- Average transaction amount over the last 30 days
- Account age
- Distance from usual location
- Failed transactions in the last 24 hours

### Feature Engineering

An additional feature was created:

```text
amount_ratio = amount / avg_amount_last_30d
