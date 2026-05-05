import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

df = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
df.drop(columns=["difficulty"], inplace=True)

df["label"] = df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)

X = df.drop(columns=["label"])
y = df["label"]

le_protocol = LabelEncoder()
le_service = LabelEncoder()
le_flag = LabelEncoder()

X["protocol_type"] = le_protocol.fit_transform(X["protocol_type"])
X["service"] = le_service.fit_transform(X["service"])
X["flag"] = le_flag.fit_transform(X["flag"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(classification_report(y_test, dt_predictions, target_names=["Normal", "Attack"]))

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
print(classification_report(y_test, xgb_predictions, target_names=["Normal", "Attack"]))

better_model = "XGBoost" if xgb_accuracy >= dt_accuracy else "Decision Tree"

print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"Better Model: {better_model}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = ["Decision Tree", "XGBoost"]
accuracies = [dt_accuracy * 100, xgb_accuracy * 100]

axes[0].bar(model_names, accuracies, width=0.4)
axes[0].set_title("Model Accuracy Comparison")
axes[0].set_ylabel("Accuracy (%)")

feature_names = X.columns.tolist()
importances = dt_model.feature_importances_

top_idx = np.argsort(importances)[::-1][:10]
top_features = [feature_names[i] for i in top_idx]
top_importances = importances[top_idx]

axes[1].barh(range(10), top_importances[::-1])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(top_features[::-1])
axes[1].set_title("Top 10 Feature Importances")

plt.tight_layout()
plt.show()

sample = X_test.iloc[[0]]

dt_pred = dt_model.predict(sample)[0]
xgb_pred = xgb_model.predict(sample)[0]

print("Decision Tree:", "Attack" if dt_pred else "Normal")
print("XGBoost:", "Attack" if xgb_pred else "Normal")

import joblib

encoders = {
    1: le_protocol,
    2: le_service,
    3: le_flag
}

joblib.dump((xgb_model, encoders), "model.pkl")

print("model.pkl saved successfully")