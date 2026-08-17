 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load data
data = pd.read_csv("creditcard.csv")   # Columns: Time, V1-V28, Amount, Class

# 2. Split features (X) and target (y)
X = data.drop("Class", axis=1)
y = data["Class"]   # 0 = normal, 1 = fraud

print("Fraud cases:", y.sum(), "out of", len(y))

# 3. Train/test split (stratify keeps the same fraud ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Fix imbalance: create synthetic fraud samples using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE: ", y_train_balanced.value_counts().to_dict())

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# 6. Predict on test data
y_pred = model.predict(X_test)

# 7. Evaluate (accuracy is misleading here - use precision/recall/f1 instead)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
