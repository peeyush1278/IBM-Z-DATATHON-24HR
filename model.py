import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

# ==============================
# 🔍 STEP 1: LOAD & DIAGNOSE
# ==============================
df = pd.read_csv("court_cases.csv")
print("="*70)
print("📊 DATASET DIAGNOSTIC REPORT")
print("="*70)
print(f"\n1️⃣ Dataset Shape: {df.shape}")
print(f"   Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# ==============================
# 🎯 STEP 2: ANALYZE TARGET VARIABLE
# ==============================
print(f"\n2️⃣ TARGET VARIABLE ANALYSIS (priority_score)")
print("-"*70)
print(f"Unique values: {df['priority_score'].nunique()}")
print(f"Value range: {df['priority_score'].min()} to {df['priority_score'].max()}")
print(f"Data type: {df['priority_score'].dtype}")
print(f"Missing values: {df['priority_score'].isnull().sum()}")

print(f"\n📊 Class Distribution:")
class_dist = df['priority_score'].value_counts().sort_index()
print(class_dist)
print(f"\n📈 Class Percentages:")
print((class_dist / len(df) * 100).round(2))

# Check for class imbalance
max_class_pct = (class_dist.max() / len(df)) * 100
min_class_pct = (class_dist.min() / len(df)) * 100
print(f"\n⚠️ Imbalance Ratio: {max_class_pct:.2f}% (max) vs {min_class_pct:.2f}% (min)")

if df['priority_score'].nunique() > 10:
    print("\n🚨 WARNING: You have more than 10 classes! This is very difficult to predict.")
    print("   Consider grouping classes into broader categories (e.g., Low/Medium/High)")

# ==============================
# 🔬 STEP 3: FEATURE ANALYSIS
# ==============================
print(f"\n3️⃣ FEATURE ANALYSIS")
print("-"*70)

# Drop ID columns
drop_cols = ["case_id", "cnr_number", "fir_number"]
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Separate features by type
numeric_features = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'priority_score' in numeric_features:
    numeric_features.remove('priority_score')

categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
date_features = [col for col in df_clean.columns if 'date' in col.lower()]

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Date features: {len(date_features)}")

# Check for features with zero variance
print(f"\n🔍 Checking for low-variance features...")
for col in numeric_features:
    unique_ratio = df_clean[col].nunique() / len(df_clean)
    if unique_ratio < 0.01:
        print(f"   ⚠️ {col}: only {df_clean[col].nunique()} unique values ({unique_ratio:.2%})")

# ==============================
# 🛠️ STEP 4: INTELLIGENT PREPROCESSING
# ==============================
print(f"\n4️⃣ PREPROCESSING")
print("-"*70)

# Handle dates intelligently
for col in date_features:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        df_clean[f'{col}_year'] = df_clean[col].dt.year
        df_clean[f'{col}_month'] = df_clean[col].dt.month
        df_clean[f'{col}_dayofweek'] = df_clean[col].dt.dayofweek
        df_clean[f'{col}_quarter'] = df_clean[col].dt.quarter
        df_clean = df_clean.drop(columns=[col])
        print(f"✅ Converted {col} to temporal features")

# Fill missing values
numeric_cols = df_clean.select_dtypes(include=["float", "int"]).columns.tolist()
if 'priority_score' in numeric_cols:
    numeric_cols.remove('priority_score')

for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna("Unknown")

# Encode categorical features
label_encoders = {}
for col in df_clean.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
# ==============================
# 🎯 STEP 5: SIMPLIFY TARGET (IF NEEDED)
# ==============================
X = df_clean.drop(columns=["priority_score"])
y_original = df_clean["priority_score"].fillna(df_clean["priority_score"].median()).astype(int)

# Create simplified target if too many classes
n_classes = y_original.nunique()
if n_classes > 5:
    print(f"\n🔄 Simplifying target from {n_classes} classes to 5 classes...")
    # Convert to 5 classes: Very Low / Low / Medium / High / Very High
    y_simplified = pd.cut(y_original, bins=5, labels=[0, 1, 2, 3, 4])
    y_simplified = y_simplified.astype(int)
    print(f"   New class distribution:")
    print(pd.Series(y_simplified).value_counts().sort_index())
    use_simplified = True
else:
    y_simplified = y_original.copy()
    use_simplified = False

# ==============================
# 📊 STEP 6: TRAIN MULTIPLE MODELS
# ==============================
print(f"\n5️⃣ MODEL COMPARISON")
print("-"*70)
print(len(X.columns.to_list()))
print(X.columns.to_list())
print(X.head())
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_simplified, test_size=0.2, random_state=42, stratify=y_simplified
)

# Standardize features for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        class_weight="balanced",
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Use scaled data for Logistic Regression
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"✅ {name} Accuracy: {acc:.4f}")
    
    if acc > best_score:
        best_score = acc
        best_model = (name, model)

# ==============================
# 🏆 STEP 7: BEST MODEL DETAILS
# ==============================
print(f"\n6️⃣ BEST MODEL RESULTS")
print("="*70)
print(f"🏆 Winner: {best_model[0]} with {best_score:.4f} accuracy")

# Detailed results for best model
if best_model[0] == "Logistic Regression":
    y_pred_best = best_model[1].predict(X_test_scaled)
else:
    y_pred_best = best_model[1].predict(X_test)

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_best))

print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Feature importance (if available)
if hasattr(best_model[1], 'feature_importances_'):
    print(f"\n🔝 Top 10 Important Features:")
    fi = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model[1].feature_importances_
    }).sort_values('importance', ascending=False)
    print(fi.head(10).to_string(index=False))

# ==============================
# 💾 STEP 8: SAVE BEST MODEL
# ==============================
print(f"\n7️⃣ SAVING MODEL")
print("-"*70)

joblib.dump(best_model[1], "best_priority_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

metadata = {
    'model_type': best_model[0],
    'accuracy': best_score,
    'n_features': X.shape[1],
    'n_classes': y_simplified.nunique(),
    'simplified_target': use_simplified,
    'feature_names': X.columns.tolist()
}
joblib.dump(metadata, "model_metadata.pkl")

print("✅ Best model saved!")
print(f"\n📋 Summary:")
print(f"   • Model: {best_model[0]}")
print(f"   • Accuracy: {best_score:.4f}")
print(f"   • Features: {X.shape[1]}")
print(f"   • Classes: {y_simplified.nunique()}")
print(f"   • Target simplified: {use_simplified}")

# ==============================
# 💡 RECOMMENDATIONS
# ==============================
print(f"\n8️⃣ RECOMMENDATIONS")
print("="*70)

if best_score < 0.5:
    print("🚨 CRITICAL: Accuracy is still very low!")
    print("\n   Possible issues:")
    print("   1. Target variable (priority_score) may not be predictable from features")
    print("   2. Features may lack signal - consider feature engineering")
    print("   3. Data quality issues - check for noise or errors")
    print("   4. Target may be randomly assigned or based on external factors")
    print("\n   Next steps:")
    print("   • Examine how priority_score is actually calculated")
    print("   • Check correlation between features and target")
    print("   • Consider if this is truly a classification problem")
elif best_score < 0.7:
    print("⚠️ Accuracy is moderate. Room for improvement:")
    print("   • Try feature engineering (interactions, ratios)")
    print("   • Collect more relevant features")
    print("   • Try XGBoost or LightGBM")
else:
    print("✅ Good accuracy! Model is ready for deployment.")