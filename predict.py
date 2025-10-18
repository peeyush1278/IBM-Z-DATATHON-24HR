import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==============================
# 🔧 LOAD SAVED ARTIFACTS
# ==============================
print("🔄 Loading trained model and encoders...")

try:
    model = joblib.load("best_priority_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    priority_bins = joblib.load("feature_scaler.pkl")
    metadata = joblib.load("model_metadata.pkl")
    print("✅ All artifacts loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please run the training script first to generate model files.")
    exit()

print(f"\n📊 Model Info:")
print(f"   Type: {metadata['model_type']}")
print(f"   Accuracy: {metadata['accuracy']:.4f}")
print(f"   Features: {metadata['n_features']}")

# ==============================
# 🔍 PREPROCESSING FUNCTION
# ==============================
def preprocess_case(case_df):
    """Preprocess a single case or batch of cases"""
    
    df = case_df.copy()
    
    # Drop ID columns if present
    drop_cols = ["case_id", "cnr_number", "fir_number"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # Handle dates
    date_cols = ["filed_date", "last_hearing_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_days'] = (df[col] - pd.Timestamp("2000-01-01")).dt.days
            df = df.drop(columns=[col])
    
    # Create interaction features
    if 'case_age_days' in df.columns and 'adjournments_count' in df.columns:
        df['delay_per_adjournment'] = df['case_age_days'] / (df['adjournments_count'] + 1)
    
    if 'undertrial_duration_months' in df.columns and 'evidence_complexity_score' in df.columns:
        df['complexity_duration_ratio'] = df['evidence_complexity_score'] * df['undertrial_duration_months']
    
    if 'number_of_petitioners' in df.columns and 'number_of_respondents' in df.columns:
        df['total_parties'] = df['number_of_petitioners'] + df['number_of_respondents']
        df['party_ratio'] = df['number_of_petitioners'] / (df['number_of_respondents'] + 1)
    
    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical values
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")
    
    # Encode categorical features
    for col in df.select_dtypes(include=["object"]).columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            df[col] = le.transform(df[col].astype(str))
        else:
            # If encoder doesn't exist, just convert to numeric
            df[col] = pd.factorize(df[col])[0]
    
    # Ensure all required features are present
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0  # Add missing features with default value
    
    # Select only the features used in training
    df = df[feature_names]
    
    return df

# ==============================
# 🎯 PREDICTION FUNCTION
# ==============================
def predict_priority(case_data):
    """
    Predict priority for one or more cases
    
    Parameters:
    -----------
    case_data : pd.DataFrame or str
        Either a DataFrame with case data or path to CSV file
    
    Returns:
    --------
    pd.DataFrame with predictions
    """
    
    # Load data if path provided
    if isinstance(case_data, str):
        df = pd.read_csv(case_data)
    else:
        df = case_data.copy()
    
    # Store case IDs if present
    case_ids = df['case_id'].values if 'case_id' in df.columns else range(len(df))
    
    # Preprocess
    X = preprocess_case(df)
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Map predictions to labels
    priority_map = {
    0: 'Very Low Priority',
    1: 'Low Priority',
    2: 'Medium Priority',
    3: 'High Priority',
    4: 'Critical Priority'  # ← add this if model has 4 classes
    }
   
    # Create results DataFrame
    results = pd.DataFrame({
        'case_id': case_ids,
        'predicted_priority': [priority_map[p] for p in predictions],
        'priority_class': predictions,
        'low_probability': probabilities[:, 0],
        'medium_probability': probabilities[:, 1],
        'high_probability': probabilities[:, 2],
        'confidence': probabilities.max(axis=1)
    })
    
    return results

# ==============================
# 📝 EXAMPLE USAGE
# ==============================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔮 COURT CASE PRIORITY PREDICTOR")
    print("="*70)
    
    # Example 1: Predict from CSV file
    print("\n1️⃣ Predicting from CSV file...")
    try:
        predictions = predict_priority("court_cases.csv")
        print(f"✅ Predicted priorities for {len(predictions)} cases")
        print("\n📊 Sample predictions:")
        print(predictions.head(10).to_string(index=False))
        
        # Save predictions
        predictions.to_csv("case_predictions.csv", index=False)
        print("\n💾 Predictions saved to 'case_predictions.csv'")
        
        # Show distribution
        print("\n📈 Priority Distribution:")
        dist = predictions['predicted_priority'].value_counts()
        for priority, count in dist.items():
            print(f"   {priority}: {count:,} cases ({count/len(predictions)*100:.1f}%)")
        
        # Show high confidence cases
        high_conf = predictions[predictions['confidence'] > 0.9]
        print(f"\n🎯 High confidence predictions (>90%): {len(high_conf)} cases")
        
    except FileNotFoundError:
        print("⚠️ court_cases.csv not found. Using example case...")
        
        # Example 2: Predict single case
        print("\n2️⃣ Predicting single case...")
        example_case = pd.DataFrame([{
            'case_id': 'CASE20250001',
            'cnr_number': 'KA4302-946594-2024',
            'district': 'Madhyamgram',
            'court_type': 'District Court',
            'case_type': 'Family',
            'case_subtype': 'Divorce',
            'acts_and_sections': 'Civil Procedure Code',
            'case_status': 'Summons',
            'filed_date': '2011-08-29',
            'last_hearing_date': '2013-07-20',
            'petitioner_status': 'Normal',
            'number_of_petitioners': 5,
            'number_of_respondents': 1,
            'petitioner_has_legal_aid': False,
            'is_government_litigant': False,
            'judge_id': 'JUDGE-13',
            'judge_efficiency_score': 1.15,
            'evidence_complexity_score': 4,
            'is_bailable': None,
            'fir_number': None,
            'undertrial_duration_months': 0,
            'dispute_value_inr': 0,
            'is_commercial_dispute': False,
            'case_age_days': 5157,
            'adjournments_count': 14
        }])
        
        result = predict_priority(example_case)
        print("\n📋 Prediction Result:")
        print(f"   Case ID: {result['case_id'].values[0]}")
        print(f"   Priority: {result['predicted_priority'].values[0]}")
        print(f"   Confidence: {result['confidence'].values[0]:.2%}")
        print(f"\n   Probabilities:")
        print(f"      Low:    {result['low_probability'].values[0]:.2%}")
        print(f"      Medium: {result['medium_probability'].values[0]:.2%}")
        print(f"      High:   {result['high_probability'].values[0]:.2%}")
    
    print("\n" + "="*70)
    print("✅ Prediction complete!")
    print("="*70)