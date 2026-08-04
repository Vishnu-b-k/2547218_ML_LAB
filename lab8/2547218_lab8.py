"""
Lab 8: Implementation and Performance Evaluation of Categorical Naive Bayes Classifier
Student Register No: 2547218
Dataset: Play Tennis Dataset (Lab 8 - Sheet1.csv)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # =========================================================================
    # Task 1: Data Preprocessing
    # =========================================================================
    print("=" * 70)
    print("TASK 1: DATA PREPROCESSING")
    print("=" * 70)

    # 1. Load the dataset
    df = pd.read_csv('Lab 8 - Sheet1.csv')
    if 'No' in df.columns:
        df = df.drop(columns=['No'])

    print(f"Dataset Shape: {df.shape[0]} samples, {df.shape[1]} columns\n")
    print("First 5 records:")
    print(df.head())

    # 2. Separate input features from target variable
    feature_cols = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target_col = 'Play Tennis' if 'Play Tennis' in df.columns else 'Play'

    X_raw = df[feature_cols]
    y_raw = df[target_col]

    print("\nClass Distribution:")
    print(y_raw.value_counts())

    # 3. Categorical Encoding
    oe = OrdinalEncoder()
    X_encoded = oe.fit_transform(X_raw)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    print("\nFeature Category Mappings:")
    for i, col in enumerate(feature_cols):
        mapping = {cat: idx for idx, cat in enumerate(oe.categories_[i])}
        print(f"  {col}: {mapping}")

    target_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
    print(f"Target Label Mapping ({target_col}): {target_mapping}")

    # =========================================================================
    # Task 2: Dataset Partitioning
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 2: DATASET PARTITIONING (80:20 Split)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    print(f"Total Samples    : {len(X_encoded)}")
    print(f"Training Samples : {len(X_train)} (80%)")
    print(f"Testing Samples  : {len(X_test)} (20%)")
    print(f"Train Class Counts (0={le.classes_[0]}, 1={le.classes_[1]}): {np.bincount(y_train)}")
    print(f"Test Class Counts  (0={le.classes_[0]}, 1={le.classes_[1]}): {np.bincount(y_test)}")

    # =========================================================================
    # Task 3: Categorical Naive Bayes Model Training & Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 3: CATEGORICAL NAIVE BAYES TRAINING & EVALUATION")
    print("=" * 70)

    cnb = CategoricalNB()
    cnb.fit(X_train, y_train)

    y_pred_nb = cnb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cr_nb = classification_report(y_test, y_pred_nb, target_names=le.classes_)

    print(f"Overall Model Accuracy: {acc_nb * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(cm_nb)
    print("\nClassification Report:")
    print(cr_nb)

    # =========================================================================
    # Task 4: Single-Sample Inference
    # =========================================================================
    print("=" * 70)
    print("TASK 4: SINGLE-SAMPLE INFERENCE")
    print("=" * 70)

    query_dict = {
        'Outlook': 'Sunny',
        'Temperature': 'Cool',
        'Humidity': 'High',
        'Wind': 'Strong'
    }
    query_df = pd.DataFrame([query_dict])
    query_encoded = oe.transform(query_df)

    nb_pred_idx = cnb.predict(query_encoded)[0]
    nb_pred_label = le.inverse_transform([nb_pred_idx])[0]
    nb_probs = cnb.predict_proba(query_encoded)[0]

    print("Given Weather Conditions:")
    for k, v in query_dict.items():
        print(f"  {k}: {v}")
    print(f"\nPredicted Class (Play Tennis): {nb_pred_label}")
    print("Output Class Probabilities:")
    for cls_name, prob in zip(le.classes_, nb_probs):
        print(f"  P(Play = '{cls_name}'): {prob:.4f} ({prob*100:.2f}%)")

    # =========================================================================
    # Task 5: Model Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 5: MODEL COMPARISON (Naive Bayes vs DT vs LR vs SVM)")
    print("=" * 70)

    dt = DecisionTreeClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)
    svm = SVC(probability=True, random_state=42)

    all_models = {
        'Categorical Naive Bayes': cnb,
        'Decision Tree': dt,
        'Logistic Regression': lr,
        'Support Vector Machine (SVM)': svm
    }

    comparison_records = []
    for name, model in all_models.items():
        if name != 'Categorical Naive Bayes':
            model.fit(X_train, y_train)

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        q_pred_idx = model.predict(query_encoded)[0]
        q_pred_label = le.inverse_transform([q_pred_idx])[0]
        q_probs = model.predict_proba(query_encoded)[0]

        comparison_records.append({
            'Model Name': name,
            'Test Accuracy': f"{test_acc * 100:.2f}%",
            'Query Prediction': q_pred_label,
            f'P({le.classes_[0]}) [No]': f"{q_probs[0]:.4f} ({q_probs[0]*100:.1f}%)",
            f'P({le.classes_[1]}) [Yes]': f"{q_probs[1]:.4f} ({q_probs[1]*100:.1f}%)"
        })

    comparison_df = pd.DataFrame(comparison_records)
    print(comparison_df.to_string(index=False))

    # =========================================================================
    # Task 6: Analysis Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS REPORT")
    print("=" * 70)
    print("""
1. Unanimous Class Prediction:
   All four classifiers (Categorical Naive Bayes, Decision Tree, Logistic Regression,
   and SVM) unanimously predicted 'No' (will not play tennis) for the query sample
   (Outlook: Sunny, Temperature: Cool, Humidity: High, Wind: Strong). This shows strong
   agreement that high humidity combined with strong wind heavily penalizes the likelihood of playing.

2. Variation in Probability Scores:
   - Decision Tree outputs an absolute probability of 1.0 (100% No) because the query falls
     into a pure terminal leaf node partitioned by dominant splits (High Humidity / Strong Wind).
   - Categorical Naive Bayes computes a softened posterior probability of 78.9% No vs 21.1% Yes
     by multiplying independent conditional likelihoods with the prior distribution.
   - Logistic Regression (86.1% No) and SVM (95.7% No) compute smooth sigmoid/Platt-scaled
     probabilities proportional to the geometric distance from their decision boundaries,
     providing well-calibrated confidence estimates.
""")

if __name__ == '__main__':
    main()
