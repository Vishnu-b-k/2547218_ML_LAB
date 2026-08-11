"""
Lab 9: Support Vector Machine (SVM) and Principal Component Analysis (PCA)
Student Register No: 2547218
Datasets: Breast Cancer Wisconsin (Diagnostic) & UCI Wine Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def part_a_svm():
    print("=" * 60)
    print("PART A: Support Vector Machine (SVM)")
    print("=" * 60)

    # 1. Load Dataset
    cancer_data = load_breast_cancer()
    X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y_cancer = cancer_data.target

    print(f"Breast Cancer Dataset Shape: {X_cancer.shape}")

    # 2. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X_cancer, y_cancer, test_size=0.20, random_state=42, stratify=y_cancer
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train SVM and Hyperparameter Tuning
    print("\nTuning SVM (Linear Kernel) with GridSearchCV...")
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(SVC(kernel='linear', random_state=42), param_grid, cv=5)
    grid.fit(X_train_scaled, y_train)

    print(f"Best Parameters: {grid.best_params_}")
    best_svm = grid.best_estimator_

    # 4. Evaluation
    y_pred = best_svm.predict(X_test_scaled)
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def part_b_pca():
    print("\n" + "=" * 60)
    print("PART B: Principal Component Analysis (PCA)")
    print("=" * 60)

    # 1. Load Dataset
    wine_data = load_wine()
    X_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    y_wine = wine_data.target

    print(f"Wine Dataset Shape: {X_wine.shape}")

    # 2. Scale
    scaler = StandardScaler()
    X_wine_scaled = scaler.fit_transform(X_wine)

    # 3. PCA (2 Components)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_wine_scaled)
    
    print(f"Explained Variance Ratio (2 PCs): {pca.explained_variance_ratio_}")
    total_var = np.sum(pca.explained_variance_ratio_)
    print(f"Total Variance Retained by 2 PCs: {total_var * 100:.2f}%")

    # 4. Cumulative Variance (all components)
    pca_full = PCA()
    pca_full.fit(X_wine_scaled)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    min_comp_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"Minimum PCs required to retain >= 95% variance: {min_comp_95}")


def extra_credit_lda():
    print("\n" + "=" * 60)
    print("EXTRA CREDIT: Linear Discriminant Analysis (LDA)")
    print("=" * 60)

    wine_data = load_wine()
    X_wine = wine_data.data
    y_wine = wine_data.target
    
    scaler = StandardScaler()
    X_wine_scaled = scaler.fit_transform(X_wine)

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_wine_scaled, y_wine)

    print(f"Explained Variance Ratio (2 LDs): {lda.explained_variance_ratio_}")
    print(f"Total Variance Retained by 2 LDs: {np.sum(lda.explained_variance_ratio_) * 100:.2f}%")
    print("LDA provides supervised dimensionality reduction and clearly separates the classes better than PCA.")


if __name__ == '__main__':
    part_a_svm()
    part_b_pca()
    extra_credit_lda()
