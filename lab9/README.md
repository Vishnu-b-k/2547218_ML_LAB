# Lab 9: SVM for Classification and PCA for Dimensionality Reduction

This directory contains the implementation and evaluation of Support Vector Machine (SVM) and Principal Component Analysis (PCA) applied to real-world datasets. It also includes an extra credit implementation of Linear Discriminant Analysis (LDA).

---

## Directory Structure
- `2547218_lab9.ipynb`: Interactive Jupyter Notebook containing all executed code cells, exploratory data analysis, plots, metrics, and markdown explanations.
- `2547218_lab9.html`: Standalone HTML export of the executed notebook for easy web viewing.
- `2547218_lab9.py`: Clean, modular, standalone executable Python script fulfilling all tasks.

---

## Part A: Support Vector Machine (SVM)
**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Tasks Completed:**
- Loaded and preprocessed the dataset using standard scaling (crucial for distance-based SVM).
- Split the dataset into 80% training and 20% testing sets.
- Trained a Linear Kernel SVM.
- Conducted Hyperparameter Tuning on the `C` parameter using `GridSearchCV`.
- Evaluated the best estimator using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

**Observations:**
The linear kernel SVM achieved exceptionally high performance metrics (Accuracy typically >95%) after standardization, indicating that the diagnostic dataset is highly linearly separable. Tuning the `C` parameter demonstrated the trade-off between margin width and misclassification tolerance.

---

## Part B: Principal Component Analysis (PCA)
**Dataset:** UCI Wine Dataset

**Tasks Completed:**
- Standardized the 13-feature wine dataset.
- Applied PCA to project the dataset down to 2 Principal Components.
- Calculated the explained variance ratio for the 2 PCs (retaining ~55.4% of total variance).
- Analyzed the cumulative explained variance, determining that 10 components are required to retain at least 95% of the total variance.
- Visualized the 2D transformed dataset using a scatter plot, showing distinct clustering of the three wine classes.

**Observations:**
PCA effectively reduces dimensionality, aiding computational efficiency and visualization, while removing multicollinearity. However, to preserve 95% of the variance, 10 components were still needed, showing that PCA's unsupervised nature doesn't explicitly prioritize class separability.

---

## Extra Credit: Linear Discriminant Analysis (LDA)
**Tasks Completed:**
- Applied LDA to reduce the Wine dataset to 2 Linear Discriminants.
- Visualized and compared the results against PCA.

**Observations:**
Unlike PCA (unsupervised), LDA is supervised and explicitly maximizes the separation between multiple classes. The LDA scatter plot demonstrated far superior, tightly clustered class separation. Furthermore, the 2 linear discriminants in LDA captured 100% of the discriminative variance between the 3 classes, whereas PCA required 10 components to capture 95% of the dataset's total variance.
