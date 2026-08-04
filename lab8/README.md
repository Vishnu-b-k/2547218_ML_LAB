# Lab 8: Implementation and Performance Evaluation of Categorical Naive Bayes Classifier

This directory contains the complete implementation and evaluation of the **Categorical Naive Bayes Classifier** on the Play Tennis dataset (`Lab 8 - Sheet1.csv`), along with comparative benchmarks against Decision Trees, Logistic Regression, and Support Vector Machines.

---

## Directory Structure
- `2547218_lab8.ipynb`: Interactive Jupyter Notebook containing all executed code cells, plots, metrics, query inference, and markdown explanations.
- `2547218_lab8.html`: Standalone HTML export of the executed notebook for easy web viewing.
- `2547218_lab8.py`: Clean, modular, standalone executable Python script fulfilling all tasks.
- `Lab 8 - Sheet1.csv`: Standard Play Tennis dataset containing weather attributes and target play decision.

---

## Tasks & Workflow

### 1. Data Preprocessing
- **Dataset Loading:** Loaded 50 records of Play Tennis dataset with 4 feature columns (`Outlook`, `Temperature`, `Humidity`, `Wind`) and 1 target column (`Play Tennis`).
- **Encoding:** Applied `OrdinalEncoder` for categorical feature values and `LabelEncoder` for binary target classes (`No` -> `0`, `Yes` -> `1`).

### 2. Dataset Partitioning
- Split dataset using an **80:20 train-test ratio** (`random_state=42`, stratified by target class).
- Training Set: 40 samples (80%).
- Testing Set: 10 samples (20%).

### 3. Categorical Naive Bayes Model Training & Evaluation
- Trained `CategoricalNB()` from `sklearn.naive_bayes`.
- **Overall Model Accuracy:** **90.00%** on the test dataset.
- Evaluated with Confusion Matrix and Classification Report (Precision, Recall, F1-Score).

### 4. Single-Sample Inference
- **Query Condition:** `Outlook: Sunny`, `Temperature: Cool`, `Humidity: High`, `Wind: Strong`.
- **Prediction:** **`No` (Will Not Play Tennis)**
- **Class Probabilities:**
  - $P(\text{Play}=\text{'No'}) = 0.7894$ ($78.94\%$)
  - $P(\text{Play}=\text{'Yes'}) = 0.2106$ ($21.06\%$)

### 5. Model Comparison
Trained and benchmarked four distinct classification paradigms on the identical training set:

| Model Name | Test Accuracy | Query Prediction | $P(\text{No})$ | $P(\text{Yes})$ |
| :--- | :---: | :---: | :---: | :---: |
| **Categorical Naive Bayes** | 90.00% | **No** | 0.7894 (78.9%) | 0.2106 (21.1%) |
| **Decision Tree** | 100.00% | **No** | 1.0000 (100.0%) | 0.0000 (0.0%) |
| **Logistic Regression** | 90.00% | **No** | 0.8613 (86.1%) | 0.1387 (13.9%) |
| **Support Vector Machine (SVM)** | 100.00% | **No** | 0.9569 (95.7%) | 0.0431 (4.3%) |

---

## Analysis & Insights
1. **Unanimous Agreement:** All four classifiers unanimously predicted **'No'** for the custom query instance (*Sunny, Cool, High Humidity, Strong Wind*), confirming that strong winds paired with high humidity strongly correlate with unfavorable tennis conditions.
2. **Probability Estimation Mechanics:**
   - **Decision Trees** produce discrete, rigid probabilities ($1.0$) because the instance falls into a pure leaf node partitioned by dominant split criteria.
   - **Categorical Naive Bayes** assumes feature conditional independence given the class, yielding a smoothed probability score ($78.94\%$) by multiplying prior and likelihood probabilities.
   - **Logistic Regression & SVM** evaluate functional distances from their continuous decision hyperplanes and calibrate probabilities using sigmoid/Platt scaling, producing well-calibrated intermediate confidence estimates ($86.1\%$ and $95.7\%$).
