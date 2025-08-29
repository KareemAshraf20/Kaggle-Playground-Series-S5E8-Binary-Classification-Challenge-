
# 📁 Project 1: Binary Classification with a Bank Marketing Dataset

## 📌 Overview
This project focuses on building a machine learning model to predict whether a client will subscribe to a term deposit (binary classification) based on marketing campaign data from a bank.

## 🎯 Objective
To preprocess the data, train multiple classification models, and select the best-performing one to predict the `y` column (whether the client subscribed).

## 📊 Dataset
- **train.csv**: Contains 750,000 rows and 18 columns (including the target `y`).
- **test.csv**: Contains 250,000 rows and 17 columns (without `y`).

## 🧹 Data Preprocessing
- Handled missing values for numerical and categorical columns.
- Applied **Label Encoding** to convert categorical columns into numerical values.
- Used **StandardScaler** to normalize numerical features.

## 🤖 Models Trained
We experimented with multiple models:
- `RandomForestClassifier`
- `LogisticRegression`
- `BaggingClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`
- `DecisionTreeClassifier` ← **Selected for submission** (achieved ~90.5% accuracy)

## 📈 Results
The `DecisionTreeClassifier` performed well on the validation set with an accuracy of **90.528%**.

## 📁 Files
- `Binary_Classification_with_a_Bank_Dataset.ipynb`: Main notebook
- `train.csv`: Training data
- `test.csv`: Test data
- `submission5.csv`: Predictions on test set

## 🚀 How to Run
1. Upload the notebook to Google Colab or run locally.
2. Ensure `train.csv` and `test.csv` are in the same directory.
3. Run all cells to preprocess, train, and predict.
4. Download `submission5.csv` for the final predictions.

---

# 📘 Code Explanation (By Section)

## 🔻 Importing Libraries
```python
import numpy as np           # Numerical operations
import pandas as pd          # Data manipulation
import matplotlib.pyplot as plt # Plotting
import seaborn as sns        # Statistical visualizations
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
```

## 🔻 Load Data
```python
df_train = pd.read_csv('/content/train.csv')
df_test = pd.read_csv('/content/test.csv')
```

## 🔻 EDA (Exploratory Data Analysis)
- Checked shape, info, describe, unique values, and missing values.
- No missing values found in either dataset.

## 🔻 Preprocessing
- Filled missing values (though none were found).
- Label encoded categorical columns.
- Scaled numerical features using `StandardScaler`.

## 🔻 Modeling
Trained and evaluated multiple models. Example:
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
de = DecisionTreeClassifier()
de.fit(x_train, y_train)
y_pred = de.predict(x_test)
print(accuracy_score(y_test, y_pred))
```

## 🔻 Prediction & Submission
```python
submission_df = pd.DataFrame({
    'id': df_test['id'],
    'y': de.predict(df_test).astype(bool)
})
submission_df.to_csv('submission5.csv', index=False)
```

---

# ✅ Final Submission
The predictions are saved in `submission5.csv` with two columns:
- `id`: Client ID
- `y`: Prediction (True/False) for subscription.
