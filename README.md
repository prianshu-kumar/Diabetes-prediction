# 🩺 Diabetes Prediction using Machine Learning

A Machine Learning project that predicts whether a patient is diabetic or not based on medical diagnostic measurements.

This project demonstrates the complete ML workflow including:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature scaling
- Model training
- Pipeline creation
- Model evaluation
- Feature importance analysis

---

# 🚀 Project Overview

The objective of this project is to build classification models capable of predicting diabetes using patient health data.

The dataset contains several medical predictor variables such as:
- Glucose level
- Blood pressure
- BMI
- Insulin
- Age
- Pregnancies

The target variable:
- `0` → Non-Diabetic
- `1` → Diabetic

---

# 📂 Dataset

Dataset used:
- **Pima Indians Diabetes Dataset**

Features included:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of patient |
| Outcome | Diabetes prediction (0 or 1) |

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📊 Exploratory Data Analysis

Performed:
- Correlation heatmaps
- Pairplots
- Distribution plots
- Boxplots
- Outlier analysis

Example insights:
- Glucose showed strong correlation with diabetes outcome
- Some features required scaling for better model performance

---

# 🤖 Machine Learning Models Used

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (optional)
- Support Vector Machine (optional)

---

# ⚙️ Pipeline Workflow

Implemented Scikit-learn Pipelines for:
- Feature scaling
- Feature selection
- Model training

Example workflow:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectFromModel(RandomForestClassifier())),
    ("model", LogisticRegression())
])
```

---

# 📈 Model Evaluation

Evaluation metrics used:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation

Current baseline performance:
- Logistic Regression → ~75%
- Decision Tree → ~74%

Future improvements:
- Hyperparameter tuning
- Ensemble methods
- XGBoost
- Better feature engineering

---

# 📌 Project Structure

```bash
Diabetes-Prediction/
│
├── notebook.ipynb
├── diabetes.csv
├── README.md
├── requirements.txt
└── model/
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/prianshu-kumar/Diabetes-prediction.git
```

Move into project folder:

```bash
cd Diabetes-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

# 📷 Visualizations

## Correlation Heatmap
- Understand feature relationships

## Pairplot
- Analyze feature distributions

## Feature Importance
- Identify most important medical indicators

---

# 🔮 Future Improvements

- Deploy using Streamlit
- Add model saving with Pickle
- Improve accuracy using ensemble learning
- Add hyperparameter tuning with GridSearchCV
- Build a complete ML pipeline

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Prianshu Kumar

GitHub:
https://github.com/prianshu-kumar