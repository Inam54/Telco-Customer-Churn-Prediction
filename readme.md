# 📊 Customer Churn Prediction System

## 📄 Project Overview

This project predicts whether a customer will **churn (leave a service)** using supervised machine learning.
It demonstrates a complete ML pipeline including **data preprocessing, feature encoding, model training, and evaluation** using Python and scikit-learn.

The project supports **multiple models** and is designed using **Object-Oriented Programming (OOP)** principles.

---

## 🗂 Dataset

**Source:** Telecom Customer Churn Dataset

### Description:

The dataset contains customer demographics, service usage, billing information, and contract details.

**Target Variable:**

* `Churn` → Yes (1) / No (0)

**Note:**
The dataset is excluded from this repository using `.gitignore` to comply with data usage best practices.

---

## 🧰 Project Files

```
churn-prediction/
├── Dataset/                # Ignored (contains CSV file)
├── churn_prediction.py     # Main ML pipeline
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Key Techniques Used

### 🔹 Data Preprocessing

* Label Encoding for binary categorical features
* Ordinal Encoding for ordered features (`InternetService`)
* One-Hot Encoding for `PaymentMethod`
* Handling missing values in `TotalCharges`
* Feature Scaling using `StandardScaler`

### 🔹 Models Implemented

* Random Forest Classifier
* Decision Tree Classifier

Model selection is done dynamically via user input.

---

## 📊 Model Evaluation Metrics

The following metrics are used to evaluate model performance:

* Accuracy
* Precision
* Recall
* F1 Score

These metrics provide a balanced understanding of classification performance, especially for churn prediction problems.

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Script

```bash
python churn_prediction.py
```

### Step 3: Choose Model

When prompted, enter:

```
RandomForest
```

or

```
DecisionTree
```

---

## 💡 Learning Outcomes

* Implementing end-to-end ML pipelines
* Handling categorical and numerical data correctly
* Applying OOP principles in ML projects
* Understanding trade-offs between precision and recall
* Building reusable and modular ML code

---

## 🛠 Tech Stack

* Python
* pandas
* scikit-learn

---

## 🔮 Future Improvements

* Add Logistic Regression and XGBoost
* Hyperparameter tuning with GridSearchCV
* Cross-validation
* Feature importance visualization
* Convert script into a REST API using FastAPI

---

## 👤 Author

**Inam Ur Rehman**
BS Computer Engineering (ITU Lahore)
Focus: Machine Learning | AI Engineering
