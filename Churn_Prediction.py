# Importing Necessary Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChurnPrediction:
    def __init__(self, path):
        self.path = path
        self.le = LabelEncoder()
        self.oe = OrdinalEncoder(categories=[['No', 'DSL', 'Fiber optic']])
        self.scaler = StandardScaler()

    def load_data(self):
        # Load dataset
        self.df = pd.read_csv(self.path)
        print("Dataset Loaded Successfully")

        # Optional dataset preview
        show_dataset = input("Do you want to see the dataset (yes/no): ")
        if show_dataset.lower() == "yes":
            nor = int(input("Enter number of rows to display: "))
            print(self.df.head(nor))

        # Basic column inspection
        for col in self.df.columns:
            print(f"\nColumn Name: {col}")
            print(self.df[col].value_counts())
            print("Total values:", self.df[col].value_counts().sum())
            print("Missing values:", self.df[col].isnull().sum())

    def preprocess(self):
        # Columns with Yes/No or similar categorical values
        le_cols = [
            "Partner", "gender", "Dependents", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "Churn"
        ]

        # Applying Label Encoding
        for col in le_cols:
            self.df[col] = self.le.fit_transform(self.df[col])

        # Encode InternetService with a meaningful order
        self.df["InternetService"] = self.oe.fit_transform(
            self.df[["InternetService"]]
        )

        # Convert TotalCharges to numeric (handle spaces)
        self.df["TotalCharges"] = pd.to_numeric(
            self.df["TotalCharges"], errors="coerce"
        )

        # Scale numerical features
        scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.df[scale_cols] = self.scaler.fit_transform(self.df[scale_cols])

        # One-hot encode PaymentMethod
        self.df = pd.get_dummies(
            self.df, columns=["PaymentMethod"], drop_first=True
        )

        # Ensure one-hot columns are integers
        payment_cols = [
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check"
        ]
        self.df[payment_cols] = self.df[payment_cols].astype(int)
        print("Preprocessing completed")
        print(self.df)
    def data_splitting(self):
        # Separate features and target
        X = self.df.drop(columns=["Churn", "customerID"])
        y = self.df["Churn"]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def model_training(self, model):
        # Train model
        model.fit(self.X_train, self.y_train)
        # Predictions
        self.y_pred = model.predict(self.X_test)

    def model_evaluation(self):
        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)

        # Print results
        print("Model Performance:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

model_name = input("Enter model name (RandomForest or DecisionTree): ")

if model_name == "RandomForest":
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
elif model_name == "DecisionTree":
    model = DecisionTreeClassifier(random_state=42)
else:
    raise ValueError("Invalid model name")

# Create object
churn_model = ChurnPrediction("Dataset/Churn_Data.csv")

# Run pipeline
churn_model.load_data()
churn_model.preprocess()
churn_model.data_splitting()
churn_model.model_training(model)
churn_model.model_evaluation()