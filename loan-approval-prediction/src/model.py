import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless-safe: saves charts to file instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    return df


# ── 2. Explore Data (EDA) ─────────────────────────────────────────────────────
def explore_data(df):
    print("\n── Missing Values ──")
    print(df.isnull().sum())

    print("\n── Data Types ──")
    print(df.dtypes)

    print("\n── Loan Status Distribution ──")
    print(df['Loan_Status'].value_counts())


# ── 3. Preprocess Data ────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Drop Loan_ID (not useful for prediction)
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    # Fill missing values
    # NOTE: df['col'].fillna(..., inplace=True) silently does nothing under
    # pandas' copy-on-write behavior (default since pandas 2.x, enforced in 3.0) -
    # it fills a temporary copy, not the real column. Reassigning avoids that.
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Log transform to handle skewness
    df['LoanAmount'] = np.log1p(df['LoanAmount'])
    df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
    df['CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])

    print("\nPreprocessing complete. Shape:", df.shape)
    return df


# ── 4. Train Model ────────────────────────────────────────────────────────────
def train_model(df):
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    print("\n── Random Forest ──")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    print(classification_report(y_test, rf_preds))

    print("\n── Logistic Regression ──")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2%}")
    print(classification_report(y_test, lr_preds))

    return rf_model, X_test, y_test, rf_preds, X.columns


# ── 5. Visualize Results ──────────────────────────────────────────────────────
def visualize(model, X_test, y_test, preds, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Loan Approval Prediction - Results', fontsize=15, fontweight='bold')

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_title('Feature Importance')
    axes[1].set_xlabel('Importance Score')

    plt.tight_layout()
    plt.savefig('loan_prediction_results.png', dpi=150, bbox_inches='tight')
    print("\nChart saved as 'loan_prediction_results.png'")
    plt.close(fig)


# ── 6. Predict New Applicant ──────────────────────────────────────────────────
def predict_applicant(model, feature_names):
    """Example: predict for a new loan applicant"""
    # Sample input (encoded values)
    sample = {
        'Gender': 1,           # Male
        'Married': 1,          # Yes
        'Dependents': 0,       # 0
        'Education': 0,        # Graduate
        'Self_Employed': 0,    # No
        'ApplicantIncome': np.log1p(5000),
        'CoapplicantIncome': np.log1p(2000),
        'LoanAmount': np.log1p(150),
        'Loan_Amount_Term': 360,
        'Credit_History': 1,   # Good
        'Property_Area': 2     # Urban
    }

    input_df = pd.DataFrame([sample])[feature_names]
    result = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    status = "✅ APPROVED" if result == 1 else "❌ REJECTED"
    print(f"\n── Prediction for New Applicant ──")
    print(f"Result: {status}")
    print(f"Approval Probability: {prob[1]:.2%}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("data/loan_data.csv")
    explore_data(df)
    df_clean = preprocess(df)
    model, X_test, y_test, preds, features = train_model(df_clean)
    visualize(model, X_test, y_test, preds, features)
    predict_applicant(model, features)
