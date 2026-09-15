"""
Customer Intelligence Engine
RFM-Based Customer Segmentation & Churn Prediction

End-to-end pipeline:
1. Load UCI Online Retail Excel data
2. Clean and preprocess transactions
3. Create RFM customer features and segments
4. Generate EDA/RFM charts
5. Build a leakage-aware churn target using a future 90-day window
6. Train Decision Tree and Random Forest classifiers
7. Evaluate models
8. Save feature importance, predictions, models and charts
9. Store final outputs in SQLite
"""

from pathlib import Path
import sqlite3
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# -------------------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = BASE_DIR / "charts"
MODEL_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"

for folder in [DATA_DIR, OUTPUT_DIR, CHART_DIR, MODEL_DIR, DATABASE_DIR]:
    folder.mkdir(exist_ok=True)


RAW_FILE = DATA_DIR / "Online Retail.xlsx"
CLEAN_FILE = DATA_DIR / "cleaned_retail_data.csv"
RFM_FILE = OUTPUT_DIR / "customer_rfm_segments.csv"
CHURN_FILE = OUTPUT_DIR / "churn_prediction_dataset.csv"
COMPARISON_FILE = OUTPUT_DIR / "model_comparison.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"
PREDICTION_FILE = OUTPUT_DIR / "test_predictions.csv"
DATABASE_FILE = DATABASE_DIR / "customer_intelligence.db"


# -------------------------------------------------------------------
# DISPLAY HELPERS
# -------------------------------------------------------------------

def title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# -------------------------------------------------------------------
# 1. LOAD AND PREPROCESS
# -------------------------------------------------------------------

def load_and_clean_data():
    title("STEP 1 - DATA LOADING AND PREPROCESSING")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_FILE}\n\n"
            "Place 'Online Retail.xlsx' inside the data folder."
        )

    df = pd.read_excel(RAW_FILE)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    # Remove exact duplicate transactions.
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {before - len(df):,}")

    # CustomerID is required for customer-level analysis.
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    print(f"Missing CustomerID rows removed: {before - len(df):,}")

    # C-prefixed invoices represent cancellations/returns in this dataset.
    before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"Cancelled transactions removed: {before - len(df):,}")

    # Keep only valid purchases.
    before = len(df)
    df = df[df["Quantity"] > 0]
    print(f"Invalid quantity rows removed: {before - len(df):,}")

    before = len(df)
    df = df[df["UnitPrice"] > 0]
    print(f"Invalid price rows removed: {before - len(df):,}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    df.to_csv(CLEAN_FILE, index=False)

    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Saved: {CLEAN_FILE}")

    return df


# -------------------------------------------------------------------
# 2. RFM ANALYSIS AND SEGMENTATION
# -------------------------------------------------------------------

def create_rfm(df):
    title("STEP 2 - RFM ANALYSIS AND CUSTOMER SEGMENTATION")

    analysis_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (analysis_date - x.max()).days,
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()

    # Rank before qcut so tied values do not break the quantile bins.
    rfm["RecencyScore"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)

    rfm["FrequencyScore"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["MonetaryScore"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["RecencyScore"].astype(str)
        + rfm["FrequencyScore"].astype(str)
        + rfm["MonetaryScore"].astype(str)
    )

    rfm["RFM_Total"] = (
        rfm["RecencyScore"]
        + rfm["FrequencyScore"]
        + rfm["MonetaryScore"]
    )

    def segment(row):
        score = row["RFM_Total"]
        if score >= 13:
            return "Champions"
        if score >= 10:
            return "Loyal Customers"
        if score >= 8:
            return "Potential Loyalists"
        if score >= 6:
            return "At Risk"
        return "Lost Customers"

    rfm["Segment"] = rfm.apply(segment, axis=1)

    rfm.to_csv(RFM_FILE, index=False)

    print(f"Analysis date: {analysis_date}")
    print(f"Unique customers: {len(rfm):,}")
    print("\nSegment distribution:")
    print(rfm["Segment"].value_counts())

    print(f"\nSaved: {RFM_FILE}")

    return rfm


# -------------------------------------------------------------------
# 3. VISUALIZATIONS
# -------------------------------------------------------------------

def create_rfm_charts(rfm):
    title("STEP 3 - RFM VISUALIZATION")

    # Chart 1: segment distribution
    plt.figure(figsize=(10, 6))
    rfm["Segment"].value_counts().plot(kind="bar")
    plt.title("Customer Segment Distribution")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "customer_segment_distribution.png", dpi=300)
    plt.close()

    # Chart 2: recency vs monetary
    plt.figure(figsize=(10, 6))
    plt.scatter(rfm["Recency"], rfm["Monetary"], alpha=0.5)
    plt.title("Recency vs Monetary Value")
    plt.xlabel("Recency (Days)")
    plt.ylabel("Monetary Value")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "recency_vs_monetary.png", dpi=300)
    plt.close()

    # Chart 3: frequency vs monetary
    plt.figure(figsize=(10, 6))
    plt.scatter(rfm["Frequency"], rfm["Monetary"], alpha=0.5)
    plt.title("Frequency vs Monetary Value")
    plt.xlabel("Purchase Frequency")
    plt.ylabel("Monetary Value")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "frequency_vs_monetary.png", dpi=300)
    plt.close()

    # Chart 4: average monetary value by segment
    segment_monetary = (
        rfm.groupby("Segment")["Monetary"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    segment_monetary.plot(kind="bar")
    plt.title("Average Monetary Value by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Average Monetary Value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "monetary_by_segment.png", dpi=300)
    plt.close()

    print("Created 4 RFM/EDA charts.")


# -------------------------------------------------------------------
# 4. CHURN DATASET
# -------------------------------------------------------------------

def create_churn_dataset(df):
    title("STEP 4 - CHURN LABEL CREATION")

    # Historical features are calculated before the snapshot.
    # The target is based on the following 90 days.
    snapshot_date = pd.Timestamp("2011-09-01")
    future_end_date = snapshot_date + pd.Timedelta(days=90)

    historical = df[df["InvoiceDate"] < snapshot_date].copy()

    rfm_ml = historical.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (snapshot_date - x.max()).days,
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()

    rfm_ml["AverageOrderValue"] = (
        rfm_ml["Monetary"] / rfm_ml["Frequency"]
    )

    future = df[
        (df["InvoiceDate"] >= snapshot_date)
        & (df["InvoiceDate"] < future_end_date)
    ].copy()

    future_activity = future.groupby("CustomerID").agg(
        FutureOrders=("InvoiceNo", "nunique")
    ).reset_index()

    rfm_ml = rfm_ml.merge(
        future_activity,
        on="CustomerID",
        how="left",
    )

    rfm_ml["FutureOrders"] = rfm_ml["FutureOrders"].fillna(0)

    # Churn = no purchase during the 90-day observation window.
    rfm_ml["Churn"] = (
        rfm_ml["FutureOrders"] == 0
    ).astype(int)

    rfm_ml.to_csv(CHURN_FILE, index=False)

    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Future observation end: {future_end_date.date()}")
    print(f"Customers available for prediction: {len(rfm_ml):,}")
    print("\nChurn distribution:")
    print(rfm_ml["Churn"].value_counts())
    print("\nChurn percentage:")
    print((rfm_ml["Churn"].value_counts(normalize=True) * 100).round(2))

    print(f"\nSaved: {CHURN_FILE}")

    return rfm_ml


# -------------------------------------------------------------------
# 5. MACHINE LEARNING
# -------------------------------------------------------------------

def train_and_evaluate_models(churn_df):
    title("STEP 5 - MACHINE LEARNING")

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "AverageOrderValue",
    ]

    X = churn_df[features]
    y = churn_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    dt_pred = decision_tree.predict(X_test)
    rf_pred = random_forest.predict(X_test)

    def evaluate(name, y_true, pred):
        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(
                y_true, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, pred, zero_division=0
            ),
            "F1_Score": f1_score(
                y_true, pred, zero_division=0
            ),
        }

        print(f"\n{name}")
        print("-" * 60)
        print(f"Accuracy : {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall   : {result['Recall']:.4f}")
        print(f"F1 Score : {result['F1_Score']:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                pred,
                target_names=["Active", "Churned"],
                zero_division=0,
            )
        )

        return result

    dt_result = evaluate(
        "Decision Tree",
        y_test,
        dt_pred,
    )

    rf_result = evaluate(
        "Random Forest",
        y_test,
        rf_pred,
    )

    comparison = pd.DataFrame(
        [dt_result, rf_result]
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    print("\nModel comparison:")
    print(comparison)

    # Confusion matrix - Decision Tree
    dt_cm = confusion_matrix(y_test, dt_pred)
    print("\nDecision Tree Confusion Matrix:")
    print(dt_cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=dt_cm,
        display_labels=["Active", "Churned"],
    )
    disp.plot()
    plt.title("Decision Tree - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "decision_tree_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    # Confusion matrix - Random Forest
    rf_cm = confusion_matrix(y_test, rf_pred)
    print("\nRandom Forest Confusion Matrix:")
    print(rf_cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=rf_cm,
        display_labels=["Active", "Churned"],
    )
    disp.plot()
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "random_forest_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    # Feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": random_forest.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False,
    )

    print("\nRandom Forest Feature Importance:")
    print(importance)

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    plt.figure(figsize=(9, 6))
    plt.bar(
        importance["Feature"],
        importance["Importance"],
    )
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Customer Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "random_forest_feature_importance.png",
        dpi=300,
    )
    plt.close()

    # Test predictions
    test_predictions = X_test.copy()
    test_predictions["Actual_Churn"] = y_test.to_numpy()
    test_predictions["Decision_Tree_Prediction"] = dt_pred
    test_predictions["Random_Forest_Prediction"] = rf_pred

    test_predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    # Save trained models
    with open(
        MODEL_DIR / "decision_tree_model.pkl",
        "wb",
    ) as file:
        pickle.dump(decision_tree, file)

    with open(
        MODEL_DIR / "random_forest_model.pkl",
        "wb",
    ) as file:
        pickle.dump(random_forest, file)

    print(f"\nSaved: {COMPARISON_FILE}")
    print(f"Saved: {IMPORTANCE_FILE}")
    print(f"Saved: {PREDICTION_FILE}")
    print("Saved both trained model files.")

    return comparison


# -------------------------------------------------------------------
# 6. SQLITE DATABASE
# -------------------------------------------------------------------

def create_database(rfm, churn_df, comparison):
    title("STEP 6 - SQLITE DATABASE")

    connection = sqlite3.connect(DATABASE_FILE)

    # Reload predictions if available.
    predictions = pd.read_csv(PREDICTION_FILE)

    rfm.to_sql(
        "customers_rfm",
        connection,
        if_exists="replace",
        index=False,
    )

    churn_df.to_sql(
        "churn_predictions",
        connection,
        if_exists="replace",
        index=False,
    )

    comparison.to_sql(
        "model_comparison",
        connection,
        if_exists="replace",
        index=False,
    )

    predictions.to_sql(
        "test_predictions",
        connection,
        if_exists="replace",
        index=False,
    )

    cursor = connection.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' ORDER BY name"
    )

    tables = [row[0] for row in cursor.fetchall()]

    print("Database tables:")
    for table in tables:
        print(f"- {table}")

    query = """
        SELECT
            Segment,
            COUNT(*) AS CustomerCount,
            ROUND(AVG(Monetary), 2) AS AverageMonetary
        FROM customers_rfm
        GROUP BY Segment
        ORDER BY CustomerCount DESC;
    """

    result = pd.read_sql_query(
        query,
        connection,
    )

    print("\nCustomer segment summary from SQLite:")
    print(result)

    connection.close()

    print(f"\nDatabase saved: {DATABASE_FILE}")


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def main():
    title("CUSTOMER INTELLIGENCE ENGINE - FULL PIPELINE")

    # NumPy is part of the project stack; print version for reproducibility.
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")

    df = load_and_clean_data()

    rfm = create_rfm(df)

    create_rfm_charts(rfm)

    churn_df = create_churn_dataset(df)

    comparison = train_and_evaluate_models(churn_df)

    create_database(
        rfm,
        churn_df,
        comparison,
    )

    title("PROJECT COMPLETED SUCCESSFULLY")

    print("All major pipeline stages completed:")
    print("1. Data preprocessing")
    print("2. RFM analysis")
    print("3. Customer segmentation")
    print("4. Exploratory visualization")
    print("5. Churn label creation")
    print("6. Decision Tree")
    print("7. Random Forest")
    print("8. Model evaluation")
    print("9. Feature importance")
    print("10. SQLite database")

    print("\nProject outputs are stored in:")
    print(f"- {DATA_DIR}")
    print(f"- {CHART_DIR}")
    print(f"- {MODEL_DIR}")
    print(f"- {OUTPUT_DIR}")
    print(f"- {DATABASE_DIR}")


if __name__ == "__main__":
    main()