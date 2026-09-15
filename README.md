# Customer Intelligence Engine

## RFM-Based Customer Segmentation & Churn Prediction

### 1. Project Overview

The **Customer Intelligence Engine** is a Python-based data analytics
and machine-learning project that analyzes customer purchasing behavior
and predicts potential customer churn.

The system uses **RFM analysis**:

-   **Recency** --- how recently a customer purchased
-   **Frequency** --- how often a customer purchased
-   **Monetary** --- how much a customer spent

Customers are grouped into business-oriented segments such as:

-   Champions
-   Loyal Customers
-   Potential Loyalists
-   At Risk
-   Lost Customers

A churn-prediction pipeline then compares:

-   Decision Tree
-   Random Forest

The project evaluates the models using accuracy, precision, recall,
F1-score, and confusion matrices. Random Forest feature importance is
also analyzed.

------------------------------------------------------------------------

## 2. Technology Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   SQLite
-   Excel / CSV
-   Jupyter-compatible Python environment
-   VS Code

------------------------------------------------------------------------

## 3. Dataset

The project uses the **UCI Online Retail** dataset.

Source:

UCI Machine Learning Repository --- Online Retail\
https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains online retail transactions from a UK-based
non-store retailer between December 2010 and December 2011.

Original dataset size:

-   541,909 transactions
-   8 transaction-level columns used by this implementation

Important fields:

-   InvoiceNo
-   StockCode
-   Description
-   Quantity
-   InvoiceDate
-   UnitPrice
-   CustomerID
-   Country

------------------------------------------------------------------------

## 4. Project Workflow

``` text
Online Retail.xlsx
        |
        v
Data Preprocessing
        |
        +--> Remove duplicates
        +--> Remove missing CustomerID
        +--> Remove cancellations
        +--> Remove invalid quantity
        +--> Remove invalid price
        +--> Create TotalAmount
        |
        v
RFM Feature Engineering
        |
        +--> Recency
        +--> Frequency
        +--> Monetary
        |
        v
RFM Scoring & Customer Segmentation
        |
        v
EDA / Matplotlib Visualizations
        |
        v
Churn Label Creation
        |
        v
Decision Tree + Random Forest
        |
        v
Model Evaluation
        |
        +--> Accuracy
        +--> Precision
        +--> Recall
        +--> F1 Score
        +--> Confusion Matrix
        +--> Feature Importance
        |
        v
SQLite Database
```

------------------------------------------------------------------------

## 5. Data Preprocessing

The original dataset is cleaned before customer-level analysis.

The preprocessing pipeline:

1.  Removes exact duplicate rows.
2.  Removes transactions without `CustomerID`.
3.  Removes cancelled invoices whose invoice number begins with `C`.
4.  Removes transactions with non-positive quantity.
5.  Removes transactions with non-positive unit price.
6.  Converts `InvoiceDate` to datetime.
7.  Creates:

``` text
TotalAmount = Quantity × UnitPrice
```

For the current dataset and preprocessing rules, the pipeline produces
approximately 392,692 valid transaction rows.

------------------------------------------------------------------------

## 6. RFM Analysis

### Recency

Recency is calculated as the number of days between the analysis date
and a customer's most recent purchase.

Lower recency is better.

### Frequency

Frequency is the number of unique invoices associated with a customer.

Higher frequency indicates more repeat purchasing.

### Monetary

Monetary value is the customer's total spending.

``` text
Monetary = sum(TotalAmount)
```

Higher monetary value indicates greater customer value.

### RFM Scoring

Each RFM dimension is divided into five quantile groups.

-   Recency is reverse-scored because lower recency is better.
-   Frequency is scored from low to high.
-   Monetary is scored from low to high.

The three scores are combined into an RFM score and an overall numerical
RFM total.

------------------------------------------------------------------------

## 7. Customer Segmentation

The implementation uses the overall RFM total to create five segments:

    RFM Total Segment
  ----------- ---------------------
       13--15 Champions
       10--12 Loyal Customers
         8--9 Potential Loyalists
         6--7 At Risk
         3--5 Lost Customers

The exact segment counts produced by the current run are:

  Segment                 Customers
  --------------------- -----------
  Loyal Customers             1,012
  Champions                     929
  Lost Customers                885
  At Risk                       808
  Potential Loyalists           704

Total analyzed customers: **4,338**.

------------------------------------------------------------------------

## 8. Churn Definition

The transaction dataset does not contain a ready-made churn label.

Therefore, this project creates a time-based target.

### Snapshot

``` text
2011-09-01
```

Customer behavior before this date is used to create the prediction
features.

### Observation window

The following 90 days are used to determine whether the customer
purchases again:

``` text
2011-09-01 to 2011-11-30
```

### Churn rule

``` text
No purchase during the future 90-day window
                =
              Churn
```

This approach avoids defining churn directly from the same future
behavior that the model is supposed to predict.

------------------------------------------------------------------------

## 9. Machine Learning

The following features are used:

-   Recency
-   Frequency
-   Monetary
-   AverageOrderValue

### Models

#### Decision Tree

A Decision Tree classifier is trained with:

``` text
max_depth = 5
random_state = 42
```

#### Random Forest

A Random Forest classifier is trained with:

``` text
n_estimators = 200
max_depth = 8
random_state = 42
```

------------------------------------------------------------------------

## 10. Model Results

Results from the completed project run:

  Model               Accuracy    Precision       Recall     F1 Score
  --------------- ------------ ------------ ------------ ------------
  Decision Tree         66.27%       59.88%   **69.76%**   **64.44%**
  Random Forest     **66.42%**   **60.76%**       65.98%       63.26%

### Interpretation

Random Forest achieved the slightly higher overall accuracy.

Decision Tree achieved higher churn recall and F1-score in this run,
meaning it identified a larger proportion of the actual churned
customers.

Therefore:

-   **Random Forest:** slightly better by accuracy
-   **Decision Tree:** better when the priority is detecting more
    churned customers

The choice of model should therefore depend on the business objective
rather than accuracy alone.

------------------------------------------------------------------------

## 11. Feature Importance

Random Forest feature importance from the completed run:

  Feature               Importance
  ------------------- ------------
  Monetary                  32.83%
  Recency                   24.93%
  Frequency                 23.71%
  AverageOrderValue         18.53%

The result indicates that **Monetary** was the strongest feature among
the four features used by the Random Forest model in this run.

------------------------------------------------------------------------

## 12. Generated Outputs

### Data

``` text
data/
├── Online Retail.xlsx
└── cleaned_retail_data.csv
```

### Charts

``` text
charts/
├── customer_segment_distribution.png
├── recency_vs_monetary.png
├── frequency_vs_monetary.png
├── monetary_by_segment.png
├── decision_tree_confusion_matrix.png
├── random_forest_confusion_matrix.png
└── random_forest_feature_importance.png
```

### Machine Learning Outputs

``` text
outputs/
├── customer_rfm_segments.csv
├── churn_prediction_dataset.csv
├── model_comparison.csv
├── feature_importance.csv
└── test_predictions.csv
```

### Trained Models

``` text
models/
├── decision_tree_model.pkl
└── random_forest_model.pkl
```

### Database

``` text
database/
└── customer_intelligence.db
```

SQLite tables:

-   `customers_rfm`
-   `churn_predictions`
-   `model_comparison`
-   `test_predictions`

------------------------------------------------------------------------

## 13. How to Run

### Step 1 --- Open the project

Open the project folder in VS Code.

### Step 2 --- Activate the virtual environment

PowerShell:

``` powershell
.venv\Scripts\activate
```

### Step 3 --- Install dependencies

``` powershell
python -m pip install -r requirements.txt
```

### Step 4 --- Make sure the dataset exists

The file must be:

``` text
data/Online Retail.xlsx
```

### Step 5 --- Run the complete pipeline

``` powershell
python main.py
```

The complete pipeline will regenerate the cleaned data, RFM analysis,
charts, churn dataset, machine-learning models, evaluation results, and
SQLite database.

------------------------------------------------------------------------

## 14. Project Structure

``` text
Customer_Intelligence_Project/
│
├── .venv/
│
├── data/
│   ├── Online Retail.xlsx
│   └── cleaned_retail_data.csv
│
├── charts/
│   ├── customer_segment_distribution.png
│   ├── recency_vs_monetary.png
│   ├── frequency_vs_monetary.png
│   ├── monetary_by_segment.png
│   ├── decision_tree_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── random_forest_feature_importance.png
│
├── models/
│   ├── decision_tree_model.pkl
│   └── random_forest_model.pkl
│
├── outputs/
│   ├── customer_rfm_segments.csv
│   ├── churn_prediction_dataset.csv
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── test_predictions.csv
│
├── database/
│   └── customer_intelligence.db
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 15. Key Business Insights

1.  RFM analysis separates customers according to purchasing behavior
    rather than treating every customer equally.
2.  Champions represent the strongest combination of recency, frequency
    and monetary value.
3.  At Risk and Lost Customers are useful targets for retention
    campaigns.
4.  Monetary value was the most important feature in the Random Forest
    model in this run.
5.  Decision Tree achieved higher churn recall than Random Forest in
    this run.
6.  Model selection should consider the business cost of missing a
    churned customer, not only overall accuracy.
7.  SQLite provides a lightweight relational storage layer for
    customer-level analysis and model results.

------------------------------------------------------------------------

## 16. Limitations

-   The dataset is historical and ends in 2011.
-   The churn label is a business-defined target rather than a label
    supplied by the original dataset.
-   The model uses a limited set of customer-level behavioral features.
-   The reported model performance is specific to this dataset,
    preprocessing pipeline and time window.
-   Production deployment would require current customer data, model
    monitoring, retraining and validation on future unseen periods.

------------------------------------------------------------------------

## 17. Resume Project Description

**Customer Intelligence Engine --- RFM-Based Customer Segmentation &
Churn Prediction**

-   Developed a data-driven customer intelligence system using RFM
    analysis to segment customers based on purchasing behavior.
-   Performed data preprocessing, feature engineering and exploratory
    data analysis using Python, Pandas and NumPy.
-   Built and evaluated Decision Tree and Random Forest classifiers to
    predict potential customer churn using time-based customer behavior
    features.
-   Evaluated models using accuracy, precision, recall, F1-score,
    confusion matrices and Random Forest feature importance.
-   Visualized customer segments and model results using Matplotlib and
    stored analytical outputs in SQLite.

------------------------------------------------------------------------

## 18. Interview Explanation

A concise explanation for an interview:

> "I developed a customer intelligence system using the UCI Online
> Retail dataset. I first cleaned the transaction data by removing
> duplicates, missing customer IDs, cancellations and invalid
> transactions. Then I engineered Recency, Frequency and Monetary
> features to segment customers using RFM scoring. For churn prediction,
> I created a time-based churn label by checking whether customers
> purchased again during a future 90-day window. I trained Decision Tree
> and Random Forest classifiers and evaluated them using accuracy,
> precision, recall, F1-score and confusion matrices. Finally, I used
> feature importance to understand the main behavioral drivers and
> stored the customer and model outputs in SQLite."
