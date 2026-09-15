import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence Engine",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# PAGE CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #888888;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PDF CHART CREATOR
# =========================================================

def create_pdf_chart(chart_type, results):

    rfm = results["rfm"]

    fig, ax = plt.subplots(
        figsize=(5.2, 2.4)
    )

    if chart_type == "segments":

        segment_counts = (
            rfm["Segment"]
            .value_counts()
        )

        segment_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Customer Segment Distribution",
            fontsize=9
        )

        ax.set_xlabel(
            "Segment",
            fontsize=7
        )

        ax.set_ylabel(
            "Customers",
            fontsize=7
        )

        ax.tick_params(
            axis="x",
            labelsize=6,
            rotation=25
        )

        ax.tick_params(
            axis="y",
            labelsize=6
        )

    elif chart_type == "rfm":

        ax.scatter(
            rfm["Recency"],
            rfm["Monetary"],
            alpha=0.45
        )

        ax.set_title(
            "Recency vs Monetary",
            fontsize=9
        )

        ax.set_xlabel(
            "Recency (Days)",
            fontsize=7
        )

        ax.set_ylabel(
            "Monetary (£)",
            fontsize=7
        )

        ax.tick_params(
            labelsize=6
        )

    elif chart_type == "accuracy":

        model_results = (
            results["model_results"]
        )

        accuracy = (
            model_results
            .set_index("Model")["Accuracy"]
            * 100
        )

        accuracy.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Model Accuracy",
            fontsize=9
        )

        ax.set_ylabel(
            "Accuracy (%)",
            fontsize=7
        )

        ax.set_ylim(
            0,
            100
        )

        ax.tick_params(
            labelsize=6
        )

    plt.tight_layout()

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    image_buffer.seek(0)

    return image_buffer


# =========================================================
# PDF REPORT GENERATOR
# =========================================================

def generate_pdf(results):

    buffer = BytesIO()

    # Landscape A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=6 * mm,
        rightMargin=6 * mm,
        topMargin=5 * mm,
        bottomMargin=5 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=1 * mm
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=9,
        alignment=TA_CENTER,
        spaceAfter=2 * mm
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=9.5,
        spaceBefore=0.5 * mm,
        spaceAfter=1 * mm
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=7.5
    )

    tiny_style = ParagraphStyle(
        "Tiny",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=5.8,
        leading=6.8
    )

    story = []

    clean_df = results["cleaned_data"]
    rfm = results["rfm"]
    churn = results["churn"]
    model_results = results["model_results"]
    importance = results["feature_importance"]

    # =====================================================
    # TITLE
    # =====================================================

    story.append(
        Paragraph(
            "Customer Intelligence Engine",
            title_style
        )
    )

    story.append(
        Paragraph(
            "RFM-Based Customer Segmentation & Churn Prediction",
            subtitle_style
        )
    )

    # =====================================================
    # SUMMARY BOX
    # =====================================================

    total_revenue = (
        clean_df["TotalAmount"].sum()
    )

    summary_data = [
        [
            Paragraph(
                "<b>Original Transactions</b>",
                tiny_style
            ),
            Paragraph(
                f"<b>{results['original_rows']:,}</b>",
                tiny_style
            ),
            Paragraph(
                "<b>Clean Transactions</b>",
                tiny_style
            ),
            Paragraph(
                f"<b>{len(clean_df):,}</b>",
                tiny_style
            ),
            Paragraph(
                "<b>Customers</b>",
                tiny_style
            ),
            Paragraph(
                f"<b>{len(rfm):,}</b>",
                tiny_style
            ),
            Paragraph(
                "<b>Total Revenue</b>",
                tiny_style
            ),
            Paragraph(
                f"<b>£{total_revenue:,.2f}</b>",
                tiny_style
            )
        ]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            32 * mm,
            25 * mm,
            31 * mm,
            25 * mm,
            22 * mm,
            20 * mm,
            27 * mm,
            34 * mm
        ],
        rowHeights=[
            9 * mm
        ]
    )

    summary_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#E9ECEF")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#888888")
            ),
            (
                "ALIGN",
                (1, 0),
                (-1, 0),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                2
            )
        ])
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(
            1,
            1.5 * mm
        )
    )

    # =====================================================
    # CUSTOMER SEGMENTATION TABLE
    # =====================================================

    segment_counts = (
        rfm["Segment"]
        .value_counts()
    )

    segment_data = [
        [
            Paragraph(
                "<b>Segment</b>",
                tiny_style
            ),
            Paragraph(
                "<b>Customers</b>",
                tiny_style
            )
        ]
    ]

    for segment, count in segment_counts.items():

        segment_data.append(
            [
                Paragraph(
                    str(segment),
                    tiny_style
                ),
                Paragraph(
                    f"{count:,}",
                    tiny_style
                )
            ]
        )

    segment_table = Table(
        segment_data,
        colWidths=[
            39 * mm,
            22 * mm
        ]
    )

    segment_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#333333")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                2
            ),
            (
                "ALIGN",
                (1, 0),
                (1, -1),
                "CENTER"
            )
        ])
    )

    # =====================================================
    # PREPROCESSING TABLE
    # =====================================================

    preprocessing_data = [
        [
            Paragraph(
                "<b>Step</b>",
                tiny_style
            ),
            Paragraph(
                "<b>Removed</b>",
                tiny_style
            )
        ],
        [
            Paragraph(
                "Duplicates",
                tiny_style
            ),
            f'{results["duplicates_removed"]:,}'
        ],
        [
            Paragraph(
                "Missing CustomerID",
                tiny_style
            ),
            f'{results["missing_customer_removed"]:,}'
        ],
        [
            Paragraph(
                "Cancellations",
                tiny_style
            ),
            f'{results["cancelled_removed"]:,}'
        ],
        [
            Paragraph(
                "Invalid Quantity",
                tiny_style
            ),
            f'{results["invalid_quantity_removed"]:,}'
        ],
        [
            Paragraph(
                "Invalid Price",
                tiny_style
            ),
            f'{results["invalid_price_removed"]:,}'
        ]
    ]

    preprocessing_table = Table(
        preprocessing_data,
        colWidths=[
            39 * mm,
            22 * mm
        ]
    )

    preprocessing_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#333333")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                2
            ),
            (
                "ALIGN",
                (1, 0),
                (1, -1),
                "CENTER"
            )
        ])
    )

    # =====================================================
    # CHURN TABLE
    # =====================================================

    if len(churn) > 0:

        active_count = int(
            (
                churn["Churn"] == 0
            ).sum()
        )

        churn_count = int(
            (
                churn["Churn"] == 1
            ).sum()
        )

        churn_rate = (
            churn_count
            / len(churn)
            * 100
        )

    else:

        active_count = 0
        churn_count = 0
        churn_rate = 0

    churn_data = [
        [
            Paragraph(
                "<b>Churn Metric</b>",
                tiny_style
            ),
            Paragraph(
                "<b>Result</b>",
                tiny_style
            )
        ],
        [
            "Customers Analyzed",
            f"{len(churn):,}"
        ],
        [
            "Active Customers",
            f"{active_count:,}"
        ],
        [
            "Churned Customers",
            f"{churn_count:,}"
        ],
        [
            "Churn Rate",
            f"{churn_rate:.2f}%"
        ]
    ]

    churn_table = Table(
        churn_data,
        colWidths=[
            39 * mm,
            22 * mm
        ]
    )

    churn_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#333333")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                2
            ),
            (
                "ALIGN",
                (1, 0),
                (1, -1),
                "CENTER"
            )
        ])
    )

    # =====================================================
    # LEFT COLUMN
    # =====================================================

    left_column = [
        Paragraph(
            "Customer Segmentation",
            section_style
        ),

        segment_table,

        Spacer(
            1,
            1.5 * mm
        ),

        Paragraph(
            "Data Preprocessing",
            section_style
        ),

        preprocessing_table,

        Spacer(
            1,
            1.5 * mm
        ),

        Paragraph(
            "Churn Analysis",
            section_style
        ),

        churn_table
    ]

    left_wrapper = Table(
        [[left_column]],
        colWidths=[
            64 * mm
        ]
    )

    left_wrapper.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                0
            )
        ])
    )

    # =====================================================
    # MIDDLE COLUMN
    # =====================================================

    segment_chart = Image(
        create_pdf_chart(
            "segments",
            results
        ),
        width=82 * mm,
        height=38 * mm
    )

    rfm_chart = Image(
        create_pdf_chart(
            "rfm",
            results
        ),
        width=82 * mm,
        height=38 * mm
    )

    middle_column = [
        Paragraph(
            "Customer Insights",
            section_style
        ),

        segment_chart,

        Spacer(
            1,
            1 * mm
        ),

        rfm_chart
    ]

    middle_wrapper = Table(
        [[middle_column]],
        colWidths=[
            88 * mm
        ]
    )

    middle_wrapper.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                0
            )
        ])
    )

    # =====================================================
    # RIGHT COLUMN
    # =====================================================

    right_column = []

    right_column.append(
        Paragraph(
            "Machine Learning Performance",
            section_style
        )
    )

    if model_results is not None:

        ml_data = [
            [
                Paragraph(
                    "<b>Model</b>",
                    tiny_style
                ),
                Paragraph(
                    "<b>Accuracy</b>",
                    tiny_style
                ),
                Paragraph(
                    "<b>Precision</b>",
                    tiny_style
                ),
                Paragraph(
                    "<b>Recall</b>",
                    tiny_style
                ),
                Paragraph(
                    "<b>F1</b>",
                    tiny_style
                )
            ]
        ]

        for _, row in model_results.iterrows():

            ml_data.append(
                [
                    row["Model"],
                    f'{row["Accuracy"] * 100:.2f}%',
                    f'{row["Precision"] * 100:.2f}%',
                    f'{row["Recall"] * 100:.2f}%',
                    f'{row["F1 Score"] * 100:.2f}%'
                ]
            )

        ml_table = Table(
            ml_data,
            colWidths=[
                34 * mm,
                21 * mm,
                21 * mm,
                21 * mm,
                17 * mm
            ]
        )

        ml_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#333333")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    2
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (-1, -1),
                    "CENTER"
                )
            ])
        )

        right_column.append(
            ml_table
        )

        right_column.append(
            Spacer(
                1,
                1 * mm
            )
        )

        accuracy_chart = Image(
            create_pdf_chart(
                "accuracy",
                results
            ),
            width=96 * mm,
            height=38 * mm
        )

        right_column.append(
            accuracy_chart
        )

        right_column.append(
            Spacer(
                1,
                1 * mm
            )
        )

        right_column.append(
            Paragraph(
                "Feature Importance",
                section_style
            )
        )

        feature_data = [
            [
                Paragraph(
                    "<b>Feature</b>",
                    tiny_style
                ),
                Paragraph(
                    "<b>Importance</b>",
                    tiny_style
                )
            ]
        ]

        if importance is not None:

            for _, row in importance.iterrows():

                feature_data.append(
                    [
                        row["Feature"],
                        f'{row["Importance"] * 100:.2f}%'
                    ]
                )

        feature_table = Table(
            feature_data,
            colWidths=[
                48 * mm,
                30 * mm
            ]
        )

        feature_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#333333")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    2
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6.5
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (1, -1),
                    "CENTER"
                )
            ])
        )

        right_column.append(
            feature_table
        )

    else:

        right_column.append(
            Paragraph(
                "Machine learning could not be trained for this dataset.",
                small_style
            )
        )

    right_wrapper = Table(
        [[right_column]],
        colWidths=[
            101 * mm
        ]
    )

    right_wrapper.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                0
            )
        ])
    )

    # =====================================================
    # THREE COLUMN REPORT
    # =====================================================

    report_table = Table(
        [
            [
                left_wrapper,
                middle_wrapper,
                right_wrapper
            ]
        ],
        colWidths=[
            66 * mm,
            90 * mm,
            103 * mm
        ]
    )

    report_table.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                2
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                2
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                0
            )
        ])
    )

    story.append(
        report_table
    )

    story.append(
        Spacer(
            1,
            1 * mm
        )
    )

    # =====================================================
    # KEY FINDINGS
    # =====================================================

    if len(segment_counts) > 0:

        top_segment = (
            segment_counts.index[0]
        )

    else:

        top_segment = "N/A"

    if (
        importance is not None
        and len(importance) > 0
    ):

        top_feature = (
            importance.iloc[0]["Feature"]
        )

    else:

        top_feature = "N/A"

    findings_text = (
        f"<b>Key Findings:</b> "
        f"Largest customer segment: {top_segment}. "
        f"Most influential Random Forest feature: {top_feature}. "
        f"Churn rate: {churn_rate:.2f}%. "
        f"Churn is defined as no purchase during the 90-day observation period. "
        f"All results are calculated from the uploaded transaction dataset."
    )

    story.append(
        Paragraph(
            findings_text,
            small_style
        )
    )

    story.append(
        Spacer(
            1,
            0.5 * mm
        )
    )

    story.append(
        Paragraph(
            "Generated by Customer Intelligence Engine",
            tiny_style
        )
    )

    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# DATA ANALYSIS
# =========================================================

def analyze_data(df):

    # -----------------------------------------------------
    # REQUIRED COLUMNS
    # -----------------------------------------------------

    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The uploaded Excel file is missing these required "
            "columns: "
            + ", ".join(missing_columns)
        )

    df = df.copy()

    # -----------------------------------------------------
    # ORIGINAL ROWS
    # -----------------------------------------------------

    original_rows = len(df)

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    df = df.drop_duplicates()

    duplicates_removed = (
        original_rows
        - len(df)
    )

    # -----------------------------------------------------
    # REMOVE MISSING CUSTOMER IDS
    # -----------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["CustomerID"]
    )

    missing_customer_removed = (
        before
        - len(df)
    )

    # -----------------------------------------------------
    # REMOVE CANCELLATIONS
    # -----------------------------------------------------

    before = len(df)

    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    cancelled_removed = (
        before
        - len(df)
    )

    # -----------------------------------------------------
    # REMOVE INVALID QUANTITY
    # -----------------------------------------------------

    before = len(df)

    df = df[
        df["Quantity"] > 0
    ]

    invalid_quantity_removed = (
        before
        - len(df)
    )

    # -----------------------------------------------------
    # REMOVE INVALID PRICE
    # -----------------------------------------------------

    before = len(df)

    df = df[
        df["UnitPrice"] > 0
    ]

    invalid_price_removed = (
        before
        - len(df)
    )

    # -----------------------------------------------------
    # DATE CONVERSION
    # -----------------------------------------------------

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["InvoiceDate"]
    )

    # -----------------------------------------------------
    # TOTAL AMOUNT
    # -----------------------------------------------------

    df["TotalAmount"] = (
        df["Quantity"]
        * df["UnitPrice"]
    )

    if df.empty:

        raise ValueError(
            "No valid transactions remain after preprocessing."
        )

    # =====================================================
    # RFM
    # =====================================================

    analysis_date = (
        df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x:
                (
                    analysis_date
                    - x.max()
                ).days
            ),

            Frequency=(
                "InvoiceNo",
                "nunique"
            ),

            Monetary=(
                "TotalAmount",
                "sum"
            )
        )
        .reset_index()
    )

    if len(rfm) < 5:

        raise ValueError(
            "At least 5 customers are required for RFM analysis."
        )

    # =====================================================
    # RFM SCORING
    # =====================================================

    rfm["RecencyScore"] = (
        pd.qcut(
            rfm["Recency"].rank(
                method="first"
            ),
            5,
            labels=[
                5,
                4,
                3,
                2,
                1
            ]
        )
        .astype(int)
    )

    rfm["FrequencyScore"] = (
        pd.qcut(
            rfm["Frequency"].rank(
                method="first"
            ),
            5,
            labels=[
                1,
                2,
                3,
                4,
                5
            ]
        )
        .astype(int)
    )

    rfm["MonetaryScore"] = (
        pd.qcut(
            rfm["Monetary"].rank(
                method="first"
            ),
            5,
            labels=[
                1,
                2,
                3,
                4,
                5
            ]
        )
        .astype(int)
    )

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

    # =====================================================
    # SEGMENTATION
    # =====================================================

    def assign_segment(score):

        if score >= 13:
            return "Champions"

        elif score >= 10:
            return "Loyal Customers"

        elif score >= 8:
            return "Potential Loyalists"

        elif score >= 6:
            return "At Risk"

        else:
            return "Lost Customers"

    rfm["Segment"] = (
        rfm["RFM_Total"]
        .apply(assign_segment)
    )

    # =====================================================
    # CHURN DATASET
    # =====================================================

    max_date = (
        df["InvoiceDate"].max()
    )

    # Historical period ends 90 days before the latest date.
    snapshot_date = (
        max_date
        - pd.Timedelta(days=90)
    )

    # Historical transactions.
    historical = df[
        df["InvoiceDate"]
        < snapshot_date
    ].copy()

    # Future observation period.
    future = df[
        df["InvoiceDate"]
        >= snapshot_date
    ].copy()

    rfm_ml = (
        historical
        .groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x:
                (
                    snapshot_date
                    - x.max()
                ).days
            ),

            Frequency=(
                "InvoiceNo",
                "nunique"
            ),

            Monetary=(
                "TotalAmount",
                "sum"
            )
        )
        .reset_index()
    )

    if not rfm_ml.empty:

        rfm_ml["AverageOrderValue"] = (
            rfm_ml["Monetary"]
            / rfm_ml["Frequency"]
        )

        future_activity = (
            future
            .groupby("CustomerID")
            .agg(
                FutureOrders=(
                    "InvoiceNo",
                    "nunique"
                )
            )
            .reset_index()
        )

        rfm_ml = rfm_ml.merge(
            future_activity,
            on="CustomerID",
            how="left"
        )

        rfm_ml["FutureOrders"] = (
            rfm_ml["FutureOrders"]
            .fillna(0)
        )

        rfm_ml["Churn"] = (
            rfm_ml["FutureOrders"]
            == 0
        ).astype(int)

    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "AverageOrderValue"
    ]

    model_results = None

    feature_importance = None

    dt_cm = None

    rf_cm = None

    test_predictions = None

    if (
        len(rfm_ml) >= 20
        and "Churn" in rfm_ml.columns
        and rfm_ml["Churn"].nunique() == 2
        and rfm_ml["Churn"].value_counts().min() >= 2
    ):

        X = rfm_ml[
            features
        ]

        y = rfm_ml[
            "Churn"
        ]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        # -------------------------------------------------
        # DECISION TREE
        # -------------------------------------------------

        decision_tree = (
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )

        # -------------------------------------------------
        # RANDOM FOREST
        # -------------------------------------------------

        random_forest = (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        )

        decision_tree.fit(
            X_train,
            y_train
        )

        random_forest.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------

        dt_pred = (
            decision_tree.predict(
                X_test
            )
        )

        rf_pred = (
            random_forest.predict(
                X_test
            )
        )

        # -------------------------------------------------
        # MODEL RESULTS
        # -------------------------------------------------

        model_results = pd.DataFrame(
            [
                {
                    "Model":
                        "Decision Tree",

                    "Accuracy":
                        accuracy_score(
                            y_test,
                            dt_pred
                        ),

                    "Precision":
                        precision_score(
                            y_test,
                            dt_pred,
                            zero_division=0
                        ),

                    "Recall":
                        recall_score(
                            y_test,
                            dt_pred,
                            zero_division=0
                        ),

                    "F1 Score":
                        f1_score(
                            y_test,
                            dt_pred,
                            zero_division=0
                        )
                },

                {
                    "Model":
                        "Random Forest",

                    "Accuracy":
                        accuracy_score(
                            y_test,
                            rf_pred
                        ),

                    "Precision":
                        precision_score(
                            y_test,
                            rf_pred,
                            zero_division=0
                        ),

                    "Recall":
                        recall_score(
                            y_test,
                            rf_pred,
                            zero_division=0
                        ),

                    "F1 Score":
                        f1_score(
                            y_test,
                            rf_pred,
                            zero_division=0
                        )
                }
            ]
        )

        # -------------------------------------------------
        # CONFUSION MATRICES
        # -------------------------------------------------

        dt_cm = confusion_matrix(
            y_test,
            dt_pred
        )

        rf_cm = confusion_matrix(
            y_test,
            rf_pred
        )

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        feature_importance = (
            pd.DataFrame(
                {
                    "Feature":
                        features,

                    "Importance":
                        random_forest
                        .feature_importances_
                }
            )
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        # -------------------------------------------------
        # TEST PREDICTIONS
        # -------------------------------------------------

        test_predictions = (
            X_test.copy()
        )

        test_predictions["ActualChurn"] = (
            y_test.values
        )

        test_predictions["DecisionTreePrediction"] = (
            dt_pred
        )

        test_predictions["RandomForestPrediction"] = (
            rf_pred
        )

    # =====================================================
    # RETURN EVERYTHING
    # =====================================================

    return {

        "cleaned_data":
            df,

        "rfm":
            rfm,

        "churn":
            rfm_ml,

        "model_results":
            model_results,

        "feature_importance":
            feature_importance,

        "dt_cm":
            dt_cm,

        "rf_cm":
            rf_cm,

        "test_predictions":
            test_predictions,

        "original_rows":
            original_rows,

        "duplicates_removed":
            duplicates_removed,

        "missing_customer_removed":
            missing_customer_removed,

        "cancelled_removed":
            cancelled_removed,

        "invalid_quantity_removed":
            invalid_quantity_removed,

        "invalid_price_removed":
            invalid_price_removed,

        "analysis_date":
            analysis_date,

        "snapshot_date":
            snapshot_date
    }


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Intelligence Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">RFM-Based Customer Segmentation & Churn Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a transaction-level Excel dataset to analyze "
    "customer purchasing behavior, RFM segments and potential churn."
)

st.divider()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload Transaction Excel File",
    type=["xlsx"],
    help="Upload an Excel (.xlsx) transaction dataset."
)


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:

    st.session_state.results = None


if "pdf_bytes" not in st.session_state:

    st.session_state.pdf_bytes = None


if "uploaded_name" not in st.session_state:

    st.session_state.uploaded_name = None


# =========================================================
# RUN ANALYSIS
# =========================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Run Customer Intelligence Analysis",
        type="primary"
    ):

        try:

            with st.spinner(
                "Processing Excel file and building machine learning models..."
            ):

                uploaded_file.seek(0)

                df = pd.read_excel(
                    uploaded_file
                )

                results = analyze_data(
                    df
                )

                pdf_bytes = generate_pdf(
                    results
                )

                st.session_state.results = (
                    results
                )

                st.session_state.pdf_bytes = (
                    pdf_bytes
                )

                st.session_state.uploaded_name = (
                    uploaded_file.name
                )

            st.success(
                "✅ Analysis completed successfully!"
            )

        except Exception as e:

            st.session_state.results = None

            st.session_state.pdf_bytes = None

            st.error(
                "❌ Analysis failed."
            )

            st.exception(e)


# =========================================================
# GET RESULTS
# =========================================================

results = (
    st.session_state.results
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

if results is not None:

    rfm = results["rfm"]

    churn = results["churn"]

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    st.header(
        "📋 Dataset Summary"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Original Transactions",
            f'{results["original_rows"]:,}'
        )

    with col2:

        st.metric(
            "Clean Transactions",
            f'{len(results["cleaned_data"]):,}'
        )

    with col3:

        st.metric(
            "Customers",
            f'{len(rfm):,}'
        )

    with col4:

        st.metric(
            "Total Revenue",
            f'£{results["cleaned_data"]["TotalAmount"].sum():,.2f}'
        )

    # =====================================================
    # PREPROCESSING
    # =====================================================

    st.subheader(
        "Data Preprocessing"
    )

    preprocessing = pd.DataFrame(
        {
            "Processing Step": [
                "Original Rows",
                "Duplicate Rows Removed",
                "Missing CustomerID Removed",
                "Cancelled Transactions Removed",
                "Invalid Quantity Removed",
                "Invalid Price Removed",
                "Final Clean Rows"
            ],

            "Result": [
                results["original_rows"],
                results["duplicates_removed"],
                results["missing_customer_removed"],
                results["cancelled_removed"],
                results["invalid_quantity_removed"],
                results["invalid_price_removed"],
                len(results["cleaned_data"])
            ]
        }
    )

    st.dataframe(
        preprocessing,
        width="stretch",
        hide_index=True
    )

    # =====================================================
    # CUSTOMER SEGMENTATION
    # =====================================================

    st.header(
        "👥 Customer Segmentation"
    )

    segment_counts = (
        rfm["Segment"]
        .value_counts()
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.subheader(
            "Customer Segment Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        segment_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Customer Segment"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    with col2:

        st.subheader(
            "Segment Summary"
        )

        segment_summary = (
            rfm
            .groupby("Segment")
            .agg(
                Customers=(
                    "CustomerID",
                    "count"
                ),

                AverageMonetary=(
                    "Monetary",
                    "mean"
                ),

                AverageFrequency=(
                    "Frequency",
                    "mean"
                ),

                AverageRecency=(
                    "Recency",
                    "mean"
                )
            )
            .sort_values(
                "Customers",
                ascending=False
            )
        )

        st.dataframe(
            segment_summary.round(2),
            width="stretch"
        )

    # =====================================================
    # RFM ANALYSIS
    # =====================================================

    st.header(
        "📈 RFM Analysis"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.subheader(
            "Recency vs Monetary"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            rfm["Recency"],
            rfm["Monetary"],
            alpha=0.5
        )

        ax.set_xlabel(
            "Recency (Days)"
        )

        ax.set_ylabel(
            "Monetary Value (£)"
        )

        ax.set_title(
            "Recency vs Monetary Value"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    with col2:

        st.subheader(
            "Frequency vs Monetary"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            rfm["Frequency"],
            rfm["Monetary"],
            alpha=0.5
        )

        ax.set_xlabel(
            "Purchase Frequency"
        )

        ax.set_ylabel(
            "Monetary Value (£)"
        )

        ax.set_title(
            "Frequency vs Monetary Value"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    # =====================================================
    # MONETARY BY SEGMENT
    # =====================================================

    st.subheader(
        "Average Monetary Value by Segment"
    )

    monetary_segment = (
        rfm
        .groupby("Segment")["Monetary"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    monetary_segment.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Customer Segment"
    )

    ax.set_ylabel(
        "Average Monetary Value (£)"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    # =====================================================
    # CHURN ANALYSIS
    # =====================================================

    st.header(
        "⚠️ Customer Churn Analysis"
    )

    if len(churn) > 0:

        active_count = int(
            (
                churn["Churn"] == 0
            ).sum()
        )

        churn_count = int(
            (
                churn["Churn"] == 1
            ).sum()
        )

        churn_percentage = (
            churn_count
            / len(churn)
            * 100
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Customers Analyzed",
                f"{len(churn):,}"
            )

        with col2:

            st.metric(
                "Active Customers",
                f"{active_count:,}"
            )

        with col3:

            st.metric(
                "Churned Customers",
                f"{churn_count:,}"
            )

        st.info(
            f"Churn rate: **{churn_percentage:.2f}%**"
        )

        st.caption(
            "Churn definition: a customer is labelled as churned "
            "when they have no purchase during the 90-day observation period."
        )

    else:

        churn_percentage = 0

        st.warning(
            "There is not enough historical data to perform churn analysis."
        )

    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    st.header(
        "🤖 Machine Learning Results"
    )

    if results["model_results"] is not None:

        model_results = (
            results["model_results"]
        )

        display_results = (
            model_results.copy()
        )

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]:

            display_results[column] = (
                display_results[column]
                * 100
            ).round(
                2
            ).astype(
                str
            ) + "%"

        st.subheader(
            "Model Performance Comparison"
        )

        st.dataframe(
            display_results,
            width="stretch",
            hide_index=True
        )

        # -------------------------------------------------
        # MODEL ACCURACY
        # -------------------------------------------------

        st.subheader(
            "Model Accuracy"
        )

        accuracy_chart = (
            model_results
            .set_index("Model")["Accuracy"]
            * 100
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        accuracy_chart.plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel(
            "Accuracy (%)"
        )

        ax.set_ylim(
            0,
            100
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

        # -------------------------------------------------
        # CONFUSION MATRICES
        # -------------------------------------------------

        st.subheader(
            "Confusion Matrices"
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            ConfusionMatrixDisplay(
                confusion_matrix=results["dt_cm"],
                display_labels=[
                    "Active",
                    "Churned"
                ]
            ).plot(
                ax=ax
            )

            ax.set_title(
                "Decision Tree"
            )

            st.pyplot(
                fig
            )

            plt.close(fig)

        with col2:

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            ConfusionMatrixDisplay(
                confusion_matrix=results["rf_cm"],
                display_labels=[
                    "Active",
                    "Churned"
                ]
            ).plot(
                ax=ax
            )

            ax.set_title(
                "Random Forest"
            )

            st.pyplot(
                fig
            )

            plt.close(fig)

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        st.subheader(
            "Random Forest Feature Importance"
        )

        importance = (
            results["feature_importance"]
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel(
            "Customer Feature"
        )

        ax.set_ylabel(
            "Importance"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

        importance_display = (
            importance.copy()
        )

        importance_display["Importance"] = (
            importance_display["Importance"]
            * 100
        ).round(
            2
        )

        st.dataframe(
            importance_display,
            width="stretch",
            hide_index=True
        )

    else:

        st.warning(
            "Machine learning could not be trained for this dataset. "
            "More historical data or both churn classes are required."
        )

    # =====================================================
    # CUSTOMER RFM TABLE
    # =====================================================

    st.header(
        "🔎 Customer RFM Data"
    )

    st.dataframe(
        rfm.head(100),
        width="stretch"
    )

    # =====================================================
    # CSV DOWNLOAD
    # =====================================================

    csv_data = (
        rfm
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Customer RFM CSV",
        data=csv_data,
        file_name="customer_rfm_segments.csv",
        mime="text/csv"
    )

    # =====================================================
    # PDF DOWNLOAD
    # =====================================================

    st.divider()

    st.header(
        "📄 Final Report"
    )

    if st.session_state.pdf_bytes is not None:

        st.download_button(
            label="📥 Download 1-Page PDF Report",
            data=st.session_state.pdf_bytes,
            file_name="Customer_Intelligence_Report.pdf",
            mime="application/pdf"
        )

        st.caption(
            "The PDF report is generated from the uploaded Excel dataset."
        )

    # =====================================================
    # FINAL STATUS
    # =====================================================

    st.divider()

    st.success(
        "✅ Customer Intelligence Analysis Completed"
    )

    st.caption(
        "Customer Intelligence Engine | "
        "Python • Pandas • NumPy • Scikit-learn • "
        "Matplotlib • SQLite"
    )
# =========================================================
# INITIAL PAGE
# =========================================================

else:

    st.info(
        "👆 Upload an Excel transaction dataset above "
        "to begin the analysis."
    )

    st.markdown(
        """
        ### 📌 Expected Excel Format

        Your transaction file should contain:

        `InvoiceNo` | `StockCode` | `Description` | `Quantity` |
        `InvoiceDate` | `UnitPrice` | `CustomerID` | `Country`
        """
    )

    # ---------------------------------------------------------
    # SAMPLE DATASET DOWNLOAD
    # ---------------------------------------------------------

    st.subheader("🧪 Try the Project with Sample Data")

    st.write(
        "Don't have an Excel dataset? Download the sample "
        "Online Retail dataset and upload it above."
    )

    try:

        with open(
            "data/Sample_Online_Retail.xlsx",
            "rb"
        ) as sample_file:

            st.download_button(
                label="⬇️ Download Sample Excel Dataset",
                data=sample_file,
                file_name="Sample_Online_Retail.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )

    except FileNotFoundError:

        st.warning(
            "Sample dataset is not available."
        )

    st.markdown(
        """
        ### 🔬 Analysis Pipeline

        **Excel Upload**
        → **Preprocessing**
        → **RFM Analysis**
        → **Customer Segmentation**
        → **Churn Analysis**
        → **Decision Tree**
        → **Random Forest**
        → **Model Evaluation**
        → **Graphs**
        → **CSV Report**
        → **PDF Report**
        """
    )