"""Generate a downloadable exploratory data analysis report as a PDF."""

from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from eda import (
    category_popularity,
    customer_activity_distribution,
    kpi_summary,
    most_purchased_products,
    most_viewed_products,
    revenue_by_category,
    revenue_by_month,
    revenue_trend_daily,
    top_customers,
)
from regression import load_or_train as load_regression_model


PRIMARY = colors.HexColor("#1F4E78")
ACCENT = colors.HexColor("#2E86AB")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_GREY = colors.HexColor("#F3F4F6")
TEXT = colors.HexColor("#243447")
CHART_COLORS = [
    colors.HexColor("#2E86AB"),
    colors.HexColor("#F18F01"),
    colors.HexColor("#6A994E"),
    colors.HexColor("#9B5DE5"),
    colors.HexColor("#E63946"),
    colors.HexColor("#00A6A6"),
    colors.HexColor("#FFB703"),
    colors.HexColor("#64748B"),
]


class HorizontalBarChart(Flowable):
    """Small dependency-free bar chart for the PDF report."""

    def __init__(self, labels, values, value_formatter, width=170 * mm):
        self.labels = [str(label) for label in labels]
        self.values = [float(value) for value in values]
        self.value_formatter = value_formatter
        self.width = width
        self.height = max(34 * mm, (len(self.labels) * 8 + 12) * mm / 3)

    def draw(self):
        canvas = self.canv
        label_width = 43 * mm
        value_width = 28 * mm
        bar_width = self.width - label_width - value_width
        row_height = (self.height - 8) / max(len(self.labels), 1)
        maximum = max(self.values, default=0)

        for index, (label, value) in enumerate(zip(self.labels, self.values)):
            y = self.height - ((index + 1) * row_height)
            canvas.setFillColor(TEXT)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(0, y + 2, label[:24])

            canvas.setFillColor(LIGHT_GREY)
            canvas.rect(label_width, y, bar_width, max(row_height - 4, 3), fill=1, stroke=0)
            if maximum > 0:
                filled_width = bar_width * value / maximum
                canvas.setFillColor(ACCENT)
                canvas.rect(
                    label_width, y, filled_width, max(row_height - 4, 3),
                    fill=1, stroke=0,
                )

            canvas.setFillColor(TEXT)
            canvas.drawRightString(
                self.width, y + 2, self.value_formatter(value)
            )


class PieChart(Flowable):
    """Pie chart with a compact legend suitable for PDF output."""

    def __init__(self, labels, values, width=170 * mm):
        pairs = [
            (str(label), float(value))
            for label, value in zip(labels, values)
            if float(value) > 0
        ]
        self.labels = [pair[0] for pair in pairs]
        self.values = [pair[1] for pair in pairs]
        self.width = width
        self.height = 58 * mm

    def draw(self):
        canvas = self.canv
        total = sum(self.values)
        if total <= 0:
            return

        diameter = 48 * mm
        x = 6 * mm
        y = 5 * mm
        angle = 90
        canvas.setLineWidth(0.5)
        canvas.setStrokeColor(colors.white)

        for index, value in enumerate(self.values):
            extent = 360 * value / total
            canvas.setFillColor(CHART_COLORS[index % len(CHART_COLORS)])
            canvas.wedge(
                x, y, x + diameter, y + diameter, angle, extent,
                fill=1, stroke=1,
            )
            angle += extent

        legend_x = 66 * mm
        legend_y = self.height - 8 * mm
        canvas.setFont("Helvetica", 8)
        for index, (label, value) in enumerate(zip(self.labels, self.values)):
            row_y = legend_y - index * 6 * mm
            canvas.setFillColor(CHART_COLORS[index % len(CHART_COLORS)])
            canvas.rect(legend_x, row_y - 2, 8, 8, fill=1, stroke=0)
            canvas.setFillColor(TEXT)
            canvas.drawString(
                legend_x + 12,
                row_y,
                f"{label[:22]} ({value / total:.1%})",
            )


class ActualPredictedChart(Flowable):
    """Scatter plot comparing actual and predicted values."""

    def __init__(self, actual, predicted, width=165 * mm):
        pairs = list(zip(actual, predicted))
        step = max(1, len(pairs) // 600)
        sampled = pairs[::step][:600]
        self.actual = [float(pair[0]) for pair in sampled]
        self.predicted = [float(pair[1]) for pair in sampled]
        self.width = width
        self.height = 72 * mm

    def draw(self):
        canvas = self.canv
        if not self.actual:
            return

        left = 18 * mm
        bottom = 12 * mm
        chart_width = self.width - 24 * mm
        chart_height = self.height - 18 * mm
        lower = min(self.actual + self.predicted)
        upper = max(self.actual + self.predicted)
        value_range = max(upper - lower, 1)

        def x_position(value):
            return left + ((value - lower) / value_range) * chart_width

        def y_position(value):
            return bottom + ((value - lower) / value_range) * chart_height

        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.rect(left, bottom, chart_width, chart_height, fill=0, stroke=1)
        canvas.setStrokeColor(colors.HexColor("#E63946"))
        canvas.setDash(4, 3)
        canvas.line(left, bottom, left + chart_width, bottom + chart_height)
        canvas.setDash()

        canvas.setFillColor(ACCENT)
        for actual, predicted in zip(self.actual, self.predicted):
            canvas.circle(
                x_position(actual), y_position(predicted), 1.1,
                fill=1, stroke=0,
            )

        canvas.setFillColor(TEXT)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(
            left + chart_width / 2, 2 * mm, "Actual price (INR)"
        )
        canvas.saveState()
        canvas.translate(4 * mm, bottom + chart_height / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "Predicted price (INR)")
        canvas.restoreState()
        canvas.drawString(left, bottom - 4 * mm, _number(lower))
        canvas.drawRightString(
            left + chart_width, bottom - 4 * mm, _number(upper)
        )


def _currency(value: float) -> str:
    return f"INR {value:,.2f}"


def _number(value: float) -> str:
    return f"{value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _safe(value) -> str:
    return escape(str(value))


def _table(data, column_widths=None, header=True):
    table = Table(data, colWidths=column_widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT_GREY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    table.setStyle(TableStyle(commands))
    return table


def _page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(20 * mm, 9 * mm, "Retail Sales Analytics - EDA Report")
    canvas.drawRightString(190 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_eda_pdf(df: pd.DataFrame) -> bytes:
    """Build a complete EDA report from the featured retail dataset."""
    if df.empty:
        raise ValueError("Cannot generate an EDA report from an empty dataset.")

    required = {
        "event_time", "event_type", "price", "user_id", "product_id",
        "category_main", "brand", "hour", "is_weekend",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "EDA report requires these missing columns: " + ", ".join(sorted(missing))
        )

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Retail Sales Exploratory Data Analysis Report",
        author="Retail Sales Analytics Dashboard",
        pageCompression=0,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        textColor=PRIMARY,
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#64748B"),
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        textColor=PRIMARY,
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        textColor=TEXT,
        fontSize=9,
        leading=13,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Insight",
        parent=styles["BodyText"],
        textColor=TEXT,
        backColor=LIGHT_BLUE,
        borderColor=ACCENT,
        borderWidth=0.5,
        borderPadding=7,
        fontSize=9,
        leading=13,
        spaceAfter=7,
    ))

    story = []
    kpis = kpi_summary(df)
    purchases = df[df["event_type"] == "purchase"]
    event_counts = df["event_type"].value_counts()
    category_revenue = revenue_by_category(df)
    monthly_revenue = revenue_by_month(df)
    daily_revenue = revenue_trend_daily(df)
    customers = top_customers(df, 10)
    viewed_products = most_viewed_products(df, 10)
    purchased_products = most_purchased_products(df, 10)
    category_events = category_popularity(df)
    activity = customer_activity_distribution(df)
    regression = load_regression_model(df)

    start_date = df["event_time"].min().date()
    end_date = df["event_time"].max().date()
    purchase_rate = len(purchases) / len(df)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    purchasing_customer_counts = purchases.groupby("user_id").size()
    repeat_rate = (
        float((purchasing_customer_counts > 1).mean())
        if not purchasing_customer_counts.empty else 0.0
    )

    top_category = category_revenue.iloc[0] if not category_revenue.empty else None
    best_month = monthly_revenue.loc[monthly_revenue["revenue"].idxmax()]
    peak_day = daily_revenue.loc[daily_revenue["revenue"].idxmax()]
    hourly_revenue = (
        purchases.groupby("hour")["price"].sum().sort_values(ascending=False)
    )
    peak_hour = int(hourly_revenue.index[0]) if not hourly_revenue.empty else 0
    weekend_revenue_share = (
        purchases.loc[purchases["is_weekend"], "price"].sum()
        / max(purchases["price"].sum(), 1)
    )

    story.extend([
        Spacer(1, 10 * mm),
        Paragraph("Retail Sales Exploratory Data Analysis", styles["ReportTitle"]),
        Paragraph(
            "<b>Report version 2.0</b><br/>"
            f"Data period: {_safe(start_date)} to {_safe(end_date)}<br/>"
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            styles["ReportSubtitle"],
        ),
        Paragraph("Executive Summary", styles["Section"]),
        _table([
            ["Metric", "Value", "Metric", "Value"],
            ["Total Revenue", _currency(kpis["total_revenue"]),
             "Purchase Events", f"{kpis['total_purchases']:,}"],
            ["Unique Customers", f"{kpis['total_customers']:,}",
             "Unique Products", f"{kpis['total_products']:,}"],
            ["Average Order Value", _currency(kpis["avg_order_value"]),
             "Purchase Rate", _percent(purchase_rate)],
        ], [42 * mm, 43 * mm, 42 * mm, 43 * mm]),
        Spacer(1, 5 * mm),
    ])

    if top_category is not None:
        category_share = top_category["revenue"] / max(kpis["total_revenue"], 1)
        story.append(Paragraph(
            f"<b>Headline:</b> {_safe(top_category['category_main']).title()} is the "
            f"leading category with {_currency(top_category['revenue'])} in revenue "
            f"({_percent(category_share)} of total revenue). The strongest month was "
            f"{_safe(best_month['period'])}, and peak daily revenue occurred on "
            f"{_safe(peak_day['date'])}.",
            styles["Insight"],
        ))

    story.extend([
        Paragraph("1. Dataset and Data Quality", styles["Section"]),
        Paragraph(
            "The report uses the cleaned, feature-engineered event dataset currently "
            "loaded by the dashboard. Missing categorical values are labeled as "
            "'unknown', duplicates are removed, timestamps are parsed, and invalid "
            "purchase prices are excluded before analysis.",
            styles["BodySmall"],
        ),
        _table([
            ["Quality Measure", "Result"],
            ["Rows", f"{len(df):,}"],
            ["Columns", f"{df.shape[1]:,}"],
            ["Date coverage", f"{start_date} to {end_date}"],
            ["Missing cells after cleaning", f"{missing_values:,}"],
            ["Duplicate rows after cleaning", f"{duplicate_rows:,}"],
            ["Unique categories", f"{df['category_main'].nunique():,}"],
            ["Unique brands", f"{df['brand'].nunique():,}"],
        ], [85 * mm, 85 * mm]),
        Spacer(1, 4 * mm),
        Paragraph("Event Distribution", styles["Section"]),
        PieChart(
            [name.title() for name in event_counts.index],
            event_counts.values,
        ),
        Paragraph(
            f"Purchases represent {_percent(purchase_rate)} of all interactions. "
            "The difference between browsing, cart activity, and completed purchases "
            "is the primary conversion opportunity.",
            styles["Insight"],
        ),
        PageBreak(),
        Paragraph("2. Sales and Revenue Analysis", styles["Section"]),
    ])

    price_summary = purchases["price"].describe(percentiles=[0.25, 0.5, 0.75])
    story.extend([
        _table([
            ["Purchase Price Statistic", "Value"],
            ["Mean", _currency(price_summary["mean"])],
            ["Median", _currency(price_summary["50%"])],
            ["Standard deviation", _currency(price_summary["std"])],
            ["Minimum", _currency(price_summary["min"])],
            ["25th percentile", _currency(price_summary["25%"])],
            ["75th percentile", _currency(price_summary["75%"])],
            ["Maximum", _currency(price_summary["max"])],
        ], [85 * mm, 85 * mm]),
        Spacer(1, 5 * mm),
        Paragraph("Revenue by Category", styles["Section"]),
        HorizontalBarChart(
            category_revenue.head(8)["category_main"].str.title(),
            category_revenue.head(8)["revenue"],
            _currency,
        ),
        Spacer(1, 3 * mm),
        Paragraph("Category Revenue Share", styles["Section"]),
        PieChart(
            category_revenue.head(8)["category_main"].str.title(),
            category_revenue.head(8)["revenue"],
        ),
    ])

    category_rows = [["Category", "Revenue", "Share", "Events"]]
    event_lookup = category_events.set_index("category_main")["event_count"]
    for row in category_revenue.head(10).itertuples():
        category_rows.append([
            str(row.category_main).title(),
            _currency(row.revenue),
            _percent(row.revenue / max(kpis["total_revenue"], 1)),
            f"{int(event_lookup.get(row.category_main, 0)):,}",
        ])
    story.extend([
        Spacer(1, 4 * mm),
        _table(category_rows, [54 * mm, 46 * mm, 30 * mm, 40 * mm]),
        Spacer(1, 5 * mm),
        Paragraph("Monthly Revenue", styles["Section"]),
        _table(
            [["Month", "Revenue"]] + [
                [str(row.period), _currency(row.revenue)]
                for row in monthly_revenue.itertuples()
            ],
            [85 * mm, 85 * mm],
        ),
        Paragraph(
            f"The best month was <b>{_safe(best_month['period'])}</b> with "
            f"<b>{_currency(best_month['revenue'])}</b>. The highest-revenue hour "
            f"was <b>{peak_hour:02d}:00</b>, while weekends contributed "
            f"<b>{_percent(weekend_revenue_share)}</b> of revenue.",
            styles["Insight"],
        ),
        PageBreak(),
        Paragraph("3. Customer Analysis", styles["Section"]),
        Paragraph(
            f"Customers generated an average of {activity.mean():.2f} events each. "
            f"Among customers who purchased, {_percent(repeat_rate)} made more than "
            "one purchase in the observed period.",
            styles["BodySmall"],
        ),
    ])

    customer_rows = [["Customer ID", "Total Spend", "Purchases"]]
    for row in customers.itertuples():
        customer_rows.append([
            str(row.user_id),
            _currency(row.total_spend),
            f"{int(row.num_purchases):,}",
        ])
    story.extend([
        _table(customer_rows, [60 * mm, 65 * mm, 45 * mm]),
        Spacer(1, 5 * mm),
        Paragraph("4. Product Analysis", styles["Section"]),
    ])

    product_rows = [["Rank", "Most Viewed Product", "Views",
                     "Most Purchased Product", "Purchases"]]
    product_count = max(len(viewed_products), len(purchased_products))
    for index in range(product_count):
        viewed = viewed_products.iloc[index] if index < len(viewed_products) else None
        bought = (
            purchased_products.iloc[index]
            if index < len(purchased_products) else None
        )
        product_rows.append([
            str(index + 1),
            str(int(viewed["product_id"])) if viewed is not None else "-",
            f"{int(viewed['view_count']):,}" if viewed is not None else "-",
            str(int(bought["product_id"])) if bought is not None else "-",
            f"{int(bought['purchase_count']):,}" if bought is not None else "-",
        ])
    story.extend([
        _table(product_rows, [15 * mm, 50 * mm, 25 * mm, 55 * mm, 25 * mm]),
        Spacer(1, 5 * mm),
        PageBreak(),
        Paragraph("5. Predictive Analysis", styles["Section"]),
        Paragraph(
            "A linear regression model estimates product price using hour, weekday, "
            "encoded category and brand, and category-level mean price. The dataset "
            "is split into training and test sets so the metrics below measure "
            "performance on records the model did not train on.",
            styles["BodySmall"],
        ),
        _table([
            ["Metric", "Value", "Interpretation"],
            ["MAE", _currency(regression["mae"]),
             "Average absolute prediction error"],
            ["RMSE", _currency(regression["rmse"]),
             "Error measure that penalizes large misses"],
            ["R-squared", f"{regression['r2']:.4f}",
             "Share of price variation explained"],
            ["Training records", f"{regression['n_train']:,}",
             "Records used to fit the model"],
            ["Test records", f"{regression['n_test']:,}",
             "Held-out records used for evaluation"],
        ], [35 * mm, 38 * mm, 97 * mm]),
        Spacer(1, 4 * mm),
        Paragraph("Actual vs. Predicted Prices", styles["Section"]),
        ActualPredictedChart(regression["y_test"], regression["y_pred"]),
        Paragraph(
            "Points near the red diagonal indicate accurate predictions. Wider "
            "dispersion shows uncertainty and product-price patterns that a simple "
            "linear model cannot fully capture. Predictions should be treated as "
            "estimates rather than exact prices.",
            styles["Insight"],
        ),
        Paragraph("6. Key Findings and Recommendations", styles["Section"]),
    ])

    recommendations = [
        (
            "Improve conversion",
            f"The purchase rate is {_percent(purchase_rate)}. Prioritize cart recovery, "
            "checkout simplification, and retargeting experiments.",
        ),
        (
            "Protect category availability",
            f"{str(top_category['category_main']).title()} leads revenue. Maintain stock "
            "coverage and focus supplier negotiations on this category.",
        ),
        (
            "Plan around peak periods",
            f"Revenue peaked in {best_month['period']} and at {peak_hour:02d}:00. "
            "Align campaigns, staffing, and inventory replenishment with these periods.",
        ),
        (
            "Develop customer retention",
            f"Only {_percent(repeat_rate)} of purchasing customers bought more than once. "
            "Use loyalty incentives and post-purchase messaging to encourage repeat orders.",
        ),
        (
            "Validate with controlled experiments",
            "EDA identifies associations rather than causes. Test promotions and UX "
            "changes with measurable A/B experiments before broad rollout.",
        ),
    ]
    for title, text in recommendations:
        story.append(Paragraph(
            f"<b>{_safe(title)}:</b> {_safe(text)}",
            styles["Insight"],
        ))

    story.extend([
        Spacer(1, 4 * mm),
        Paragraph("Methodology and Scope", styles["Section"]),
        Paragraph(
            "This report presents descriptive exploratory analysis of event-level "
            "e-commerce data. Revenue is the sum of price for purchase events. "
            "Percentages use the cleaned dataset as their denominator. Results should "
            "be interpreted in the context of the available date range and data source; "
            "synthetic fallback data is suitable for demonstration but not for making "
            "real commercial decisions.",
            styles["BodySmall"],
        ),
    ])

    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return output.getvalue()
