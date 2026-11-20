"""Forecast next-day order volume from historical UCI Online Retail invoices."""

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import matplotlib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import dates as mdates, pyplot as plt, ticker


DAY = timedelta(days=1)
WEEK = 7
HOLDOUT_DAYS = 28
SOURCE_COLUMNS = {"InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"}


def daily_order_counts(path):
    """Count distinct paid invoices by day on a continuous calendar."""
    invoice_dates = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = SOURCE_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            invoice = (row["InvoiceNo"] or "").strip()
            if len(invoice) != 6 or not invoice.isascii() or not invoice.isdigit():
                continue
            try:
                quantity = int(row["Quantity"])
                price = float(row["UnitPrice"])
                if quantity <= 0 or not 0 < price < float("inf"):
                    continue
                day = datetime.strptime(row["InvoiceDate"], "%m/%d/%Y %H:%M").date()
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid quantity, price, or date on CSV line {reader.line_num}") from error
            invoice_dates[invoice] = min(day, invoice_dates.get(invoice, day))

    if not invoice_dates:
        raise ValueError("CSV contains no eligible invoices")

    counts = Counter(invoice_dates.values())
    # The first and last dates in the source may be partial days.
    first, last = min(counts) + DAY, max(counts) - DAY
    days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
    return days, [counts[day] for day in days]


def order_features(day, previous_week):
    """Build the inputs available just before the target day begins."""
    return {
        "weekday": str(day.weekday()),
        "orders_yesterday": previous_week[-1],
        "orders_last_week": previous_week[0],
        "average_orders_previous_week": mean(previous_week),
    }


def save_chart(path, days, actual, ridge, baseline):
    figure, axes = plt.subplots(figsize=(10, 4.4), layout="constrained")
    axes.set_title("Next-day order volume · 28-day backtest", loc="left", fontsize=16, pad=26)
    axes.plot(days, actual, label="Actual", color="#24364a", linewidth=2.2)
    axes.plot(days, ridge, label="Ridge forecast", color="#007b70", linewidth=1.8, marker="o", markersize=3)
    axes.plot(days, baseline, label="Same weekday last week", color="#a45220", linewidth=1.7, linestyle="--")
    axes.set_ylabel("Orders per day")
    axes.set_ylim(bottom=0)
    axes.grid(axis="y", color="#d9e1e8", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)
    axes.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    axes.legend(loc="upper left", bbox_to_anchor=(0, 1.11), ncols=3, frameon=False, borderaxespad=0)
    figure.savefig(path, dpi=130, facecolor="white")
    plt.close(figure)


def run_backtest(input_path, output_dir):
    csv_path = output_dir / "backtest.csv"
    chart_path = output_dir / "backtest.png"
    if input_path.resolve() in {csv_path.resolve(), chart_path.resolve()}:
        raise ValueError("output files must differ from the input CSV")
    days, counts = daily_order_counts(input_path)
    minimum_days = WEEK + HOLDOUT_DAYS + WEEK
    if len(days) < minimum_days:
        raise ValueError(f"need at least {minimum_days} complete calendar days for this backtest")

    features = [order_features(days[index], counts[index - WEEK:index]) for index in range(WEEK, len(days))]
    targets = counts[WEEK:]
    test_days, test_actual = days[-HOLDOUT_DAYS:], targets[-HOLDOUT_DAYS:]
    test_features = features[-HOLDOUT_DAYS:]
    baseline = [values["orders_last_week"] for values in test_features]

    model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), Ridge(alpha=1.0))
    model.fit(features[:-HOLDOUT_DAYS], targets[:-HOLDOUT_DAYS])
    predictions = [max(0.0, float(value)) for value in model.predict(test_features)]
    baseline_mae = mean_absolute_error(test_actual, baseline)
    ridge_mae = mean_absolute_error(test_actual, predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("date", "actual_orders", "ridge_forecast", "same_weekday_last_week"))
        writer.writerows(zip(test_days, test_actual, (f"{value:.4f}" for value in predictions), baseline, strict=True))
    save_chart(chart_path, test_days, test_actual, predictions, baseline)

    print(f"Complete calendar: {days[0]} to {days[-1]} ({len(days)} days, {sum(counts):,} orders)")
    print(f"Training: {len(features) - HOLDOUT_DAYS} days. Test: {test_days[0]} to {test_days[-1]}.")
    print("Mean absolute error (orders/day)")
    print(f"  Same weekday last week: {baseline_mae:.2f}")
    print(f"  Ridge forecast:         {ridge_mae:.2f}")
    print(f"Wrote {csv_path} and {chart_path}")

    # Fit on every complete day only after recording the held-out evaluation.
    next_day = days[-1] + DAY
    model.fit(features, targets)
    next_forecast = model.predict([order_features(next_day, counts[-WEEK:])])[0]
    print(f"Example for {next_day}: weekly baseline {counts[-WEEK]} orders, "
          f"Ridge {max(0, round(float(next_forecast)))} orders")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/online-retail.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    try:
        run_backtest(args.input, args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
