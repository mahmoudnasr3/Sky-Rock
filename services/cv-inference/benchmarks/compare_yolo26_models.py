from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RUNS = {
    "YOLO26 Small": Path("runs/detect/skyrock_yolo26s_300e/results.csv"),
    "YOLO26 Medium": Path("runs/detect/skyrock_yolo26m_300e/results.csv"),
    "YOLO26 Large": Path("runs/detect/skyrock_yolo26l_300e/results.csv"),
}

REPORTS_DIR = Path("reports/model_comparison")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    frames = []

    for model_name, csv_path in RUNS.items():
        if not csv_path.exists():
            print(f"Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df["model"] = model_name

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No results.csv files found.")

    return pd.concat(frames, ignore_index=True)


def find_column(df, keywords):
    for col in df.columns:
        normalized = col.lower().replace(" ", "")
        if all(keyword in normalized for keyword in keywords):
            return col

    return None


def plot_metric(df, metric_col, filename, title, ylabel):
    plt.figure(figsize=(10, 6))

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        plt.plot(model_df["epoch"], model_df[metric_col], label=model_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=200)
    plt.close()


def create_final_summary(df):
    metric_columns = {
        "mAP50": find_column(df, ["map50"]),
        "mAP50-95": find_column(df, ["map50-95"]),
        "precision": find_column(df, ["precision"]),
        "recall": find_column(df, ["recall"]),
    }

    rows = []

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        last_row = model_df.iloc[-1]

        row = {
            "model": model_name,
            "epochs_completed": int(last_row["epoch"]) + 1,
        }

        for metric_name, col in metric_columns.items():
            if col is not None:
                row[metric_name] = float(last_row[col])

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS_DIR / "final_model_summary.csv", index=False)

    return summary


def main():
    df = load_results()

    map50_col = find_column(df, ["map50"])
    map5095_col = find_column(df, ["map50-95"])
    precision_col = find_column(df, ["precision"])
    recall_col = find_column(df, ["recall"])

    if map50_col:
        plot_metric(df, map50_col, "map50_comparison.png", "mAP50 Comparison", "mAP50")

    if map5095_col:
        plot_metric(df, map5095_col, "map50_95_comparison.png", "mAP50-95 Comparison", "mAP50-95")

    if precision_col:
        plot_metric(df, precision_col, "precision_comparison.png", "Precision Comparison", "Precision")

    if recall_col:
        plot_metric(df, recall_col, "recall_comparison.png", "Recall Comparison", "Recall")

    summary = create_final_summary(df)

    print("Model comparison completed.")
    print(summary)
    print(f"Reports saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()