import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

KEY_METRICS = ["total_pnl", "return_on_margin", "calmar_ratio", "sharpe_ratio", "max_drawdown", "avg_profit_per_day"]

def load_metrics(metrics_file: str) -> pd.DataFrame:
    return pd.read_csv(metrics_file)

def plot_metric_by_param(metrics_df: pd.DataFrame, param: str, metric: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_df, x=param, y=metric)
    plt.title(f"{metric} by {param}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{metric}_by_{param}.png")
    plt.savefig(filename)
    plt.close()

def plot_metric_by_slice(metrics_df: pd.DataFrame, slice_var: str, metric: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_df, x=slice_var, y=metric)
    plt.title(f"{metric} by slice: {slice_var}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{metric}_by_{slice_var}.png")
    plt.savefig(filename)
    plt.close()

def aggregate_metrics(metrics_df: pd.DataFrame, groupby_vars: list, metric: str, aggfunc="mean") -> pd.DataFrame:
    return metrics_df.groupby(groupby_vars)[metric].agg(aggfunc).reset_index()

def top_configs(metrics_df: pd.DataFrame, metric: str, top_n=10, ascending=False) -> pd.DataFrame:
    return metrics_df.sort_values(by=metric, ascending=ascending).head(top_n)

def compare_slices(metrics_df: pd.DataFrame, slice_vars: list, metric: str, output_dir: str = "plots", aggfunc="mean"):
    os.makedirs(output_dir, exist_ok=True)
    grouped = metrics_df.groupby(slice_vars)[metric].agg(aggfunc).reset_index()
    pivoted = grouped.pivot(index=slice_vars[0], columns=slice_vars[1], values=metric)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivoted, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"{metric} across slices: {slice_vars[0]} x {slice_vars[1]}")
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{metric}_slice_{slice_vars[0]}_x_{slice_vars[1]}.png")
    plt.savefig(filename)
    plt.close()

def summarize_metrics_by_param(metrics_df: pd.DataFrame, slice_var: str, param: str, key_metrics: list = KEY_METRICS) -> dict:
    summary_by_slice = {}
    for slice_value in metrics_df[slice_var].unique():
        filtered = metrics_df[metrics_df[slice_var] == slice_value]
        if not filtered.empty:
            grouped = filtered.groupby(param)[key_metrics].mean().reset_index()
            summary_by_slice[slice_value] = grouped.sort_values(by=key_metrics[0], ascending=False)
    return summary_by_slice

def display_summary_tables(summary_dict: dict, slice_var: str, param: str):
    for slice_val, df in summary_dict.items():
        print(f"\n===== {slice_var.upper()} = {slice_val} | Param = {param} =====")
        print(df.to_string(index=False))
