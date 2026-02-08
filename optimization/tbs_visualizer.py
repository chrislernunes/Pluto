# tbs_visualizer.py

from visualizer_generic import *

def visualize_tbs_results(metrics_file="reports/optimization/tbs_optimization_results.csv"):
    metrics_df = load_metrics(metrics_file)

    print("Top Configs by Calmar Ratio:")
    print(top_configs(metrics_df, "calmar_ratio", top_n=10))

    print("\nTop Configs by Sharpe Ratio:")
    print(top_configs(metrics_df, "sharpe_ratio", top_n=10))

    # Plot individual param performance
    plot_metric_by_param(metrics_df, "sl_pct", "total_pnl")
    plot_metric_by_param(metrics_df, "sl_pct", "calmar_ratio")
    plot_metric_by_param(metrics_df, "selector_val", "return_on_margin")

    # Slice-wise performance (e.g., underlying)
    plot_metric_by_slice(metrics_df, "underlying", "total_pnl")
    plot_metric_by_slice(metrics_df, "underlying", "sharpe_ratio")

    # Combined slice view (if multiple slices used)
    if "session" in metrics_df.columns and "underlying" in metrics_df.columns:
        compare_slices(metrics_df, ["session", "underlying"], "calmar_ratio")
        compare_slices(metrics_df, ["session", "underlying"], "return_on_margin")

# Run if needed
if __name__ == "__main__":
    visualize_tbs_results()
