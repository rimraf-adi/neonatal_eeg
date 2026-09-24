import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

PLOTLY_TEMPLATE = "plotly_white"
AXIS_CONFIG = dict(
    xaxis=dict(
        title_font=dict(color="black", size=18),
        tickfont=dict(color="black", size=14),
        color="black",
        gridcolor="#e0e0e0",
    ),
    yaxis=dict(
        title_font=dict(color="black", size=18),
        tickfont=dict(color="black", size=14),
        color="black",
        gridcolor="#e0e0e0",
    ),
)
LAYOUT_CONFIG = dict(
    template=PLOTLY_TEMPLATE,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", size=14),
    **AXIS_CONFIG
)

def save_plot_chart(df, x_col, y_col, title, filename, color_col=None, color_scale="OrRd", hlines=None):
    if df.empty:
        return

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col if color_col else y_col,
        color_continuous_scale=color_scale,
        labels={y_col: y_col}
    )
    
    if hlines:
        for y_val, color, text in hlines:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, annotation_text=text)

    fig.update_layout(xaxis_tickangle=-90, **LAYOUT_CONFIG)
    
    try:
        pio.write_image(fig, filename, width=1200, height=800, scale=2)
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

def create_wavelet_plots():
    results_dir = os.path.join("legacy", "ttest_results")
    summary_csv = os.path.join(results_dir, "wavelet_ttest_summary.csv")
    
    if not os.path.exists(summary_csv):
        print(f"Summary CSV not found at {summary_csv}")
        return
        
    df = pd.read_csv(summary_csv)
    
    # 1. Significance plot
    df_p = df.rename(columns={'Mean_neg_log10p': '-log10p'}).sort_values("-log10p", ascending=False)
    save_plot_chart(
        df_p,
        x_col="Feature",
        y_col="-log10p",
        title="Feature Significance: -log10(p) (wavelet)",
        filename=os.path.join(results_dir, "wavelet_significance.png"),
        color_scale="OrRd",
        hlines=None
    )
    
    # 2. T-stat plot
    df_t = df.rename(columns={'Mean_t_stat': 't_stat'}).sort_values("t_stat", ascending=False)
    save_plot_chart(
        df_t,
        x_col="Feature",
        y_col="t_stat",
        title="Feature Direction: T-Statistic (wavelet)",
        filename=os.path.join(results_dir, "wavelet_t_stat.png"),
        color_scale="RdBu",
        color_col="t_stat"
    )

if __name__ == "__main__":
    create_wavelet_plots()
