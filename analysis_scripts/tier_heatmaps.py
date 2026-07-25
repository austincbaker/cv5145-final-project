"""Generate per-tier heatmaps: Basic, Compound, Detailed -- baseline and role_graph."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

CSV = "combined_results/baseline_accuracy_summary.csv"
df = pd.read_csv(CSV, header=2)
df = df.rename(columns={df.columns[0]: "Model"})
df = df[df["Model"].notna() & (df["Model"] != "") & df["Model"].str.strip().astype(bool)]

for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce")

BASELINE_MODELS = {
    "gemma-4-26B-A4B-it": "Gemma-4-26B",
    "InternVideo2.5-8B": "InternVideo2.5-8B",
    "InternVL2.5-8B": "InternVL2.5-8B",
    "InternVL3-9B": "InternVL3-9B",
    "InternVL3.5-8B": "InternVL3.5-8B",
    "LLaVA-Video-7B-Qwen2": "LLaVA-Video-7B",
    "Ovis2.5-9B Austin": "Ovis2.5-9B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "VideoLLaMA3-7B": "VideoLLaMA3-7B",
    "Qwen2.5-VL-72B-Instruct-AWQ": "Qwen2.5-VL-72B",
    "gpt": "GPT5.1",
    "Qwen3-VL-8B-Thinking vllm": "Qwen3-VL-8B-Think",
    "InternVL3.5-8B detailed cot prompt": "InternVL3.5-8B-CoT",
    "InternVL3.5-8B dream of thoughts": "InternVL3.5-8B-DoT",
}

ROLE_GRAPH_MODELS = {"gemma-4-26B-A4B-it": "Gemma-4-26B", "Ovis2.5-9B": "Ovis2.5-9B"}
rg_idx = df[df["Model"].str.strip() == "Role prompt"].index
if len(rg_idx) > 0:
    rg_df = df.loc[rg_idx[0] + 1:].copy()
    rg_df = rg_df[rg_df["Primary Action"].notna()]
    rg_df = rg_df[rg_df["Model"].isin(ROLE_GRAPH_MODELS)]
    rg_df["Model"] = rg_df["Model"].map(ROLE_GRAPH_MODELS)
else:
    rg_df = pd.DataFrame()

rg_cutoff = rg_idx[0] if len(rg_idx) > 0 else len(df)
baseline_df = df.loc[:rg_cutoff - 1].copy()
baseline_df = baseline_df[baseline_df["Model"].isin(BASELINE_MODELS)]
baseline_df = baseline_df[baseline_df["Primary Action"].notna()]
baseline_df["Model"] = baseline_df["Model"].map(BASELINE_MODELS)

TIERS = {
    "Basic": {
        "cols": ["Primary Action", "Role ID", "Aggressor ID", "Victim Recognition"],
        "labels": ["Action", "Role ID", "Aggressor ID", "Victim"],
    },
    "Compound": {
        "cols": ["Action+Aggressors", "Action+Victims", "Aggressor+Victim"],
        "labels": ["Action+Aggressor", "Action+Victim", "Aggressor+Victim"],
    },
    "Detailed": {
        "cols": ["Aggressor+Action+Victim", "Sequence Verification"],
        "labels": ["Agg+Action+Victim", "Sequence Verify"],
    },
}


def make_heatmap(data_df, tier_name, labels, cols, suffix=""):
    data = data_df[["Model"] + cols].copy()
    data = data.dropna(subset=cols, how="all")
    data = data.set_index("Model")
    data.columns = labels

    if len(data) == 0:
        print("No data for %s%s, skipping" % (tier_name, suffix))
        return

    row_means = data.mean(axis=1)
    data = data.loc[row_means.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), max(4, len(data) * 0.45)))
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto", vmin=10, vmax=70)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)

    for i in range(len(data)):
        for j in range(len(labels)):
            val = data.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 25 or val > 60 else "black"
                ax.text(j, i, "%.1f" % val, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    title_suffix = " (Role Graph)" if suffix else ""
    ax.set_title("%s Question Types%s" % (tier_name, title_suffix), fontsize=14, fontweight="bold", pad=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)

    plt.tight_layout()
    tag = "_role_graph" if suffix else ""
    out = "analysis_scripts/output/%s_heatmap%s" % (tier_name.lower(), tag)
    plt.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
    plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved %s" % out)


for tier_name, tier_info in TIERS.items():
    make_heatmap(baseline_df, tier_name, tier_info["labels"], tier_info["cols"], suffix="")
    if len(rg_df) > 0:
        make_heatmap(rg_df, tier_name, tier_info["labels"], tier_info["cols"], suffix="_rg")
