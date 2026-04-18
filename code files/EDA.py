from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "Data_Cleaned_v3.csv"
PLOTS_DIR = Path("Plots")
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ---- nicer global style ----
plt.rcParams.update({
    "figure.dpi": 140,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": "#cccccc",
    "grid.color": "#dddddd",
    "font.family": "DejaVu Sans",
})

def save_fig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=250, bbox_inches="tight")
    plt.close()

# delete unwanted outputs if they exist
for fname in ["02_activity_by_time_small_multiples.png", "class_balance_summary.txt"]:
    p = PLOTS_DIR / fname
    if p.exists():
        p.unlink()

# 1) Target distributions
cols = ["Time of Day", "Weather", "Activity"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, c, col in zip(axes, cols, colors):
    vc = df[c].value_counts()
    bars = ax.bar(vc.index.astype(str), vc.values, color=col, alpha=0.85)

    ax.set_title(c)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, padding=3, fontsize=9)

    # clean top/right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

save_fig("01_target_distributions.png")

# 2) Description length histogram
lens = df["Description"].astype(str).str.split().str.len()

plt.figure(figsize=(7.5, 4.5))
plt.hist(lens, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")

plt.title("Description Length (Words)")
plt.xlabel("Words per description")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.axvline(lens.mean(), color="black", linestyle="--", linewidth=1,
            label=f"Mean = {lens.mean():.1f}")
plt.legend(frameon=False)

save_fig("02_description_length_hist.png")

# 3) Top-20 words
STOP = {
    "the","a","an","and","or","of","to","in","on","at","for","with","from","by","as","is",
    "are","was","were","be","been","this","that","it","its","their","his","her","they",
    "you","i","we","he","she","them","there","here","into","over","under","near","around"
}

text = " ".join(df["Description"].astype(str).str.lower().tolist())
tokens = [w for w in re.findall(r"[a-z]+", text) if w not in STOP and len(w) > 2]
freq = pd.Series(tokens).value_counts().head(20).sort_values()

plt.figure(figsize=(8.5, 5.5))
bars = plt.barh(freq.index, freq.values, color="#55A868", alpha=0.85)

plt.title("Top 20 Words in Descriptions (Stopwords removed)")
plt.xlabel("Frequency")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.bar_label(bars, padding=3, fontsize=9)

# clean borders
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

save_fig("03_top20_words.png")

print(f"Done. Plots saved in: {PLOTS_DIR.resolve()}")
