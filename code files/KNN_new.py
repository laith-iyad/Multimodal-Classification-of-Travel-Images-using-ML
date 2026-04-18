from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# ---- settings ----
PLOTS_DIR = Path("Plots")
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("Data_Cleaned_v3.csv")

z = np.load("X_fused_clip.npz")
X = z["X"]
row_id = z.get("row_id", np.arange(len(X)))

targets = ["Activity", "Weather", "Time of Day"]
Ks = [1, 3]

print("Model: KNN (cosine)")
for target in targets:
    y = df.loc[row_id, target]
    mask = y.notna()
    Xc, yc = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )

    for k in Ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        print(f"{target:10s} | k={k} | Acc={acc:.4f} | Macro-F1={f1:.4f}")

        # Confusion matrix (normalized by true class)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            normalize="true",
            xticks_rotation=45,
            ax=ax
        )
        ax.set_title(f"KNN (k={k}) - {target} (normalized)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"cm_knn_k{k}_{target.replace(' ', '_').lower()}.png", dpi=250)
        plt.close(fig)
