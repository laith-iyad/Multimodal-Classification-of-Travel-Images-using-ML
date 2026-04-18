import numpy as np, pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path("Plots")
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("Data_Cleaned_v3.csv")
z = np.load("X_fused_clip.npz")
X = z["X"].astype(np.float32)
row_id = z.get("row_id", np.arange(len(X)))

# re-normalize fused embeddings
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

targets = ["Activity", "Weather", "Time of Day"]
C_grid = {"C": [0.01, 0.1, 1, 10, 100]}  # >=4 values

print("Model: LinearSVC")
for target in targets:
    y = df.loc[row_id, target]
    m = y.notna()
    valid_row_ids = row_id[m]
    Xc, yc = X[m], y[m]

    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        Xc, yc, valid_row_ids, test_size=0.2, random_state=42, stratify=yc
    )

    grid = GridSearchCV(
        LinearSVC(class_weight="balanced", dual="auto"),
        C_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1_macro"   # tune for macro-F1
    )
    grid.fit(Xtr, ytr)

    pred = grid.best_estimator_.predict(Xte)
    print(f"\n--- {target} ---")
    print("Best Params:", grid.best_params_)
    print(f"Acc={accuracy_score(yte,pred):.4f}, Macro-F1={f1_score(yte,pred,average='macro'):.4f}")

    # Visualize Misclassified Images (Weather only)
    if target == "Weather":
        # Find indices where prediction does not match truth
        mis_mask = (pred != yte)
        mis_indices = np.where(mis_mask)[0]
        
        if len(mis_indices) > 0:
            # Select up to 3 random samples
            n_view = 3
            n_samples = min(n_view, len(mis_indices))
            # rng = np.random.RandomState(42)  <-- Fixed seed removed
            rng = np.random.RandomState()      # Random seed
            selected_pos = rng.choice(mis_indices, n_samples, replace=False)
            
            # Setup grid: 1 row x 3 cols
            n_cols = 3
            n_rows = 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
            axes_flat = axes.flatten()
            
            for i in range(len(axes_flat)):
                ax = axes_flat[i]
                if i < n_samples:
                    pos = selected_pos[i]
                    orig_idx = idx_te[pos]
                    true_label = yte.iloc[pos]
                    pred_label = pred[pos]
                    
                    img_rel_path = df.loc[orig_idx, "Image Path"]
                    img_path = Path(img_rel_path)
                    
                    # Load and display
                    try:
                        import PIL.Image
                        if img_path.exists():
                            img = PIL.Image.open(img_path)
                            ax.imshow(img)
                        elif (Path.cwd() / img_rel_path).exists():
                            img = PIL.Image.open(Path.cwd() / img_rel_path)
                            ax.imshow(img)
                        else:
                            ax.text(0.5, 0.5, f"Img Missing:\n{img_rel_path}", ha='center', fontsize=8)
                    except Exception as e:
                        print(f"Error loading {img_rel_path}: {e}")
                        ax.text(0.5, 0.5, f"Load Error:\n{str(e)[:50]}", ha='center', fontsize=8, wrap=True)
                        
                    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color="red", fontsize=9)
                
                ax.axis("off")
            
            plt.suptitle(f"Misclassified Examples (Random 3) - {target}", fontsize=14)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"misclassified_svm_{target.replace(' ', '_').lower()}.png", dpi=200)
            plt.close()
