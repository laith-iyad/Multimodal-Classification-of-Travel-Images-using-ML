# Assignment 3 — CLIP Embeddings Classification

This project predicts **Time of Day**, **Weather**, and **Activity** using **fused CLIP (image + text) embeddings**.

---

## Required files to download
- `Images/` (folder) from https://drive.google.com/file/d/15wb7L8syKdKX8VWxZQ_DFrrBvmt8z-LH/view
- `Data_Cleaned_v3.csv`
- `ExtractTextImageFeatures.py`
- Model scripts:
  - `KNN.py`
  - `LogisticRegression.py`
  - `SVM.py`

---

## Install requirements
```bash
pip install numpy pandas scikit-learn matplotlib pillow tqdm torch open-clip-torch
```

---

## Run the project
### 1) Extract CLIP features (creates `.npz` files)
```bash
python ExtractTextImageFeatures.py
```

This generates:
- `X_img_clip.npz`
- `X_txt_clip.npz`
- `X_fused_clip.npz`

### 2) Run models and enjoy
```bash
python EDA.py
python KNN.py
python LogisticRegression.py
python SVM.py
```

---

## Outputs
- Model results printed in terminal (Accuracy + Macro-F1)
- Plots/confusion matrices saved in `Plots/` (if enabled in the scripts)

---

## Notes
- `Data_Cleaned_v3.csv` must contain `Image Path`, `Description`, and target columns (e.g., `Activity`, `Weather`, `Time of Day`).
- Make sure all paths in `Image Path` correctly point to files inside `Images/`.