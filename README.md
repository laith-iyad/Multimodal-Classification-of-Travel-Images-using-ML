# Multimodal Classification of Travel Images using ML

This project is about classifying travel images into three categories: **Time of Day**, **Weather**, and **Activity**. The idea is that instead of using just the image or just the text, we combine both — that's the multimodal part. We use **CLIP embeddings** (a model from OpenAI) to extract features from both images and their text descriptions, then fuse them together and train classifiers on top.

---

## What are we trying to predict?

Given a travel image and its description, the models try to predict:
- **Time of Day** — like morning, afternoon, night, etc.
- **Weather** — sunny, cloudy, rainy, etc.
- **Activity** — what's happening in the image (hiking, swimming, sightseeing, etc.)

---

## Project structure

```
├── Data_Cleaned_v3.csv           # the dataset
├── Drive_Link_imgs.txt           # link to download the images folder
├── Assignment3_Report.pdf        # full report
├── code files/
│   ├── ExtractTextImageFeatures.py   # extract CLIP features
│   ├── EDA.py                        # exploratory data analysis
│   ├── KNN_new.py                    # KNN classifier
│   ├── LogisticRegression.py         # Logistic Regression classifier
│   ├── SVM.py                        # SVM classifier
│   ├── clean__data_algorithm.py      # data cleaning
│   └── description.py               # helper for descriptions
```

---

## How the code works

### 1. `ExtractTextImageFeatures.py`
This is the first thing you need to run. It loads the CLIP model (`ViT-L-14`) and encodes both the images and their text descriptions into feature vectors. The two are then concatenated to make a fused embedding. It saves three `.npz` files:
- `X_img_clip.npz` — image features only
- `X_txt_clip.npz` — text features only
- `X_fused_clip.npz` — image + text combined (this is what the classifiers use)

It uses batch processing (batch size 32) and runs on GPU if available, otherwise falls back to CPU.

### 2. `EDA.py`
Does some basic exploratory analysis on the dataset and saves plots to a `Plots/` folder:
- Bar charts showing the distribution of each target label (Time of Day, Weather, Activity)
- A histogram of how long the descriptions are (in words)
- A bar chart of the top 20 most common words in the descriptions (stopwords removed)

Pretty useful to run first and get a feel for the data before jumping into models.

### 3. `KNN_new.py`
Runs a K-Nearest Neighbors classifier on the fused CLIP embeddings using **cosine similarity**. It tests two values of k (1 and 3) for each of the three targets. Prints accuracy and Macro-F1 for each, and saves normalized confusion matrices to `Plots/`.

### 4. `LogisticRegression.py`
Trains a Logistic Regression model using GridSearchCV to find the best regularization parameter C (tested over: 0.001, 0.01, 0.1, 1, 10, 100, 1000). Uses `class_weight="balanced"` to handle any imbalance in the labels. Also saves confusion matrices, and for **Activity** and **Time of Day**, it visualizes 3 randomly selected misclassified images so you can see where the model went wrong.

### 5. `SVM.py`
Same idea as logistic regression but uses a **LinearSVC** instead. Also does GridSearchCV for C, and visualizes 3 misclassified images specifically for the **Weather** target.

---

## How to run it

### Step 0 — Download the images
Get the images folder from the link in `Drive_Link_imgs.txt` and place it in the same directory as `Data_Cleaned_v3.csv`.

### Step 1 — Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib pillow tqdm torch open-clip-torch
```

### Step 2 — Extract features
```bash
python "code files/ExtractTextImageFeatures.py"
```
This will create the `.npz` files. Takes a few minutes depending on your machine.

### Step 3 — Run EDA (optional but recommended)
```bash
python "code files/EDA.py"
```

### Step 4 — Train and evaluate models
```bash
python "code files/KNN_new.py"
python "code files/LogisticRegression.py"
python "code files/SVM.py"
```

All outputs (accuracy, F1, confusion matrices, misclassified images) will be printed or saved in the `Plots/` folder.

---

## Requirements

- Python 3.8+
- A GPU is recommended for the feature extraction step, but not required
- The `Data_Cleaned_v3.csv` file must have columns: `Image Path`, `Description`, `Activity`, `Weather`, `Time of Day`
