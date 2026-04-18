from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import open_clip  # pip install open_clip_torch
from tqdm import tqdm

# ---- Settings ----
CSV_FILE = "Data_Cleaned_v3.csv"   # in same folder as this script
IMAGE_COL = "Image Path"
TEXT_COL  = "Description"

CLIP_MODEL   = "ViT-L-14"
CLIP_WEIGHTS = "openai"
BATCH = 32


def root_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def encode_images(model, preprocess, paths, device):
    out = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(paths), BATCH), desc="Encoding Images"):
            imgs = []
            for p in paths[i:i + BATCH]:
                with Image.open(p) as im:
                    imgs.append(preprocess(im.convert("RGB")))
            x = torch.stack(imgs).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def encode_text(model, tokenizer, texts, device):
    out = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), BATCH), desc="Encoding Text"):
            tok = tokenizer(texts[i:i + BATCH]).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def main():
    root = root_dir()
    df = pd.read_csv(root / CSV_FILE)

    paths = [(root / p).resolve() for p in df[IMAGE_COL].astype(str)]
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    row_id = np.arange(len(df), dtype=np.int32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval().to(device)

    X_img = encode_images(model, preprocess, paths, device)

    X_txt = encode_text(model, tokenizer, texts, device)
    X_fused = np.concatenate([X_img, X_txt], axis=1).astype(np.float32)

    np.savez_compressed(root / "X_img_clip.npz",   X=X_img,   row_id=row_id)
    np.savez_compressed(root / "X_txt_clip.npz",   X=X_txt,   row_id=row_id)
    np.savez_compressed(root / "X_fused_clip.npz", X=X_fused, row_id=row_id)

    print("Saved: X_img_clip.npz",   X_img.shape)
    print("Saved: X_txt_clip.npz",   X_txt.shape)
    print("Saved: X_fused_clip.npz", X_fused.shape)


if __name__ == "__main__":
    main()
