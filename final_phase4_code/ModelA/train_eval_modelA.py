# =============================================================
# Train/Eval Script for Model A (ResNet18 + GRU Fusion)
# =============================================================

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from modelA_fusion import ModelA_MultimodalEmotionNet


# =============================================================
# CONFIG
# =============================================================
TRAIN_DIR = "/scratch/gilbreth/ichaudha/balanced-raf-db/train"
VAL_DIR   = "/scratch/gilbreth/ichaudha/balanced-raf-db/val"

TEXT_CSV = "/scratch/gilbreth/ichaudha/Phase4/emotions.csv"  # <-- YOUR TEXT DATA HERE

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0001
MAX_LEN = 40      # max tokens per text


# =============================================================
# TEXT FUNCTIONS
# =============================================================
def build_vocab(texts, min_freq=2):
    freq = {}
    for sentence in texts:
        for w in sentence.split():
            freq[w] = freq.get(w, 0) + 1

    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in freq.items():
        if f >= min_freq:
            vocab[w] = len(vocab)

    return vocab


def encode(sentence, vocab, max_len):
    tokens = []
    for w in sentence.split():
        tokens.append(vocab.get(w, vocab["<UNK>"]))

    # pad or truncate
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    return torch.tensor(tokens)


# =============================================================
# CUSTOM DATASET
# =============================================================
class RAFDB_TextFusionDataset(Dataset):
    def __init__(self, img_folder, text_df, vocab, transform):
        self.img_data = datasets.ImageFolder(img_folder, transform=transform)
        self.text_df = text_df
        self.vocab = vocab

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img, img_label = self.img_data[idx]
        text = self.text_df.iloc[idx]['text']
        text_enc = encode(text, self.vocab, MAX_LEN)
        return img, text_enc, img_label


# =============================================================
# MAIN TRAINING PIPELINE
# =============================================================
def main():
    print("[INFO] Loading text dataset CSV...")
    df = pd.read_csv(TEXT_CSV)

    print("[INFO] Building vocabulary...")
    vocab = build_vocab(df['text'].tolist())
    vocab_size = len(vocab)
    print(f"[INFO] Vocab size: {vocab_size}")

    # image transforms
    img_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    print("[INFO] Creating datasets...")
    train_ds = RAFDB_TextFusionDataset(TRAIN_DIR, df, vocab, img_tf)
    val_ds   = RAFDB_TextFusionDataset(VAL_DIR, df, vocab, img_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("[INFO] Initializing Model A...")
    model = ModelA_MultimodalEmotionNet(vocab_size=vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create logs folder
    out_dir = f"outputs/{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Logs & checkpoints saved to: {out_dir}")

    # =========================================================
    # TRAINING LOOP
    # =========================================================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for imgs, txt, labels in train_loader:
            imgs, txt, labels = imgs.to(device), txt.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, txt)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"[EPOCH {epoch+1}] Train Loss = {avg_train_loss:.4f}")

        # =====================================================
        # VALIDATION
        # =====================================================
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, txt, labels in val_loader:
                imgs, txt, labels = imgs.to(device), txt.to(device), labels.to(device)

                logits = model(imgs, txt)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"          Val Loss = {avg_val_loss:.4f} | Acc = {accuracy:.4f}")

    # Save model
    torch.save(model.state_dict(), f"{out_dir}/modelA.pth")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
