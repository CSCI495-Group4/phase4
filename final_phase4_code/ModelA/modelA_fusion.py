# =============================================================
# Model A : ResNet-18 + GRU Multimodal Fusion Architecture
# =============================================================

import torch
import torch.nn as nn
import torchvision.models as models


class ModelA_MultimodalEmotionNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, num_classes=7):
        super(ModelA_MultimodalEmotionNet, self).__init__()

        # -----------------------------
        # IMAGE ENCODER (ResNet-18)
        # -----------------------------
        base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(base_resnet.children())[:-1]    # remove classifier head
        self.image_encoder = nn.Sequential(*modules)
        self.image_fc = nn.Linear(512, hidden_dim)     # map to 512-D

        # -----------------------------
        # TEXT ENCODER (Embedding + GRU)
        # -----------------------------
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # -----------------------------
        # FUSION + CLASSIFICATION
        # -----------------------------
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, images, text):

        # ---- Image features ----
        img_feat = self.image_encoder(images)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.image_fc(img_feat)

        # ---- Text features ----
        embedded = self.embedding(text)
        _, hidden = self.gru(embedded)
        text_feat = hidden.squeeze(0)

        # ---- Fusion ----
        fused = torch.cat([img_feat, text_feat], dim=1)
        fused = self.dropout(fused)

        # ---- Classifier head ----
        logits = self.fc(fused)

        return logits
