# -*- coding: utf-8 -*-
"""Lightweight track centerline model ? MobileNetV2 + reduced UNet++.

~2.8M params (vs 32.5M original), ~10x compression.
Encoder:    MobileNetV2 (pretrained, ~2.2M)
Decoder:    UNet++ channels (128, 64, 32, 16, 16)
Semantic:   DepthwiseSepConv -> 1ch
Embedding:  Conv->BN->ReLU->Conv -> 4dim
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class LightweightBackbone(nn.Module):
    """MobileNetV2 encoder + reduced-channel UNet++ decoder."""

    def __init__(
        self,
        encoder_name: str = "mobilenet_v2",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        decoder_channels: tuple = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            decoder_channels=decoder_channels,
        )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.out_channels = decoder_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class LightSemanticHead(nn.Module):
    """1-channel centerline logits with depthwise separable conv."""

    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class LightEmbeddingHead(nn.Module):
    """4-dimensional per-pixel embedding vectors."""

    def __init__(self, in_channels: int = 16, embedding_dim: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embedding_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class LightTrackSegModel(nn.Module):
    """MobileNetV2 + UNet++ (reduced) + SemanticHead + EmbeddingHead."""

    def __init__(
        self,
        encoder_name: str = "mobilenet_v2",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        embedding_dim: int = 4,
        decoder_channels: tuple = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.backbone = LightweightBackbone(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
        )
        feat_channels = self.backbone.out_channels
        self.semantic_head = LightSemanticHead(in_channels=feat_channels)
        self.embedding_head = LightEmbeddingHead(
            in_channels=feat_channels, embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        semantic = self.semantic_head(features)
        embedding = self.embedding_head(features)
        return semantic, embedding


if __name__ == "__main__":
    model = LightTrackSegModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.2f}M")
    x = torch.randn(1, 3, 256, 256)
    sem, emb = model(x)
    print(f"Semantic: {sem.shape}, Embedding: {emb.shape}")
