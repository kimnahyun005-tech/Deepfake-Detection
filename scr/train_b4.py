import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from main_trainer import HybridDeepfakeDataset, DeepfakeDetector  

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),  # EfficientNet-B4의 최적 입력 해상도
        transforms.RandomCrop(380),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_sources = [
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/train/real", 0),
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/train/fake", 1)
    ]

    val_sources = [
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/validation/real", 0),
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/validation/fake", 1)
    ]

    train_dataset = HybridDeepfakeDataset(train_sources, transform=train_transform)
    val_dataset = HybridDeepfakeDataset(val_sources, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print("-> EfficientNet-B4 사전 학습 가중치 다운로드 및 모델 생성 중...")
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    backbone = efficientnet_b4(weights=weights)
    
    in_features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )

    model = DeepfakeDetector(backbone, lr=cfg["learning_rate"])

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.get("monitor_metric", "val_loss"),
        dirpath="models",
        filename="efficientnet_b4_best",  
        save_top_k=1,
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.get("monitor_metric", "val_loss"),
        patience=cfg.get("early_stopping_patience", 7),
        mode="min"
    )

    trainer = Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    print("▶️ EfficientNet-B4 딥페이크 학습을 시작합니다!")
    trainer.fit(model, train_loader, val_loader)
