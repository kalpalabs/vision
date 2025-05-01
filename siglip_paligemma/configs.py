from dataclasses import dataclass
import torch
from torch.amp import GradScaler

@dataclass
class DataConfig:
    batch_size: int = 2048
    base_train_path: str = "/home/azureuser/.cache/huggingface/datasets/timm___imagenet-1k-wds/webdatasets/imagenet1k-train-"
    base_test_path: str = "/home/azureuser/.cache/huggingface/datasets/timm___imagenet-1k-wds/webdatasets/imagenet1k-validation-{00..63}.tar"
    buffer_size: int = 10000 # should be greater than batch_size
    train_split: float = 0.8
    num_workers: int = 12
    persistent_workers: bool = True
    pin_memory: bool = True
    img_processor_resize: bool = False # transforms.ToTensor() already scales input img to [0, 1]
    img_processor_rescale: bool = False # already done in ImageDataLoader.transforms

@dataclass
class TrainConfig:
    batch_size: int = 2048
    num_epochs: int = 5
    lr: float = 1e-4
    min_lr: float = 1e-6
    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scaler = GradScaler

@dataclass 
class ModelConfig:
    model_name: str = "google/siglip-so400m-patch14-224"
    # model_name: str = "google/paligemma-3b-pt-224"
    num_classes: int = 1000
    
@dataclass 
class Config:
    data_config: DataConfig = DataConfig()
    train_config: TrainConfig = TrainConfig()
    model_config: ModelConfig = ModelConfig()
