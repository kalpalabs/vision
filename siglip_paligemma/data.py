from siglip_paligemma.configs import DataConfig
import torch
import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
from siglip_paligemma import IMAGENET_TRAIN_LEN, IMAGENET_VAL_LEN
import glob
from transformers import SiglipImageProcessor


class WebDataLoader(DataLoader):
    def __init__(self, len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = len
        
    def __len__(self):
        return self.len

class ImageDataLoader:
    def __init__(self, config: DataConfig):
        
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(256),                     # Resize shorter side to 256
            transforms.CenterCrop(224),                 # Center crop to 224x224
            transforms.ToTensor(),
        ])
        
        self.img_processor = SiglipImageProcessor(do_resize=config.img_processor_resize, do_rescale=config.img_processor_rescale)
        
        num_shards = len(glob.glob(f"{config.base_train_path}*"))
        shard_ids = list(range(num_shards))

        num_train = int(config.train_split * num_shards)
        train_ids = shard_ids[:num_train]
        val_ids = shard_ids[num_train:]

        train_shards = [f"{config.base_train_path}{i:04d}.tar" for i in train_ids]
        val_shards = [f"{config.base_train_path}{i:04d}.tar" for i in val_ids]

        total_train_samples = int(IMAGENET_TRAIN_LEN/num_shards) * num_train
        total_val_samples = IMAGENET_TRAIN_LEN - total_train_samples
    
        train_dataset = (
            wds.WebDataset(train_shards)
            .shuffle(config.buffer_size)
            .decode("pil")  # decode images using PIL
            .to_tuple("jpg", "cls")
            .map_tuple(self._process_train, torch.tensor)
        )

        val_dataset = (
            wds.WebDataset(val_shards)
            .shuffle(config.buffer_size)
            .decode("pil")  # decode images using PIL
            .to_tuple("jpg", "cls")
            .map_tuple(self._process_val, torch.tensor)
        )
        
        test_dataset = (
            wds.WebDataset(config.base_test_path)
            .shuffle(config.buffer_size)
            .decode("pil")  # decode images using PIL
            .to_tuple("jpg", "cls")
            .map_tuple(self._process_val, torch.tensor)
        )
        print(f"{total_train_samples=}, {total_val_samples/1e6=}")
        
        train_loader_len = math.ceil(total_train_samples/config.batch_size)
        val_loader_len =  math.ceil(total_val_samples/config.batch_size)
        test_loader_len = math.ceil(IMAGENET_VAL_LEN/config.batch_size)
        
        self.train_loader = WebDataLoader(train_loader_len, train_dataset.batched(batchsize=config.batch_size), batch_size=None, num_workers=config.num_workers, persistent_workers=config.persistent_workers, pin_memory=config.pin_memory)
        self.val_loader = WebDataLoader(val_loader_len, val_dataset.batched(batchsize=config.batch_size), batch_size=None, num_workers=config.num_workers, persistent_workers=config.persistent_workers, pin_memory=config.pin_memory)
        self.test_loader = WebDataLoader(test_loader_len, test_dataset.batched(batchsize=config.batch_size), batch_size=None, num_workers=config.num_workers, persistent_workers=config.persistent_workers, pin_memory=config.pin_memory)
        
    def get_dataloaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def _process_train(self, image):
        image = self.train_transforms(image)
        image = self.img_processor(images=image, return_tensors="pt")
        return image['pixel_values'].squeeze(0)

    def _process_val(self, image):
        image = self.val_transforms(image)
        image = self.img_processor(images=image, return_tensors="pt")
        return image['pixel_values'].squeeze(0)