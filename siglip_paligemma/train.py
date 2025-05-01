from siglip_paligemma.configs import Config
import wandb
from tqdm import tqdm
from torch.amp import autocast
from dataclasses import asdict
import torch
from safetensors.torch import save_file


class Trainer:
    def __init__(self, config: Config, device, model, train_loader, val_loader, test_loader):
        wandb.init(project="siglip-imagenet-1k", config=asdict(config))
        self.train_config = config.train_config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.optimizer = self.train_config.optimizer([p for p in self.model.parameters() if p.requires_grad], lr=self.train_config.lr)
        self.scheduler = self.train_config.scheduler(self.optimizer, T_max=self.train_config.num_epochs, eta_min=self.train_config.min_lr)
        self.scaler = self.train_config.scaler()
        
    def train(self):
        for _ in range(self.train_config.num_epochs):
            # Training
            self.model.train()
            for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training", position=0)):
                images, labels = batch
                labels=labels.to(self.device, non_blocking=True)
                images=images.to(self.device, non_blocking=True)
                
                with autocast(self.device.type): # 'cuda'
                    out = self.model(pixel_values=images)
                    loss = torch.nn.functional.cross_entropy(out, labels)
                
                # Backpropagation
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Log metrics
                wandb.log({
                    "batch_loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0]
                })
                
                # Validating every 100 batches instead of every epoch
                # if (1 + batch_id) % 100 == 0:

            self.scheduler.step()
            self.validate()

        #self.save_checkpoint()
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", position=1):
                images, labels = batch
                labels=labels.to(self.device, non_blocking=True)
                images=images.to(self.device, non_blocking=True)
                
                with autocast(self.device.type): # 'cuda'
                    out = self.model(pixel_values=images)
                    curr_loss = torch.nn.functional.cross_entropy(out, labels)
                    
                val_loss += curr_loss.item()
                predicted = torch.argmax(out, dim=1)
                correct += (predicted == labels).sum().item()
                total += len(labels)
                
        val_loss /= len(self.val_loader)
        val_acc = 100 * correct/total
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                images, labels = batch
                labels=labels.to(self.device, non_blocking=True)
                images=images.to(self.device, non_blocking=True)
                
                with autocast(self.device.type): # 'cuda'
                    out = self.model(pixel_values=images)
                    curr_loss = torch.nn.functional.cross_entropy(out, labels)
                    
                test_loss += curr_loss.item()
                predicted = torch.argmax(out, dim=1)
                correct += (predicted == labels).sum().item()
                total += len(labels)
                
        test_loss /= len(self.test_loader)
        test_acc = 100 * correct/total
        print(f"Evaluation: {test_loss=}, {test_acc=}")
        
    def save_checkpoint(self, name = "siglip.safetensors"):
        save_file(self.model.state_dict(), name)
        