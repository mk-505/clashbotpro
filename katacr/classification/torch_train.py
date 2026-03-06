import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
from katacr.classification.train import CardDataset, TrainConfig, DatasetBuilder, ModelConfig
from katacr.build_dataset.constant import path_dataset as path_dataset_root
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch
import re
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

EMPTY_CARD_INDEX = 1


class DatasetBuilder:
  def __init__(self, path_dataset, seed=42, augment_prob=0.1):
    self.path_dataset = path_dataset
    self.aug_prob = augment_prob
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    self.preprocss()
    self.split_data(val_ratio=0.2)
  
  def preprocss(self):
    path_dirs = sorted(list(x for x in Path(self.path_dataset).glob('*')))
    path_dirs = [p for p in path_dirs if re.search(r'^[a-zA-Z]', p.name)]
    cl = self.card_list = [x.name for x in path_dirs]
    self.augments = sorted(list(x for x in (Path(self.path_dataset)/"_augmentation").glob('*.png')))
    if cl[EMPTY_CARD_INDEX] != 'empty':
      idx1 = EMPTY_CARD_INDEX
      idx2 = self.card_list.index('empty')
      cl[idx1], cl[idx2] = cl[idx2], cl[idx1]
    self.idx2card = dict(enumerate(self.card_list))
    self.card2idx = {c: i for i, c in self.idx2card.items()}
    self.images, self.labels = [], []
    for d in path_dirs:
      cls = d.name
      for p in d.glob('*.jpg'):
        self.images.append(cv2.imread(str(p)))
        self.labels.append(self.card2idx[cls])
    self.aug_images = []
    for p in self.augments:
      self.aug_images.append(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))

  def split_data(self, val_ratio=0.2):
    all_data = list(zip(self.images, self.labels))
    train_data, val_data = train_test_split(all_data, test_size=val_ratio, stratify=self.labels, random_state=42)
    self.train_images, self.train_labels = zip(*train_data)
    self.val_images, self.val_labels = zip(*val_data)
  
  def get_dataloader(self, train_cfg: TrainConfig, mode='train'):
    images = self.train_images if mode == 'train' else self.val_images
    labels = self.train_labels if mode == 'train' else self.val_labels
    return DataLoader(
      CardDataset(images, labels, mode=mode, cfg=train_cfg, aug_images=self.aug_images, aug_prob=self.aug_prob),
      batch_size=train_cfg.batch_size,
      shuffle=mode=='train',
      num_workers=train_cfg.num_workers,
      persistent_workers=train_cfg.num_workers > 0,
      drop_last=mode=='train',
    )
  



def get_args_and_writer():
    from katacr.utils.parser import Parser, datetime
    import sys

    # Clear Jupyter's extra command-line args
    sys.argv = ['']  # Removes unwanted --f=... args

    parser = Parser(model_name="CardClassification", wandb_project_name="ClashRoyale Card")
    parser.add_argument("--image-size", type=tuple, default=(64, 80))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)

    args = parser.get_args()
    args.lr = args.learning_rate
    args.run_name = f"{args.model_name}__lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    writer = parser.get_writer(args)

    return args, writer



args, writer = get_args_and_writer()
ds_builder=DatasetBuilder(str(path_dataset_root / "images/card_classification"), args.seed)
args.num_class = len(ds_builder.card_list)
train_cfg = TrainConfig(**vars(args))
model_cfg = ModelConfig(**vars(args))


import torch
import torch.nn as nn

class BottleneckResNetBlock(nn.Module):
    def __init__(self, in_channels, filters, strides=(1, 1)):
        super().__init__()
        out_channels = filters * 2
        self.strides = strides

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.9)
        
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.9)

        self.conv3 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        nn.init.zeros_(self.bn3.weight)

        if strides != (1, 1) or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
            )
        else:
            self.proj = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.proj(identity)
        out += identity
        return self.relu(out)
    

class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_channels = 1

        self.conv1=nn.Conv2d(self.in_channels, self.cfg.filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.act1=nn.ReLU()

        self.stages = nn.ModuleList()
        in_channels = self.cfg.filters

        for i, num_blocks in enumerate(self.cfg.stage_sizes):
            blocks = []
            filters = self.cfg.filters * (2 ** i)
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(BottleneckResNetBlock(
                    in_channels=in_channels,
                    filters=filters,
                    strides=stride
                ))
                in_channels = filters * 2  # because each block outputs filters * 2
            self.stages.append(nn.Sequential(*blocks))
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_out_channels = self.cfg.filters * (2 ** (len(self.cfg.stage_sizes) - 1)) * 2
        self.fc = nn.Linear(final_out_channels, self.cfg.num_class, bias=False)


        
    def forward(self, x):
        x=self.conv1(x)
        x=self.act1(x)

        for stage in self.stages:
            x=stage(x)

        x = self.avgpool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        x = self.fc(x)  # [B, num_class]

        return(x)


## Scheduler
def get_cosine_schedule_with_warmup(optimizer, train_cfg):
    warmup_steps = train_cfg.warmup_epochs * train_cfg.steps_per_epoch
    total_steps = train_cfg.total_epochs * train_cfg.steps_per_epoch
    min_lr_ratio = 0.01  # Final LR is 1% of base LR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: from 1 to min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))  # in [0,1]
            return cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler



## Loss function
loss_fn=nn.CrossEntropyLoss()



## Selected optimizer with weight decay 
def get_optimizer(model, train_cfg):
    # Separate parameters for selective weight decay
    decay = []
    no_decay = []


    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen parameters
        if param.ndim > 1:
            decay.append(param)  # e.g., weights
        else:
            no_decay.append(param)  # e.g., biases, LayerNorm, BatchNorm scales

    param_groups = [
        {'params': decay, 'weight_decay': train_cfg.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(
        param_groups,
        lr=train_cfg.lr,
        betas=(train_cfg.betas[0], train_cfg.betas[1])
    )

    return optimizer


def train_step(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        x = x.permute(0, 3, 1, 2)
        x = x.to(torch.float32) / 255.0
        y = y.to(device).long()

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 3, 1, 2)
            x = x.to(torch.float32) / 255.0
            y = y.to(device).long()

            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy


model=ResNet(model_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_ds = ds_builder.get_dataloader(train_cfg, mode='train')
val_ds = ds_builder.get_dataloader(train_cfg, mode='val')
args.steps_per_epoch = train_cfg.steps_per_epoch = len(train_ds)
args.card2idx = ds_builder.card2idx
args.idx2card = ds_builder.idx2card
optimizer = get_optimizer(model, train_cfg)
scheduler = get_cosine_schedule_with_warmup(optimizer, train_cfg)
loss_fn = nn.CrossEntropyLoss()


'''

If do Training:

for epoch in range(10):
    train_loss, train_acc = train_step(model, train_ds, optimizer, scheduler, loss_fn, device)
    val_loss, val_acc = validate(model, val_ds, loss_fn, device)

    print(f"Epoch {epoch+1}/{train_cfg.total_epochs}: "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")


'''