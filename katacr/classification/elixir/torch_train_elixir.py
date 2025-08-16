from katacr.classification.elixir.train import  TrainConfig, ModelConfig, ElixirDataset
from katacr.classification.torch_train import ResNet, get_optimizer, get_cosine_schedule_with_warmup, train_step, validate
from pathlib import Path
import cv2
import math
import numpy as np
import torch.nn as nn
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader



def get_args_and_writer():
    from katacr.utils.parser import Parser, datetime
    import sys

    # Clear Jupyter's extra command-line args
    sys.argv = ['']  # Removes unwanted --f=... args

    parser = Parser(model_name="ElixirClassification", wandb_project_name="ClashRoyale Elixir")
    parser.add_argument("--image-size", type=tuple, default=(32, 32))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)

    args = parser.get_args()
    args.lr = args.learning_rate
    args.run_name = f"{args.model_name}__lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    writer = parser.get_writer(args)

    return args, writer


def save_model_ckpt(model, val_loss, best_val_loss, path='D:\RL_Finance\KataCR\elixir_resnet.pth'):
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), path)
        print(f"✅ Saved new best model with val_loss = {val_loss:.4f}")

class DatasetBuilder:
  def __init__(self, path_dataset, seed=42):
    self.path_dataset = path_dataset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    self.preprocss()
    self.split_data(val_ratio=0.2)
  
  def preprocss(self):
    path_dirs = sorted(list(x for x in Path(self.path_dataset).glob('*')))
    self.elixir_list = [x.name for x in path_dirs]
    self.idx2elixir = dict(enumerate(self.elixir_list))
    self.elixir2idx = {c: i for i, c in enumerate(self.elixir_list)}
    self.images, self.labels = [], []
    for d in path_dirs:
      cls = d.name
      for p in d.glob('*.jpg'):
        self.images.append(cv2.imread(str(p)))
        self.labels.append(self.elixir2idx[cls])
  def split_data(self, val_ratio=0.2):
      all_data = list(zip(self.images, self.labels))
      train_data, val_data = train_test_split(all_data, test_size=val_ratio, stratify=self.labels, random_state=42)
      self.train_images, self.train_labels = zip(*train_data)
      self.val_images, self.val_labels = zip(*val_data)
  
  def get_dataloader(self, train_cfg: TrainConfig, mode='train'):
    images = self.train_images if mode == 'train' else self.val_images
    labels = self.train_labels if mode == 'train' else self.val_labels
    return DataLoader(
      ElixirDataset(images, labels, mode=mode, cfg=train_cfg),
      batch_size=train_cfg.batch_size,
      shuffle=mode=='train',
      num_workers=train_cfg.num_workers,
      persistent_workers=train_cfg.num_workers > 0,
      drop_last=mode=='train',
    )
  

args, writer = get_args_and_writer()
path_dataset_root = Path("C:/Disk_D/RL_Finance/ImageLabeling")
ds_builder=DatasetBuilder(str(path_dataset_root / "images/shawn_elixir_classification"), args.seed)
args.num_class = len(ds_builder.elixir_list)
train_cfg = TrainConfig(**vars(args))
model_cfg = ModelConfig(**vars(args))



model=ResNet(model_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_ds = ds_builder.get_dataloader(train_cfg, mode='train')
val_ds = ds_builder.get_dataloader(train_cfg, mode='val')
args.steps_per_epoch = train_cfg.steps_per_epoch = len(train_ds)
args.elixir2idx = ds_builder.elixir2idx
args.idx2elixir = ds_builder.idx2elixir
optimizer = get_optimizer(model, train_cfg)
scheduler = get_cosine_schedule_with_warmup(optimizer, train_cfg)
loss_fn = nn.CrossEntropyLoss()


def run():
    best_val_loss = 10
    for epoch in range(10):
        train_loss, train_acc = train_step(model, train_ds, optimizer, scheduler, loss_fn, device)
        val_loss, val_acc = validate(model, val_ds, loss_fn, device)


        print(f"Epoch {epoch+1}/{train_cfg.total_epochs}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        save_model_ckpt(model, val_loss, best_val_loss)
        best_val_loss = min(best_val_loss, val_loss)
'''
If we would like to traing the model

best_val_loss = 10
for epoch in range(10):
    train_loss, train_acc = train_step(model, train_ds, optimizer, scheduler, loss_fn, device)
    val_loss, val_acc = validate(model, val_ds, loss_fn, device)


    print(f"Epoch {epoch+1}/{train_cfg.total_epochs}: "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    save_model_ckpt(model, val_loss, best_val_loss)
    best_val_loss = min(best_val_loss, val_loss)



'''
