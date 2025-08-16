from katacr.classification.elixir.train import TrainConfig, ModelConfig
from katacr.classification.elixir.torch_train_elixir import DatasetBuilder, get_args_and_writer
from katacr.classification.torch_train import ResNet
from katacr.build_dataset.utils.split_part import extract_bbox
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np



args, writer = get_args_and_writer()
## Define local path_dataset_root
path_dataset_root = Path(r'C:\Disk_D\RL_Finance\ImageLabeling')
ds_builder=DatasetBuilder(str(path_dataset_root / "images/shawn_elixir_classification"), args.seed)
args.num_class = len(ds_builder.elixir_list)
train_cfg = TrainConfig(**vars(args))
model_cfg = ModelConfig(**vars(args))

class ElixirClassifier:
    def __init__(self, model_path, args):
        self.elixir2idx = ds_builder.elixir2idx
        self.idx2elixir = ds_builder.idx2elixir
        model_cfg = ModelConfig(**vars(args))
        train_cfg = TrainConfig(**vars(args))
        self.img_size = train_cfg.image_size

        # Load trained PyTorch model
        self.model =ResNet(model_cfg)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def preprocess(self, imgs):
        processed = []
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if gray.shape[:2][::-1] != self.img_size:
                gray = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_CUBIC)
            gray = gray.astype(np.float32) / 255.
            gray = gray[None, :, :]  # (1, H, W)
            processed.append(gray)
        x = np.stack(processed)  # (B, 1, H, W)
        return torch.tensor(x, dtype=torch.float32)

    def __call__(self, x, keepdim=False, cvt_label=True, verbose=False):
        if not isinstance(x, list): x = [x]
        x_tensor = self.preprocess(x)  # (B, 1, H, W)

        with torch.no_grad():
            logits = self.model(x_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        if cvt_label:
            labels = [self.idx2elixir[i] for i in preds]
        else:
            labels = preds

        if verbose:
            print("Logits:", logits[0], "Prediction:", labels)
            cv2.imshow("img", x[0])
            cv2.waitKey(0)

        if len(x) == 1 and not keepdim:
            return labels[0]
        return labels
