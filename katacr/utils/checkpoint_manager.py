import os
import json
import shutil
from pathlib import Path
import torch

class CheckpointManager:
    def __init__(self, path_save, max_to_keep=1, remove_old=False):
        self.path_save = Path(path_save).resolve()
        self.max_to_keep = max_to_keep
        if remove_old and self.path_save.exists():
            shutil.rmtree(self.path_save)
        self.path_save.mkdir(parents=True, exist_ok=True)

    def _get_ckpt_path(self, epoch: int):
        # Folder for this epoch, zero-padded 3 digits like "001"
        return self.path_save / f"{epoch:03}"

    def _cleanup_old_checkpoints(self):
        # Keep only the newest max_to_keep checkpoints
        ckpts = sorted([d for d in self.path_save.iterdir() if d.is_dir()])
        while len(ckpts) > self.max_to_keep:
            old_ckpt = ckpts.pop(0)
            shutil.rmtree(old_ckpt)

    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, config: dict = None, verbose: bool = True):
        ckpt_dir = self._get_ckpt_path(epoch)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state_dict
        model_path = ckpt_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

        # Optionally save optimizer state_dict
        if optimizer is not None:
            optim_path = ckpt_dir / "optimizer.pth"
            torch.save(optimizer.state_dict(), optim_path)

        # Save config as JSON (also save epoch for info)
        if config is None:
            config = {}
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
        config['_epoch'] = epoch
        config_path = ckpt_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        if verbose:
            print(f"Checkpoint saved to {ckpt_dir}")

        # Cleanup older checkpoints
        self._cleanup_old_checkpoints()

    def load(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, map_location=None):
        ckpt_dir = self._get_ckpt_path(epoch)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_dir} does not exist")

        model_path = ckpt_dir / "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=map_location))

        if optimizer is not None:
            optim_path = ckpt_dir / "optimizer.pth"
            if optim_path.exists():
                optimizer.load_state_dict(torch.load(optim_path, map_location=map_location))
            else:
                print(f"Warning: optimizer state not found in {optim_path}")

        config_path = ckpt_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        return config
