import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

if project_root not in sys.path:
    sys.path.append(project_root)
from policy.offline.parse_and_logs import logs
from policy.offline.torch_parse_and_logs import parse_args_and_writer
from utils.checkpoint_manager import CheckpointManager
from constants.label_list import unit_list
from policy.offline.dataset import DatasetBuilder

from policy.offline.torch_starformer import StARformer, TrainConfig, get_torch_scheduler, model_step
from policy.offline.torch_argparser import get_cfg
import torch
from torch.optim import AdamW
## sagemaker location '/opt/ml/input/data/training'
## local machine location 'C:/Disk_D/RL_Finance/KataCR/logs/offline_preprocess'

def train():
    args, writer = parse_args_and_writer([
    '--replay-dataset', '/opt/ml/input/data/training',
    '--batch-size', '32',
    '--wandb', 'False'
    ], log_dir="/opt/ml/output/tensorboard")

    # Now pass to DatasetBuilder
    ds_builder = DatasetBuilder(path_dataset=args.replay_dataset, n_step=args.n_step)
        


    train_ds = ds_builder.get_dataset(
                args.batch_size, args.num_workers, random_interval=args.random_interval,
                max_delay=args.max_delay, card_shuffle=args.card_shuffle, use_card_idx=True)

    args.validation_path="/opt/ml/input/data/validation" ## for cloud validation location
    val_builder = DatasetBuilder(path_dataset=args.validation_path, n_step=args.n_step)
    val_ds=val_builder.get_dataset(
            args.batch_size, args.num_workers, random_interval=args.random_interval,
            max_delay=args.max_delay, card_shuffle=args.card_shuffle, use_card_idx=True)
        
    args.n_unit = len(unit_list)
    args.n_cards = ds_builder.n_cards
    args.max_timestep = int(max(ds_builder.data['timestep']))
    args.steps_per_epoch = len(train_ds)
    ### Model ###


    cfg = get_cfg()

    model = StARformer(cfg)  # cfg is your full config object
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_cfg = TrainConfig(
    steps_per_epoch=500,
    n_step=10,
    accumulate=1
    )
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = get_torch_scheduler(optimizer, train_cfg)
    ### Checkpoint ###
    ## this is local save path: ckpt_manager = CheckpointManager(str(args.path_logs / 'ckpt'), max_to_keep=10)

    save_dir = os.environ.get("SM_MODEL_DIR", "./saved_model")  # /opt/ml/model on SageMaker
    ckpt_manager = CheckpointManager(str(Path(save_dir) / "ckpt"), max_to_keep=10)
    write_tfboard_freq = min(100, len(train_ds))


    global_step = 0 
    val_best_loss=1000
    ### Train and Evaluate ###
    for ep in range(args.total_epochs):
        print(f"Epoch: {ep+1}/{args.total_epochs}")
        print("Training...")
        logs.reset()
        train_epoch_loss=0

        train_total_metrics = {
                            'loss_select': 0.0,
                            'loss_pos': 0.0,
                            'acc_select': 0.0,
                            'acc_pos': 0.0,
                            'acc_delay': 0.0,
                            'acc_select_and_pos': 0.0
                        }
        for batch_idx, batch in enumerate(train_ds):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(train_ds)}")
            s, a, rtg, timestep, y = batch
            B = y['select'].shape[0]
            # a is real action in each frame, y is target action with future action time delay predict
            # we need select=0 and pos=(0,-1) as start action padding idx.
            # Padding a['select'] with 0 at the beginning
            select_pad = torch.zeros((B, 1), dtype=torch.int32)
            a['select'] = torch.cat([select_pad, a['select'][:, :-1]], dim=1)  # (B, l)

            # Padding a['pos'] with [0, -1] at the beginning
            pos_pad = torch.stack([
                torch.zeros((B, 1), dtype=torch.int32),
                -torch.ones((B, 1), dtype=torch.int32)
            ], dim=-1)  # shape: (B, 1, 2)

            a['pos'] = torch.cat([pos_pad, a['pos'][:, :-1]], dim=1)  # (B, l, 2)
            loss, metrics = model_step(model, optimizer, scheduler, s, a, rtg, timestep, y, cfg, device, train=True)
            for key in train_total_metrics:
                train_total_metrics[key] += metrics[key]
                
            logs.update(
            ['train_loss', 'train_loss_select', 'train_loss_pos',
            'train_acc_select', 'train_acc_pos', 'train_acc_delay',
            'train_acc_select_and_pos'],
            [loss.item(), metrics['loss_select'], metrics['loss_pos'], metrics['acc_select'], metrics['acc_pos'], metrics['acc_delay'], metrics['acc_select_and_pos']])

            if global_step % write_tfboard_freq == 0:
                logs.update(
                    ['SPS', 'epoch', 'learning_rate'],
                    [write_tfboard_freq / logs.get_time_length(), ep + 1, optimizer.param_groups[0]['lr']]
                )
                logs.writer_tensorboard(writer, global_step)
                logs.reset()

            global_step += 1
            train_epoch_loss += loss.item()
        
        train_avg_epoch_metrics = {key: val/len(train_ds) for key, val in train_total_metrics.items()}
        print(f"Train-Epoch-Loss:{train_epoch_loss/len(train_ds)}, Detailed-Loss:{train_avg_epoch_metrics}, Epoch:{ep+1}/{args.total_epochs}")
        
        val_epoch_loss=0
        val_total_metrics = {
                            'loss_select': 0.0,
                            'loss_pos': 0.0,
                            'acc_select': 0.0,
                            'acc_pos': 0.0,
                            'acc_delay': 0.0,
                            'acc_select_and_pos': 0.0
                        }
        for batch_idx, batch in enumerate(val_ds):
            s, a, rtg, timestep, y = batch
            B = y['select'].shape[0]
            # a is real action in each frame, y is target action with future action time delay predict
            # we need select=0 and pos=(0,-1) as start action padding idx.
            # Padding a['select'] with 0 at the beginning
            B=y['select'].shape[0]
            select_pad = torch.zeros((B, 1), dtype=torch.int32)
            a['select'] = torch.cat([select_pad, a['select'][:, :-1]], dim=1)  # (B, l)

            # Padding a['pos'] with [0, -1] at the beginning
            pos_pad = torch.stack([
                torch.zeros((B, 1), dtype=torch.int32),
                -torch.ones((B, 1), dtype=torch.int32)
            ], dim=-1)  # shape: (B, 1, 2)

            a['pos'] = torch.cat([pos_pad, a['pos'][:, :-1]], dim=1)  # (B, l, 2)
            loss, metrics = model_step(model, optimizer, scheduler, s, a, rtg, timestep, y, cfg, device, train=False)
            for key in val_total_metrics:
                val_total_metrics[key] += metrics[key]
            val_epoch_loss += loss.item()

        val_loss = val_epoch_loss/len(val_ds)
        avg_epoch_metrics = {key: val/len(val_ds) for key, val in val_total_metrics.items()}
        print(f"Validation-Epoch-Loss:{val_loss}, Detailed-Loss:{avg_epoch_metrics}, {ep+1}/{args.total_epochs}")
        if val_loss <= val_best_loss:
            ckpt_manager.save(
                epoch=ep + 1,
                model=model,
                optimizer=optimizer,
                config={
                    **vars(args),
                    "global_step": global_step,
                },
                verbose=True
                )
            val_best_loss=val_loss
            print(f"save model with best validation error:{val_best_loss}, {ep+1}/{args.total_epochs}")
    writer.close()
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
