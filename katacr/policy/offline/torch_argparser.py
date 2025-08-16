import argparse
import numpy as np
from constants.label_list import unit_list
from policy.offline.torch_starformer import CNNBlockConfig, CNNBlock, ArenaCNNBlock  # Adjust import path

BAR_SIZE = (24, 8)
BAR_RGB = False

def get_parser():
    parser = argparse.ArgumentParser(description="StARTransformer configuration")

    # Model architecture
    parser.add_argument("--n-embd-global", type=int, default=48)
    parser.add_argument("--n-head-global", type=int, default=4)
    parser.add_argument("--n-embd-local", type=int, default=32)
    parser.add_argument("--n-head-local", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-block", type=int, default=1)

    # Data related
    parser.add_argument("--n-cards", type=int, default=9)
    parser.add_argument("--n-unit", type=int, default=len(unit_list))
    parser.add_argument("--n-step", type=int, default=30)
    parser.add_argument("--max-timestep", type=int, default=300)

    # Token/sequence config
    parser.add_argument("--group-token-length", type=int, default=150)
    parser.add_argument("--step-length", type=int, default=30)

    # Dropout settings
    parser.add_argument("--p_drop_attn", type=float, default=0.1)
    parser.add_argument("--p_drop_resid", type=float, default=0.1)

    return parser

def get_cfg():
    parser = get_parser()
    args = parser.parse_args(args=[])  

    # Convert to config object
    class Config: pass
    cfg = Config()
    cfg.__dict__.update(vars(args))

    # Add additional manually defined values
    cfg.n_elixir = 10
    cfg.bar_size = BAR_SIZE
    cfg.n_bar_size = int(np.prod(BAR_SIZE) * (3 if BAR_RGB else 1))

    cfg.bar_cfg = CNNBlockConfig(filters=[16, 32, 3], kernels=[6, 3, 3], strides=[2, 2, 2])
    cfg.arena_cfg = CNNBlockConfig(filters=[64, 128, 128], kernels=[6, 3, 3], strides=[2, 2, 2])

    cfg.CNN = CNNBlock
    cfg.arena_CNN = ArenaCNNBlock

    cfg.arena_cnn_output_dim = 1536   # This might need to be dynamically computed
    cfg.patch_size = (2, 2)
    cfg.num_patches = (32 // 2) * (18 // 2)  # Assuming arena output shape is (32,18)
    cfg.arena_patch_dim = 60  # You might revise this
    cfg.p_drop_embd = 0.1
    cfg.pred_card_idx = False
    cfg.max_delay = 20

    return cfg
