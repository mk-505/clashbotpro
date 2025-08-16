import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # pip install einops

import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, SequentialLR


## Remember to import the cfg for Starformer

## ... 

## Embedding Layer and Cnn block for bars and arena
class Embed(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.zeros_(self.weight)


import math
class CNNBlockConfig:
    def __init__(self, filters=[32, 64, 64], kernels=[8, 4, 3], strides=[4, 2, 1]):
        self.filters = filters
        self.kernels = kernels
        self.strides = strides

class CNNBlock(nn.Module):
    def __init__(self, cfg: CNNBlockConfig, in_channels=1):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for f, k, s in zip(cfg.filters, cfg.kernels, cfg.strides):
            pad = math.floor((k - 1) / 2)
            conv = nn.Conv2d(in_channels=current_in_channels, out_channels=f,
                             kernel_size=k, stride=s, padding=pad)
            layers.extend([conv, nn.ReLU()])
            current_in_channels = f
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, H, W, bar_h, bar_w, C)
        B, N, H, W, ph, pw, C = x.shape

        # Rearrange to: (B*N*H*W, C, ph, pw)
        x = x.permute(0, 1, 2, 3, 6, 4, 5).reshape(B * N * H * W, C, ph, pw)

        out = self.net(x)  # (B*N*H*W, C_out, ph_out, pw_out)

        # Equivalent of jax's mean(-1): average over last spatial dim (width)
        out = out.mean(dim=-1)  # (B*N*H*W, C_out, ph_out)

        # Equivalent of [..., 0, :] in jax: select first "row" along height dimension
        out = out[:, :, 0]  # (B*N*H*W, C_out)

        # Reshape back to (B, N, H, W, C_out)
        out = out.view(B, N, H, W, -1)

        return out


class ArenaCNNBlock(nn.Module):
    def __init__(self, cfg: CNNBlockConfig, in_channels=15):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for f, k, s in zip(cfg.filters, cfg.kernels, cfg.strides):
            pad = math.floor((k - 1) / 2)
            layers.append(nn.Conv2d(current_in_channels, f, kernel_size=k, stride=s, padding=pad))
            layers.append(nn.ReLU())
            current_in_channels = f
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, N, H, W, C)
        B, N, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3)        # -> (B, N, C, H, W)
        x = x.reshape(B * N, C, H, W)       # -> (B*N, C, H, W)
        out = self.net(x)                   # -> (B*N, C_out, H', W')
        out = out.reshape(B, N, -1)            # -> (B, N, C_out * H' * W')
        return out
    

## Define the Training Config

from typing import Callable, Tuple

class TrainConfig:
    def __init__(
        self,
        steps_per_epoch: int,
        n_step: int,
        accumulate: int,
        seed: int = 42,
        weight_decay: float = 0.1,
        lr: float = 6e-4,
        total_epochs: int = 10,
        batch_size: int = 16,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_tokens: int = 512 * 20,
        clip_global_norm: float = 1.0,
        lr_fn: Callable = None,
        **kwargs
    ):
        self.steps_per_epoch = steps_per_epoch
        self.n_step = n_step
        self.accumulate = accumulate
        self.seed = seed
        self.weight_decay = weight_decay
        self.lr = lr
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.betas = betas
        self.warmup_tokens = warmup_tokens
        self.clip_global_norm = clip_global_norm
        self.lr_fn = lr_fn

        # Any extra kwargs passed in
        for k, v in kwargs.items():
            setattr(self, k, v)




## Attention block, Transformer block, StarBlock and the final Starformer
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, input_dim, p_drop_attn=0.1, p_drop_resid=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Combined projection for q, k, v
        self.q_proj = nn.Linear(input_dim, n_embd)
        self.k_proj = nn.Linear(input_dim, n_embd)
        self.v_proj = nn.Linear(input_dim, n_embd)
        
        # Final projection
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        self.dropout_attn = nn.Dropout(p_drop_attn)
        self.dropout_resid = nn.Dropout(p_drop_resid)
        
        # Register causal mask buffer (to prevent attending to future tokens)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        )

    def forward(self, x, mask=None):
        """
        x: [B, L, n_embd]
        mask: optional bool tensor of shape [B, 1, L, L], where 0 means masked positions
        """
        B, L, _ = x.size()
        
        # Project q, k, v
        q = self.q_proj(x)  # [B, L, n_embd]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (B, n_head, L, head_dim)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, L, head_dim]
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, n_head, L, L]
        
        # Apply causal mask
        causal_mask = self.mask[:, :, :L, :L]  # slice mask to seq length
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply external mask if provided
        if mask is not None:
            # Assume mask shape is [B, 1, L, L] or broadcastable
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, n_head, L, L]
        attn_probs = self.dropout_attn(attn_probs)
        
        # Attention output
        y = torch.matmul(attn_probs, v)  # [B, n_head, L, head_dim]
        
        # Reshape back and project out
        y = y.transpose(1, 2).contiguous().view(B, L, self.n_embd)  # [B, L, n_embd]
        y = self.out_proj(y)
        y = self.dropout_resid(y)
        
        return y



class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, input_dim, p_drop_resid=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        
        # LayerNorm layers (two separate ones for attention and MLP)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # Causal Self-Attention block
        # If input_dim is None, default to n_embd (usually input and hidden dims are the same here)
        if input_dim is None:
            input_dim = n_embd
        self.proj = nn.Identity() if input_dim==n_embd else nn.Linear(input_dim,n_embd)
        self.csa = CausalSelfAttention(n_embd, n_head, n_embd, p_drop_resid=p_drop_resid)
        
        # MLP: two linear layers with GELU activation in between
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        
        self.dropout = nn.Dropout(p_drop_resid)
        
    def forward(self, x, mask=None):
        x = self.proj(x)
        # x shape: [B, L, n_embd]
        # Attention block with residual
        x = x + self.csa(self.ln1(x), mask=mask)
        
        # MLP block with residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class StarBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


        self.local_block= TransformerBlock(n_embd=self.cfg.n_embd_local, 
                                           n_head=self.cfg.n_head_local, 
                                           input_dim=self.cfg.n_embd_local,)
        self.global_block= TransformerBlock(n_embd=cfg.n_embd_global,
                                            n_head=cfg.n_head_global,
                                            input_dim=cfg.n_embd_global,)
        self.dense= nn.Linear(self.cfg.n_embd_local*(self.cfg.group_token_length+3), self.cfg.n_embd_global)
        nn.init.normal_(self.dense.weight, std=0.02)

    def forward(self, xl, xg):
        """
        xl: [B, N, M, nl]
        xg: [B, 2N, ng]
        """
        B, N, M, nl = xl.shape
        _, N2, ng = xg.shape

        # Step 1: Local block applied to each timestep group
        xl_reshaped = xl.view(B * N, M, nl)                  # [B*N, M, nl]
        xl_out = self.local_block(xl_reshaped)               # [B*N, M, nl]
        xl = xl_out.view(B, N, M, nl)                        # [B, N, M, nl]

        # Step 2: Project local output into global embedding space
        xl_flat = xl.view(B, N, M * nl)                      # [B, N, M*nl]
        zg = self.dense(xl_flat).unsqueeze(2)   # [B, N, 1, ng]

        # Step 3: Reshape xg into [B, N, 2, ng] and concat with zg → [B, N, 3, ng]
        xg = xg.view(B, N, 2, ng)                            # [B, N, 2, ng]
        zg = torch.cat([zg, xg], dim=2)                      # [B, N, 3, ng]
        zg = zg.view(B, 3 * N, ng)                           # [B, 3N, ng]

        # Step 4: Construct causal attention mask
        # Causal mask: lower triangular + custom dependencies
        mask = torch.tril(torch.ones((3 * N, 3 * N), device=xl.device))  # [3N, 3N]

        # Custom dependencies as in the JAX code
        for i in range(N):
            mask[i * 3, i * 3 + 1] = 1
            mask[i * 3, i * 3 + 2] = 1
            mask[i * 3 + 1, i * 3 + 2] = 1

        # Expand mask to batch shape: [B, 1, 3N, 3N]
        attn_mask = mask.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1)

        # Step 5: Global transformer block
        zg = self.global_block(zg, mask=attn_mask)           # [B, 3N, ng]

        # Step 6: Extract updated global tokens (elixir + arena) from [1:] slice
        zg = zg.view(B, N, 3, ng)[:, :, 1:, :]               # [B, N, 2, ng]
        xg = zg.contiguous().view(B, N2, ng)                 # [B, 2N, ng]

        return xl, xg



class StARformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Embeddings
        self.arena_cls_embed = Embed(cfg.n_unit + 2, 8)  
        self.arena_bel_embed = nn.Identity()  # for reshaping bel to (...,1)
        
        # CNN blocks for bars
        self.bar_cnn1 = cfg.CNN(cfg.bar_cfg)
        self.bar_cnn2 = cfg.CNN(cfg.bar_cfg)
        
        # Global token embeddings
        self.pos_embd = Embed(2 * cfg.step_length, cfg.n_embd_global)  # example max seq length
        self.cards_g_embed = Embed(cfg.n_cards, cfg.n_embd_global // 6)
        self.elixir_g_embed = Embed(cfg.n_elixir + 2, cfg.n_embd_global // 6)
        ## Cnn block for the arena information aggregation
        self.arena_cnn = nn.Sequential(
            cfg.arena_CNN(cfg.arena_cfg),
            nn.Linear(cfg.arena_cnn_output_dim, cfg.n_embd_global)  # output dense layer for arena
        )
        
        # Local token embeddings
        self.card_embd_local = Embed(cfg.n_cards, cfg.n_embd_local)
        self.pos_local_embed = Embed(32 * 18 + 2, cfg.n_embd_local)
        self.elixir_local_embed = Embed(cfg.n_elixir + 2, cfg.n_embd_local)
        self.arena_patch_embed = Embed(cfg.num_patches, cfg.n_embd_local)
        self.arena_dense = nn.Linear(cfg.arena_patch_dim, cfg.n_embd_local)
        
        # Reward dense layer
        self.reward_dense = nn.Linear(1, cfg.n_embd_local)
        
        # Dropout layers
        self.drop_embd = nn.Dropout(cfg.p_drop_embd)
        
        # StARBlock layers - repeated
        self.starb_blocks = nn.ModuleList([
            StarBlock(cfg) for _ in range(cfg.n_block)
        ])
        
        # LayerNorm for global tokens before final prediction heads
        self.ln_global = nn.LayerNorm(cfg.n_embd_global)
        
        # Prediction heads
        self.select_head = nn.Linear(cfg.n_embd_global, 4 if cfg.pred_card_idx else cfg.n_cards, bias=False)
        self.y_head = nn.Linear(cfg.n_embd_global, 32, bias=False)
        self.x_head = nn.Linear(cfg.n_embd_global, 18, bias=False)
        self.delay_head = nn.Linear(cfg.n_embd_global, cfg.max_delay + 1, bias=False)
    
    def forward(self, s, a, r, timestep, train=True):
        B, N, H, W = s['arena'].shape[:-1]
        arena = s['arena']
        arena_mask = s['arena_mask']
        cards = s['cards']
        elixir = s['elixir']
        
        # Arena embedding (simplified)
        cls = arena[..., 0]
        cls = torch.where(cls < 0, torch.tensor(0, device=cls.device), cls + 1)  
        cls = self.arena_cls_embed(cls)
        bel = arena[..., 1].unsqueeze(-1).float()
        
        bar1 = arena[..., -2 * self.cfg.n_bar_size:-self.cfg.n_bar_size].reshape(B, N, H, W, self.cfg.bar_size[1],self.cfg.bar_size[0], -1).float() / 255.
        bar2 = arena[..., -self.cfg.n_bar_size:].reshape(B, N, H, W, self.cfg.bar_size[1],self.cfg.bar_size[0], -1).float() / 255.
        bar1 = self.bar_cnn1(bar1)
        bar2 = self.bar_cnn2(bar2)


        arena_emb = torch.cat([cls, bel, bar1, bar2], dim=-1)
        arena_emb = arena_emb * arena_mask.unsqueeze(-1)
        

        
        # Global embedding tokens
        pos_ids = torch.arange(2 * N, device=arena.device).unsqueeze(0)
        pos_embd = self.pos_embd(pos_ids)
        
        cards_g = self.cards_g_embed(cards).view(B, N, -1)
        elixir = elixir + 1
        elixir_g = self.elixir_g_embed(elixir)
        z1 = torch.cat([cards_g, elixir_g], dim=-1)
        z2 = self.arena_cnn(arena_emb).view(B, N, -1)
        xg = torch.cat([z1, z2], dim=-1).view(B, 2 * N, -1) + pos_embd

        

        # Local token embeddings (simplified)
        if self.cfg.pred_card_idx:
            select = self.card_embd_local(a['select']).view(B, N, 1, -1)
        else:
            select = self.card_embd_local(a['select']).view(B, N, 1, -1)
        a['pos'][..., 1] = a['pos'][..., 1] + 1
        pos_emb = self.pos_local_embed(a['pos'][..., 0] * 18 + a['pos'][..., 1]).view(B, N, 1, -1)
        a_emb = torch.cat([select, pos_emb], dim=2)
        
        cards_local = self.card_embd_local(cards)
        elixir_local = self.elixir_local_embed(elixir).view(B, N, 1, -1)
        
        arena_patch = rearrange(arena_emb, 'B N (H p1) (W p2) C -> B N (H W) (p1 p2 C)', p1=self.cfg.patch_size[0], p2=self.cfg.patch_size[1])
        patch_pos_emb = self.arena_patch_embed(torch.arange(arena_patch.size(2), device=arena_patch.device))
        arena_patch_emb = self.arena_dense(arena_patch) + patch_pos_emb.unsqueeze(0).unsqueeze(0)
        
        s_emb = torch.cat([arena_patch_emb, cards_local, elixir_local], dim=2)
        
        r_emb = torch.tanh(self.reward_dense(r.unsqueeze(-1))).view(B, N, 1, -1)
    
        xl = torch.cat([a_emb, s_emb, r_emb], dim=2)
        
        # Time embedding
        time_emb = self.pos_local_embed(timestep).view(B, N, 1, -1)
        xl = xl + time_emb.repeat(1, 1, xl.size(2), 1)
        
        # Dropout on embeddings
        xl = self.drop_embd(xl)
        xg = self.drop_embd(xg)

        # StARBlock forward passes
        for block in self.starb_blocks:
            xl, xg = block(xl, xg)
        
        xg = self.ln_global(xg).view(B, N, 2, -1)
        
        card = xg[..., 0, :]
        arena = xg[..., 1, :]
        
        select = self.select_head(card)
        y = self.y_head(arena)
        x = self.x_head(arena)
        delay = self.delay_head(arena)
        
        return select, y, x, delay
    


## Provide the optimizer and model step in the training script

def masked_cross_entropy(logits, target, mask=None, eps=1e-6):
    logits = logits.view(-1, logits.shape[-1])             # (B * T, C)
    target = target.view(-1)                               # (B * T,)
    loss = F.cross_entropy(logits, target, reduction='none')  # (B * T,)
    if mask is not None:
        mask = mask.view(-1).float()
        return (loss * mask).sum() / (mask.sum() + eps)
    else:
        return loss.mean()
    
def masked_accuracy(logits, target, mask=None, eps=1e-6):
    pred = logits.argmax(dim=-1).view(-1)
    target = target.view(-1)
    correct = (pred == target).float()
    if mask is not None:
        mask = mask.view(-1).float()
        return (correct * mask).sum() / (mask.sum() + eps), correct
    else:
        return correct.mean(), correct


def model_step(model, optimizer, scheduler, s, a, r, timestep, target, cfg, device, train=True):
    model.to(device)
    model.train() if train else model.eval()

    # Move inputs to device
    s = {k: v.to(device) for k, v in s.items()}
    a = {k: v.to(device) for k, v in a.items()}
    r = r.to(device)
    timestep = timestep.to(device)
    y_select = target['select'].to(device).long()
    y_pos = target['pos'].to(device).long()
    y_delay = target['delay'].to(device).long()

    mask = (y_delay < cfg.max_delay).float()

    if train:
        # --- Training mode ---
        select_logits, y_logits, x_logits, delay_logits = model(s, a, r, timestep)
        loss_select = masked_cross_entropy(select_logits, y_select, mask)
        loss_y = masked_cross_entropy(y_logits, y_pos[..., 0], mask)
        loss_x = masked_cross_entropy(x_logits, y_pos[..., 1], mask)
        loss_delay = masked_cross_entropy(delay_logits, y_delay, mask)
        loss_pos = loss_y + loss_x
        loss = s['arena'].shape[0] * (loss_select + loss_pos + loss_delay)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    else:
        # --- Validation mode ---
        with torch.no_grad():
            select_logits, y_logits, x_logits, delay_logits = model(s, a, r, timestep)
            loss_select = masked_cross_entropy(select_logits, y_select, mask)
            loss_y = masked_cross_entropy(y_logits, y_pos[..., 0], mask)
            loss_x = masked_cross_entropy(x_logits, y_pos[..., 1], mask)
            loss_delay = masked_cross_entropy(delay_logits, y_delay, mask)
            loss_pos = loss_y + loss_x
            loss = s['arena'].shape[0] * (loss_select + loss_pos + loss_delay)

    # Accuracy (same for both modes)
    acc_select, flag_select = masked_accuracy(select_logits, y_select, mask)
    acc_y, flag_y = masked_accuracy(y_logits, y_pos[..., 0], mask)
    acc_x, flag_x = masked_accuracy(x_logits, y_pos[..., 1], mask)
    acc_delay, flag_delay = masked_accuracy(delay_logits, y_delay, mask)

    flag_pos = flag_y * flag_x
    n = mask.sum() + 1e-6
    acc_pos = (flag_pos * mask.view(-1)).sum() / n
    acc_select_and_pos = (flag_select * flag_pos * mask.view(-1)).sum() / n
    acc_all = (flag_select * flag_pos * flag_delay * mask.view(-1)).sum() / n

    metrics = {
        'loss_select': loss_select,
        'loss_pos': loss_pos,
        'loss_delay': loss_delay,
        'acc_select': acc_select,
        'acc_pos': acc_pos,
        'acc_delay': acc_delay,
        'acc_select_and_pos': acc_select_and_pos,
        'acc_all': acc_all,
    }

    return loss, metrics



## Optimizer

def get_torch_scheduler(optimizer, train_cfg):
    warmup_steps = train_cfg.warmup_tokens // (train_cfg.n_step * train_cfg.batch_size * train_cfg.accumulate)

    total_steps = train_cfg.total_epochs * train_cfg.steps_per_epoch // train_cfg.accumulate
    decay_steps = max(total_steps - warmup_steps, 1)

    def warmup_lambda(step):
        if warmup_steps == 0:
            return 1.0
        return step / float(warmup_steps)

    def cosine_lambda(step):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        return 0.1 + 0.9 * cosine_decay  # minimum 0.1

    # second stage scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = LambdaLR(optimizer, lr_lambda=cosine_lambda)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    return scheduler


