import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DownAndUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownAndUp, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv1(x)


class RMSNorm(nn.Module):
    """
    沿通道维度的 RMSNorm。
    支持两种输入:
      - stack 后的张量 [N, B, C, H, W]  (channel_dim=2, 用于 AttnRes 聚合)
      - 普通特征图    [B, C, H, W]      (channel_dim=1, 用于单张特征归一化)
    """
    def __init__(self, dim, eps=1e-8, channel_dim=2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.channel_dim = channel_dim

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=self.channel_dim, keepdim=True) + self.eps)
        x_norm = x / rms
        shape = [1] * x.dim()
        shape[self.channel_dim] = -1
        return x_norm * self.scale.view(*shape)




class SAPQAttnRes(nn.Module):
    def __init__(self, channels, init_method='zero', temperature=1.0):
        super().__init__()
        self.channels = channels
        self.temperature = temperature

        # SAPQ: 1×1 conv 在每个空间位置独立生成 C 维 query
        self.query_generator = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = RMSNorm(channels, channel_dim=2)

        if init_method == 'zero':
            nn.init.zeros_(self.query_generator.weight)
        elif init_method == 'random':
            nn.init.normal_(self.query_generator.weight, std=0.01)

    def _reinit_weights(self):
        """供外部 initialize() 后重新设置 zero-init."""
        nn.init.zeros_(self.query_generator.weight)

    def forward(self, block_reps: List[torch.Tensor], current_feature: torch.Tensor):
        if len(block_reps) == 0:
            return current_feature

        # V: [N+1, B, C, H, W]
        V = torch.stack(block_reps + [current_feature], dim=0)
        K = self.norm(V)

        # SAPQ: 从 current_feature 动态生成空间 query field
        W = self.query_generator(current_feature)  # [B, C, H, W]

        # 空间位置独立的 attention score
        logits = torch.einsum('bchw,nbchw->nbhw', W, K)
        logits = logits / (self.channels ** 0.5 * self.temperature)
        attn = logits.softmax(dim=0)

        # 加权聚合
        h = torch.einsum('nbhw,nbchw->bchw', attn, V)
        return h


class DecoupledKVSkip(nn.Module):
    def __init__(self, memory_channels, query_channels, d_inter):
        super().__init__()
        self.d_inter = d_inter
        self.query_channels = query_channels

        # Φ_kv: 融合 K+V 投影, 一次 conv 同时算出 K 和 V
        self.phi_kv = nn.Conv2d(memory_channels, d_inter + query_channels,
                                kernel_size=1, bias=False)
        # Φ_q: D_up → query
        self.phi_q = nn.Conv2d(query_channels, d_inter, kernel_size=1, bias=False)
        # Key 归一化
        self.norm_k = RMSNorm(d_inter, channel_dim=2)

    def forward(self, memory_bank: List[torch.Tensor], D_up: torch.Tensor):
        """
        memory_bank: encoder 对应 stage 内所有 block 输出, 每个 [B, C_mem, H, W]
        D_up: decoder 上采样后的特征 [B, C_q, H, W], 空间尺寸与 memory 一致
        Returns: f_skip [B, C_q, H, W]
        """
        N_mem = len(memory_bank)
        B = memory_bank[0].shape[0]
        H, W = memory_bank[0].shape[2], memory_bank[0].shape[3]

        # --- 并行化: N 个 memory 沿 batch cat, 单次 conv ---
        M_cat = torch.cat(memory_bank, dim=0)            # [N*B, C_mem, H, W]
        KV_flat = self.phi_kv(M_cat)                      # [N*B, d_inter+C_q, H, W]
        K_flat, V_flat = KV_flat.split([self.d_inter, self.query_channels], dim=1)
        K_all = K_flat.view(N_mem, B, self.d_inter, H, W)
        V_all = V_flat.view(N_mem, B, self.query_channels, H, W)
        K_all = self.norm_k(K_all)

        # Q 投影
        Q = self.phi_q(D_up)                              # [B, d_inter, H, W]

        # QK 点积 + softmax
        scores = torch.einsum('bchw,nbchw->nbhw', Q, K_all)
        scores = scores / (self.d_inter ** 0.5)
        attn = scores.softmax(dim=0)

        # 加权聚合 V
        f_skip = torch.einsum('nbhw,nbchw->bchw', attn, V_all)
        return f_skip

class AttnResEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks,
                 enable_intra_attn=True, attn_init='zero'):
        super().__init__()
        self.n_blocks = n_blocks
        self.out_channels = out_channels

        blocks = [DownAndUp(in_channels, out_channels)]
        for _ in range(n_blocks - 1):
            blocks.append(DownAndUp(out_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)

        if n_blocks >= 2 and enable_intra_attn:
            self.attn_res = nn.ModuleList([
                SAPQAttnRes(out_channels, init_method=attn_init)
                for _ in range(n_blocks - 1)
            ])
        else:
            self.attn_res = None

    def forward(self, x):
        """Returns: (output, memory_bank)"""
        memory = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.attn_res is not None:
                x = self.attn_res[i - 1](memory, x)
            x = block(x)
            memory.append(x)
        return x, memory



class AttnResDecoderStage(nn.Module):
    def __init__(self, out_channels, n_blocks,
                 enable_intra_attn=True, attn_init='zero'):
        super().__init__()
        self.n_blocks = n_blocks
        self.out_channels = out_channels

        blocks = [DownAndUp(out_channels, out_channels) for _ in range(n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        if n_blocks >= 2 and enable_intra_attn:
            self.attn_res = nn.ModuleList([
                SAPQAttnRes(out_channels, init_method=attn_init)
                for _ in range(n_blocks - 1)
            ])
        else:
            self.attn_res = None

    def forward(self, x):
        memory = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.attn_res is not None:
                x = self.attn_res[i - 1](memory, x)
            x = block(x)
            memory.append(x)
        return x



class Model(nn.Module):
    def __init__(self,
                 img_channels: int = 3,
                 n_classes: int = 1,
                 channels: tuple = (18, 36, 72, 144, 144),
                 n_blocks_enc: tuple = (2, 2, 3, 3, 3),
                 n_blocks_dec: tuple = (2, 2, 2, 2),
                 encoder_attn_res: bool = False,
                 decoder_attn_res: bool = True,
                 kv_inter_dim_ratio: float = 0.5,
                 attn_init: str = 'zero'):
        super(Model, self).__init__()
        assert len(channels) == len(n_blocks_enc), \
            f"channels({len(channels)}) must match n_blocks_enc({len(n_blocks_enc)})"
        n_enc = len(channels)
        n_dec = n_enc - 1
        assert len(n_blocks_dec) == n_dec, \
            f"n_blocks_dec({len(n_blocks_dec)}) must be {n_dec}"

        self.channels = channels
        self.n_classes = n_classes
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.enc_stages = nn.ModuleList()
        for s in range(n_enc):
            in_c = img_channels if s == 0 else channels[s - 1]
            self.enc_stages.append(
                AttnResEncoderStage(
                    in_channels=in_c,
                    out_channels=channels[s],
                    n_blocks=n_blocks_enc[s],
                    enable_intra_attn=encoder_attn_res,
                    attn_init=attn_init,
                )
            )

        dec_out_channels = []
        for s in range(n_dec):
            if s < n_dec - 1:
                dec_out_channels.append(channels[n_enc - 3 - s])
            else:
                dec_out_channels.append(channels[0])
        self.dec_out_channels = dec_out_channels

        self.kv_skips = nn.ModuleList()    # 创新点 3
        self.fusions = nn.ModuleList()     # concat(D_up, f_skip) → dec_out_c
        self.dec_stages = nn.ModuleList()  # 创新点 1+2

        for s in range(n_dec):
            enc_skip_c = channels[n_enc - 2 - s]   # 对应 encoder stage 的通道数
            dec_out_c = dec_out_channels[s]
            d_inter = max(8, int(enc_skip_c * kv_inter_dim_ratio))

            # DecoupledKVSkip: encoder memory → f_skip
            self.kv_skips.append(
                DecoupledKVSkip(
                    memory_channels=enc_skip_c,
                    query_channels=enc_skip_c,  # D_up 通道数 = enc_skip_c
                    d_inter=d_inter,
                )
            )

            # Fusion: concat(D_up, f_skip) = 2*enc_skip_c → dec_out_c
            self.fusions.append(
                nn.Sequential(
                    nn.Conv2d(2 * enc_skip_c, dec_out_c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(dec_out_c),
                    nn.ReLU(inplace=True),
                )
            )

            # Decoder stage with intra-stage AttnRes
            self.dec_stages.append(
                AttnResDecoderStage(
                    out_channels=dec_out_c,
                    n_blocks=n_blocks_dec[s],
                    enable_intra_attn=decoder_attn_res,
                    attn_init=attn_init,
                )
            )

        self.out_conv = nn.Conv2d(dec_out_channels[-1], n_classes,
                                  kernel_size=1, stride=1, padding=0)
    def _upsample_and_pad(self, x, target):

        x = self.upsample(x)
        diffY = target.size(2) - x.size(2)
        diffX = target.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x, mode=None):
        enc_outputs = []      # 每个 stage 的最后输出 (用于获取空间尺寸)
        enc_memories = []     # 每个 stage 的完整 memory bank (供 KV skip)

        for s, stage in enumerate(self.enc_stages):
            if s > 0:
                x = self.maxpool(x)
            x, memory = stage(x)
            enc_outputs.append(x)
            enc_memories.append(memory)

        n_enc = len(self.enc_stages)
        for s in range(len(self.dec_stages)):
            # 1) 上采样 + 空间对齐
            skip_idx = n_enc - 2 - s
            D_up = self._upsample_and_pad(x, enc_outputs[skip_idx])

            # 2) DecoupledKVSkip: 从 encoder memory bank 检索
            f_skip = self.kv_skips[s](enc_memories[skip_idx], D_up)

            # 3) Fusion
            fused = self.fusions[s](torch.cat([D_up, f_skip], dim=1))

            # 4) Decoder stage (intra-stage SAPQ AttnRes)
            x = self.dec_stages[s](fused)

        return self.out_conv(x)


class ModelDecoderOnly(Model):
    def __init__(self, *args, **kwargs):
        kwargs['encoder_attn_res'] = False
        kwargs['decoder_attn_res'] = True
        super().__init__(*args, **kwargs)


class ModelBaseline(nn.Module):
    def __init__(self,
                 img_channels: int = 3,
                 n_classes: int = 1,
                 channels: tuple = (18, 36, 72, 144, 144),
                 n_blocks_enc: tuple = (2, 2, 3, 3, 3),
                 n_blocks_dec: tuple = (2, 2, 2, 2)):
        super().__init__()
        n_enc = len(channels)
        n_dec = n_enc - 1

        self.channels = channels
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.enc_stages = nn.ModuleList()
        for s in range(n_enc):
            in_c = img_channels if s == 0 else channels[s - 1]
            blocks = [DownAndUp(in_c, channels[s])]
            for _ in range(n_blocks_enc[s] - 1):
                blocks.append(DownAndUp(channels[s], channels[s]))
            self.enc_stages.append(nn.Sequential(*blocks))

        # Decoder
        dec_out_channels = []
        for s in range(n_dec):
            dec_out_channels.append(channels[n_enc - 3 - s] if s < n_dec - 1 else channels[0])

        self.dec_convs = nn.ModuleList()
        for s in range(n_dec):
            enc_skip_c = channels[n_enc - 2 - s]
            in_c = enc_skip_c * 2  # concat(D_up, skip)
            out_c = dec_out_channels[s]
            blocks = [DownAndUp(in_c, out_c)]
            for _ in range(n_blocks_dec[s] - 1):
                blocks.append(DownAndUp(out_c, out_c))
            self.dec_convs.append(nn.Sequential(*blocks))

        self.out_conv = nn.Conv2d(dec_out_channels[-1], n_classes, 1)

    def forward(self, x, mode=None):
        enc_outs = []
        for s, stage in enumerate(self.enc_stages):
            if s > 0:
                x = self.maxpool(x)
            x = stage(x)
            enc_outs.append(x)

        n_enc = len(self.enc_stages)
        for s in range(len(self.dec_convs)):
            skip_idx = n_enc - 2 - s
            x = self.upsample(x)
            target = enc_outs[skip_idx]
            diffY = target.size(2) - x.size(2)
            diffX = target.size(3) - x.size(3)
            if diffY != 0 or diffX != 0:
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([target, x], dim=1)
            x = self.dec_convs[s](x)

        return self.out_conv(x)