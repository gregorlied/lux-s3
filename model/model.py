import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class ScaleOrShift(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.gamma1 = ScaleOrShift(input_dim, dropout)
        self.alpha1 = ScaleOrShift(input_dim, dropout)
        self.beta1 = ScaleOrShift(input_dim, dropout)

        self.gamma2 = ScaleOrShift(input_dim, dropout)
        self.alpha2 = ScaleOrShift(input_dim, dropout)
        self.beta2 = ScaleOrShift(input_dim, dropout)

        # Two-layer MLP
        self.mlp_layer1 = nn.Linear(input_dim, dim_feedforward)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_act = nn.GELU()
        self.mlp_layer2 = nn.Linear(dim_feedforward, input_dim)

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_cond, mask=None):

        # Scale and Shift
        gamma1 = self.gamma1(x_cond).unsqueeze(1)
        alpha1 = self.alpha1(x_cond).unsqueeze(1)
        beta1 = self.beta1(x_cond).unsqueeze(1)

        gamma2 = self.gamma2(x_cond).unsqueeze(1)
        alpha2 = self.alpha2(x_cond).unsqueeze(1)
        beta2 = self.beta2(x_cond).unsqueeze(1)

        # Attention part
        pre_attn = self.norm1(x) * gamma1 + beta1
        attn_out = self.self_attn(pre_attn, mask=mask)
        x = x + self.dropout(attn_out) * alpha1

        # MLP
        pre_linear = self.norm2(x) * gamma2 + beta2
        linear_out = self.mlp_layer1(pre_linear)
        linear_out = self.mlp_dropout(linear_out)
        linear_out = self.mlp_act(linear_out)
        linear_out = self.mlp_layer2(linear_out)
        x = x + self.dropout(linear_out) * alpha2

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, x_cond, mask=None):
        for layer in self.layers:
            x = layer(x, x_cond, mask=mask)
        return x
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid(self.fc2(self.act(self.fc1(y))))
        y = y.reshape(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se_layer = SELayer(out_channels)
        if in_channels != out_channels:
            self.reshape = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.reshape = nn.Identity()
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.bn2(self.conv2(x))
        x = self.se_layer(x)
        x = x + self.reshape(residual)
        x = self.act2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels),
            *[ResidualBlock(out_channels, out_channels) for i in range(1, num_blocks)],
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, num_blocks=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                ResidualBlock(in_channels, out_channels, in_channels // 2),
                *[ResidualBlock(out_channels, out_channels) for i in range(1, num_blocks)],
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                *[ResidualBlock(out_channels, out_channels) for i in range(1, num_blocks)],
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_channels, num_global_channels, num_classes, num_agents, num_blocks=1, bilinear=True, dit_head_bool=False):
        super(UNet, self).__init__()
        self.num_channels = num_channels
        self.num_global_channels = num_global_channels
        self.num_classes = num_classes
        self.num_agents = num_agents
        self.bilinear = bilinear
        self.dit_head_bool = dit_head_bool
        
        factor = 2 if bilinear else 1

        self.cond_embed = nn.Embedding(num_agents, 64)

        self.inc = ResidualBlock(num_channels, 64)
        self.down1 = Down(64, 128, num_blocks=num_blocks)
        self.down2 = Down(128, 256, num_blocks=num_blocks)
        self.down3 = Down(256, 512 // factor, num_blocks=num_blocks)
        self.up1 = Up(512 + num_global_channels, 256 // factor, bilinear=bilinear, num_blocks=num_blocks)
        self.up2 = Up(256, 128 // factor, bilinear=bilinear, num_blocks=num_blocks)
        self.up3 = Up(128, 64, bilinear=bilinear, num_blocks=num_blocks)

        if self.dit_head_bool:
            self.encoder = TransformerEncoder(num_layers=4, input_dim=64, num_heads=4, dim_feedforward=4*64)
            self.head = nn.Linear(64, num_classes)
        else:
            self.heads = nn.ModuleList([OutConv(64, num_classes) for _ in range(num_agents)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, agent_ids):
        b, _, h, w = x.shape

        x, x_global = x[:, :self.num_channels, :, :], x[:, self.num_channels:, :, :]
        x_cond = self.cond_embed(agent_ids)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = torch.cat((x4, x_global[:, :, :x4.shape[2], :x4.shape[3]]), dim=1)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        if self.dit_head_bool:
            x = x.permute(0, 2, 3, 1)                               # (b, h, w, c)
            x = x.reshape(b, h * w, -1)                             # (b, h * w, c)
            mask = mask.reshape(b, h * w)                           # (b, h * w)
            mask = mask[:, None, None, :]                           # (b, 1, 1, h * w)
            x = self.encoder(x, x_cond, mask=mask)                  # (b, h * w, c)

            logits = self.head(x)                                   # (b, h * w, num_classes)
            logits = logits.reshape(b, h, w, -1)                    # (b, h, w, num_classes)
            logits = logits.permute(0, 3, 1, 2)                     # (b, num_classes, h, w)
            return logits

        logits = []
        for head in self.heads:
            out = head(x)                                           # (b, num_classes, h, w)
            logits.append(out.unsqueeze(1))                         # (b, 1, num_classes, h, w)
        logits = torch.cat(logits, dim=1)                           # (b, num_agents, h * w, num_classes)

        agent_ids = agent_ids.reshape(-1, 1, 1, 1, 1)               # (b, 1, 1, 1, 1)
        agent_ids = agent_ids.expand(b, 1, self.num_classes, h, w)  # (b, 1, num_classes, h, w)
        logits = torch.gather(logits, dim=1, index=agent_ids)       # (b, 1, num_classes, h, w)
        logits = logits.squeeze(1)                                  # (b, num_classes, h, w)
        return logits
