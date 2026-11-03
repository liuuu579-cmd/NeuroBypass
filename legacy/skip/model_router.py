# model_router.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RouterTransformer(nn.Module):
    def __init__(self, input_dim: int, num_opt_layers: int = 40, nhead: int = 8, num_encoder_layers: int = 4, d_hid: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Attention Pooling 层
        self.attention_pool = nn.Linear(input_dim, 1)
        
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, d_hid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hid // 2, num_opt_layers)
        )
        self.input_dim = input_dim

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src * math.sqrt(self.input_dim))
        src = src.permute(1, 0, 2)

        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask) # [B, S, D]
        
        attn_weights = self.attention_pool(output) # [B, S, 1]
        
        if src_padding_mask is not None:
            # 【修复点】：使用 -1e4 替代 -1e9，防止 float16 溢出
            attn_weights = attn_weights.masked_fill(src_padding_mask.unsqueeze(-1), -1e4)
            
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        pooled_output = torch.sum(output * attn_weights, dim=1) # [B, D]
        
        logits = self.output_head(pooled_output) # [B, 40]
        return logits
        

class LightweightRouter(nn.Module):
    def __init__(self, input_dim: int = 5120, num_opt_layers: int = 40, d_hid: int = 1024):
        super().__init__()
        # 【极其重要】：大模型的中间层必须先做 LayerNorm，否则数值极速膨胀导致 Loss 卡死
        self.norm = nn.LayerNorm(input_dim)
        
        # 简单的 Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # 轻量级 MLP 预测器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, d_hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hid, d_hid // 2),
            nn.GELU(),
            nn.Linear(d_hid // 2, num_opt_layers)
        )

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # src: [B, S, D]
        src = self.norm(src)
        
        attn_weights = self.attn_pool(src) # [B, S, 1]
        
        if src_padding_mask is not None:
            # Mask 掉 padding 部分
            attn_weights = attn_weights.masked_fill(src_padding_mask.unsqueeze(-1), -1e4)
            
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled_output = torch.sum(src * attn_weights, dim=1) # [B, D]
        
        logits = self.classifier(pooled_output) # [B, 40]
        return logits