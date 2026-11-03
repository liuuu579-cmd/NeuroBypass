# predictor_model2B.py
import torch
import torch.nn as nn
import math


class PredictorMLP(nn.Module):
    def __init__(self, input_dim=5120, num_opt_layers=40, hidden_dims=[2048, 1024], dropout=0.1):
        super().__init__()
        self.num_opt_layers = num_opt_layers
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, num_opt_layers)

    def forward(self, src, src_padding_mask=None):
        # src shape: (B, S, E)
        B, S, E = src.shape
        
        # 将 batch 和 sequence 合并，让每个 token 独立通过 MLP
        src_flat = src.view(B * S, E)
        
        hidden_output = self.hidden_layers(src_flat)
        scores_flat = self.output_layer(hidden_output) # shape: (B*S, NUM_LAYERS)
        
        # 恢复形状
        scores = scores_flat.view(B, S, self.num_opt_layers) # shape: (B, S, L)
        
        # 调整为损失函数期望的形状 (B, L, S)
        scores = scores.permute(0, 2, 1)
        
        return scores
class PositionalEncoding(nn.Module):
    """标准的Transformer位置编码"""
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class PredictorTransformer(nn.Module):
    def __init__(self, input_dim: int, num_opt_layers: int, nhead: int = 8, num_encoder_layers: int = 4, d_hid: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # <<< MODIFICATION START >>>
        # 将单一的线性输出头替换为一个更强大的MLP
        # 这为最终的决策步骤提供了更多的非线性能力
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, d_hid // 2), # 第一个线性层，可以适当减小维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid // 2, num_opt_layers) # 第二个线性层，输出最终logits
        )
        # <<< MODIFICATION END >>>
        
        self.input_dim = input_dim
        self.num_opt_layers = num_opt_layers

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # forward函数无需任何修改
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src * math.sqrt(self.input_dim))
        src = src.permute(1, 0, 2)

        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask) 
        
        scores = self.output_head(output)
        
        scores = scores.permute(0, 2, 1)
        
        return scores