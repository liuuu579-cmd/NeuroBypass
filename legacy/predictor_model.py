# predictor_model.py
import torch
import torch.nn as nn
import math

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
    def __init__(self,
                 input_dim: int = 5120,    # D_model of OPT-13B
                 predictor_dim: int = 768,   # Internal dimension of our predictor
                 n_layers: int = 2,        # Number of Transformer Encoder layers
                 n_heads: int = 8,         # Number of attention heads
                 num_opt_layers: int = 40, # Number of layers in the target LLM (OPT-13B)
                 opt_head_dim: int = 128,  # Head_Dim of the target LLM
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. 降维: 从OPT的大维度投影到我们轻量级模型的维度
        self.input_projection = nn.Linear(input_dim, predictor_dim)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(predictor_dim, dropout)
        
        # 3. 轻量级Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=n_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            batch_first=True  # ！！！重要：让输入/输出为 [batch, seq, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. 输出头: 将最终的表征映射到目标预测形状
        # 目标是为每一层生成一个代理查询向量 (Num_Layers, Head_Dim)
        self.output_head = nn.Linear(predictor_dim, num_opt_layers * opt_head_dim)

        self.num_opt_layers = num_opt_layers
        self.opt_head_dim = opt_head_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [Batch, Seq_Len, D_model_opt] -> e.g., [16, 2000, 5120]
        
        # 1. 降维
        x = self.input_projection(embeddings)  # -> [B, S, predictor_dim] e.g., [16, 2000, 768]
        
        # 2. 添加位置编码 (需要调整形状以匹配非batch_first的pos_encoder)
        # PyTorch的PositionalEncoding默认是 [S, B, D], 而我们的数据是 [B, S, D]
        x = x.permute(1, 0, 2) # -> [S, B, predictor_dim]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # -> [B, S, predictor_dim]
        
        # 3. 通过Transformer编码器
        encoded_seq = self.transformer_encoder(x) # -> [B, S, predictor_dim]
        
        # 4. 提取序列的最终表征 (使用最后一个token的输出)
        # 这代表了整个序列处理完毕后，用于预测下一步的"意图"
        last_token_repr = encoded_seq[:, -1, :] # -> [B, predictor_dim]
        
        # 5. 经过输出头生成预测
        predictions = self.output_head(last_token_repr) # -> [B, num_opt_layers * opt_head_dim]
        
        # 6. Reshape为最终的代理查询矩阵
        # -> [B, num_opt_layers, opt_head_dim] e.g., [16, 40, 128]
        predicted_query_proxy = predictions.view(-1, self.num_opt_layers, self.opt_head_dim)
        
        return predicted_query_proxy