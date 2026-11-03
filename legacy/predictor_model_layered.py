# predictor_model_layered.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLayerPredictorMLP(nn.Module):
    """
    一个专门为单个注意力层预测分数排序的MLP。
    它包含LayerNorm以提高训练稳定性。
    
    输入: 最后一个token的embedding, shape: [BatchSize, D_MODEL]
    输出: 对整个序列的预测分数, shape: [BatchSize, SEQ_LEN]
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[2048, 2048], dropout=0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            # 关键改进: 在线性和激活之间加入LayerNorm
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x 的 shape 应该是 [B, D_MODEL]
        return self.mlp(x)