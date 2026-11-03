import torch
import torch.nn as nn

class DynamicConditionalRouter(nn.Module):
    def __init__(self, embedding_dim=5120, hidden_dim=512, num_layers=40, max_skip=16):
        super().__init__()
        self.num_layers = num_layers
        
        # 条件编码：将要剪枝的层数 N (1~16) 映射为嵌入向量
        self.depth_emb = nn.Embedding(max_skip + 1, hidden_dim)
        
        # 文本表征降维
        self.feature_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 融合网络
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_layers)
        )

    def forward(self, features, target_k, src_padding_mask=None):
        # features: (Batch, Seq_Len, Embedding_Dim)
        # target_k: (Batch,)
        
        # 平均池化提取句子级上下文 (排除 padding)
        if src_padding_mask is not None:
            mask_expanded = (~src_padding_mask).unsqueeze(-1).float()
            pooled_feat = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled_feat = features.mean(dim=1)
            
        proj_feat = self.feature_proj(pooled_feat) # (Batch, hidden_dim)
        cond = self.depth_emb(target_k)            # (Batch, hidden_dim)
        
        # 融合目标跳过深度信号
        fusion = proj_feat + cond
        logits = self.classifier(fusion)          # (Batch, num_layers)
        return logits