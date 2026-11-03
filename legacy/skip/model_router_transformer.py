import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConditionalRouterTransformer(nn.Module):
    """
    基于 Transformer Encoder 与 K-Conditioning Token 的动态路由网络
    输入: Layer 1 特征 [B, S, D_MODEL] (D_MODEL=5120)
    输出: OPT-13B 各层的路由 Logits [B, NUM_OPT_LAYERS]
    """
    def __init__(
        self,
        input_dim: int = 5120,       # OPT-13B Hidden Size
        hidden_dim: int = 1024,      # Router Transformer 内部维度
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        num_opt_layers: int = 40,    # OPT-13B 总层数
        max_skip: int = 16,          # 最大剪枝步数
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_opt_layers = num_opt_layers
        self.max_skip = max_skip

        # 1. 输入投影层：5120 -> hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.proj_ln = nn.LayerNorm(hidden_dim)

        # 2. 条件 Token 嵌入 (1 到 max_skip)
        # 索引 0 作为 padding/占位，有效 K 取 1..max_skip
        self.k_embedding = nn.Embedding(max_skip + 1, hidden_dim)

        # 3. 2 层 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 结构更利于训练稳定性
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 4. 路由分类头 (基于 Conditioning Token 隐状态)
        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_opt_layers)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.k_embedding.weight, std=0.02)
        for p in self.router_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeddings: torch.Tensor, target_k: torch.Tensor, src_padding_mask: torch.Tensor = None):
        """
        参数:
            embeddings: [Batch, Seq_Len, 5120] (OPT Layer 1 隐藏状态)
            target_k: [Batch] 整型 Tensor (取值 1 ~ max_skip)
            src_padding_mask: [Batch, Seq_Len] 布尔类型 (True 代表填充位 Padding)
        返回:
            logits: [Batch, num_opt_layers] (未经过 Sigmoid 的原始得分)
        """
        # 锁定在 float32 运算，避免 FP16 下溢
        embeddings = embeddings.float()
        B, S, _ = embeddings.shape

        # 1. 序列投影
        h_seq = self.proj_ln(self.in_proj(embeddings))  # [B, S, hidden_dim]

        # 2. 注入 K-Conditioning Token
        # target_k 裁剪到合法范围
        clamped_k = torch.clamp(target_k.long(), min=1, max=self.max_skip)
        k_tokens = self.k_embedding(clamped_k).unsqueeze(1)  # [B, 1, hidden_dim]

        # 拼在序列最前面 (SeqLen 变为 S + 1)
        h = torch.cat([k_tokens, h_seq], dim=1)  # [B, 1 + S, hidden_dim]

        # 3. 构造 Padding Mask (Conditioning Token 永远有效，即 False)
        if src_padding_mask is not None:
            cond_mask = torch.zeros((B, 1), dtype=torch.bool, device=embeddings.device)
            full_key_padding_mask = torch.cat([cond_mask, src_padding_mask.bool()], dim=1)
        else:
            full_key_padding_mask = None

        # 4. Transformer 全注意力交互
        h_out = self.encoder(h, src_key_padding_mask=full_key_padding_mask)
        h_out = self.final_norm(h_out)

        # 5. 提取第 0 个位置 (Conditioning Token) 作为路由决策表征
        cond_rep = h_out[:, 0, :]  # [B, hidden_dim]

        # 6. 计算层路由 Logits
        logits = self.router_head(cond_rep)  # [B, num_opt_layers]
        return logits