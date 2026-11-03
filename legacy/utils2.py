# utils2.py
import torch
import torch.nn as nn

class AttentionStateCatcher:
    """
    一个用于捕获 Transformer 模型内部状态的类。
    - 捕获原始的 Token Embeddings。
    - 捕获每一层注意力模块的 Q 和 K 投影。
    """
    def __init__(self, model):
        self.model = model
        # 定位到关键模块
        self._embedding_layer = model.model.decoder.embed_tokens
        self._attention_layers = model.model.decoder.layers
        # 初始化
        self.captured_data = {}
        self._hooks = []

    def _capture_embeddings_hook(self, module, args, output):
        """钩子函数: 捕获 embedding 层的输出。"""
        # output shape: (batch_size, seq_len, hidden_dim)
        self.captured_data['embeddings'] = output.detach().cpu()

    def _capture_projection_hook(self, name, layer_idx):
        """
        创建一个钩子函数，用于捕获指定层 (q_proj 或 k_proj) 的输出。
        """
        def hook(module, args, output):
            # 确保字典和列表已初始化
            if name not in self.captured_data:
                self.captured_data[name] = [None] * len(self._attention_layers)
            
            # output shape: (batch_size, seq_len, hidden_dim)
            self.captured_data[name][layer_idx] = output.detach().cpu()
        return hook

    def register_hooks(self):
        """
        在 Embedding 层和每一层的 Q/K 投影上注册前向钩子。
        """
        self.reset()
        
        # 1. 注册 Embedding 钩子
        hook = self._embedding_layer.register_forward_hook(self._capture_embeddings_hook)
        self._hooks.append(hook)
        
        # 2. 注册每一层注意力中的 Q 和 K 投影钩子
        for i, layer in enumerate(self._attention_layers):
            q_proj_hook = layer.self_attn.q_proj.register_forward_hook(self._capture_projection_hook('queries', i))
            k_proj_hook = layer.self_attn.k_proj.register_forward_hook(self._capture_projection_hook('keys', i))
            self._hooks.extend([q_proj_hook, k_proj_hook])
            
    def remove_hooks(self):
        """移除所有已注册的钩子，防止内存泄漏。"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_captured_data(self):
        """
        整理捕获的数据，特别是将 Q/K 张量重塑为多头注意力形式。
        """
        if 'queries' not in self.captured_data or 'keys' not in self.captured_data or 'embeddings' not in self.captured_data:
            print("Warning: Not all required data was captured.")
            return None

        # 从模型配置中获取注意力头的数量和维度
        config = self.model.config
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        
        # 创建一个新的字典来存放处理后的数据
        processed_data = {}

        # --- 核心修正逻辑 ---
        for name in ['queries', 'keys']:
            # 1. 将层的列表堆叠成一个张量。
            #    初始形状: (num_layers, batch_size, seq_len, hidden_size)
            stacked_tensor = torch.stack(self.captured_data[name], dim=0)
            
            # 2. 获取维度信息
            num_layers, batch_size, seq_len, _ = stacked_tensor.shape

            # 3. 重塑张量，将 hidden_size 拆分为 num_heads 和 head_dim
            #    形状变为: (num_layers, batch_size, seq_len, num_heads, head_dim)
            viewed_tensor = stacked_tensor.view(num_layers, batch_size, seq_len, num_heads, head_dim)
            
            # 4. 调整维度顺序以匹配常见的注意力计算方式 (虽然对于你的 calculate_target_scores 不是必须的，但这是个好习惯)
            #    形状变为: (batch_size, num_layers, num_heads, seq_len, head_dim)
            permuted_tensor = viewed_tensor.permute(1, 0, 3, 2, 4)
            
            # 5. 移除 batch 维度 (因为在数据生成时 batch_size=1)
            #    最终形状: (num_layers, num_heads, seq_len, head_dim)
            #    注意: 你的 calculate_target_scores 函数期望的输入是 (L, S, H, D_h)，所以我们需要再次调整
            final_tensor = permuted_tensor.squeeze(0).permute(0, 2, 1, 3) # (L, H, S, D_h) -> (L, S, H, D_h)
            
            processed_data[name] = final_tensor

        # 同样移除 embeddings 的 batch 维度
        processed_data['embeddings'] = self.captured_data['embeddings'].squeeze(0)
            
        return processed_data

    def reset(self):
        """清空捕获的数据，为下一次运行做准备。"""
        self.captured_data = {}