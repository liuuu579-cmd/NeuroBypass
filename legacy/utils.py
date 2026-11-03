#utils.py
import torch

class AttentionStateCatcher:
    def __init__(self, model):
        self.model = model
        self.captured_data = {}
        self._hooks = []
        self._embedding_layer = model.model.decoder.embed_tokens
        self._attention_layers = model.model.decoder.layers

    def _capture_embeddings(self, module, args, output):
        self.captured_data['embeddings'] = output.detach().cpu()

    def _capture_tensors(self, name, layer_idx):
        def hook(module, args, output):
            if name not in self.captured_data:
                self.captured_data[name] = [None] * len(self._attention_layers)
            # The output of q_proj/k_proj is shape (batch, seq, hidden_size)
            self.captured_data[name][layer_idx] = output.detach().cpu()
        return hook

    def register_hooks(self):
        self.reset()
        hook = self._embedding_layer.register_forward_hook(self._capture_embeddings)
        self._hooks.append(hook)
        
        for i, layer in enumerate(self._attention_layers):
            q_proj_hook = layer.self_attn.q_proj.register_forward_hook(self._capture_tensors('queries', i))
            k_proj_hook = layer.self_attn.k_proj.register_forward_hook(self._capture_tensors('keys', i))
            self._hooks.extend([q_proj_hook, k_proj_hook])
            
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_captured_data(self):
        if 'queries' not in self.captured_data or 'keys' not in self.captured_data:
            return None

        # --- 关键的修复在这里 ---
        config = self.model.config
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        for name in ['queries', 'keys']:
            # 1. Stack the list of tensors. 
            #    Shape becomes: (num_layers, batch, seq_len, hidden_size) -> 4D
            stacked_tensor = torch.stack(self.captured_data[name], dim=0)
            
            # 2. Get batch_size and seq_len for the view operation
            num_layers, batch_size, seq_len, _ = stacked_tensor.shape
            
            # 3. Manually split the hidden_size dimension into num_heads and head_dim.
            #    This is the step our hook missed.
            #    Shape becomes: (num_layers, batch, seq_len, num_heads, head_dim) -> 5D
            viewed_tensor = stacked_tensor.view(num_layers, batch_size, seq_len, num_heads, head_dim)
            
            # 4. Permute the dimensions to match the Zarr storage layout.
            #    Target shape: (batch, num_layers, seq_len, num_heads, head_dim)
            permuted_tensor = viewed_tensor.permute(1, 0, 2, 3, 4)
            
            # 5. Squeeze the batch dimension (since it's 1) and store.
            #    Final shape: (num_layers, seq_len, num_heads, head_dim)
            self.captured_data[name] = permuted_tensor.squeeze(0)

        self.captured_data['embeddings'] = self.captured_data['embeddings'].squeeze(0)
            
        return self.captured_data

    def reset(self):
        self.captured_data = {}